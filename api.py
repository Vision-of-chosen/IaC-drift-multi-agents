#!/usr/bin/env python3
"""
FastAPI Web Application for Terraform Drift Detection & Remediation System

This module provides a web-based API interface that coordinates the multi-agent
system with step-by-step user interaction between each agent execution.

Architecture:
- Step-by-step agent coordination with user approval
- RESTful API endpoints for each workflow step
- Session management for workflow state
- Real-time progress updates
"""

import logging
import os
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import boto3
from boto3.session import Session as Boto3Session
os.environ["BYPASS_TOOL_CONSENT"] = "True"
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import json
import glob
import zipfile
import tempfile
import shutil

# Add tools to path
sys.path.append("tools/src")

from strands.models.bedrock import BedrockModel
from agents import OrchestrationAgent, DetectAgent, DriftAnalyzerAgent, RemediateAgent, ReportAgent, NotificationAgent
from shared_memory import shared_memory
from config import BEDROCK_MODEL_ID, BEDROCK_REGION, TERRAFORM_DIR
from useful_tools.terraform_tools import terraform_plan, terraform_apply

# Configure database-backed shared memory if enabled
USE_DB_STORAGE = os.environ.get("USE_DB_STORAGE", "false").lower() == "true"
DB_STORAGE_PATH = os.environ.get("DB_STORAGE_PATH", "shared_memory.db")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Terraform Drift Detection & Remediation API",
    description="""
    Step-by-step coordination of drift detection, analysis, and remediation agents
    
    Features:
    - Traditional REST API endpoints for one-time requests
    - Multi-agent orchestration with intelligent routing
    - Session-based conversation management
    """,
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TerraformUploadResponse(BaseModel):
    message: str
    filename: str
    session_id: str
    extracted_files: list[str] = Field(default_factory=list, description="List of .tf files extracted from the upload")
    terraform_plan_result: Dict[str, Any]
    terraform_apply_result: Dict[str, Any]
    success: bool
    timestamp: datetime

# Global state management
session_states: Dict[str, Dict[str, Any]] = {}



# Pydantic Models
class ChatMessage(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="User ID for AWS credential association")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    routed_agent: Optional[str] = Field(None, description="Which agent was invoked, if any")
    agent_result: Optional[str] = Field(None, description="Result from the invoked agent")
    conversation_state: str
    timestamp: datetime
    suggestions: list[str] = Field(default_factory=list, description="Suggested next actions")


class WorkflowStatus(BaseModel):
    session_id: str
    conversation_state: str
    message: str
    timestamp: datetime
    shared_memory_keys: list[str]


class SystemStatus(BaseModel):
    total_sessions: int
    active_sessions: int
    system_health: str
    terraform_dir: str

class AWSCredentials(BaseModel):
    aws_access_key_id: str = Field(..., description="AWS Access Key ID")
    aws_secret_access_key: str = Field(..., description="AWS Secret Access Key")
    aws_region: str = Field("ap-southeast-2", description="AWS Region")
    user_id: Optional[str] = Field(None, description="User ID for credential association")

class NotificationConfig(BaseModel):
    recipient_email: str = Field(..., description="Email address to receive notifications")
    resource_types: Optional[List[str]] = Field(None, description="List of AWS resource types to monitor")
    setup_aws_config: bool = Field(False, description="Whether to set up AWS Config for drift detection")
    include_global_resources: bool = Field(True, description="Whether to include global resources in monitoring")

# Chat-based Orchestrator Class
class ChatOrchestrator:
    """Intelligent conversation orchestrator that routes messages to appropriate agents"""
    
    def __init__(self):
        self.model = self._create_model()
        self.agents = self._create_agents()
        self._boto3_sessions: Dict[str, Boto3Session] = {} # Dictionary to store boto3 sessions
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _create_model(self) -> BedrockModel:
        """Create Bedrock model instance"""
        return BedrockModel(
            model_id=BEDROCK_MODEL_ID,
            region_name=BEDROCK_REGION,
        )
    
    def _create_agents(self) -> Dict[str, Any]:
        """Create all agent instances"""
        return {
            'orchestration': OrchestrationAgent(self.model),
            'detect': DetectAgent(self.model),
            'analyzer': DriftAnalyzerAgent(self.model),
            'remediate': RemediateAgent(self.model),
            'report': ReportAgent(self.model),
            'notification': NotificationAgent(self.model)
        }
    
    async def initialize_session(self, session_id: Optional[str] = None) -> str:
        """Initialize a new conversation session"""
        if session_id and session_id in session_states:
            # Set the current session in shared memory
            shared_memory.set_session(session_id)
            return session_id
        
        session_id = str(uuid.uuid4())
        
        # Initialize session state
        session_state = {
            "session_id": session_id,
            "conversation_state": "idle",
            "conversation_history": [],
            "timestamp": datetime.now(),
            "context": {}
        }
        
        session_states[session_id] = session_state
        
        # Set the current session in shared memory
        shared_memory.set_session(session_id)
        
        # Initialize shared memory for this session
        shared_memory.set("state", "idle", session_id)
        shared_memory.set("history", [], session_id)
        
        return session_id
    
    async def process_chat_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """Process a chat message and route to appropriate agent if needed"""
        if session_id not in session_states:
            session_id = await self.initialize_session(session_id)
        
        # Set the current session in shared memory
        shared_memory.set_session(session_id)
        
        session_state = session_states[session_id]
        
        try:
            # Add user message to conversation history
            session_state["conversation_history"].append({
                "role": "user",
                "message": message,
                "timestamp": datetime.now()
            })
            
            # Update shared memory with conversation context (JSON-serializable only)
            serializable_history = self._make_json_serializable(session_state["conversation_history"])
            serializable_context = self._make_json_serializable(session_state.get("context", {}))
            
            shared_memory.set("history", serializable_history, session_id)
            shared_memory.set("current_user_message", message, session_id)
            shared_memory.set("context", serializable_context, session_id)
            
            # Let the orchestration agent decide what to do
            orchestration_agent = self.agents['orchestration'].get_agent()
            # Update shared memory using the proper method
            self.agents['orchestration'].update_shared_memory()
            
            # Create a context-aware prompt for the orchestrator
            context_prompt = f"""
User message: "{message}"

Conversation history: {session_state["conversation_history"][-3:]}  # Last 3 messages for context

Current session state: {session_state["conversation_state"]}

Based on this message and context, decide if you need to route to a specialized agent or handle this conversationally.

If routing to an agent:
1. Explain what agent you're invoking and why
2. Execute the agent
3. Summarize the results for the user
4. Suggest next steps

If handling conversationally:
1. Provide helpful guidance
2. Answer questions about the system
3. Ask clarifying questions if needed

You can also run multiple agents in parallel if needed for better performance.
"""
            
            # Execute orchestration agent
            orchestration_result = orchestration_agent(context_prompt)
            
            # Determine if orchestrator wants to route to a specific agent
            routed_agent = None
            agent_result = None
            
            orchestration_response = str(orchestration_result)
            
            # Check for parallel execution requests
            if "parallel" in orchestration_response.lower() and ("detect" in orchestration_response.lower() or "analyzer" in orchestration_response.lower()):
                # Execute detect and analyzer agents in parallel
                parallel_agents = []
                if "detect" in orchestration_response.lower():
                    parallel_agents.append("detect")
                if "analyzer" in orchestration_response.lower():
                    parallel_agents.append("analyzer")
                if "remediate" in orchestration_response.lower():
                    parallel_agents.append("remediate")
                if "report" in orchestration_response.lower():
                    parallel_agents.append("report")
                
                logger.info(f"Executing agents in parallel: {parallel_agents}")
                parallel_results = await self.execute_agents_in_parallel(parallel_agents, message, session_id)
                
                # Combine results
                routed_agent = "Multiple Agents (Parallel)"
                agent_result = "\n\n".join([f"{agent.upper()} RESULT:\n{result}" for agent, result in parallel_results.items()])
                
                # Update session state based on executed agents
                if "detect" in parallel_agents:
                    session_state["conversation_state"] = "detection_complete"
                if "analyzer" in parallel_agents and "detect" in parallel_agents:
                    session_state["conversation_state"] = "analysis_complete"
                if "remediate" in parallel_agents:
                    session_state["conversation_state"] = "remediation_complete"
                if "report" in parallel_agents:
                    session_state["conversation_state"] = "report_complete"
                
            # Check if orchestrator indicated it wants to route to a specific agent
            elif "routing to DetectAgent" in orchestration_response.lower() or "invoking detectagent" in orchestration_response.lower():
                routed_agent = "DetectAgent"
                agent_result = self._execute_agent('detect', message, session_id)
                session_state["conversation_state"] = "detection_complete"
                
            elif "routing to driftanalyzeragent" in orchestration_response.lower() or "invoking analyzer" in orchestration_response.lower():
                routed_agent = "DriftAnalyzerAgent"
                agent_result = self._execute_agent('analyzer', message, session_id)
                session_state["conversation_state"] = "analysis_complete"
                
            elif "routing to remediateagent" in orchestration_response.lower() or "invoking remediate" in orchestration_response.lower():
                routed_agent = "RemediateAgent"
                agent_result = self._execute_agent('remediate', message, session_id)
                session_state["conversation_state"] = "remediation_complete"

            elif "routing to reportagent" in orchestration_response.lower() or "invoking report" in orchestration_response.lower():
                routed_agent = "ReportAgent"
                agent_result = self._execute_agent('report', message, session_id)
                session_state["conversation_state"] = "report_complete"
            
            # Generate suggestions based on current state
            suggestions = self._generate_suggestions(session_state["conversation_state"], routed_agent)
            
            # Add assistant response to conversation history
            session_state["conversation_history"].append({
                "role": "assistant",
                "message": orchestration_response,
                "routed_agent": routed_agent,
                "agent_result": agent_result,
                "timestamp": datetime.now()
            })
            
            # Update session timestamp
            session_state["timestamp"] = datetime.now()
            
            # Update session state in shared memory
            shared_memory.set("state", session_state["conversation_state"], session_id)
            
            return {
                "session_id": session_id,
                "response": orchestration_response,
                "routed_agent": routed_agent,
                "agent_result": agent_result[:1000] + "..." if agent_result and len(agent_result) > 1000 else agent_result,
                "conversation_state": session_state["conversation_state"],
                "timestamp": datetime.now(),
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            error_response = f"I encountered an error while processing your message: {str(e)}. Please try again or rephrase your request."
            
            session_state["conversation_history"].append({
                "role": "assistant",
                "message": error_response,
                "error": True,
                "timestamp": datetime.now()
            })
            
            return {
                "session_id": session_id,
                "response": error_response,
                "routed_agent": None,
                "agent_result": None,
                "conversation_state": session_state["conversation_state"],
                "timestamp": datetime.now(),
                "suggestions": ["Try rephrasing your request", "Ask for help", "Check system status"]
            }

    
    def _execute_agent(self, agent_type: str, user_message: str, session_id: str = None) -> str:
        """Execute a specific agent and return its result"""
        try:
            # Set the current session in shared memory
            if session_id:
                shared_memory.set_session(session_id)
            
            agent = self.agents[agent_type].get_agent()
            
            # Update agent's shared memory
            self.agents[agent_type].update_shared_memory()
            
            # Check if there's a user_id associated with the current request
            user_id = shared_memory.get("current_user_id", session_id=session_id)
            
            # If user_id exists, get boto3 session for this agent execution
            if user_id:
                logger.info(f"Setting up AWS session for {agent_type} agent using user_id: {user_id}")
                
                # Get user-specific boto3 session
                user_session = get_user_boto3_session(user_id)
                
                if user_session:
                    # Store boto3 session in a global dictionary keyed by agent_type and session_id
                    # This makes it available for the agent to use
                    session_key = f"{agent_type}_{session_id}"
                    self._boto3_sessions[session_key] = user_session
                    
                    # Update shared memory with session information
                    shared_memory.set(f"{agent_type}_aws_session_available", True, session_id)
                    shared_memory.set(f"{agent_type}_aws_region", user_session.region_name or "ap-southeast-2", session_id)
            
            # Execute the agent
            result = agent(user_message)
            
            # Store result in shared memory
            shared_memory.set(f"{agent_type}_last_result", str(result), session_id)
            
            # Clean up the boto3 session reference after execution
            session_key = f"{agent_type}_{session_id}"
            if session_key in self._boto3_sessions:
                del self._boto3_sessions[session_key]
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error executing {agent_type} agent: {e}")
            return f"Error executing {agent_type}: {str(e)}"
            
    async def _execute_agent_async(self, agent_type: str, user_message: str, session_id: str = None) -> str:
        """Execute a specific agent asynchronously and return its result"""
        # Create a wrapper to run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: self._execute_agent(agent_type, user_message, session_id)
        )
        return result
        
    async def execute_agents_in_parallel(self, agent_types: List[str], user_message: str, session_id: str = None) -> Dict[str, str]:
        """Execute multiple agents in parallel and return their results"""
        tasks = []
        for agent_type in agent_types:
            if agent_type in self.agents:
                task = self._execute_agent_async(agent_type, user_message, session_id)
                tasks.append((agent_type, task))
            else:
                logger.warning(f"Agent type '{agent_type}' not found in available agents")
        
        # Execute all tasks in parallel
        results = {}
        for agent_type, task in tasks:
            try:
                result = await task
                results[agent_type] = result
            except Exception as e:
                logger.error(f"Error executing {agent_type} agent in parallel: {e}")
                results[agent_type] = f"Error: {str(e)}"
        
        return results
    

    
    def _generate_suggestions(self, conversation_state: str, last_routed_agent: Optional[str]) -> list[str]:
        """Generate contextual suggestions for the user"""
        if conversation_state == "idle":
            return [
                "Ask me to check for infrastructure drift",
                "Request a scan of your Terraform resources",
                "Get help with the system capabilities"
            ]
        elif conversation_state == "detection_complete":
            return [
                "Ask me to analyze the detected drift",
                "Request impact assessment of the changes",
                "Ask about the severity of the issues found"
            ]
        elif conversation_state == "analysis_complete":
            return [
                "Ask me to apply the recommended fixes",
                "Request remediation of specific issues",
                "Ask for more details about the recommendations"
            ]
        elif conversation_state == "remediation_complete":
            return [
                "Ask me to run another drift detection",
                "Request validation of the applied fixes",
                "Start a new drift detection session"
            ]
        else:
            return [
                "Ask for status update",
                "Request help with available commands",
                "Get system information"
            ]


# Initialize the orchestrator
orchestrator = ChatOrchestrator()


# API Endpoints
@app.get("/", summary="API Health Check")
async def root():
    """Health check endpoint"""
    return {
        "message": "Terraform Drift Detection & Remediation API",
        "status": "healthy",
        "version": "1.1.0",
        "terraform_dir": TERRAFORM_DIR,
        "endpoints": {
            "rest_chat": "/chat",
            "parallel_agents": "/parallel-agents",
            "test_client": "/test-client"
        },
        "features": {
            "parallel_execution": True,
            "async_processing": True,
            "multi_agent_coordination": True
        },
        "storage_mode": "database" if USE_DB_STORAGE else "memory",
        "db_path": DB_STORAGE_PATH if USE_DB_STORAGE else None
    }



async def get_session_id_from_header(x_session_id: Optional[str] = Header(None, alias="X-Session-ID")) -> Optional[str]:
    """Extract session ID from header."""
    return x_session_id

class ParallelAgentRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="User ID for AWS credential association")
    agents: List[str] = Field(..., description="List of agents to run in parallel")

class ParallelAgentResponse(BaseModel):
    session_id: str
    results: Dict[str, str] = Field(..., description="Results from each agent")
    conversation_state: str
    timestamp: datetime
    suggestions: list[str] = Field(default_factory=list, description="Suggested next actions")

@app.post("/chat", response_model=ChatResponse, summary="Chat with Terraform Drift Assistant")
async def chat(
    request: ChatMessage, 
    x_user_id: Optional[str] = Header(None),
    session_id_header: Optional[str] = Depends(get_session_id_from_header)
):
    """Send a message to the intelligent Terraform drift assistant"""
    try:
        # Initialize session if needed
        # Use session_id from header if available, otherwise from request
        session_id = session_id_header or request.session_id
        
        if not session_id:
            session_id = await orchestrator.initialize_session()
        else:
            # Set the current session in shared memory
            shared_memory.set_session(session_id)
        
        # Get user ID from request or header
        user_id = request.user_id or x_user_id
        
        # If user_id is provided, store it in session and shared memory
        if user_id:
            # Store user_id in session state for future reference
            if session_id in session_states:
                session_states[session_id]["user_id"] = user_id
                
            # Store user_id in shared memory for agents to use
            shared_memory.set("user_id", user_id, session_id)
            shared_memory.set("current_user_id", user_id, session_id)
            
            # Get boto3 session for this user if credentials exist
            user_session = get_user_boto3_session(user_id)
            if user_session:
                # Store session information in shared memory
                # (We don't store actual credentials in shared memory for security)
                shared_memory.set("aws_session_available", True, session_id)
                shared_memory.set("aws_region", user_session.region_name or "ap-southeast-2", session_id)
                
                logger.info(f"Using AWS credentials for user {user_id} in chat request")
        
        # Process the chat message
        result = await orchestrator.process_chat_message(session_id, request.message)
        
        # Add user_id to response if available
        if user_id:
            result["user_id"] = user_id
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in chat processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat message: {e}")

@app.post("/parallel-agents", response_model=ParallelAgentResponse, summary="Execute Multiple Agents in Parallel")
async def run_parallel_agents(
    request: ParallelAgentRequest,
    x_user_id: Optional[str] = Header(None),
    session_id_header: Optional[str] = Depends(get_session_id_from_header)
):
    """
    Execute multiple agents in parallel for better performance
    
    This endpoint:
    1. Takes a list of agent types to run in parallel
    2. Executes all specified agents concurrently
    3. Returns combined results from all agents
    
    Available agents: detect, analyzer, remediate, report, notification
    """
    try:
        # Initialize session if needed
        session_id = session_id_header or request.session_id
        
        if not session_id:
            session_id = await orchestrator.initialize_session()
        else:
            # Set the current session in shared memory
            shared_memory.set_session(session_id)
        
        # Get user ID from request or header
        user_id = request.user_id or x_user_id
        
        # If user_id is provided, store it in session and shared memory
        if user_id:
            # Store user_id in session state for future reference
            if session_id in session_states:
                session_states[session_id]["user_id"] = user_id
                
            # Store user_id in shared memory for agents to use
            shared_memory.set("user_id", user_id, session_id)
            shared_memory.set("current_user_id", user_id, session_id)
        
        # Validate requested agents
        valid_agents = ["detect", "analyzer", "remediate", "report", "notification"]
        agents_to_run = [agent for agent in request.agents if agent in valid_agents]
        
        if not agents_to_run:
            raise HTTPException(
                status_code=400, 
                detail=f"No valid agents specified. Available agents: {valid_agents}"
            )
        
        # Add user message to conversation history
        if session_id in session_states:
            session_state = session_states[session_id]
            session_state["conversation_history"].append({
                "role": "user",
                "message": request.message,
                "timestamp": datetime.now()
            })
            
            # Update shared memory with conversation context
            serializable_history = orchestrator._make_json_serializable(session_state["conversation_history"])
            shared_memory.set("history", serializable_history, session_id)
            shared_memory.set("current_user_message", request.message, session_id)
        
        # Execute agents in parallel
        logger.info(f"Executing agents in parallel: {agents_to_run}")
        parallel_results = await orchestrator.execute_agents_in_parallel(agents_to_run, request.message, session_id)
        
        # Update session state based on executed agents
        if session_id in session_states:
            session_state = session_states[session_id]
            
            if "detect" in agents_to_run:
                session_state["conversation_state"] = "detection_complete"
            if "analyzer" in agents_to_run and "detect" in agents_to_run:
                session_state["conversation_state"] = "analysis_complete"
            if "remediate" in agents_to_run:
                session_state["conversation_state"] = "remediation_complete"
            if "report" in agents_to_run:
                session_state["conversation_state"] = "report_complete"
            
            # Add assistant response to conversation history
            session_state["conversation_history"].append({
                "role": "assistant",
                "message": f"Executed {len(agents_to_run)} agents in parallel",
                "parallel_agents": agents_to_run,
                "timestamp": datetime.now()
            })
            
            # Generate suggestions based on current state
            suggestions = orchestrator._generate_suggestions(session_state["conversation_state"], "Multiple Agents")
            
            return ParallelAgentResponse(
                session_id=session_id,
                results=parallel_results,
                conversation_state=session_state["conversation_state"],
                timestamp=datetime.now(),
                suggestions=suggestions
            )
        else:
            raise HTTPException(status_code=404, detail="Session not found")
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error running parallel agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run parallel agents: {str(e)}")




@app.post("/start-session", summary="Start New Chat Session")
async def start_session():
    """Initialize a new conversation session"""
    try:
        session_id = await orchestrator.initialize_session()
        
        return {
            "session_id": session_id,
            "message": "New conversation session started. How can I help you with Terraform drift detection today?",
            "conversation_state": "idle",
            "timestamp": datetime.now(),
            "suggestions": [
                "Ask me to check for infrastructure drift",
                "Request a scan of your Terraform resources", 
                "Get help with the system capabilities"
            ]
        }
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start session: {e}")


@app.get("/status", response_model=WorkflowStatus, summary="Get Session Status")
async def get_session_status(
    session_id: Optional[str] = None,
    session_id_header: Optional[str] = Depends(get_session_id_from_header)
):
    """
    Get current status of a conversation session.
    
    Args:
        session_id: Optional session ID as query parameter
        session_id_header: Optional session ID from X-Session-ID header
    """
    # Use session_id from header if available, otherwise from parameter
    session_id = session_id_header or session_id
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
        
    if session_id not in session_states:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Set the current session in shared memory
    shared_memory.set_session(session_id)
    
    session_state = session_states[session_id]
    
    # Get session-specific memory keys
    session_keys = shared_memory.keys(session_id, include_global=False)
    
    # Get session state from shared memory
    memory_state = shared_memory.get("state", session_state["conversation_state"], session_id)
    
    return WorkflowStatus(
        session_id=session_id,
        conversation_state=memory_state,
        message=f"Session active. Current state: {memory_state}. {len(session_state.get('conversation_history', []))} messages in conversation.",
        timestamp=session_state["timestamp"],
        shared_memory_keys=session_keys
    )


@app.get("/conversation", summary="Get Conversation History")
async def get_conversation_history(
    session_id: Optional[str] = None,
    session_id_header: Optional[str] = Depends(get_session_id_from_header)
):
    """
    Get the conversation history for a session.
    
    Args:
        session_id: Optional session ID as query parameter
        session_id_header: Optional session ID from X-Session-ID header
    """
    # Use session_id from header if available, otherwise from parameter
    session_id = session_id_header or session_id
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
        
    if session_id not in session_states:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_state = session_states[session_id]
    
    return {
        "session_id": session_id,
        "conversation_history": session_state.get("conversation_history", []),
        "conversation_state": session_state["conversation_state"],
        "total_messages": len(session_state.get("conversation_history", [])),
        "timestamp": session_state["timestamp"]
    }


@app.get("/shared-memory", summary="View Shared Memory")
async def get_shared_memory(
    session_id: Optional[str] = None, 
    include_global: bool = True,
    session_id_header: Optional[str] = Depends(get_session_id_from_header)
):
    """
    Get current shared memory contents.
    
    Args:
        session_id: Optional session ID as query parameter
        include_global: Whether to include global (non-session) memory
        session_id_header: Optional session ID from X-Session-ID header
    """
    # Use session_id from header if available, otherwise from parameter
    session_id = session_id_header or session_id
    
    if session_id:
        # Set the current session in shared memory
        shared_memory.set_session(session_id)
        
        return {
            "shared_memory": shared_memory.get_all(session_id, include_global),
            "keys": shared_memory.keys(session_id, include_global),
            "size": shared_memory.size(session_id),
            "session_id": session_id,
            "storage_mode": "database" if USE_DB_STORAGE else "memory",
            "timestamp": datetime.now()
        }
    else:
        # Get all sessions
        sessions = shared_memory.get_all_sessions()
        
        # Get global memory
        global_memory = shared_memory.get_all()
        
        return {
            "shared_memory": global_memory,
            "keys": shared_memory.keys(),
            "size": shared_memory.size(),
            "available_sessions": sessions,
            "session_count": len(sessions),
            "storage_mode": "database" if USE_DB_STORAGE else "memory",
            "timestamp": datetime.now()
        }




@app.get("/system-status", response_model=SystemStatus, summary="Get System Status")
async def get_system_status():
    """Get overall system status"""
    active_sessions = len([s for s in session_states.values() 
                          if s.get("workflow_status") not in ["completed", "failed"]])
    
    return SystemStatus(
        total_sessions=len(session_states),
        active_sessions=active_sessions,
        system_health="healthy",
        terraform_dir=TERRAFORM_DIR
    )

@app.delete("/session", summary="Clear Session")
async def clear_session(
    session_id: Optional[str] = None,
    session_id_header: Optional[str] = Depends(get_session_id_from_header)
):
    """
    Clear a specific session.
    
    Args:
        session_id: Optional session ID as query parameter
        session_id_header: Optional session ID from X-Session-ID header
    """
    # Use session_id from header if available, otherwise from parameter
    session_id = session_id_header or session_id
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
        
    if session_id not in session_states:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Clear session-specific shared memory
    deleted_count = shared_memory.clear_session_data(session_id)
    
    # Clear session state
    del session_states[session_id]
    
    return {
        "message": f"Session {session_id} cleared successfully",
        "deleted_memory_items": deleted_count
    }


@app.delete("/sessions", summary="Clear All Sessions")
async def clear_all_sessions():
    """Clear all sessions and reset shared memory"""
    global session_states
    
    # Get all sessions
    sessions = list(session_states.keys())
    
    # Clear each session's data from shared memory
    deleted_items = 0
    for session_id in sessions:
        deleted_items += shared_memory.clear_session_data(session_id)
    
    # Clear session states
    session_states.clear()
    
    # Clear shared memory (but keep global data)
    shared_memory.clear_session()
    
    return {
        "message": "All sessions cleared, and session-specific shared memory reset",
        "cleared_sessions": len(sessions),
        "deleted_memory_items": deleted_items
    }

@app.post("/upload-terraform", response_model=TerraformUploadResponse, summary="Upload Terraform File/Folder and Run Plan")
async def upload_terraform_file(
    file: UploadFile = File(...),
    session_id_header: Optional[str] = Depends(get_session_id_from_header)
):
    """
    Upload a .tf file or a zip folder containing .tf files, replace existing terraform files, and run terraform plan
    
    This endpoint:
    1. Validates that the uploaded file is a .tf file or .zip folder
    2. If zip file, extracts only .tf files from the folder structure
    3. Clears the current terraform directory
    4. Saves the .tf files to the terraform directory
    5. Runs terraform plan to generate/update the .tfstate file
    6. Stores file information in session-specific shared memory
    
    Returns the terraform plan results including any changes detected.
    """
    try:
        # Validate file extension
        if not file.filename:
            raise HTTPException(
                status_code=400, 
                detail="No filename provided."
            )
        
        is_zip = file.filename.lower().endswith('.zip')
        is_tf = file.filename.lower().endswith('.tf')
        
        if not (is_zip or is_tf):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only .tf files or .zip folders are allowed."
            )
        
        # Initialize session if needed
        session_id = session_id_header
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Created new session for terraform upload: {session_id}")
        
        # Set the current session in shared memory
        shared_memory.set_session(session_id)
        
        logger.info(f"Processing upload of {'zip folder' if is_zip else 'terraform file'}: {file.filename} for session {session_id}")
        
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Clear existing files in terraform directory (but preserve .terraform and .tfstate files)
        terraform_dir = os.path.abspath(TERRAFORM_DIR)
        
        # Create session-specific terraform directory
        session_terraform_dir = os.path.join(terraform_dir, f"session_{session_id}")
        os.makedirs(session_terraform_dir, exist_ok=True)
        
        # Ensure main terraform directory exists
        os.makedirs(terraform_dir, exist_ok=True)
        
        # Remove existing .tf files in the session-specific directory
        for tf_file in glob.glob(os.path.join(session_terraform_dir, "*.tf")):
            try:
                os.remove(tf_file)
                logger.info(f"Removed existing terraform file: {tf_file}")
            except Exception as e:
                logger.warning(f"Could not remove {tf_file}: {e}")
        
        # Remove other non-essential files but keep .terraform directory and .tfstate files
        for file_pattern in ["*.md", "*.txt"]:
            for file_to_remove in glob.glob(os.path.join(session_terraform_dir, file_pattern)):
                try:
                    os.remove(file_to_remove)
                    logger.info(f"Removed file: {file_to_remove}")
                except Exception as e:
                    logger.warning(f"Could not remove {file_to_remove}: {e}")
        
        extracted_files = []
        file_contents = {}  # Store file contents for shared memory
        
        if is_zip:
            # Handle zip folder upload
            try:
                # Create a temporary directory to extract files
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save zip file to temp directory
                    temp_zip_path = os.path.join(temp_dir, file.filename)
                    with open(temp_zip_path, 'wb') as f:
                        f.write(content)
                    
                    # Extract zip file
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Find all .tf files in the extracted directory
                    tf_files_found = []
                    for root, dirs, files in os.walk(temp_dir):
                        for file_name in files:
                            if file_name.lower().endswith('.tf'):
                                tf_files_found.append(os.path.join(root, file_name))
                    
                    if not tf_files_found:
                        raise HTTPException(
                            status_code=400,
                            detail="No .tf files found in the uploaded zip folder."
                        )
                    
                    # Copy .tf files to session-specific terraform directory
                    for tf_file_path in tf_files_found:
                        file_name = os.path.basename(tf_file_path)
                        dest_path = os.path.join(session_terraform_dir, file_name)
                        
                        # Handle duplicate filenames by adding a number suffix
                        counter = 1
                        original_name = file_name
                        while os.path.exists(dest_path):
                            name_without_ext = os.path.splitext(original_name)[0]
                            ext = os.path.splitext(original_name)[1]
                            file_name = f"{name_without_ext}_{counter}{ext}"
                            dest_path = os.path.join(session_terraform_dir, file_name)
                            counter += 1
                        
                        # Copy file to session directory
                        shutil.copy2(tf_file_path, dest_path)
                        
                        # Also copy to main terraform directory for compatibility
                        main_dest_path = os.path.join(terraform_dir, file_name)
                        shutil.copy2(tf_file_path, main_dest_path)
                        
                        # Read file content for shared memory
                        with open(tf_file_path, 'r') as f:
                            try:
                                file_content = f.read()
                                file_contents[file_name] = file_content
                            except UnicodeDecodeError:
                                # Handle binary files
                                file_contents[file_name] = "Binary file - content not stored in memory"
                        
                        extracted_files.append(file_name)
                        logger.info(f"Extracted and saved terraform file: {file_name}")
                
                logger.info(f"Successfully extracted {len(extracted_files)} .tf files from zip folder")
                
            except zipfile.BadZipFile:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid zip file format."
                )
            except Exception as e:
                logger.error(f"Error processing zip file: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process zip file: {str(e)}"
                )
        else:
            # Handle single .tf file upload
            # Save to session-specific directory
            session_file_path = os.path.join(session_terraform_dir, file.filename)
            with open(session_file_path, 'wb') as f:
                f.write(content)
            
            # Also save to main terraform directory for compatibility
            main_file_path = os.path.join(terraform_dir, file.filename)
            with open(main_file_path, 'wb') as f:
                f.write(content)
            
            # Store file content in memory
            try:
                file_contents[file.filename] = content.decode('utf-8')
            except UnicodeDecodeError:
                file_contents[file.filename] = "Binary file - content not stored in memory"
            
            extracted_files.append(file.filename)
            logger.info(f"Saved terraform file to session directory: {session_file_path} and main directory: {main_file_path}")
        
        # Run terraform plan using MCP tools with ExecuteTerraformCommand from awblab
        logger.info("Running terraform plan using ExecuteTerraformCommand from awblab MCP...")
        
        # First, try to initialize terraform if needed
        logger.info("Initializing terraform...")
        # init_result = terraform_run_command(
        #     command="init",
        #     working_directory=terraform_dir
        # )
        
        # if not init_result.get("success", False):
        #     logger.warning(f"Terraform init failed: {init_result.get('output', 'Unknown error')}")
            # Continue anyway, as some configurations might not need init
        
        # Use terraform_run_command with ExecuteTerraformCommand from awblab MCP
        logger.info(f"Running terraform plan using ExecuteTerraformCommand in session directory: {session_terraform_dir}")
        
        # First try with session directory
        plan_result = terraform_plan(
            terraform_dir=session_terraform_dir
        )
        
        # If plan fails in session directory, try with main directory
        if not plan_result.get("success", False):
            logger.warning(f"Terraform plan failed in session directory, trying main directory: {terraform_dir}")
            plan_result = terraform_plan(
                terraform_dir=terraform_dir
            )
        
        # Run terraform apply
        apply_result = terraform_apply(
            terraform_dir=session_terraform_dir if plan_result.get("success", False) else terraform_dir,
            auto_approve=True
        )
        
        # Update shared memory with the upload results (session-specific)
        shared_memory.set("last_uploaded_file", file.filename, session_id)
        shared_memory.set("extracted_terraform_files", extracted_files, session_id)
        shared_memory.set("terraform_file_contents", file_contents, session_id)
        shared_memory.set("terraform_plan_result", plan_result, session_id)
        shared_memory.set("terraform_apply_result", apply_result, session_id)
        shared_memory.set("terraform_directory", session_terraform_dir, session_id)
        shared_memory.set("main_terraform_directory", terraform_dir, session_id)
        shared_memory.set("upload_timestamp", datetime.now().isoformat(), session_id)
        
        # Store file content in shared memory
        for file_name, content in file_contents.items():
            shared_memory.set(f"terraform_file_{file_name}", content, session_id)
        
        success = plan_result.get("success", False)
        
        if success:
            if is_zip:
                message = f"Successfully uploaded zip folder '{file.filename}' with {len(extracted_files)} .tf files and ran terraform plan"
            else:
                message = f"Successfully uploaded {file.filename} and ran terraform plan"
        else:
            if is_zip:
                message = f"Uploaded zip folder '{file.filename}' with {len(extracted_files)} .tf files but terraform plan encountered issues"
            else:
                message = f"Uploaded {file.filename} but terraform plan encountered issues"
                
        # Add session ID to session_states if not already there
        if session_id not in session_states:
            session_states[session_id] = {
                "session_id": session_id,
                "conversation_state": "terraform_uploaded",
                "conversation_history": [],
                "timestamp": datetime.now(),
                "context": {
                    "terraform_files": extracted_files,
                    "terraform_directory": session_terraform_dir
                }
            }
        
        return TerraformUploadResponse(
            message=message,
            filename=file.filename,
            session_id=session_id,
            extracted_files=extracted_files,
            terraform_plan_result=plan_result,
            terraform_apply_result=apply_result,
            success=success,
            timestamp=datetime.now()
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error uploading terraform file: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to upload and process terraform file: {str(e)}"
        )


@app.post("/generate-report", summary="Generate Drift Report")
async def generate_report():
    """Generate a JSON report of drift detection results"""
    try:
        # Access the report agent
        report_agent = orchestrator.agents.get('report')
        if not report_agent:
            raise HTTPException(status_code=404, detail="Report agent not available")
            
        # Generate the report
        report = report_agent.generate_json_report("report.json")
        
        # Read the report file to return in response
        try:
            with open("report.json", "r") as f:
                report_content = json.load(f)
        except Exception as e:
            logger.error(f"Error reading report file: {e}")
            report_content = {"error": f"Could not read report file: {e}"}
        
        # Return report details and content
        return {
            "message": "Report generated successfully",
            "file_path": "report.json",
            "scan_details": {
                "id": report.get("scanDetails", {}).get("id"),
                "fileName": report.get("scanDetails", {}).get("fileName"),
                "scanDate": report.get("scanDetails", {}).get("scanDate"),
                "status": report.get("scanDetails", {}).get("status"),
                "totalResources": report.get("scanDetails", {}).get("totalResources"),
                "driftCount": report.get("scanDetails", {}).get("driftCount"),
                "riskLevel": report.get("scanDetails", {}).get("riskLevel")
            },
            "total_drifts": len(report.get("drifts", [])),
            "report_content": report_content
        }
            
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")


@app.get("/terraform-status", summary="Get Terraform Files Status")
async def get_terraform_status():
    """Get information about Terraform files in the working directory"""
    try:
        from pathlib import Path
        
        # Check if terraform directory exists
        if os.path.exists(TERRAFORM_DIR):
            tf_files = list(Path(TERRAFORM_DIR).glob("*.tf"))
            state_files = list(Path(TERRAFORM_DIR).glob("*.tfstate"))
            
            # Get file details
            tf_file_details = []
            for file in tf_files:
                file_stat = os.stat(file)
                tf_file_details.append({
                    "name": file.name,
                    "size": file_stat.st_size,
                    "last_modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                })
                
            state_file_details = []
            for file in state_files:
                file_stat = os.stat(file)
                state_file_details.append({
                    "name": file.name,
                    "size": file_stat.st_size,
                    "last_modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                })
                
            return {
                "terraform_dir": TERRAFORM_DIR,
                "exists": True,
                "tf_files_count": len(tf_files),
                "state_files_count": len(state_files),
                "tf_files": tf_file_details,
                "state_files": state_file_details
            }
        else:
            return {
                "terraform_dir": TERRAFORM_DIR,
                "exists": False,
                "error": f"Terraform directory {TERRAFORM_DIR} not found"
            }
    except Exception as e:
        logger.error(f"Error checking Terraform status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check Terraform status: {e}")

def get_user_boto3_session(user_id: str = None) -> Optional[Boto3Session]:
    """
    Get a boto3 session configured with user-specific AWS credentials.
    
    Args:
        user_id: The user ID to retrieve credentials for
        
    Returns:
        A configured boto3 Session object
    """
    if not user_id:
        # Use default credentials from environment if no user_id provided
        return boto3.session.Session()
    
    # Try to get user-specific credentials from shared memory
    user_credentials_key = f"aws_credentials_{user_id}"
    user_credentials = shared_memory.get(user_credentials_key)
    
    if not user_credentials:
        # Fall back to environment credentials if user credentials not found
        logger.warning(f"No credentials found for user {user_id}, using environment credentials")
        return boto3.session.Session()
    
    # Create a session with user-specific credentials
    return boto3.session.Session(
        aws_access_key_id=user_credentials.get("access_key"),
        aws_secret_access_key=user_credentials.get("secret_key"),
        region_name=user_credentials.get("region")
    )

@app.post("/set-aws-credentials", summary="Set AWS Credentials")
async def set_aws_credentials(
    credentials: AWSCredentials,
    session_id: Optional[str] = None,
    session_id_header: Optional[str] = Depends(get_session_id_from_header)
):
    """
    Set AWS credentials for the user and session.
    
    This endpoint:
    1. Validates the provided AWS credentials
    2. Stores them in shared memory for reference
    3. Returns confirmation of the setup
    """
    try:
        # Use session_id from header if available, otherwise from parameter
        session_id = session_id_header or session_id
        
        # Generate a user ID if not provided
        user_id = credentials.user_id or str(uuid.uuid4())
        
        # Set session in shared memory if provided
        if session_id:
            shared_memory.set_session(session_id)
        
        # Store credentials in shared memory
        aws_credentials = {
            "access_key": credentials.aws_access_key_id,
            "secret_key": credentials.aws_secret_access_key,
            "region": credentials.aws_region,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store globally for user-specific access
        user_credentials_key = f"aws_credentials_{user_id}"
        shared_memory.set(user_credentials_key, aws_credentials)
        
        # If session provided, also store user_id in session-specific memory
        if session_id:
            shared_memory.set("user_id", user_id, session_id)
            shared_memory.set("aws_region", credentials.aws_region, session_id)
            shared_memory.set("aws_session_available", True, session_id)
        
        logger.info(f"AWS credentials set for user: {user_id}")
        
        return {
            "message": "AWS credentials set successfully",
            "user_id": user_id,
            "session_id": session_id,
            "aws_region": credentials.aws_region,
            "aws_access_key_id": credentials.aws_access_key_id[:10] + "..." if len(credentials.aws_access_key_id) > 10 else credentials.aws_access_key_id,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error setting AWS credentials: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set AWS credentials: {str(e)}"
        )


@app.get("/aws-credentials-status", summary="Get AWS Credentials Status")
async def get_aws_credentials_status():
    """
    Check if AWS credentials are set and return their status.
    """
    try:
        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_region = os.environ.get("AWS_REGION")
        
        credentials_set = all([aws_access_key_id, aws_secret_access_key, aws_region])
        
        return {
            "credentials_set": credentials_set,
            "aws_region": aws_region,
            "aws_access_key_id_present": bool(aws_access_key_id),
            "aws_secret_access_key_present": bool(aws_secret_access_key),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking AWS credentials status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check AWS credentials status: {str(e)}"
        )

@app.get("/help", summary="Get Help Information")
async def get_help():
    """Retrieve help information about the system"""
    return {
        "system_overview": {
            "description": "Terraform Drift Detection & Remediation System",
            "agents": [
                {"name": "OrchestrationAgent", "role": "Coordinates the entire workflow"},
                {"name": "DetectAgent", "role": "Finds drift between Terraform state and AWS"},
                {"name": "DriftAnalyzerAgent", "role": "Analyzes impact and provides recommendations"},
                {"name": "RemediateAgent", "role": "Applies fixes to remediate drift"},
                {"name": "ReportAgent", "role": "Generates reports on drift findings"}
            ]
        },
        "commands": [
            {"name": "detect", "description": "Start drift detection process"},
            {"name": "analyze", "description": "Run analysis on detected drift"},
            {"name": "remediate", "description": "Apply recommended fixes"},
            {"name": "report", "description": "Generate JSON report"},
            {"name": "status", "description": "Show current system status"}
        ],
        "example_usage": [
            {"example": "detect", "description": "Runs drift detection on Terraform resources"},
            {"example": "analyze high priority drift", "description": "Analyze only high priority drift issues"},
            {"example": "remediate security issues only", "description": "Apply fixes to security-related issues"}
        ]
    }


@app.get("/detailed-memory", summary="Get Detailed Shared Memory")
async def get_detailed_memory():
    """Get detailed contents of the shared memory"""
    try:
        result = {}
        
        # Organize data in a readable format
        if shared_memory.data:
            # Group responses
            responses = {}
            agent_status = {}
            other_data = {}
            
            for key, value in shared_memory.data.items():
                if "_response_" in key and isinstance(value, dict):
                    agent_name = key.split('_response_')[0]
                    if agent_name not in responses:
                        responses[agent_name] = []
                    
                    # Add content with appropriate length limits
                    content_summary = {}
                    if "content" in value and value["content"]:
                        content = value["content"]
                        if isinstance(content, str):
                            content_summary["content"] = (content[:100] + '...') if len(content) > 100 else content
                            content_summary["content_length"] = len(content)
                        else:
                            content_summary["content"] = str(content)
                    
                    # Add timestamp
                    if "timestamp" in value:
                        content_summary["timestamp"] = value["timestamp"]
                        
                    responses[agent_name].append(content_summary)
                
                # Process agent status data
                elif "_status" in key and isinstance(value, dict):
                    agent_status[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            agent_status[key][sub_key] = json.dumps(sub_value)[:100] + '...' if len(json.dumps(sub_value)) > 100 else json.dumps(sub_value)
                        else:
                            sub_str = str(sub_value)
                            agent_status[key][sub_key] = sub_str[:80] + '...' if len(sub_str) > 80 else sub_str
                
                # Other data
                else:
                    if isinstance(value, str) and len(value) > 100:
                        other_data[key] = value[:100] + '...'
                    else:
                        other_data[key] = value
            
            result["responses"] = responses
            result["agent_status"] = agent_status
            result["other_data"] = other_data
            result["keys_count"] = len(shared_memory.data)
            
        else:
            result["status"] = "Empty shared memory"
            
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving detailed shared memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve detailed shared memory: {e}")


@app.post("/debug-mode/{session_id}", summary="Toggle Debug Mode")
async def toggle_debug_mode(session_id: str, enable_debug: bool = True):
    """Enable or disable debug mode for a session"""
    if session_id not in session_states:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_state = session_states[session_id]
    session_state["debug_mode"] = enable_debug
    
    return {
        "session_id": session_id,
        "debug_mode": enable_debug,
        "message": f"Debug mode {'enabled' if enable_debug else 'disable'} for session {session_id}"
    }


@app.get("/logs/{session_id}", summary="Get Debug Logs for Session")
async def get_debug_logs(session_id: str):
    """Get debug logs for a specific session"""
    if session_id not in session_states:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_state = session_states[session_id]
    debug_logs = session_state.get("debug_logs", [])
    
    return {
        "session_id": session_id,
        "log_count": len(debug_logs),
        "logs": debug_logs
    }


@app.get("/report", summary="Get Report Content")
async def get_report():
    """Get the contents of report.json file"""
    try:
        report_path = os.path.join(os.path.dirname(__file__), 'report.json')
        
        # Check if file exists
        if not os.path.exists(report_path):
            raise HTTPException(
                status_code=404,
                detail="report.json not found"
            )
            
        # Read and parse JSON file
        with open(report_path, 'r') as f:
            report_content = json.load(f)
            
        return JSONResponse(
            content=report_content,
            status_code=200
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing report.json: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error parsing report.json - invalid JSON format"
        )
    except Exception as e:
        logger.error(f"Error reading report.json: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read report.json: {str(e)}"
        )


@app.get("/report-json", summary="Get Report in JSON Format")
async def get_report_json():
    """Get the report in JSON format with the specified structure."""
    try:
        # First check if report exists in shared memory
        report = shared_memory.get("drift_json_report")
        
        if report:
            return JSONResponse(content=report)
        
        # If not in memory, try to read from file
        report_path = os.path.join(os.path.dirname(__file__), 'report.json')
        
        # Check if file exists
        if not os.path.exists(report_path):
            # No report available, generate an empty report structure
            empty_report = {
                "id": "no-scan",
                "fileName": "none",
                "scanDate": datetime.now().isoformat(),
                "status": "not_started",
                "totalResources": 0,
                "driftCount": 0,
                "riskLevel": "none",
                "duration": "0s",
                "createdBy": "system",
                "createdOn": datetime.now().isoformat(),
                "modifiedBy": "system",
                "drifts": []
            }
            return JSONResponse(content=empty_report)
        
        # Read and parse JSON file
        with open(report_path, 'r') as f:
            report_content = json.load(f)
        
        return JSONResponse(content=report_content)
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing report.json: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error parsing report.json - invalid JSON format"
        )
    except Exception as e:
        logger.error(f"Error reading report.json: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read report: {str(e)}"
        )


@app.get("/journal", summary="Get Daily Journal")
async def get_daily_journal(date: Optional[str] = None):
    """
    Retrieve the journal entries for a specific date or today's date.
    
    Args:
        date: Optional date string in YYYY-MM-DD format. Defaults to today.
    
    Returns:
        JSON response with parsed journal entries
    """
    try:
        # Get target date (today if not specified)
        target_date = date if date else datetime.now().strftime("%Y-%m-%d")
        journal_path = os.path.join("journal", f"{target_date}.md")
        
        # Check if journal file exists
        if not os.path.exists(journal_path):
            return {
                "date": target_date,
                "entries": [],
                "exists": False
            }
        
        # Read journal file
        with open(journal_path, 'r') as f:
            content = f.read()
        
        # Parse journal entries
        entries = []
        current_entry = None
        
        for line in content.split('\n'):
            if line.startswith('## '):
                # New entry found
                if current_entry:
                    entries.append(current_entry)
                
                # Extract timestamp
                timestamp = line.replace('## ', '').strip()
                current_entry = {
                    "timestamp": timestamp,
                    "content": "",
                    "full_timestamp": f"{target_date} {timestamp}"
                }
            elif current_entry and line.strip():
                # Add content to current entry
                if current_entry["content"]:
                    current_entry["content"] += "\n" + line
                else:
                    current_entry["content"] = line
        
        # Add the last entry if it exists
        if current_entry:
            entries.append(current_entry)
        
        return {
            "date": target_date,
            "entries": entries,
            "exists": True,
            "count": len(entries)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving journal: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve journal: {str(e)}"
        )
@app.post("/setup-notifications", summary="Setup Email Notifications for AWS Changes")
async def setup_notifications(config: NotificationConfig):
    """
    Configure email notifications for AWS infrastructure changes using AWS-native services.
    
    This endpoint:
    1. Sets up an SNS topic and subscribes the specified email
    2. Creates EventBridge rules to monitor infrastructure changes
    3. Optionally sets up AWS Config for drift detection
    4. Returns the monitoring configuration and setup results
    """
    try:
        # Access the notification agent
        notification_agent = orchestrator.agents.get('notification')
        if not notification_agent:
            raise HTTPException(status_code=404, detail="Notification agent not available")
        
        # Set recipient email
        email_result = notification_agent.set_recipient_email(config.recipient_email)
        if email_result.get("status") != "success":
            raise HTTPException(
                status_code=400,
                detail=f"Failed to set recipient email: {email_result.get('message')}"
            )
        
        # Store configuration in shared memory
        shared_memory.set("notification_config", {
            "recipient_email": config.recipient_email,
            "resource_types": config.resource_types,
            "setup_aws_config": config.setup_aws_config,
            "include_global_resources": config.include_global_resources,
            "setup_time": datetime.now().isoformat()
        })
        
        # Start monitoring with AWS-native services
        monitoring_result = notification_agent.start_continuous_monitoring()
        
        return {
            "message": f"AWS-native notification monitoring setup successfully for {config.recipient_email}",
            "resource_types": config.resource_types or "All critical resources",
            "aws_config_enabled": config.setup_aws_config,
            "monitoring_status": monitoring_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error setting up notifications: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to setup notifications: {str(e)}"
        )

@app.get("/notification-status", summary="Get Notification Monitoring Status")
async def get_notification_status():
    """
    Get the current status of AWS-native notification monitoring.
    """
    try:
        # Access the notification agent
        notification_agent = orchestrator.agents.get('notification')
        if not notification_agent:
            raise HTTPException(status_code=404, detail="Notification agent not available")
            
        # Get notification status
        status_result = notification_agent._check_notification_status()
        if status_result.get("status") != "success":
            raise HTTPException(
                status_code=500,
                detail=f"Failed to check notification status: {status_result.get('message')}"
            )
            
        # Get notification configuration from shared memory
        notification_config = shared_memory.get("notification_config", {})
        monitoring_active = shared_memory.get("notification_monitoring_active", False)
        last_notification = shared_memory.get("last_notification_sent")
        
        return {
            "monitoring_active": monitoring_active,
            "recipient_email": notification_config.get("recipient_email"),
            "resource_types": notification_config.get("resource_types"),
            "setup_time": notification_config.get("setup_time"),
            "last_notification_sent": last_notification,
            "system_status": status_result.get("notification_system", {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting notification status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get notification status: {str(e)}"
        )

@app.post("/run-notification-check", summary="Send Test Notification")
async def run_notification_check(custom_message: Optional[str] = None):
    """
    Send a test notification through the AWS-native notification system.
    
    Args:
        custom_message: Optional custom message for the test notification
    """
    try:
        # Access the notification agent
        notification_agent = orchestrator.agents.get('notification')
        if not notification_agent:
            raise HTTPException(status_code=404, detail="Notification agent not available")
        
        # Check if SNS topic ARN is set
        if not notification_agent.sns_topic_arn:
            # Try to retrieve from shared memory
            notification_agent.sns_topic_arn = shared_memory.get("notification_sns_topic_arn")
            
            # If still not available, set up the SNS topic
            if not notification_agent.sns_topic_arn:
                sns_result = notification_agent._setup_sns_topic()
                if sns_result.get("status") != "success":
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to set up SNS topic: {sns_result.get('message')}"
                    )
        
        # Send test notification
        subject = "AWS Drift Monitoring - Test Notification"
        message = custom_message or "This is a test notification from the AWS Drift Monitoring System."
        
        # Call the _send_test_notification method with parameters
        notification_result = notification_agent._send_test_notification(
            subject=subject,
            message=message
        )
        
        if notification_result.get("status") != "success":
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send test notification: {notification_result.get('message')}"
            )
        
        return {
            "message": "Test notification sent successfully",
            "notification_result": notification_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error sending test notification: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send test notification: {str(e)}"
        )

class AWSCredentials(BaseModel):
    aws_access_key_id: str = Field(..., description="AWS Access Key ID")
    aws_secret_access_key: str = Field(..., description="AWS Secret Access Key")
    aws_region: str = Field("ap-southeast-2", description="AWS Region")
    user_id: Optional[str] = Field(None, description="User ID for credential association")

@app.post("/aws-credentials", summary="Get AWS Credentials Export Commands")
async def get_aws_credentials(
    credentials: AWSCredentials,
    session_id: Optional[str] = None,
    session_id_header: Optional[str] = Depends(get_session_id_from_header)
):
    """
    Accept AWS credentials and return them in export command format.
    Also stores them for the user and session.
    
    Args:
        credentials: AWS credentials (access key, secret key, region)
        session_id: Optional session ID to associate credentials with
        session_id_header: Optional session ID from X-Session-ID header
        
    Returns:
        Dict with AWS credentials export commands
    """
    try:
        # Use session_id from header if available, otherwise from parameter
        session_id = session_id_header or session_id
        
        # Generate a user ID if not provided
        user_id = credentials.user_id or str(uuid.uuid4())
        
        # Set session in shared memory if provided
        if session_id:
            shared_memory.set_session(session_id)
        
        # Store credentials in shared memory with user-specific keys
        user_credentials_key = f"aws_credentials_{user_id}"
        aws_credentials = {
            "access_key": credentials.aws_access_key_id,
            "secret_key": credentials.aws_secret_access_key,
            "region": credentials.aws_region,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store globally for user-specific access
        shared_memory.set(user_credentials_key, aws_credentials)
        
        # If session provided, also store user_id in session-specific memory
        if session_id:
            shared_memory.set("user_id", user_id, session_id)
            shared_memory.set("aws_region", credentials.aws_region, session_id)
            shared_memory.set("aws_session_available", True, session_id)
        
        logger.info(f"AWS credentials set for user: {user_id}")
        
        # Format credentials as export commands
        export_commands = {
            "user_id": user_id,
            "session_id": session_id,
            "export_commands": f"export AWS_ACCESS_KEY_ID={credentials.aws_access_key_id}\nexport AWS_SECRET_ACCESS_KEY={credentials.aws_secret_access_key}\nexport AWS_REGION={credentials.aws_region}",
            "access_key": credentials.aws_access_key_id,
            "secret_key": credentials.aws_secret_access_key,
            "region": credentials.aws_region,
            "timestamp": datetime.now().isoformat(),
            "environment_updated": False
        }
        
        return export_commands
        
    except Exception as e:
        logger.error(f"Error processing AWS credentials: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process AWS credentials: {str(e)}"
        )

@app.get("/aws-resources/{user_id}", summary="List AWS Resources for User")
async def list_aws_resources(user_id: str, resource_type: str = "s3"):
    """
    List AWS resources for a specific user using their stored credentials.
    
    Args:
        user_id: The user ID to retrieve credentials for
        resource_type: Type of AWS resource to list (s3, ec2, etc.)
        
    Returns:
        List of AWS resources for the specified user
    """
    try:
        # Get boto3 session for the user
        session = get_user_boto3_session(user_id)
        
        if not session:
            return {
                "user_id": user_id,
                "error": "No credentials found for this user",
                "timestamp": datetime.now().isoformat()
            }
        
        resources = []
        
        # List resources based on type
        try:
            if resource_type.lower() == "s3":
                # List S3 buckets
                s3_client = session.client('s3')
                response = s3_client.list_buckets()
                resources = [{"name": bucket['Name'], "creation_date": bucket['CreationDate'].isoformat()} 
                            for bucket in response.get('Buckets', [])]
                
            elif resource_type.lower() == "ec2":
                # List EC2 instances
                ec2_client = session.client('ec2')
                response = ec2_client.describe_instances()
                
                for reservation in response.get('Reservations', []):
                    for instance in reservation.get('Instances', []):
                        resources.append({
                            "id": instance.get('InstanceId'),
                            "type": instance.get('InstanceType'),
                            "state": instance.get('State', {}).get('Name'),
                            "launch_time": instance.get('LaunchTime').isoformat() if 'LaunchTime' in instance else None
                        })
                        
            elif resource_type.lower() == "lambda":
                # List Lambda functions
                lambda_client = session.client('lambda')
                response = lambda_client.list_functions()
                resources = [{"name": function['FunctionName'], "runtime": function['Runtime']} 
                            for function in response.get('Functions', [])]
                            
            else:
                return {
                    "error": f"Unsupported resource type: {resource_type}",
                    "supported_types": ["s3", "ec2", "lambda"],
                    "user_id": user_id
                }
            
            return {
                "user_id": user_id,
                "resource_type": resource_type,
                "count": len(resources),
                "resources": resources,
                "aws_region": session.region_name,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as aws_error:
            return {
                "user_id": user_id,
                "resource_type": resource_type,
                "error": str(aws_error),
                "aws_region": session.region_name,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error listing AWS resources for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list AWS resources: {str(e)}"
        )

async def get_user_id_from_header(x_user_id: Optional[str] = Header(None)) -> Optional[str]:
    """
    Extract and validate user ID from request header.
    
    Args:
        x_user_id: User ID from X-User-ID header
        
    Returns:
        Validated user ID or None
    """
    return x_user_id

@app.get("/aws-resources", summary="List AWS Resources Using Header Authentication")
async def list_aws_resources_from_header(
    resource_type: str = "s3",
    user_id: str = Depends(get_user_id_from_header)
):
    """
    List AWS resources for a user identified by the X-User-ID header.
    
    Args:
        resource_type: Type of resources to list (s3, ec2, lambda)
        user_id: User ID extracted from X-User-ID header
        
    Returns:
        List of AWS resources for the specified user
    """
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="X-User-ID header is required"
        )
        
    # Reuse the existing implementation
    return await list_aws_resources(user_id, resource_type)

@app.get("/test-aws-credentials/{user_id}", summary="Test AWS Credentials for User")
async def test_aws_credentials(user_id: str):
    """
    Test the AWS credentials for a specific user.
    
    Args:
        user_id: The user ID to test credentials for
        
    Returns:
        Status of AWS credentials test
    """
    try:
        # Get boto3 session for the user
        session = get_user_boto3_session(user_id)
        
        if not session:
            return {
                "user_id": user_id,
                "credentials_valid": False,
                "error": "No credentials found for this user",
                "timestamp": datetime.now().isoformat()
            }
        
        # Test if credentials are valid
        try:
            sts_client = session.client('sts')
            identity = sts_client.get_caller_identity()
            
            return {
                "user_id": user_id,
                "credentials_valid": True,
                "aws_account_id": identity.get("Account"),
                "aws_user_arn": identity.get("Arn"),
                "aws_user_id": identity.get("UserId"),
                "aws_region": session.region_name,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as aws_error:
            return {
                "user_id": user_id,
                "credentials_valid": False,
                "error": str(aws_error),
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error testing AWS credentials for user {user_id}: {e}")
        return {
            "user_id": user_id,
            "credentials_valid": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/storage-config", summary="Get Storage Configuration")
async def get_storage_config():
    """Get the current storage configuration"""
    return {
        "use_db_storage": USE_DB_STORAGE,
        "db_path": DB_STORAGE_PATH if USE_DB_STORAGE else None,
        "storage_mode": "database" if USE_DB_STORAGE else "memory",
        "memory_size": shared_memory.size(),
        "timestamp": datetime.now()
    }

@app.post("/toggle-storage-mode", summary="Toggle Storage Mode")
async def toggle_storage_mode(use_db: bool = True, db_path: Optional[str] = None):
    """
    Toggle between database and memory storage modes.
    
    Args:
        use_db: Whether to use database storage
        db_path: Path to the SQLite database file (optional)
        
    Returns:
        Status of the operation
    """
    try:
        # Set environment variables
        os.environ["USE_DB_STORAGE"] = str(use_db).lower()
        
        if db_path:
            os.environ["DB_STORAGE_PATH"] = db_path
        
        # Restart is required to apply changes
        return {
            "message": "Storage mode updated. Restart the server to apply changes.",
            "use_db": use_db,
            "db_path": db_path or DB_STORAGE_PATH if use_db else None,
            "current_mode": "database" if USE_DB_STORAGE else "memory",
            "restart_required": True,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error toggling storage mode: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to toggle storage mode: {str(e)}"
        )

@app.get("/test-shared-memory", summary="Test Shared Memory Isolation")
async def test_shared_memory_isolation(
    session_id_1: Optional[str] = None,
    session_id_2: Optional[str] = None,
    key: str = "test_key",
    value_1: str = "value_for_session_1",
    value_2: str = "value_for_session_2"
):
    """
    Test that shared memory properly isolates data between different sessions.
    
    This endpoint:
    1. Creates two sessions if not provided
    2. Sets the same key with different values in each session
    3. Retrieves the values to verify isolation
    4. Returns the results
    
    Args:
        session_id_1: First session ID (created if not provided)
        session_id_2: Second session ID (created if not provided)
        key: Key to use for the test
        value_1: Value to set for session 1
        value_2: Value to set for session 2
    """
    try:
        # Create sessions if not provided
        if not session_id_1:
            session_id_1 = str(uuid.uuid4())
            logger.info(f"Created test session 1: {session_id_1}")
        
        if not session_id_2:
            session_id_2 = str(uuid.uuid4())
            logger.info(f"Created test session 2: {session_id_2}")
        
        # Set value for session 1
        shared_memory.set_session(session_id_1)
        shared_memory.set(key, value_1, session_id_1)
        
        # Set value for session 2
        shared_memory.set_session(session_id_2)
        shared_memory.set(key, value_2, session_id_2)
        
        # Get values from both sessions
        shared_memory.set_session(session_id_1)
        value_from_session_1 = shared_memory.get(key, session_id=session_id_1)
        
        shared_memory.set_session(session_id_2)
        value_from_session_2 = shared_memory.get(key, session_id=session_id_2)
        
        # Test cross-session access (should not leak)
        shared_memory.set_session(session_id_1)
        value_2_from_session_1 = shared_memory.get(key, session_id=session_id_2)
        
        shared_memory.set_session(session_id_2)
        value_1_from_session_2 = shared_memory.get(key, session_id=session_id_1)
        
        # Set global value
        shared_memory.clear_session()
        shared_memory.set(key, "global_value")
        
        # Get global value from both sessions
        shared_memory.set_session(session_id_1)
        global_from_session_1 = shared_memory.get(key, default="not_found")
        
        shared_memory.set_session(session_id_2)
        global_from_session_2 = shared_memory.get(key, default="not_found")
        
        # Clear session
        shared_memory.clear_session()
        
        # Return results
        return {
            "session_id_1": session_id_1,
            "session_id_2": session_id_2,
            "test_key": key,
            "session_1_value": value_1,
            "session_2_value": value_2,
            "retrieved_from_session_1": value_from_session_1,
            "retrieved_from_session_2": value_from_session_2,
            "cross_session_access_1_to_2": value_2_from_session_1,
            "cross_session_access_2_to_1": value_1_from_session_2,
            "global_value_from_session_1": global_from_session_1,
            "global_value_from_session_2": global_from_session_2,
            "isolation_working": (
                value_from_session_1 == value_1 and
                value_from_session_2 == value_2 and
                value_from_session_1 != value_from_session_2
            ),
            "storage_mode": "database" if USE_DB_STORAGE else "memory"
        }
    except Exception as e:
        logger.error(f"Error testing shared memory isolation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test shared memory isolation: {str(e)}"
        )

@app.get("/test-aws-multi-user", summary="Test AWS Multi-User Isolation")
async def test_aws_multi_user(
    user_id_1: Optional[str] = None,
    user_id_2: Optional[str] = None,
    access_key_1: Optional[str] = None,
    secret_key_1: Optional[str] = None,
    access_key_2: Optional[str] = None,
    secret_key_2: Optional[str] = None,
    region: str = "ap-southeast-2"
):
    """
    Test that AWS credentials work correctly for multiple users.
    
    This endpoint:
    1. Sets AWS credentials for two different users
    2. Creates boto3 sessions for each user
    3. Verifies that the sessions use the correct credentials
    4. Returns the results
    
    Args:
        user_id_1: First user ID (generated if not provided)
        user_id_2: Second user ID (generated if not provided)
        access_key_1: AWS access key for first user
        secret_key_1: AWS secret key for first user
        access_key_2: AWS access key for second user
        secret_key_2: AWS secret key for second user
        region: AWS region
    """
    try:
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "region": region
        }
        
        # Generate user IDs if not provided
        if not user_id_1:
            user_id_1 = str(uuid.uuid4())
            results["user_id_1_generated"] = True
        
        if not user_id_2:
            user_id_2 = str(uuid.uuid4())
            results["user_id_2_generated"] = True
        
        results["user_id_1"] = user_id_1
        results["user_id_2"] = user_id_2
        
        # Set credentials for user 1 if provided
        if access_key_1 and secret_key_1:
            user1_credentials = {
                "access_key": access_key_1,
                "secret_key": secret_key_1,
                "region": region,
                "timestamp": datetime.now().isoformat()
            }
            
            user_credentials_key_1 = f"aws_credentials_{user_id_1}"
            shared_memory.set(user_credentials_key_1, user1_credentials)
            results["user_1_credentials_set"] = True
        else:
            results["user_1_credentials_set"] = False
            results["user_1_credentials_message"] = "No credentials provided for user 1"
        
        # Set credentials for user 2 if provided
        if access_key_2 and secret_key_2:
            user2_credentials = {
                "access_key": access_key_2,
                "secret_key": secret_key_2,
                "region": region,
                "timestamp": datetime.now().isoformat()
            }
            
            user_credentials_key_2 = f"aws_credentials_{user_id_2}"
            shared_memory.set(user_credentials_key_2, user2_credentials)
            results["user_2_credentials_set"] = True
        else:
            results["user_2_credentials_set"] = False
            results["user_2_credentials_message"] = "No credentials provided for user 2"
        
        # Create boto3 sessions
        session1 = get_user_boto3_session(user_id_1)
        session2 = get_user_boto3_session(user_id_2)
        
        # Test session 1
        if session1:
            results["user_1_session_created"] = True
            
            if access_key_1:
                # Verify session 1 credentials
                try:
                    credentials1 = session1.get_credentials()
                    if credentials1:
                        frozen_creds1 = credentials1.get_frozen_credentials()
                        
                        # Check if credentials match (only check first 4 chars for security)
                        access_key_match = frozen_creds1.access_key.startswith(access_key_1[:4])
                        results["user_1_access_key_match"] = access_key_match
                        results["user_1_region"] = session1.region_name
                        
                        # Test AWS API call if possible
                        try:
                            sts_client = session1.client('sts')
                            identity = sts_client.get_caller_identity()
                            results["user_1_aws_account"] = identity.get("Account")
                            results["user_1_aws_user_id"] = identity.get("UserId")
                            results["user_1_aws_api_success"] = True
                        except Exception as aws_error:
                            results["user_1_aws_api_success"] = False
                            results["user_1_aws_api_error"] = str(aws_error)
                    else:
                        results["user_1_credentials_found"] = False
                except Exception as cred_error:
                    results["user_1_credentials_error"] = str(cred_error)
        else:
            results["user_1_session_created"] = False
        
        # Test session 2
        if session2:
            results["user_2_session_created"] = True
            
            if access_key_2:
                # Verify session 2 credentials
                try:
                    credentials2 = session2.get_credentials()
                    if credentials2:
                        frozen_creds2 = credentials2.get_frozen_credentials()
                        
                        # Check if credentials match (only check first 4 chars for security)
                        access_key_match = frozen_creds2.access_key.startswith(access_key_2[:4])
                        results["user_2_access_key_match"] = access_key_match
                        results["user_2_region"] = session2.region_name
                        
                        # Test AWS API call if possible
                        try:
                            sts_client = session2.client('sts')
                            identity = sts_client.get_caller_identity()
                            results["user_2_aws_account"] = identity.get("Account")
                            results["user_2_aws_user_id"] = identity.get("UserId")
                            results["user_2_aws_api_success"] = True
                        except Exception as aws_error:
                            results["user_2_aws_api_success"] = False
                            results["user_2_aws_api_error"] = str(aws_error)
                    else:
                        results["user_2_credentials_found"] = False
                except Exception as cred_error:
                    results["user_2_credentials_error"] = str(cred_error)
        else:
            results["user_2_session_created"] = False
        
        # Verify isolation
        if access_key_1 and access_key_2 and session1 and session2:
            creds1 = session1.get_credentials().get_frozen_credentials()
            creds2 = session2.get_credentials().get_frozen_credentials()
            
            results["credentials_isolation"] = creds1.access_key != creds2.access_key
        
        return results
        
    except Exception as e:
        logger.error(f"Error testing AWS multi-user isolation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test AWS multi-user isolation: {str(e)}"
        )

if __name__ == "__main__":
    # Ensure terraform directory exists
    os.makedirs(TERRAFORM_DIR, exist_ok=True)
    
    # Log storage mode
    logger.info(f"Starting API with storage mode: {'database' if USE_DB_STORAGE else 'memory'}")
    if USE_DB_STORAGE:
        logger.info(f"Database path: {DB_STORAGE_PATH}")
    
    # Run the FastAPI server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 