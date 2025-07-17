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
import shutil
import glob
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import json

# Add tools to path
sys.path.append("tools/src")
sys.path.append("useful_tools")

from strands.models.bedrock import BedrockModel
from agents import OrchestrationAgent, DetectAgent, DriftAnalyzerAgent, RemediateAgent
from shared_memory import shared_memory
from config import BEDROCK_MODEL_ID, BEDROCK_REGION, TERRAFORM_DIR

# Import terraform tools
from terraform_mcp_tool import terraform_run_command
from terraform_tools import terraform_plan

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
    - WebSocket streaming for real-time chat interactions
    - Multi-agent orchestration with intelligent routing
    - Session-based conversation management
    
    WebSocket Usage:
    - Connect to /ws/chat/{session_id} for streaming chat
    - Send messages as JSON: {"message": "your text here"}
    - Receive streaming responses with different message types:
      * "status": Processing updates
      * "response": Orchestrator responses
      * "agent_result": Agent execution results
      * "complete": Final response with suggestions
      * "error": Error messages
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

# Global state management
session_states: Dict[str, Dict[str, Any]] = {}

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected for session: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def send_stream_chunk(self, session_id: str, chunk: str, message_type: str = "stream"):
        """Send a streaming chunk to the client"""
        message = {
            "type": message_type,
            "content": chunk,
            "timestamp": datetime.now().isoformat()
        }
        await self.send_message(session_id, message)

# Initialize connection manager
connection_manager = ConnectionManager()


# Pydantic Models
class TerraformUploadResponse(BaseModel):
    message: str
    filename: str
    terraform_plan_result: Dict[str, Any]
    success: bool
    timestamp: datetime


class ChatMessage(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


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


# Chat-based Orchestrator Class
class ChatOrchestrator:
    """Intelligent conversation orchestrator that routes messages to appropriate agents"""
    
    def __init__(self):
        self.model = self._create_model()
        self.agents = self._create_agents()
    
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
            'remediate': RemediateAgent(self.model)
        }
    
    async def initialize_session(self, session_id: Optional[str] = None) -> str:
        """Initialize a new conversation session"""
        if session_id and session_id in session_states:
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
        
        # Initialize shared memory for this session
        shared_memory.set(f"session_{session_id}_state", "idle")
        shared_memory.set(f"session_{session_id}_history", [])
        
        return session_id
    
    async def process_chat_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """Process a chat message and route to appropriate agent if needed"""
        if session_id not in session_states:
            session_id = await self.initialize_session(session_id)
        
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
            
            shared_memory.set(f"session_{session_id}_history", serializable_history)
            shared_memory.set("current_user_message", message)
            shared_memory.set("session_context", serializable_context)
            
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
"""
            
            # Execute orchestration agent
            orchestration_result = orchestration_agent(context_prompt)
            
            # Determine if orchestrator wants to route to a specific agent
            routed_agent = None
            agent_result = None
            
            orchestration_response = str(orchestration_result)
            
            # Check if orchestrator indicated it wants to route to a specific agent
            if "routing to DetectAgent" in orchestration_response.lower() or "invoking detectagent" in orchestration_response.lower():
                routed_agent = "DetectAgent"
                agent_result = self._execute_agent('detect', message)
                session_state["conversation_state"] = "detection_complete"
                
            elif "routing to driftanalyzeragent" in orchestration_response.lower() or "invoking analyzer" in orchestration_response.lower():
                routed_agent = "DriftAnalyzerAgent"
                agent_result = self._execute_agent('analyzer', message)
                session_state["conversation_state"] = "analysis_complete"
                
            elif "routing to remediateagent" in orchestration_response.lower() or "invoking remediate" in orchestration_response.lower():
                routed_agent = "RemediateAgent"
                agent_result = self._execute_agent('remediate', message)
                session_state["conversation_state"] = "remediation_complete"
            
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

    async def process_streaming_chat_message(self, session_id: str, message: str):
        """Process a chat message with streaming responses"""
        if session_id not in session_states:
            session_id = await self.initialize_session(session_id)
        
        session_state = session_states[session_id]
        
        try:
            # Send initial acknowledgment
            await connection_manager.send_stream_chunk(session_id, "Processing your message...", "status")
            
            # Add user message to conversation history
            session_state["conversation_history"].append({
                "role": "user",
                "message": message,
                "timestamp": datetime.now()
            })
            
            # Update shared memory with conversation context (JSON-serializable only)
            serializable_history = self._make_json_serializable(session_state["conversation_history"])
            serializable_context = self._make_json_serializable(session_state.get("context", {}))
            
            shared_memory.set(f"session_{session_id}_history", serializable_history)
            shared_memory.set("current_user_message", message)
            shared_memory.set("session_context", serializable_context)
            
            # Send status update
            await connection_manager.send_stream_chunk(session_id, "Analyzing your request...", "status")
            
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
"""
            
            # Execute orchestration agent
            orchestration_result = orchestration_agent(context_prompt)
            orchestration_response = str(orchestration_result)
            
            # Stream the orchestration response
            await connection_manager.send_stream_chunk(session_id, orchestration_response, "response")
            
            # Determine if orchestrator wants to route to a specific agent
            routed_agent = None
            agent_result = None
            
            # Check if orchestrator indicated it wants to route to a specific agent
            if "routing to DetectAgent" in orchestration_response.lower() or "invoking detectagent" in orchestration_response.lower():
                routed_agent = "DetectAgent"
                await connection_manager.send_stream_chunk(session_id, "Executing drift detection...", "status")
                agent_result = await self._execute_agent_streaming(session_id, 'detect', message)
                session_state["conversation_state"] = "detection_complete"
                
            elif "routing to driftanalyzeragent" in orchestration_response.lower() or "invoking analyzer" in orchestration_response.lower():
                routed_agent = "DriftAnalyzerAgent"
                await connection_manager.send_stream_chunk(session_id, "Analyzing drift patterns...", "status")
                agent_result = await self._execute_agent_streaming(session_id, 'analyzer', message)
                session_state["conversation_state"] = "analysis_complete"
                
            elif "routing to remediateagent" in orchestration_response.lower() or "invoking remediate" in orchestration_response.lower():
                routed_agent = "RemediateAgent"
                await connection_manager.send_stream_chunk(session_id, "Applying remediation...", "status")
                agent_result = await self._execute_agent_streaming(session_id, 'remediate', message)
                session_state["conversation_state"] = "remediation_complete"
            
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
            
            # Send final response with suggestions
            final_response = {
                "type": "complete",
                "session_id": session_id,
                "routed_agent": routed_agent,
                "conversation_state": session_state["conversation_state"],
                "timestamp": datetime.now().isoformat(),
                "suggestions": suggestions
            }
            
            await connection_manager.send_message(session_id, final_response)
            
        except Exception as e:
            logger.error(f"Error processing streaming chat message: {e}")
            error_response = f"I encountered an error while processing your message: {str(e)}. Please try again or rephrase your request."
            
            session_state["conversation_history"].append({
                "role": "assistant",
                "message": error_response,
                "error": True,
                "timestamp": datetime.now()
            })
            
            await connection_manager.send_stream_chunk(session_id, error_response, "error")
            
            final_response = {
                "type": "error",
                "session_id": session_id,
                "conversation_state": session_state["conversation_state"],
                "timestamp": datetime.now().isoformat(),
                "suggestions": ["Try rephrasing your request", "Ask for help", "Check system status"]
            }
            
            await connection_manager.send_message(session_id, final_response)
    
    def _execute_agent(self, agent_type: str, user_message: str) -> str:
        """Execute a specific agent and return its result"""
        try:
            agent = self.agents[agent_type].get_agent()
            
            # Update agent's shared memory
            self.agents[agent_type].update_shared_memory()
            
            # Execute the agent
            result = agent(user_message)
            
            # Store result in shared memory
            shared_memory.set(f"{agent_type}_last_result", str(result))
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error executing {agent_type} agent: {e}")
            return f"Error executing {agent_type}: {str(e)}"
    
    async def _execute_agent_streaming(self, session_id: str, agent_type: str, user_message: str) -> str:
        """Execute a specific agent with streaming output"""
        try:
            agent = self.agents[agent_type].get_agent()
            
            # Update agent's shared memory
            self.agents[agent_type].update_shared_memory()
            
            # Send progress update
            await connection_manager.send_stream_chunk(session_id, f"Running {agent_type} agent...", "status")
            
            # Execute the agent
            result = agent(user_message)
            result_str = str(result)
            
            # Stream the result in chunks
            chunk_size = 200
            for i in range(0, len(result_str), chunk_size):
                chunk = result_str[i:i+chunk_size]
                await connection_manager.send_stream_chunk(session_id, chunk, "agent_result")
                await asyncio.sleep(0.1)  # Small delay for streaming effect
            
            # Store result in shared memory
            shared_memory.set(f"{agent_type}_last_result", result_str)
            
            return result_str
            
        except Exception as e:
            logger.error(f"Error executing {agent_type} agent: {e}")
            error_msg = f"Error executing {agent_type}: {str(e)}"
            await connection_manager.send_stream_chunk(session_id, error_msg, "error")
            return error_msg
    
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
        "version": "1.0.0",
        "terraform_dir": TERRAFORM_DIR,
        "endpoints": {
            "websocket_chat": "/ws/chat/{session_id}",
            "rest_chat": "/chat",
            "upload_terraform": "/upload-terraform",
            "test_client": "/test-client"
        }
    }


@app.get("/test-client", response_class=HTMLResponse, summary="WebSocket Test Client")
async def get_test_client():
    """Serve the HTML test client for WebSocket chat"""
    try:
        with open("websocket_client.html", "r") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test client not found")


@app.post("/chat", response_model=ChatResponse, summary="Chat with Terraform Drift Assistant")
async def chat(request: ChatMessage):
    """Send a message to the intelligent Terraform drift assistant"""
    try:
        # Initialize session if needed
        session_id = request.session_id
        if not session_id:
            session_id = await orchestrator.initialize_session()
        
        # Process the chat message
        result = await orchestrator.process_chat_message(session_id, request.message)
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in chat processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat message: {e}")


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming chat with Terraform Drift Assistant"""
    await connection_manager.connect(websocket, session_id)
    
    try:
        # Initialize session if needed
        await orchestrator.initialize_session(session_id)
        
        # Send welcome message
        welcome_message = {
            "type": "welcome",
            "content": "Connected to Terraform Drift Assistant. How can I help you today?",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "suggestions": [
                "Ask me to check for infrastructure drift",
                "Request a scan of your Terraform resources",
                "Get help with the system capabilities"
            ]
        }
        await connection_manager.send_message(session_id, welcome_message)
        
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Extract user message
                user_message = message_data.get("message", "")
                
                if user_message.strip():
                    # Process the message with streaming response
                    await orchestrator.process_streaming_chat_message(session_id, user_message)
                
            except json.JSONDecodeError:
                await connection_manager.send_stream_chunk(
                    session_id, 
                    "Invalid message format. Please send valid JSON.", 
                    "error"
                )
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await connection_manager.send_stream_chunk(
                    session_id, 
                    f"Error processing your message: {str(e)}", 
                    "error"
                )
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        connection_manager.disconnect(session_id)


@app.post("/upload-terraform", response_model=TerraformUploadResponse, summary="Upload Terraform File and Run Plan")
async def upload_terraform_file(file: UploadFile = File(...)):
    """
    Upload a .tf file, replace existing terraform files, and run terraform plan
    
    This endpoint:
    1. Validates that the uploaded file is a .tf file
    2. Clears the current terraform directory 
    3. Saves the new .tf file to the terraform directory
    4. Runs terraform plan to generate/update the .tfstate file
    
    Returns the terraform plan results including any changes detected.
    """
    try:
        # Validate file extension
        if not file.filename or not file.filename.endswith('.tf'):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only .tf files are allowed."
            )
        
        logger.info(f"Processing upload of terraform file: {file.filename}")
        
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Clear existing files in terraform directory (but preserve .terraform and .tfstate files)
        terraform_dir = os.path.abspath(TERRAFORM_DIR)
        
        # Ensure terraform directory exists
        os.makedirs(terraform_dir, exist_ok=True)
        
        # Remove existing .tf files, but preserve terraform state and cache
        for tf_file in glob.glob(os.path.join(terraform_dir, "*.tf")):
            try:
                os.remove(tf_file)
                logger.info(f"Removed existing terraform file: {tf_file}")
            except Exception as e:
                logger.warning(f"Could not remove {tf_file}: {e}")
        
        # Remove other non-essential files but keep .terraform directory and .tfstate files
        for file_pattern in ["*.md", "*.txt"]:
            for file_to_remove in glob.glob(os.path.join(terraform_dir, file_pattern)):
                try:
                    os.remove(file_to_remove)
                    logger.info(f"Removed file: {file_to_remove}")
                except Exception as e:
                    logger.warning(f"Could not remove {file_to_remove}: {e}")
        
        # Save the new terraform file
        new_file_path = os.path.join(terraform_dir, file.filename)
        
        with open(new_file_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"Saved new terraform file: {new_file_path}")
        
        # Run terraform plan using MCP tools
        logger.info("Running terraform plan...")
        
        # Use terraform_run_command to ensure proper initialization and planning
        plan_result = terraform_run_command(
            command="plan",
            working_directory=terraform_dir
        )
        
        if not plan_result.get("success", False):
            # If terraform_run_command fails, try with terraform_plan tool as fallback
            logger.warning("terraform_run_command failed, trying terraform_plan tool...")
            try:
                plan_result = terraform_plan(
                    terraform_dir=terraform_dir,
                    output_format="human"
                )
            except Exception as fallback_error:
                logger.error(f"Both terraform tools failed: {fallback_error}")
                plan_result = {
                    "error": f"Terraform plan failed: {str(fallback_error)}",
                    "output": "",
                    "success": False
                }
        
        # Update shared memory with the upload results
        shared_memory.set("last_uploaded_file", file.filename)
        shared_memory.set("terraform_plan_result", plan_result)
        shared_memory.set("terraform_directory", terraform_dir)
        
        success = plan_result.get("success", False)
        
        if success:
            message = f"Successfully uploaded {file.filename} and ran terraform plan"
        else:
            message = f"Uploaded {file.filename} but terraform plan encountered issues"
        
        return TerraformUploadResponse(
            message=message,
            filename=file.filename,
            terraform_plan_result=plan_result,
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


@app.get("/status/{session_id}", response_model=WorkflowStatus, summary="Get Session Status")
async def get_session_status(session_id: str):
    """Get current status of a conversation session"""
    if session_id not in session_states:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_state = session_states[session_id]
    
    return WorkflowStatus(
        session_id=session_id,
        conversation_state=session_state["conversation_state"],
        message=f"Session active. Current state: {session_state['conversation_state']}. {len(session_state.get('conversation_history', []))} messages in conversation.",
        timestamp=session_state["timestamp"],
        shared_memory_keys=shared_memory.keys()
    )


@app.get("/conversation/{session_id}", summary="Get Conversation History")
async def get_conversation_history(session_id: str):
    """Get the conversation history for a session"""
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
async def get_shared_memory():
    """Get current shared memory contents"""
    return {
        "shared_memory": shared_memory.data,
        "keys": shared_memory.keys(),
        "size": shared_memory.size(),
        "timestamp": datetime.now()
    }


@app.get("/websocket-connections", summary="Get Active WebSocket Connections")
async def get_websocket_connections():
    """Get information about active WebSocket connections"""
    connections = []
    for session_id, websocket in connection_manager.active_connections.items():
        session_info = session_states.get(session_id, {})
        connections.append({
            "session_id": session_id,
            "conversation_state": session_info.get("conversation_state", "unknown"),
            "last_activity": session_info.get("timestamp", "unknown"),
            "messages_count": len(session_info.get("conversation_history", []))
        })
    
    return {
        "active_connections": len(connection_manager.active_connections),
        "connections": connections,
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


@app.delete("/session/{session_id}", summary="Clear Session")
async def clear_session(session_id: str):
    """Clear a specific session"""
    if session_id not in session_states:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Disconnect WebSocket if active
    connection_manager.disconnect(session_id)
    
    # Clear session state
    del session_states[session_id]
    
    return {"message": f"Session {session_id} cleared successfully"}


@app.delete("/sessions", summary="Clear All Sessions")
async def clear_all_sessions():
    """Clear all sessions and reset shared memory"""
    global session_states
    
    # Disconnect all WebSocket connections
    for session_id in list(connection_manager.active_connections.keys()):
        connection_manager.disconnect(session_id)
    
    # Clear session states and shared memory
    session_states.clear()
    shared_memory.clear()
    
    return {"message": "All sessions cleared, WebSocket connections closed, and shared memory reset"}


if __name__ == "__main__":
    # Ensure terraform directory exists
    os.makedirs(TERRAFORM_DIR, exist_ok=True)
    
    # Run the FastAPI server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 