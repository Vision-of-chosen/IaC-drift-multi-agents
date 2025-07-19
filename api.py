#!/usr/bin/env python3
"""
FastAPI Web Application for Terraform Drift Detection & Remediation System

This module provides a web-based API interface that coordinates the multi-agent
system with step-by-step user interaction between each agent execution.

Architecture:
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
import glob
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import json
import zipfile
import tempfile
import shutil

# Add tools to path
sys.path.append("tools/src")

from strands.models.bedrock import BedrockModel
from strands.multiagent.graph import GraphBuilder
from agents import OrchestrationAgent, DetectAgent, DriftAnalyzerAgent, RemediateAgent, ReportAgent, NotificationAgent
from shared_memory import shared_memory
from config import BEDROCK_MODEL_ID, BEDROCK_REGION, TERRAFORM_DIR
from permission_handlers import permission_manager

from useful_tools.terraform_mcp_tool import terraform_run_command
from useful_tools.terraform_tools import terraform_plan, terraform_apply

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

# Global state management
session_states: Dict[str, Dict[str, Any]] = {}

class TerraformUploadResponse(BaseModel):
    message: str
    filename: str
    extracted_files: list[str] = Field(default_factory=list, description="List of .tf files extracted from the upload")
    terraform_plan_result: Dict[str, Any]
    terraform_apply_result: Dict[str, Any]
    success: bool
    timestamp: datetime

# Pydantic Models
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


class NotificationConfig(BaseModel):
    recipient_email: str = Field(..., description="Email address to send notifications to")
    interval_minutes: int = Field(15, description="Interval in minutes between monitoring checks (legacy parameter)")
    resource_types: Optional[List[str]] = Field(None, description="List of AWS resource types to monitor")
    setup_aws_config: bool = Field(True, description="Whether to set up AWS Config for drift detection")
    include_global_resources: bool = Field(True, description="Whether to include global resources like IAM in AWS Config")

# Chat-based Orchestrator Class
class ChatOrchestrator:
    """Intelligent conversation orchestrator that routes messages to appropriate agents"""
    
    def __init__(self):
        self.model = self._create_model()
        self.graph, self.agents = self._create_agents_and_graph()
    
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
    
    def _create_agents_and_graph(self):
        """Create all agent instances and build the graph"""
        # Create individual agents
        orchestration_agent = OrchestrationAgent(self.model)
        detect_agent = DetectAgent(self.model)
        drift_analyzer_agent = DriftAnalyzerAgent(self.model)
        remediate_agent = RemediateAgent(self.model)
        report_agent = ReportAgent(self.model)
        notification_agent = NotificationAgent(self.model)

        # Build the agent graph using GraphBuilder
        builder = GraphBuilder()
        
        # Add agents as nodes
        orchestration_node = builder.add_node(orchestration_agent.get_agent(), "orchestration")
        detect_node = builder.add_node(detect_agent.get_agent(), "detect")
        analyzer_node = builder.add_node(drift_analyzer_agent.get_agent(), "analyzer")
        remediate_node = builder.add_node(remediate_agent.get_agent(), "remediate")
        report_node = builder.add_node(report_agent.get_agent(), "report")
        notification_node = builder.add_node(notification_agent.get_agent(), "notification")

        # Define the workflow edges - all agents connect directly to Orchestration Agent
        # Orchestration → DetectAgent
        builder.add_edge(orchestration_node, detect_node)
        
        # Orchestration → DriftAnalyzerAgent
        builder.add_edge(orchestration_node, analyzer_node)
        
        # Orchestration → RemediateAgent
        builder.add_edge(orchestration_node, remediate_node)
        
        # Orchestration → ReportAgent
        builder.add_edge(orchestration_node, report_node)
        
        # Orchestration → NotificationAgent
        builder.add_edge(orchestration_node, notification_node)
        
        # Build the graph
        graph = builder.build()
        
        agents = {
            'orchestration': orchestration_agent,
            'detect': detect_agent, 
            'analyzer': drift_analyzer_agent,
            'remediate': remediate_agent,
            'report': report_agent,
            'notification': notification_agent
        }
        
        return graph, agents
    
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
        """Process a chat message using the multi-agent graph system"""
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
            shared_memory.set("user_request", message)
            shared_memory.set("workflow_status", "initiated")
            shared_memory.set("request_timestamp", datetime.now().isoformat())
            shared_memory.set("session_context", serializable_context)
            
            # Add request to history
            if "request_history" not in shared_memory.data:
                shared_memory.set("request_history", [])
            
            request_history = shared_memory.get("request_history", [])
            request_history.append({
                "request": message,
                "timestamp": datetime.now().isoformat()
            })
            shared_memory.set("request_history", request_history[-5:])  # Keep last 5 requests
            
            # Execute the graph with the user input
            result = self.graph.execute(message)
            
            # Process results from each agent
            response_content = ""
            routed_agents = []
            agent_results = {}
            debug_logs = []  # Thêm logs để debug
            
            for node_id, node_result in result.results.items():
                agent_results_list = node_result.get_agent_results()
                for i, agent_result in enumerate(agent_results_list):
                    # Get agent from agents dictionary
                    agent = self.agents.get(node_id)
                    if agent:
                        # Update agent status
                        agent.update_agent_status(f"Processed request: {message[:50]}...")
                    
                    # Extract content from agent result
                    content = ""
                    if hasattr(agent_result.message, 'content'):
                        message_content = agent_result.message["content"]
                        if isinstance(message_content, list):
                            for block in message_content:
                                if isinstance(block, dict) and 'text' in block:
                                    content += block['text']
                        elif isinstance(message_content, str):
                            content = message_content
                        
                        # If message is dict
                        elif isinstance(agent_result.message, dict):
                            if 'content' in agent_result.message:
                                if isinstance(agent_result.message['content'], list):
                                    for block in agent_result.message['content']:
                                        if isinstance(block, dict) and 'text' in block:
                                            content += block['text']
                                elif isinstance(agent_result.message['content'], str):
                                    content = agent_result.message['content']
                    
                    # If all fails, try to get string from message
                    if not content and hasattr(agent_result, 'message'):
                        content = str(agent_result.message)
                    
                    # Save to shared memory with valuable content
                    shared_memory.set(f"{node_id}_response_{i}", {
                        "content": content if content else agent_result.message if hasattr(agent_result, 'message') else "No extractable content",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Add to response content
                    if content:
                        response_content += f"\n\n🤖 {node_id.title()}Agent:\n{content}"
                        routed_agents.append(node_id)
                        agent_results[node_id] = content
                    
                    # Thêm debug logs
                    debug_info = {
                        "agent": node_id,
                        "message_type": str(type(agent_result.message)),
                        "content_length": len(content) if content else 0,
                        "timestamp": datetime.now().isoformat()
                    }
                    debug_logs.append(debug_info)
                
                # Update workflow status
                shared_memory.set("workflow_status", "completed")
                shared_memory.set("completion_timestamp", datetime.now().isoformat())
                
                # Update conversation state based on agents that were executed
                if "detect" in routed_agents:
                    session_state["conversation_state"] = "detection_complete"
                elif "analyzer" in routed_agents:
                    session_state["conversation_state"] = "analysis_complete"
                elif "remediate" in routed_agents:
                    session_state["conversation_state"] = "remediation_complete"
                elif "report" in routed_agents:
                    session_state["conversation_state"] = "report_complete"
                else:
                    session_state["conversation_state"] = "conversational"
                
                # Generate suggestions based on current state
                suggestions = self._generate_suggestions(session_state["conversation_state"], routed_agents[0] if routed_agents else None)
                
                # Add assistant response to conversation history
                session_state["conversation_history"].append({
                    "role": "assistant",
                    "message": response_content,
                    "routed_agents": routed_agents,
                    "agent_results": agent_results,
                    "timestamp": datetime.now()
                })
                
                # Update session timestamp
                session_state["timestamp"] = datetime.now()
                
                # Lưu debug logs vào session state
                if "debug_mode" in session_state and session_state["debug_mode"]:
                    if "debug_logs" not in session_state:
                        session_state["debug_logs"] = []
                    session_state["debug_logs"].extend(debug_logs)
                    
                    # Giới hạn số lượng logs lưu trữ
                    if len(session_state["debug_logs"]) > 100:
                        session_state["debug_logs"] = session_state["debug_logs"][-100:]
                
            return {
                "session_id": session_id,
                "response": response_content,
                "routed_agent": routed_agents[0] if routed_agents else None,
                "agent_result": str(agent_results)[:1000] + "..." if len(str(agent_results)) > 1000 else str(agent_results),
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
    
    # Note: _execute_agent and _execute_agent_streaming methods removed
    # The graph-based approach now handles agent execution directly
    
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
        elif conversation_state == "report_complete":
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
            "rest_chat": "/chat",
            "test_client": "/test-client"
        }
    }



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
    
    
    # Clear session state
    del session_states[session_id]
    
    return {"message": f"Session {session_id} cleared successfully"}


@app.delete("/sessions", summary="Clear All Sessions")
async def clear_all_sessions():
    """Clear all sessions and reset shared memory"""
    global session_states
    
    
    # Clear session states and shared memory
    session_states.clear()
    shared_memory.clear()
    
    return {"message": "All sessions cleared, and shared memory reset"}


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


@app.post("/upload-terraform", response_model=TerraformUploadResponse, summary="Upload Terraform File/Folder and Run Plan")
async def upload_terraform_file(file: UploadFile = File(...)):
    """
    Upload a .tf file or a zip folder containing .tf files, replace existing terraform files, and run terraform plan
    
    This endpoint:
    1. Validates that the uploaded file is a .tf file or .zip folder
    2. If zip file, extracts only .tf files from the folder structure
    3. Clears the current terraform directory
    4. Saves the .tf files to the terraform directory
    5. Runs terraform plan to generate/update the .tfstate file
    
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
        
        logger.info(f"Processing upload of {'zip folder' if is_zip else 'terraform file'}: {file.filename}")
        
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
        
        extracted_files = []
        
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
                    
                    # Copy .tf files to terraform directory
                    for tf_file_path in tf_files_found:
                        file_name = os.path.basename(tf_file_path)
                        dest_path = os.path.join(terraform_dir, file_name)
                        
                        # Handle duplicate filenames by adding a number suffix
                        counter = 1
                        original_name = file_name
                        while os.path.exists(dest_path):
                            name_without_ext = os.path.splitext(original_name)[0]
                            ext = os.path.splitext(original_name)[1]
                            file_name = f"{name_without_ext}_{counter}{ext}"
                            dest_path = os.path.join(terraform_dir, file_name)
                            counter += 1
                        
                        shutil.copy2(tf_file_path, dest_path)
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
            new_file_path = os.path.join(terraform_dir, file.filename)
            
            with open(new_file_path, 'wb') as f:
                f.write(content)
            
            extracted_files.append(file.filename)
            logger.info(f"Saved terraform file: {new_file_path}")
        
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
        logger.info("Running terraform plan using ExecuteTerraformCommand...")
        plan_result = terraform_plan(
            
            terraform_dir=terraform_dir
        )
        apply_result = terraform_apply(
            terraform_dir=terraform_dir,
            auto_approve=True
        )
        
        # Update shared memory with the upload results
        shared_memory.set("last_uploaded_file", file.filename)
        shared_memory.set("extracted_terraform_files", extracted_files)
        shared_memory.set("terraform_plan_result", plan_result)
        shared_memory.set("terraform_apply_result", apply_result)
        shared_memory.set("terraform_directory", terraform_dir)
        
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
        return TerraformUploadResponse(
            message=message,
            filename=file.filename,
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
        
        # Send test notification
        subject = "AWS Drift Monitoring - Test Notification"
        message = custom_message or "This is a test notification from the AWS Drift Monitoring System."
        
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