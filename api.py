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
import glob
import zipfile
import tempfile
import shutil

from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
import json

# Add tools to path
sys.path.append("tools/src")

from strands.models.bedrock import BedrockModel
from agents import OrchestrationAgent, DetectAgent, DriftAnalyzerAgent, RemediateAgent
from shared_memory import shared_memory
from config import BEDROCK_MODEL_ID, BEDROCK_REGION, TERRAFORM_DIR
from permission_handlers import permission_manager

# Add tools to path
sys.path.append("tools/src")
sys.path.append("useful_tools")

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
    - REST API endpoints for chat-based agent interaction
    - Multi-agent orchestration with intelligent routing
    - Session-based conversation management
    - Permission-based agent tool approval system
    
    Chat API Usage:
    - Use POST /chat to send messages to agents
    - Agents may request permissions for potentially dangerous operations
    - When permission is needed, the response includes permission_request data
    - Respond to permissions using POST /permission/respond
    - Continue agent execution using POST /chat/continue/{session_id}
    
    Permission System:
    - Agents request approval before executing potentially dangerous tools
    - Users can approve, deny, or set permanent policies (always/never)
    - Permission requests pause agent execution until resolved
    - REST endpoints for managing permissions and continuing execution
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

# Pydantic Models
class ChatMessage(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")

class TerraformUploadResponse(BaseModel):
    message: str
    filename: str
    extracted_files: list[str] = Field(default_factory=list, description="List of .tf files extracted from the upload")
    terraform_plan_result: Dict[str, Any]
    success: bool
    timestamp: datetime


class PermissionRequest(BaseModel):
    request_id: str
    agent_name: str
    function_name: str
    parameters: dict
    timestamp: str
    session_id: Optional[str] = None


class PermissionResponse(BaseModel):
    request_id: str = Field(..., description="The approval request ID")
    approved: bool = Field(..., description="Whether the request was approved")
    response: str = Field(..., description="Response type: yes/no/always/never/deny-all")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    routed_agent: Optional[str] = Field(None, description="Which agent was invoked, if any")
    agent_result: Optional[str] = Field(None, description="Result from the invoked agent")
    conversation_state: str
    timestamp: datetime
    suggestions: list[str] = Field(default_factory=list, description="Suggested next actions")
    permission_request: Optional[PermissionRequest] = Field(None, description="Pending permission request if agent needs approval")
    agent_paused: bool = Field(False, description="Whether agent execution is paused waiting for permission")


class WorkflowStatus(BaseModel):
    session_id: str
    conversation_state: str
    message: str
    timestamp: datetime
    shared_memory_keys: list[str]
    pending_permission: Optional[PermissionRequest] = Field(None, description="Current pending permission request")


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
        # Remove WebSocket callback setup since we're using REST API
        
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
            "context": {},
            "permission_request": None,
            "agent_paused": False
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
            # Check if there's a pending permission request that needs to be resolved first
            current_permission = permission_manager.get_current_permission_request()
            if current_permission and session_state.get("agent_paused", False):
                return {
                    "session_id": session_id,
                    "response": "⚠️ There is a pending permission request that must be resolved before continuing. Please approve or deny the current request.",
                    "routed_agent": None,
                    "agent_result": None,
                    "conversation_state": session_state["conversation_state"],
                    "timestamp": datetime.now(),
                    "suggestions": ["Check pending permissions", "Approve the current request", "Deny the current request"],
                    "permission_request": PermissionRequest(**current_permission),
                    "agent_paused": True
                }
            
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
                try:
                    agent_result = self._execute_agent('detect', message, session_id)
                    session_state["conversation_state"] = "detection_complete"
                except PermissionRequestException as e:
                    # Agent execution was paused for permission
                    session_state["agent_paused"] = True
                    session_state["permission_request"] = e.permission_request
                    
                    return {
                        "session_id": session_id,
                        "response": f"{orchestration_response}\n\n⚠️ Agent execution paused - permission required to continue.",
                        "routed_agent": routed_agent,
                        "agent_result": None,
                        "conversation_state": session_state["conversation_state"],
                        "timestamp": datetime.now(),
                        "suggestions": ["Approve the permission request", "Deny the permission request"],
                        "permission_request": PermissionRequest(**e.permission_request),
                        "agent_paused": True
                    }
                
            elif "routing to driftanalyzeragent" in orchestration_response.lower() or "invoking analyzer" in orchestration_response.lower():
                routed_agent = "DriftAnalyzerAgent"
                try:
                    agent_result = self._execute_agent('analyzer', message, session_id)
                    session_state["conversation_state"] = "analysis_complete"
                except PermissionRequestException as e:
                    # Agent execution was paused for permission
                    session_state["agent_paused"] = True
                    session_state["permission_request"] = e.permission_request
                    
                    return {
                        "session_id": session_id,
                        "response": f"{orchestration_response}\n\n⚠️ Agent execution paused - permission required to continue.",
                        "routed_agent": routed_agent,
                        "agent_result": None,
                        "conversation_state": session_state["conversation_state"],
                        "timestamp": datetime.now(),
                        "suggestions": ["Approve the permission request", "Deny the permission request"],
                        "permission_request": PermissionRequest(**e.permission_request),
                        "agent_paused": True
                    }
                
            elif "routing to remediateagent" in orchestration_response.lower() or "invoking remediate" in orchestration_response.lower():
                routed_agent = "RemediateAgent"
                try:
                    agent_result = self._execute_agent('remediate', message, session_id)
                    session_state["conversation_state"] = "remediation_complete"
                except PermissionRequestException as e:
                    # Agent execution was paused for permission
                    session_state["agent_paused"] = True
                    session_state["permission_request"] = e.permission_request
                    
                    return {
                        "session_id": session_id,
                        "response": f"{orchestration_response}\n\n⚠️ Agent execution paused - permission required to continue.",
                        "routed_agent": routed_agent,
                        "agent_result": None,
                        "conversation_state": session_state["conversation_state"],
                        "timestamp": datetime.now(),
                        "suggestions": ["Approve the permission request", "Deny the permission request"],
                        "permission_request": PermissionRequest(**e.permission_request),
                        "agent_paused": True
                    }
            
            # Clear any previous permission state since agent completed successfully
            session_state["agent_paused"] = False
            session_state["permission_request"] = None
            
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
                "suggestions": suggestions,
                "permission_request": None,
                "agent_paused": False
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
                "suggestions": ["Try rephrasing your request", "Ask for help", "Check system status"],
                "permission_request": None,
                "agent_paused": False
            }

    def _execute_agent(self, agent_type: str, user_message: str, session_id: str = None) -> str:
        """Execute a specific agent and return its result"""
        try:
            agent = self.agents[agent_type].get_agent()
            
            # Update agent's shared memory
            self.agents[agent_type].update_shared_memory()
            
            # Set session context for permission requests
            if session_id:
                shared_memory.set("current_session_id", session_id)
            
            # Clear any previous permission requests
            permission_manager.clear_current_permission_request()
            
            # Execute the agent
            result = agent(user_message)
            
            # Check if there's a pending permission request after execution
            current_permission = permission_manager.get_current_permission_request()
            if current_permission:
                # Agent execution was interrupted by permission request
                raise PermissionRequestException(current_permission)
            
            # Store result in shared memory
            shared_memory.set(f"{agent_type}_last_result", str(result))
            
            return str(result)
            
        except PermissionRequestException:
            # Re-raise permission exceptions
            raise
        except Exception as e:
            logger.error(f"Error executing {agent_type} agent: {e}")
            return f"Error executing {agent_type}: {str(e)}"

    # Remove the streaming methods since we're using REST API only
    
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


# Custom exception for permission requests
class PermissionRequestException(Exception):
    def __init__(self, permission_request: dict):
        self.permission_request = permission_request
        super().__init__(f"Permission required for {permission_request.get('function_name', 'unknown')}")


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
            "chat": "/chat",
            "start_session": "/start-session",
            "continue_execution": "/chat/continue/{session_id}",
            "permissions": {
                "respond": "/permission/respond",
                "pending": "/permissions/pending", 
                "status": "/permissions/status",
                "clear_denied": "/permissions/clear-denied"
            },
            "session_management": {
                "status": "/status/{session_id}",
                "conversation_history": "/conversation/{session_id}",
                "clear_session": "/session/{session_id}",
                "clear_all": "/sessions"
            }
        },
        "features": [
            "REST API chat with agents",
            "Permission-based tool approval",
            "Session management",
            "Multi-agent orchestration",
            "Pause/resume agent execution"
        ]
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


@app.post("/permission/respond", summary="Respond to Permission Request")
async def respond_to_permission(response: PermissionResponse):
    """Respond to a permission request from an agent"""
    try:
        # Parse response type to determine approval
        approved = response.response.lower() in ("yes", "y", "approve", "always", "a")
        
        # Handle the response
        success = permission_manager.handle_approval_response(
            response.request_id, 
            approved, 
            response.response
        )
        
        if success:
            return {
                "message": f"Permission response processed successfully",
                "request_id": response.request_id,
                "approved": approved,
                "response_type": response.response
            }
        else:
            raise HTTPException(status_code=404, detail="Permission request not found")
            
    except Exception as e:
        logger.error(f"Error processing permission response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process permission response: {e}")


@app.post("/chat/continue/{session_id}", summary="Continue Agent Execution After Permission")
async def continue_agent_execution(session_id: str):
    """Continue agent execution after a permission request has been resolved"""
    try:
        if session_id not in session_states:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_state = session_states[session_id]
        
        # Check if there's a permission request that has been resolved
        current_permission = permission_manager.get_current_permission_request()
        if current_permission:
            raise HTTPException(status_code=400, detail="Permission request still pending")
        
        # Check if agent was paused
        if not session_state.get("agent_paused", False):
            raise HTTPException(status_code=400, detail="No paused agent execution to continue")
        
        # Clear the paused state
        session_state["agent_paused"] = False
        session_state["permission_request"] = None
        
        # Get the last message to retry agent execution
        last_user_message = None
        for msg in reversed(session_state.get("conversation_history", [])):
            if msg.get("role") == "user":
                last_user_message = msg.get("message")
                break
        
        if not last_user_message:
            raise HTTPException(status_code=400, detail="No user message to retry")
        
        # Process the message again (this time permissions should be resolved)
        result = await orchestrator.process_chat_message(session_id, last_user_message)
        
        return ChatResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error continuing agent execution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to continue agent execution: {e}")


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
        init_result = terraform_run_command(
            command="init",
            working_directory=terraform_dir
        )
        
        if not init_result.get("success", False):
            logger.warning(f"Terraform init failed: {init_result.get('output', 'Unknown error')}")
            # Continue anyway, as some configurations might not need init
        
        # Use terraform_run_command with ExecuteTerraformCommand from awblab MCP
        logger.info("Running terraform plan using ExecuteTerraformCommand...")
        plan_result = terraform_run_command(
            command="plan",
            working_directory=terraform_dir
        )
        
        # Update shared memory with the upload results
        shared_memory.set("last_uploaded_file", file.filename)
        shared_memory.set("extracted_terraform_files", extracted_files)
        shared_memory.set("terraform_plan_result", plan_result)
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


@app.get("/permissions/pending", summary="Get Pending Permission Requests")
async def get_pending_permissions():
    """Get all pending permission requests"""
    try:
        pending = permission_manager.get_pending_approvals()
        return {
            "pending_permissions": pending,
            "count": len(pending),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting pending permissions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pending permissions: {e}")


@app.delete("/permissions/clear-denied", summary="Clear Denied Tools")
async def clear_denied_tools():
    """Clear all globally denied tools to allow them to be requested again"""
    try:
        permission_manager.clear_denied_tools()
        return {
            "message": "All denied tools cleared successfully",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error clearing denied tools: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear denied tools: {e}")


@app.get("/permissions/status", summary="Get Permission Manager Status")
async def get_permission_status():
    """Get current permission manager status"""
    try:
        from permission_handlers import get_permission_status
        status = get_permission_status()
        status["timestamp"] = datetime.now()
        return status
    except Exception as e:
        logger.error(f"Error getting permission status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get permission status: {e}")


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
    
    # Get current permission request if any
    current_permission = permission_manager.get_current_permission_request()
    pending_permission = None
    if current_permission:
        pending_permission = PermissionRequest(**current_permission)
    
    return WorkflowStatus(
        session_id=session_id,
        conversation_state=session_state["conversation_state"],
        message=f"Session active. Current state: {session_state['conversation_state']}. {len(session_state.get('conversation_history', []))} messages in conversation.",
        timestamp=session_state["timestamp"],
        shared_memory_keys=shared_memory.keys(),
        pending_permission=pending_permission
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
    
    return {"message": "All sessions cleared, shared memory reset"}


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