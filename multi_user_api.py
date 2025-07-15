#!/usr/bin/env python3
"""
Multi-User FastAPI Web Application for Terraform Drift Detection & Remediation System

This module provides a web-based API interface that coordinates the multi-agent
system with multi-user support, using AWS_ACCESS_KEY_ID as username.

Features:
- Multi-user support with AWS_ACCESS_KEY_ID as username
- User-specific terraform directories
- Database persistence for conversations and messages
- User-scoped shared memory
- WebSocket streaming for real-time chat interactions
- REST API endpoints for user management and operations
"""

import logging
import os
import sys
import shutil
import glob
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, UploadFile, File, Depends, Header
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
from multi_user_shared_memory import multi_user_shared_memory, get_user_memory
from database import db_manager, User, Conversation, Message
from config import BEDROCK_MODEL_ID, BEDROCK_REGION, TERRAFORM_DIR, get_user_terraform_dir

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
    title="Multi-User Terraform Drift Detection & Remediation API",
    description="""
    Multi-user coordination of drift detection, analysis, and remediation agents
    
    Features:
    - Multi-user support with AWS_ACCESS_KEY_ID as username
    - User-specific terraform file storage
    - Database persistence for conversations
    - User-scoped shared memory
    - WebSocket streaming for real-time chat interactions
    - REST API endpoints for user-specific operations
    
    Authentication:
    - Uses AWS_ACCESS_KEY_ID in headers for user identification
    - Each user gets isolated terraform directories and conversations
    
    Terraform File Upload:
    - POST /user/upload-terraform to upload .tf files
    - Files saved to user-specific directories
    - Runs terraform plan using MCP tools
    - Returns plan results and validation
    
    WebSocket Usage:
    - Connect to /ws/chat/{user_id}/{conversation_id} for streaming chat
    - Send messages as JSON: {"message": "your text here"}
    - Receive streaming responses with different message types:
      * "status": Processing updates
      * "response": Orchestrator responses
      * "agent_result": Agent execution results
      * "complete": Final response with suggestions
      * "error": Error messages
    """,
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket Connection Manager for multi-user
class MultiUserConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[int, WebSocket]] = {}  # user_id -> {conversation_id: websocket}
    
    async def connect(self, websocket: WebSocket, user_id: str, conversation_id: int):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = {}
        self.active_connections[user_id][conversation_id] = websocket
        logger.info(f"WebSocket connected for user: {user_id[:8]}... conversation: {conversation_id}")
    
    def disconnect(self, user_id: str, conversation_id: int):
        if user_id in self.active_connections and conversation_id in self.active_connections[user_id]:
            del self.active_connections[user_id][conversation_id]
            if not self.active_connections[user_id]:  # Remove user if no active conversations
                del self.active_connections[user_id]
            logger.info(f"WebSocket disconnected for user: {user_id[:8]}... conversation: {conversation_id}")
    
    async def send_message(self, user_id: str, conversation_id: int, message: dict):
        if user_id in self.active_connections and conversation_id in self.active_connections[user_id]:
            try:
                await self.active_connections[user_id][conversation_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {user_id[:8]}... conv {conversation_id}: {e}")
                self.disconnect(user_id, conversation_id)
    
    async def send_stream_chunk(self, user_id: str, conversation_id: int, chunk: str, message_type: str = "stream"):
        """Send a streaming chunk to the client"""
        message = {
            "type": message_type,
            "content": chunk,
            "timestamp": datetime.now().isoformat()
        }
        await self.send_message(user_id, conversation_id, message)

# Initialize connection manager
connection_manager = MultiUserConnectionManager()


# Pydantic Models
class UserAuth(BaseModel):
    aws_access_key_id: str = Field(..., description="AWS Access Key ID (used as username)")
    aws_secret_access_key: str = Field(..., description="AWS Secret Access Key")
    aws_region: str = Field(..., description="AWS Region (e.g., ap-southeast-2)")


class TerraformUploadResponse(BaseModel):
    message: str
    filename: str
    terraform_plan_result: Dict[str, Any]
    success: bool
    user_terraform_dir: str
    timestamp: datetime


class ChatMessage(BaseModel):
    message: str = Field(..., description="User message")
    conversation_id: Optional[int] = Field(None, description="Conversation ID for continuity")


class ChatResponse(BaseModel):
    conversation_id: int
    response: str
    routed_agent: Optional[str] = Field(None, description="Which agent was invoked, if any")
    agent_result: Optional[str] = Field(None, description="Result from the invoked agent")
    conversation_state: str
    timestamp: datetime
    suggestions: list[str] = Field(default_factory=list, description="Suggested next actions")


class ConversationSummary(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    status: str
    conversation_state: str
    message_count: int


class UserStatus(BaseModel):
    user_id: str
    aws_region: str
    created_at: datetime
    last_active: datetime
    conversation_count: int
    active_conversations: int
    terraform_dir: str


# Authentication dependency
async def get_current_user(aws_access_key_id: str = Header(...), 
                          aws_secret_access_key: str = Header(...), 
                          aws_region: str = Header(...)) -> str:
    """Extract and validate user from headers"""
    if not aws_access_key_id or not aws_access_key_id.startswith('AKIA'):
        raise HTTPException(status_code=401, detail="Invalid AWS Access Key ID format")
    
    if len(aws_access_key_id) != 20:
        raise HTTPException(status_code=401, detail="AWS Access Key ID should be 20 characters long")
    
    # Set environment variables for AWS operations
    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
    os.environ['AWS_REGION'] = aws_region
    os.environ['AWS_DEFAULT_REGION'] = aws_region
    
    # Create or update user in database
    user = db_manager.create_or_get_user(aws_access_key_id, aws_region)
    
    return aws_access_key_id


# Chat-based Multi-User Orchestrator Class
class MultiUserChatOrchestrator:
    """Intelligent conversation orchestrator with multi-user support"""
    
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
    
    async def initialize_conversation(self, user_id: str, conversation_id: Optional[int] = None) -> int:
        """Initialize a new conversation or get existing one"""
        if conversation_id:
            # Verify conversation exists and belongs to user
            conversation = db_manager.get_conversation(conversation_id)
            if conversation and conversation.user_id == user_id:
                return conversation_id
        
        # Create new conversation
        conversation = db_manager.create_conversation(user_id)
        
        # Initialize user-scoped shared memory
        user_memory = get_user_memory(user_id, conversation.id)
        user_memory.set("conversation_state", "idle")
        user_memory.set("user_terraform_dir", get_user_terraform_dir(user_id))
        
        return conversation.id
    
    async def process_chat_message(self, user_id: str, conversation_id: int, message: str) -> Dict[str, Any]:
        """Process a chat message with user-scoped context"""
        try:
            # Get user-scoped memory
            user_memory = get_user_memory(user_id, conversation_id)
            
            # Get conversation from database
            conversation = db_manager.get_conversation(conversation_id)
            if not conversation or conversation.user_id != user_id:
                raise HTTPException(status_code=403, detail="Conversation not found or access denied")
            
            # Add user message to database
            db_manager.add_message(conversation_id, "user", message)
            
            # Update user-scoped shared memory with context
            user_memory.set("current_user_message", message)
            user_memory.set("user_terraform_dir", get_user_terraform_dir(user_id))
            
            # Get recent conversation history
            recent_messages = db_manager.get_conversation_messages(conversation_id, limit=10)
            conversation_history = [
                {
                    "role": msg.role,
                    "message": msg.content,
                    "timestamp": msg.created_at.isoformat()
                }
                for msg in recent_messages[-5:]  # Last 5 messages for context
            ]
            
            user_memory.set("conversation_history", conversation_history)
            
            # Update agent shared memory
            self._update_agents_memory(user_memory)
            
            # Create context-aware prompt for orchestrator
            context_prompt = f"""
User message: "{message}"

Conversation history: {conversation_history}

Current conversation state: {conversation.conversation_state}

User terraform directory: {get_user_terraform_dir(user_id)}

Based on this message and context, decide if you need to route to a specialized agent or handle this conversationally.

If routing to an agent:
1. Explain what agent you're invoking and why
2. Execute the agent with user-specific context
3. Summarize the results for the user
4. Suggest next steps

If handling conversationally:
1. Provide helpful guidance
2. Answer questions about the system
3. Ask clarifying questions if needed
"""
            
            # Execute orchestration agent
            orchestration_agent = self.agents['orchestration'].get_agent()
            orchestration_result = orchestration_agent(context_prompt)
            orchestration_response = str(orchestration_result)
            
            # Determine if orchestrator wants to route to a specific agent
            routed_agent = None
            agent_result = None
            
            if "routing to DetectAgent" in orchestration_response.lower() or "invoking detectagent" in orchestration_response.lower():
                routed_agent = "DetectAgent"
                agent_result = self._execute_agent('detect', message, user_memory)
                db_manager.update_conversation_state(conversation_id, "detection_complete")
                
            elif "routing to driftanalyzeragent" in orchestration_response.lower() or "invoking analyzer" in orchestration_response.lower():
                routed_agent = "DriftAnalyzerAgent"
                agent_result = self._execute_agent('analyzer', message, user_memory)
                db_manager.update_conversation_state(conversation_id, "analysis_complete")
                
            elif "routing to remediateagent" in orchestration_response.lower() or "invoking remediate" in orchestration_response.lower():
                routed_agent = "RemediateAgent"
                agent_result = self._execute_agent('remediate', message, user_memory)
                db_manager.update_conversation_state(conversation_id, "remediation_complete")
            
            # Generate suggestions based on current state
            conversation = db_manager.get_conversation(conversation_id)  # Refresh state
            suggestions = self._generate_suggestions(conversation.conversation_state, routed_agent)
            
            # Add assistant response to database
            db_manager.add_message(
                conversation_id, 
                "assistant", 
                orchestration_response,
                routed_agent,
                agent_result
            )
            
            return {
                "conversation_id": conversation_id,
                "response": orchestration_response,
                "routed_agent": routed_agent,
                "agent_result": agent_result[:1000] + "..." if agent_result and len(agent_result) > 1000 else agent_result,
                "conversation_state": conversation.conversation_state,
                "timestamp": datetime.now(),
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            error_response = f"I encountered an error while processing your message: {str(e)}. Please try again or rephrase your request."
            
            # Add error message to database
            db_manager.add_message(conversation_id, "assistant", error_response, None, None, {"error": True})
            
            return {
                "conversation_id": conversation_id,
                "response": error_response,
                "routed_agent": None,
                "agent_result": None,
                "conversation_state": db_manager.get_conversation(conversation_id).conversation_state,
                "timestamp": datetime.now(),
                "suggestions": ["Try rephrasing your request", "Ask for help", "Check system status"]
            }

    async def process_streaming_chat_message(self, user_id: str, conversation_id: int, message: str):
        """Process a chat message with streaming responses"""
        try:
            # Get user-scoped memory
            user_memory = get_user_memory(user_id, conversation_id)
            
            # Get conversation from database
            conversation = db_manager.get_conversation(conversation_id)
            if not conversation or conversation.user_id != user_id:
                await connection_manager.send_stream_chunk(user_id, conversation_id, "Access denied", "error")
                return
            
            # Send initial acknowledgment
            await connection_manager.send_stream_chunk(user_id, conversation_id, "Processing your message...", "status")
            
            # Add user message to database
            db_manager.add_message(conversation_id, "user", message)
            
            # Update user-scoped shared memory
            user_memory.set("current_user_message", message)
            user_memory.set("user_terraform_dir", get_user_terraform_dir(user_id))
            
            # Get recent conversation history
            recent_messages = db_manager.get_conversation_messages(conversation_id, limit=10)
            conversation_history = [
                {
                    "role": msg.role,
                    "message": msg.content,
                    "timestamp": msg.created_at.isoformat()
                }
                for msg in recent_messages[-5:]
            ]
            
            user_memory.set("conversation_history", conversation_history)
            
            # Update agent shared memory
            self._update_agents_memory(user_memory)
            
            await connection_manager.send_stream_chunk(user_id, conversation_id, "Analyzing your request...", "status")
            
            # Create context-aware prompt for orchestrator
            context_prompt = f"""
User message: "{message}"

Conversation history: {conversation_history}

Current conversation state: {conversation.conversation_state}

User terraform directory: {get_user_terraform_dir(user_id)}

Based on this message and context, decide if you need to route to a specialized agent or handle this conversationally.
"""
            
            # Execute orchestration agent
            orchestration_agent = self.agents['orchestration'].get_agent()
            orchestration_result = orchestration_agent(context_prompt)
            orchestration_response = str(orchestration_result)
            
            # Stream the orchestration response
            await connection_manager.send_stream_chunk(user_id, conversation_id, orchestration_response, "response")
            
            # Determine if orchestrator wants to route to a specific agent
            routed_agent = None
            agent_result = None
            
            if "routing to DetectAgent" in orchestration_response.lower() or "invoking detectagent" in orchestration_response.lower():
                routed_agent = "DetectAgent"
                await connection_manager.send_stream_chunk(user_id, conversation_id, "Executing drift detection...", "status")
                agent_result = await self._execute_agent_streaming(user_id, conversation_id, 'detect', message, user_memory)
                db_manager.update_conversation_state(conversation_id, "detection_complete")
                
            elif "routing to driftanalyzeragent" in orchestration_response.lower() or "invoking analyzer" in orchestration_response.lower():
                routed_agent = "DriftAnalyzerAgent"
                await connection_manager.send_stream_chunk(user_id, conversation_id, "Analyzing drift patterns...", "status")
                agent_result = await self._execute_agent_streaming(user_id, conversation_id, 'analyzer', message, user_memory)
                db_manager.update_conversation_state(conversation_id, "analysis_complete")
                
            elif "routing to remediateagent" in orchestration_response.lower() or "invoking remediate" in orchestration_response.lower():
                routed_agent = "RemediateAgent"
                await connection_manager.send_stream_chunk(user_id, conversation_id, "Applying remediation...", "status")
                agent_result = await self._execute_agent_streaming(user_id, conversation_id, 'remediate', message, user_memory)
                db_manager.update_conversation_state(conversation_id, "remediation_complete")
            
            # Generate suggestions
            conversation = db_manager.get_conversation(conversation_id)  # Refresh state
            suggestions = self._generate_suggestions(conversation.conversation_state, routed_agent)
            
            # Add assistant response to database
            db_manager.add_message(
                conversation_id, 
                "assistant", 
                orchestration_response,
                routed_agent,
                agent_result
            )
            
            # Send final response
            final_response = {
                "type": "complete",
                "conversation_id": conversation_id,
                "routed_agent": routed_agent,
                "conversation_state": conversation.conversation_state,
                "timestamp": datetime.now().isoformat(),
                "suggestions": suggestions
            }
            
            await connection_manager.send_message(user_id, conversation_id, final_response)
            
        except Exception as e:
            logger.error(f"Error processing streaming chat message: {e}")
            error_response = f"I encountered an error while processing your message: {str(e)}. Please try again or rephrase your request."
            
            # Add error message to database
            db_manager.add_message(conversation_id, "assistant", error_response, None, None, {"error": True})
            
            await connection_manager.send_stream_chunk(user_id, conversation_id, error_response, "error")
            
            final_response = {
                "type": "error",
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "suggestions": ["Try rephrasing your request", "Ask for help", "Check system status"]
            }
            
            await connection_manager.send_message(user_id, conversation_id, final_response)
    
    def _update_agents_memory(self, user_memory):
        """Update all agents with user-scoped memory"""
        for agent_name, agent_wrapper in self.agents.items():
            # This would need to be updated in each agent to accept user memory
            agent_wrapper.user_memory = user_memory
            if hasattr(agent_wrapper, 'update_shared_memory'):
                agent_wrapper.update_shared_memory()
    
    def _execute_agent(self, agent_type: str, user_message: str, user_memory) -> str:
        """Execute a specific agent with user-scoped context"""
        try:
            agent = self.agents[agent_type].get_agent()
            
            # Update agent's memory with user context
            self._update_agents_memory(user_memory)
            
            # Execute the agent
            result = agent(user_message)
            
            # Store result in user-scoped memory
            user_memory.set(f"{agent_type}_last_result", str(result))
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error executing {agent_type} agent: {e}")
            return f"Error executing {agent_type}: {str(e)}"
    
    async def _execute_agent_streaming(self, user_id: str, conversation_id: int, agent_type: str, user_message: str, user_memory) -> str:
        """Execute a specific agent with streaming output"""
        try:
            agent = self.agents[agent_type].get_agent()
            
            # Update agent's memory with user context
            self._update_agents_memory(user_memory)
            
            # Send progress update
            await connection_manager.send_stream_chunk(user_id, conversation_id, f"Running {agent_type} agent...", "status")
            
            # Execute the agent
            result = agent(user_message)
            result_str = str(result)
            
            # Stream the result in chunks
            chunk_size = 200
            for i in range(0, len(result_str), chunk_size):
                chunk = result_str[i:i+chunk_size]
                await connection_manager.send_stream_chunk(user_id, conversation_id, chunk, "agent_result")
                await asyncio.sleep(0.1)  # Small delay for streaming effect
            
            # Store result in user-scoped memory
            user_memory.set(f"{agent_type}_last_result", result_str)
            
            return result_str
            
        except Exception as e:
            logger.error(f"Error executing {agent_type} agent: {e}")
            error_msg = f"Error executing {agent_type}: {str(e)}"
            await connection_manager.send_stream_chunk(user_id, conversation_id, error_msg, "error")
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
orchestrator = MultiUserChatOrchestrator()


# API Endpoints
@app.get("/", summary="API Health Check")
async def root():
    """Health check endpoint"""
    return {
        "message": "Multi-User Terraform Drift Detection & Remediation API",
        "status": "healthy",
        "version": "2.0.0",
        "features": [
            "Multi-user support with AWS_ACCESS_KEY_ID as username",
            "User-specific terraform directories",
            "Database persistence",
            "User-scoped shared memory",
            "WebSocket streaming chat",
            "REST API endpoints"
        ],
        "endpoints": {
            "websocket_chat": "/ws/chat/{user_id}/{conversation_id}",
            "rest_chat": "/user/chat",
            "upload_terraform": "/user/upload-terraform",
            "user_conversations": "/user/conversations",
            "user_status": "/user/status"
        }
    }


@app.post("/user/authenticate", summary="Authenticate User")
async def authenticate_user(auth: UserAuth):
    """Authenticate user and set up session"""
    # Set environment variables
    os.environ['AWS_ACCESS_KEY_ID'] = auth.aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = auth.aws_secret_access_key
    os.environ['AWS_REGION'] = auth.aws_region
    os.environ['AWS_DEFAULT_REGION'] = auth.aws_region
    
    # Create or update user in database
    user = db_manager.create_or_get_user(auth.aws_access_key_id, auth.aws_region)
    
    return {
        "message": "User authenticated successfully",
        "user_id": auth.aws_access_key_id,
        "aws_region": auth.aws_region,
        "terraform_dir": get_user_terraform_dir(auth.aws_access_key_id),
        "timestamp": datetime.now()
    }


@app.post("/user/chat", response_model=ChatResponse, summary="Chat with Terraform Drift Assistant")
async def chat(request: ChatMessage, user_id: str = Depends(get_current_user)):
    """Send a message to the intelligent Terraform drift assistant"""
    try:
        # Initialize conversation if needed
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = await orchestrator.initialize_conversation(user_id)
        
        # Process the chat message
        result = await orchestrator.process_chat_message(user_id, conversation_id, request.message)
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in chat processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat message: {e}")


@app.websocket("/ws/chat/{user_id}/{conversation_id}")
async def websocket_chat(websocket: WebSocket, user_id: str, conversation_id: int):
    """WebSocket endpoint for streaming chat with Terraform Drift Assistant"""
    await connection_manager.connect(websocket, user_id, conversation_id)
    
    try:
        # Verify user exists in database
        user = db_manager.get_user_by_aws_key(user_id)
        if not user:
            await connection_manager.send_stream_chunk(user_id, conversation_id, "User not found. Please authenticate first.", "error")
            return
        
        # Initialize conversation if needed
        if not db_manager.get_conversation(conversation_id):
            conversation_id = await orchestrator.initialize_conversation(user_id, conversation_id)
        
        # Send welcome message
        welcome_message = {
            "type": "welcome",
            "content": f"Connected to Terraform Drift Assistant. User: {user_id[:8]}..., Conversation: {conversation_id}",
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "suggestions": [
                "Ask me to check for infrastructure drift",
                "Request a scan of your Terraform resources",
                "Get help with the system capabilities"
            ]
        }
        await connection_manager.send_message(user_id, conversation_id, welcome_message)
        
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Extract user message
                user_message = message_data.get("message", "")
                
                if user_message.strip():
                    # Process the message with streaming response
                    await orchestrator.process_streaming_chat_message(user_id, conversation_id, user_message)
                
            except json.JSONDecodeError:
                await connection_manager.send_stream_chunk(
                    user_id, 
                    conversation_id,
                    "Invalid message format. Please send valid JSON.", 
                    "error"
                )
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await connection_manager.send_stream_chunk(
                    user_id, 
                    conversation_id,
                    f"Error processing your message: {str(e)}", 
                    "error"
                )
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user: {user_id[:8]}... conversation: {conversation_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id[:8]}... conversation {conversation_id}: {e}")
    finally:
        connection_manager.disconnect(user_id, conversation_id)


@app.post("/user/upload-terraform", response_model=TerraformUploadResponse, summary="Upload Terraform File")
async def upload_terraform_file(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    """
    Upload a .tf file to user-specific terraform directory
    
    This endpoint:
    1. Validates that the uploaded file is a .tf file
    2. Saves the file to the user's specific terraform directory
    3. Preserves existing files (no deletion)
    4. Runs terraform plan using MCP tools
    
    Returns the terraform plan results including any changes detected.
    """
    try:
        # Validate file extension
        if not file.filename or not file.filename.endswith('.tf'):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only .tf files are allowed."
            )
        
        logger.info(f"Processing upload of terraform file: {file.filename} for user {user_id[:8]}...")
        
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Get user-specific terraform directory
        user_terraform_dir = get_user_terraform_dir(user_id)
        
        # Save the terraform file (no deletion of existing files)
        new_file_path = os.path.join(user_terraform_dir, file.filename)
        
        with open(new_file_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"Saved terraform file: {new_file_path}")
        
        # Run terraform plan using MCP tools
        logger.info("Running terraform plan...")
        
        # Use terraform_run_command with user-specific directory
        plan_result = terraform_run_command(
            command="plan",
            working_directory=user_terraform_dir
        )
        
        if not plan_result.get("success", False):
            # If terraform_run_command fails, try with terraform_plan tool as fallback
            logger.warning("terraform_run_command failed, trying terraform_plan tool...")
            try:
                plan_result = terraform_plan(
                    terraform_dir=user_terraform_dir,
                    output_format="human"
                )
            except Exception as fallback_error:
                logger.error(f"Both terraform tools failed: {fallback_error}")
                plan_result = {
                    "error": f"Terraform plan failed: {str(fallback_error)}",
                    "output": "",
                    "success": False
                }
        
        # Update user-scoped shared memory
        user_memory = get_user_memory(user_id)
        user_memory.set("last_uploaded_file", file.filename)
        user_memory.set("terraform_plan_result", plan_result)
        user_memory.set("user_terraform_dir", user_terraform_dir)
        
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
            user_terraform_dir=user_terraform_dir,
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


@app.get("/user/conversations", response_model=List[ConversationSummary], summary="Get User Conversations")
async def get_user_conversations(user_id: str = Depends(get_current_user), limit: int = 10):
    """Get recent conversations for the authenticated user"""
    try:
        conversations = db_manager.get_user_conversations(user_id, limit)
        
        conversation_summaries = []
        for conv in conversations:
            messages = db_manager.get_conversation_messages(conv.id, limit=1)  # Get message count
            message_count = len(db_manager.get_conversation_messages(conv.id, limit=1000))  # Get actual count
            
            conversation_summaries.append(ConversationSummary(
                id=conv.id,
                title=conv.title,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                status=conv.status,
                conversation_state=conv.conversation_state,
                message_count=message_count
            ))
        
        return conversation_summaries
        
    except Exception as e:
        logger.error(f"Error getting user conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversations: {e}")


@app.get("/user/conversations/{conversation_id}/messages", summary="Get Conversation Messages")
async def get_conversation_messages(conversation_id: int, user_id: str = Depends(get_current_user), limit: int = 50):
    """Get messages for a specific conversation"""
    try:
        # Verify conversation belongs to user
        conversation = db_manager.get_conversation(conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=403, detail="Conversation not found or access denied")
        
        messages = db_manager.get_conversation_messages(conversation_id, limit)
        
        return {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "routed_agent": msg.routed_agent,
                    "agent_result": msg.agent_result,
                    "created_at": msg.created_at,
                    "metadata": msg.message_metadata
                }
                for msg in messages
            ],
            "total_messages": len(messages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {e}")


@app.post("/user/conversations", summary="Create New Conversation")
async def create_conversation(title: str = None, user_id: str = Depends(get_current_user)):
    """Create a new conversation for the user"""
    try:
        conversation = db_manager.create_conversation(user_id, title)
        
        return {
            "conversation_id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "status": conversation.status,
            "conversation_state": conversation.conversation_state
        }
        
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {e}")


@app.get("/user/status", response_model=UserStatus, summary="Get User Status")
async def get_user_status(user_id: str = Depends(get_current_user)):
    """Get status information for the authenticated user"""
    try:
        user = db_manager.get_user_by_aws_key(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        conversations = db_manager.get_user_conversations(user_id, limit=1000)
        active_conversations = len([c for c in conversations if c.status == "active"])
        
        return UserStatus(
            user_id=user_id,
            aws_region=user.aws_region,
            created_at=user.created_at,
            last_active=user.last_active,
            conversation_count=len(conversations),
            active_conversations=active_conversations,
            terraform_dir=get_user_terraform_dir(user_id)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user status: {e}")


@app.get("/user/shared-memory", summary="View User Shared Memory")
async def get_user_shared_memory(user_id: str = Depends(get_current_user), conversation_id: int = None):
    """Get user-scoped shared memory contents"""
    try:
        user_memory = get_user_memory(user_id, conversation_id)
        
        return {
            "user_id": user_id[:8] + "...",
            "conversation_id": conversation_id,
            "shared_memory": user_memory.data,
            "keys": user_memory.keys(),
            "size": user_memory.size(),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting user shared memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get shared memory: {e}")


@app.delete("/user/conversations/{conversation_id}", summary="Delete Conversation")
async def delete_conversation(conversation_id: int, user_id: str = Depends(get_current_user)):
    """Delete a specific conversation"""
    try:
        # Verify conversation belongs to user
        conversation = db_manager.get_conversation(conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=403, detail="Conversation not found or access denied")
        
        # Disconnect WebSocket if active
        connection_manager.disconnect(user_id, conversation_id)
        
        # Update conversation status to archived instead of deleting
        db_manager.update_conversation_state(conversation_id, "archived")
        
        # Clear conversation-specific shared memory
        user_memory = get_user_memory(user_id, conversation_id)
        user_memory.clear()
        
        return {"message": f"Conversation {conversation_id} archived successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {e}")


@app.get("/admin/stats", summary="Get System Statistics")
async def get_system_stats():
    """Get overall system statistics (admin endpoint)"""
    try:
        # Get database statistics
        with db_manager.get_session() as session:
            total_users = session.query(User).count()
            total_conversations = session.query(Conversation).count()
            total_messages = session.query(Message).count()
            active_conversations = session.query(Conversation).filter(Conversation.status == "active").count()
        
        # Get memory statistics
        memory_stats = {
            "total_memory_spaces": multi_user_shared_memory.size(),
            "active_users_in_memory": len(multi_user_shared_memory.get_all_users()),
            "active_websocket_connections": len(connection_manager.active_connections)
        }
        
        return {
            "database_stats": {
                "total_users": total_users,
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "active_conversations": active_conversations
            },
            "memory_stats": memory_stats,
            "system_health": "healthy",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system stats: {e}")


if __name__ == "__main__":
    # Initialize database
    from database import init_database
    init_database()
    
    # Ensure base terraform directory exists
    os.makedirs("./terraform_users", exist_ok=True)
    
    # Run the FastAPI server
    uvicorn.run(
        "multi_user_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 