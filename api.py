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
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import json

# Add tools to path
sys.path.append("tools/src")

from strands.models.bedrock import BedrockModel
from strands.multiagent.graph import GraphBuilder
from agents import OrchestrationAgent, DetectAgent, DriftAnalyzerAgent, RemediateAgent
from shared_memory import shared_memory
from config import BEDROCK_MODEL_ID, BEDROCK_REGION, TERRAFORM_DIR
from permission_handlers import permission_manager

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
        
        # Build the agent graph using GraphBuilder
        builder = GraphBuilder()
        
        # Add agents as nodes
        orchestration_node = builder.add_node(orchestration_agent.get_agent(), "orchestration")
        detect_node = builder.add_node(detect_agent.get_agent(), "detect")
        analyzer_node = builder.add_node(drift_analyzer_agent.get_agent(), "analyzer")
        remediate_node = builder.add_node(remediate_agent.get_agent(), "remediate")
        
        # Define the workflow edges - all agents connect directly to Orchestration Agent
        # Orchestration → DetectAgent
        builder.add_edge(orchestration_node, detect_node)
        
        # Orchestration → DriftAnalyzerAgent
        builder.add_edge(orchestration_node, analyzer_node)
        
        # Orchestration → RemediateAgent
        builder.add_edge(orchestration_node, remediate_node)
        
        # Build the graph
        graph = builder.build()
        
        agents = {
            'orchestration': orchestration_agent,
            'detect': detect_agent, 
            'analyzer': drift_analyzer_agent,
            'remediate': remediate_agent
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
                    if hasattr(agent_result, 'message') and agent_result.message:
                        # Try to get from message.content as list
                        if hasattr(agent_result.message, 'content'):
                            if isinstance(agent_result.message.content, list):
                                for block in agent_result.message.content:
                                    if isinstance(block, dict) and 'text' in block:
                                        content += block['text']
                            # If content is string
                            elif isinstance(agent_result.message.content, str):
                                content = agent_result.message.content
                        
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

    async def process_streaming_chat_message(self, session_id: str, message: str, connection_manager: ConnectionManager):
        """Process a chat message with streaming responses using the multi-agent graph system"""
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
            
            # Send status update
            await connection_manager.send_stream_chunk(session_id, "Executing multi-agent system...", "status")
            
            # Execute the graph with the user input
            result = self.graph.execute(message)
            
            # Process results from each agent with streaming
            response_content = ""
            routed_agents = []
            agent_results = {}
            
            for node_id, node_result in result.results.items():
                agent_results_list = node_result.get_agent_results()
                for i, agent_result in enumerate(agent_results_list):
                    # Get agent from agents dictionary
                    agent = self.agents.get(node_id)
                    if agent:
                        # Update agent status
                        agent.update_agent_status(f"Processed request: {message[:50]}...")
                        # Send streaming update
                        await connection_manager.send_stream_chunk(session_id, f"Processing with {node_id.title()}Agent...", "status")
                    
                    # Extract content from agent result
                    content = ""
                    if hasattr(agent_result, 'message') and agent_result.message:
                        # Try to get from message.content as list
                        if hasattr(agent_result.message, 'content'):
                            if isinstance(agent_result.message.content, list):
                                for block in agent_result.message.content:
                                    if isinstance(block, dict) and 'text' in block:
                                        content += block['text']
                            # If content is string
                            elif isinstance(agent_result.message.content, str):
                                content = agent_result.message.content
                        
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
                    
                    # Stream the agent result
                    if content:
                        agent_response = f"\n\n🤖 {node_id.title()}Agent:\n{content}"
                        await connection_manager.send_stream_chunk(session_id, agent_response, "agent_result")
                        response_content += agent_response
                        routed_agents.append(node_id)
                        agent_results[node_id] = content
            
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
            
            # Send final response with suggestions
            final_response = {
                "type": "complete",
                "session_id": session_id,
                "routed_agent": routed_agents[0] if routed_agents else None,
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
                    await orchestrator.process_streaming_chat_message(session_id, user_message, connection_manager)
                
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


@app.get("/approve/{approval_id}", summary="Approve or deny a pending action")
async def approve_action(approval_id: str, approved: bool):
    """Approve or deny a pending tool execution request"""
    if approval_id not in permission_manager.pending_approvals:
        raise HTTPException(status_code=404, detail="Approval ID not found or already processed")

    # Update the status of the pending approval
    if approved:
        permission_manager.pending_approvals[approval_id]["status"] = "approved"
        return {"message": f"Request {approval_id} approved."}
    else:
        permission_manager.pending_approvals[approval_id]["status"] = "denied"
        return {"message": f"Request {approval_id} denied."}


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