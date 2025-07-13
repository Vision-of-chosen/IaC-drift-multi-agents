#!/usr/bin/env python3
"""
FastAPI Application for Terraform Drift Detection & Remediation System

This module provides a REST API interface for the multi-agent Terraform drift detection system,
designed to be used with chatbot applications.
"""

import logging
import os
import sys
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add tools to path
sys.path.append("tools/src")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import system components
from shared_memory import shared_memory, initialize_shared_memory
from chat_interface import create_terraform_drift_system
from config import (
    BEDROCK_MODEL_ID,
    BEDROCK_REGION,
    TERRAFORM_DIR,
    SHARED_MEMORY_KEYS,
    CHAT_COMMANDS
)

# Initialize FastAPI app
app = FastAPI(
    title="Terraform Drift Detection API",
    description="Multi-agent system for detecting and remediating Terraform infrastructure drift",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for your environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
terraform_system = None
system_agents = None

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to process")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")

class ChatResponse(BaseModel):
    response: str = Field(..., description="System response")
    status: str = Field(..., description="Processing status")
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = Field(None, description="Session identifier")
    agent_results: Optional[Dict[str, Any]] = Field(None, description="Detailed agent results")

class StatusResponse(BaseModel):
    status: str = Field(..., description="System status")
    terraform_dir: str = Field(..., description="Terraform directory path")
    aws_region: str = Field(..., description="AWS region")
    model_id: str = Field(..., description="AI model identifier")
    agents_count: int = Field(..., description="Number of active agents")
    workflow_status: str = Field(..., description="Current workflow status")
    tf_files_count: int = Field(..., description="Number of Terraform files")
    state_files_count: int = Field(..., description="Number of state files")

class SharedMemoryResponse(BaseModel):
    shared_memory: Dict[str, Any] = Field(..., description="Current shared memory contents")
    timestamp: datetime = Field(default_factory=datetime.now)

class CommandsResponse(BaseModel):
    commands: Dict[str, str] = Field(..., description="Available commands")
    description: str = Field(..., description="System description")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    global terraform_system, system_agents
    
    try:
        logger.info("🚀 Initializing Terraform Drift Detection System...")
        
        # Ensure terraform directory exists
        os.makedirs(TERRAFORM_DIR, exist_ok=True)
        
        # Initialize shared memory
        initialize_shared_memory()
        
        # Create the system
        terraform_system, system_agents = create_terraform_drift_system()
        
        logger.info("✅ System initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down Terraform Drift Detection System")
    if shared_memory:
        shared_memory.clear()

# Dependency to get system instance
async def get_system():
    """Dependency to get the system instance"""
    if terraform_system is None:
        raise HTTPException(
            status_code=503,
            detail="System not initialized"
        )
    return terraform_system

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Terraform Drift Detection & Remediation System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "chat": "/chat",
            "status": "/status",
            "memory": "/memory",
            "commands": "/commands"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "system_initialized": terraform_system is not None
    }

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    system = Depends(get_system)
):
    """
    Main chat endpoint for processing user messages through the multi-agent system
    """
    try:
        logger.info(f"Processing chat request: {request.message}")
        
        # Store user request in shared memory
        shared_memory.set("user_request", request.message)
        shared_memory.set("workflow_status", "initiated")
        
        # Process special commands
        if request.message.lower() == "help":
            return ChatResponse(
                response=_get_help_response(),
                status="completed",
                session_id=request.session_id
            )
        
        if request.message.lower() == "status":
            status_info = await get_system_status()
            return ChatResponse(
                response=_format_status_response(status_info),
                status="completed",
                session_id=request.session_id
            )
        
        if request.message.lower() == "memory":
            memory_info = await get_shared_memory()
            return ChatResponse(
                response=_format_memory_response(memory_info),
                status="completed",
                session_id=request.session_id
            )
        
        # Execute the graph with the user input
        result = system.execute(request.message)
        
        # Process agent results
        agent_results = {}
        response_text = ""
        
        for node_id, node_result in result.results.items():
            agent_results[node_id] = []
            agent_responses = node_result.get_agent_results()
            
            for agent_result in agent_responses:
                if hasattr(agent_result, 'message') and agent_result.message:
                    content = ""
                    if hasattr(agent_result.message, 'content'):
                        for block in agent_result.message.content:
                            if isinstance(block, dict) and 'text' in block:
                                content += block['text']
                    
                    agent_results[node_id].append({
                        "content": content,
                        "timestamp": datetime.now()
                    })
                    
                    # Add to response text
                    response_text += f"\n🤖 {node_id.title()}Agent:\n{content}\n"
        
        # Update workflow status
        shared_memory.set("workflow_status", "completed")
        
        return ChatResponse(
            response=response_text.strip() if response_text else "Request processed successfully",
            status="completed",
            session_id=request.session_id,
            agent_results=agent_results
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        shared_memory.set("workflow_status", "failed")
        shared_memory.set("last_error", str(e))
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

# System status endpoint
@app.get("/status", response_model=StatusResponse)
async def get_system_status():
    """Get current system status"""
    try:
        # Count terraform files
        tf_files_count = 0
        state_files_count = 0
        
        if os.path.exists(TERRAFORM_DIR):
            tf_files = list(Path(TERRAFORM_DIR).glob("*.tf"))
            state_files = list(Path(TERRAFORM_DIR).glob("*.tfstate"))
            tf_files_count = len(tf_files)
            state_files_count = len(state_files)
        
        workflow_status = shared_memory.get("workflow_status", "idle")
        
        return StatusResponse(
            status="running",
            terraform_dir=TERRAFORM_DIR,
            aws_region=BEDROCK_REGION,
            model_id=BEDROCK_MODEL_ID,
            agents_count=len(system_agents) if system_agents else 0,
            workflow_status=workflow_status,
            tf_files_count=tf_files_count,
            state_files_count=state_files_count
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting system status: {str(e)}"
        )

# Shared memory endpoint
@app.get("/memory", response_model=SharedMemoryResponse)
async def get_shared_memory():
    """Get current shared memory contents"""
    try:
        return SharedMemoryResponse(
            shared_memory=shared_memory.data
        )
        
    except Exception as e:
        logger.error(f"Error getting shared memory: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting shared memory: {str(e)}"
        )

# Commands endpoint
@app.get("/commands", response_model=CommandsResponse)
async def get_commands():
    """Get available commands"""
    return CommandsResponse(
        commands=CHAT_COMMANDS,
        description="Available commands for the Terraform Drift Detection System"
    )

# Clear shared memory endpoint
@app.post("/memory/clear")
async def clear_shared_memory():
    """Clear shared memory"""
    try:
        shared_memory.clear()
        initialize_shared_memory()
        return {"message": "Shared memory cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing shared memory: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing shared memory: {str(e)}"
        )

# Workflow status endpoint
@app.get("/workflow/status")
async def get_workflow_status():
    """Get current workflow status"""
    return {
        "workflow_status": shared_memory.get("workflow_status", "idle"),
        "user_request": shared_memory.get("user_request", ""),
        "last_error": shared_memory.get("last_error", ""),
        "timestamp": datetime.now()
    }

# Helper functions
def _get_help_response() -> str:
    """Generate help response"""
    help_text = """
🔧 Terraform Drift Detection & Remediation System

Available Commands:
• detect - Run drift detection
• analyze - Analyze detected drift
• remediate - Apply drift remediation
• status - Check system status
• memory - View shared memory
• help - Show this help

Example Usage:
• "detect"
• "analyze high priority drift"
• "remediate security issues only"

The system uses 4 specialized agents:
• OrchestrationAgent - Coordinates the workflow
• DetectAgent - Finds drift between Terraform state and AWS
• DriftAnalyzerAgent - Analyzes impact and provides recommendations
• RemediateAgent - Applies fixes to remediate drift
"""
    return help_text.strip()

def _format_status_response(status_info: StatusResponse) -> str:
    """Format status response as text"""
    return f"""
System Status:
• Status: {status_info.status}
• Terraform Directory: {status_info.terraform_dir}
• AWS Region: {status_info.aws_region}
• Model: {status_info.model_id}
• Active Agents: {status_info.agents_count}
• Workflow Status: {status_info.workflow_status}
• Terraform Files: {status_info.tf_files_count} .tf files, {status_info.state_files_count} state files
""".strip()

def _format_memory_response(memory_info: SharedMemoryResponse) -> str:
    """Format memory response as text"""
    memory_text = "Shared Memory Contents:\n"
    for key, value in memory_info.shared_memory.items():
        memory_text += f"• {key}: {str(value)[:100]}...\n"
    return memory_text.strip()

# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).dict()
    )

# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Terraform Drift Detection API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "fastapi_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    ) 