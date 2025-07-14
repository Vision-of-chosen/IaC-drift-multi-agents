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
from agents import OrchestrationAgent, DetectAgent, DriftAnalyzerAgent, RemediateAgent
from strands.models.bedrock import BedrockModel
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

# Global system agents
individual_agents = None

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

class AgentRequest(BaseModel):
    message: str = Field(..., description="Message/instruction for the agent")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")

class DetectResponse(BaseModel):
    response: str = Field(..., description="Drift detection results")
    status: str = Field(..., description="Processing status")
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = Field(None, description="Session identifier")
    drift_results: Optional[Dict[str, Any]] = Field(None, description="Structured drift detection results")

class AnalyzeResponse(BaseModel):
    response: str = Field(..., description="Drift analysis results")
    status: str = Field(..., description="Processing status")
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = Field(None, description="Session identifier")
    analysis_results: Optional[Dict[str, Any]] = Field(None, description="Structured analysis results")

class RemediateResponse(BaseModel):
    response: str = Field(..., description="Remediation results")
    status: str = Field(..., description="Processing status")
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = Field(None, description="Session identifier")
    remediation_results: Optional[Dict[str, Any]] = Field(None, description="Structured remediation results")

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

def create_bedrock_model() -> BedrockModel:
    """Create and configure the Bedrock model"""
    return BedrockModel(
        model_id=BEDROCK_MODEL_ID,
        region_name=BEDROCK_REGION,
    )

def create_individual_agents():
    """Create individual agents without graph dependencies"""
    
    # Create model
    model = create_bedrock_model()
    
    # Create individual agents
    orchestration_agent = OrchestrationAgent(model)
    detect_agent = DetectAgent(model)
    drift_analyzer_agent = DriftAnalyzerAgent(model)
    remediate_agent = RemediateAgent(model)
    
    return {
        'orchestration': orchestration_agent,
        'detect': detect_agent,
        'analyzer': drift_analyzer_agent,
        'remediate': remediate_agent
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    global individual_agents
    
    try:
        logger.info("🚀 Initializing Terraform Drift Detection System...")
        
        # Ensure terraform directory exists
        os.makedirs(TERRAFORM_DIR, exist_ok=True)
        
        # Initialize shared memory
        initialize_shared_memory()
        
        # Create individual agents
        individual_agents = create_individual_agents()
        
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
async def get_agents():
    """Dependency to get the individual agents"""
    if individual_agents is None:
        raise HTTPException(
            status_code=503,
            detail="System not initialized"
        )
    return individual_agents

# Helper functions to execute individual agents
async def execute_detect_agent(message: str, agents: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the detect agent independently"""
    detect_agent = agents['detect']
    
    # Store the user request
    shared_memory.set("user_request", message)
    shared_memory.set("workflow_status", "detect_initiated")
    
    # Execute the detect agent
    result = detect_agent.get_agent()(message)
    
    # Extract response content
    response_content = ""
    if hasattr(result, 'message') and result.message:
        if hasattr(result.message, 'content'):
            for block in result.message.content:
                if isinstance(block, dict) and 'text' in block:
                    response_content += block['text']
    
    # Update shared memory with detect status
    shared_memory.set("workflow_status", "detect_completed")
    
    return {
        "response": result,
        "drift_results": shared_memory.get("drift_detection_results", {}),
        "status": "completed"
    }

async def execute_analyze_agent(message: str, agents: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the analyze agent independently"""
    analyze_agent = agents['analyzer']
    
    # Store the user request
    shared_memory.set("user_request", message)
    shared_memory.set("workflow_status", "analyze_initiated")
    
    # Execute the analyze agent
    result = analyze_agent.get_agent()(message)
    
    # Extract response content
    response_content = ""
    if hasattr(result, 'message') and result.message:
        if hasattr(result.message, 'content'):
            for block in result.message.content:
                if isinstance(block, dict) and 'text' in block:
                    response_content += block['text']
    
    # Update shared memory with analyze status
    shared_memory.set("workflow_status", "analyze_completed")
    
    return {
        "response": response_content,
        "analysis_results": shared_memory.get("drift_analysis_results", {}),
        "status": "completed"
    }

async def execute_remediate_agent(message: str, agents: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the remediate agent independently"""
    remediate_agent = agents['remediate']
    
    # Store the user request
    shared_memory.set("user_request", message)
    shared_memory.set("workflow_status", "remediate_initiated")
    
    # Execute the remediate agent
    result = remediate_agent.get_agent()(message)
    
    # Extract response content
    response_content = ""
    if hasattr(result, 'message') and result.message:
        if hasattr(result.message, 'content'):
            for block in result.message.content:
                if isinstance(block, dict) and 'text' in block:
                    response_content += block['text']
    
    # Update shared memory with remediate status
    shared_memory.set("workflow_status", "remediate_completed")
    
    return {
        "response": response_content,
        "remediation_results": shared_memory.get("remediation_results", {}),
        "status": "completed"
    }

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
            "detect": "/detect",
            "analyze": "/analyze",
            "remediate": "/remediate",
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
        "system_initialized": individual_agents is not None
    }

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    agents = Depends(get_agents)
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
        
        # Execute the orchestration agent
        orchestration_agent = agents['orchestration']
        
        # Execute the orchestration agent
        result = orchestration_agent.get_agent()(request.message)
        
        # Extract response content
        response_text = ""
        if hasattr(result, 'message') and result.message:
            if hasattr(result.message, 'content'):
                for block in result.message.content:
                    if isinstance(block, dict) and 'text' in block:
                        response_text += block['text']
        
        # Update workflow status
        shared_memory.set("workflow_status", "completed")
        
        return ChatResponse(
            response=response_text.strip() if response_text else "Request processed successfully",
            status="completed",
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        shared_memory.set("workflow_status", "failed")
        shared_memory.set("last_error", str(e))
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

# Individual agent endpoints
@app.post("/detect", response_model=DetectResponse)
async def detect(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    agents = Depends(get_agents)
):
    """
    Detect drift between Terraform state and actual AWS infrastructure
    """
    try:
        logger.info(f"Processing detect request: {request.message}")
        
        # Execute the detect agent
        result = await execute_detect_agent(request.message, agents)
        
        return DetectResponse(
            response=result["response"],
            status=result["status"],
            session_id=request.session_id,
            drift_results=result["drift_results"]
        )
        
    except Exception as e:
        logger.error(f"Error processing detect request: {e}")
        shared_memory.set("workflow_status", "failed")
        shared_memory.set("last_error", str(e))
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing detect request: {str(e)}"
        )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    agents = Depends(get_agents)
):
    """
    Analyze detected drift for severity and impact assessment
    """
    try:
        logger.info(f"Processing analyze request: {request.message}")
        
        # Execute the analyze agent
        result = await execute_analyze_agent(request.message, agents)
        
        return AnalyzeResponse(
            response=result["response"],
            status=result["status"],
            session_id=request.session_id,
            analysis_results=result["analysis_results"]
        )
        
    except Exception as e:
        logger.error(f"Error processing analyze request: {e}")
        shared_memory.set("workflow_status", "failed")
        shared_memory.set("last_error", str(e))
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing analyze request: {str(e)}"
        )

@app.post("/remediate", response_model=RemediateResponse)
async def remediate(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    agents = Depends(get_agents)
):
    """
    Remediate infrastructure drift by applying corrective changes
    """
    try:
        logger.info(f"Processing remediate request: {request.message}")
        
        # Execute the remediate agent
        result = await execute_remediate_agent(request.message, agents)
        
        return RemediateResponse(
            response=result["response"],
            status=result["status"],
            session_id=request.session_id,
            remediation_results=result["remediation_results"]
        )
        
    except Exception as e:
        logger.error(f"Error processing remediate request: {e}")
        shared_memory.set("workflow_status", "failed")
        shared_memory.set("last_error", str(e))
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing remediate request: {str(e)}"
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
            agents_count=len(individual_agents) if individual_agents else 0,
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

Available Endpoints:
• /chat - General coordination and guidance (uses OrchestrationAgent)
• /detect - Run drift detection independently (uses DetectAgent)
• /analyze - Analyze detected drift independently (uses DriftAnalyzerAgent)
• /remediate - Apply drift remediation independently (uses RemediateAgent)
• /status - Check system status
• /memory - View shared memory
• /commands - View available commands

Each agent works independently and returns results immediately:
• OrchestrationAgent - Provides guidance and coordination
• DetectAgent - Finds drift between Terraform state and AWS
• DriftAnalyzerAgent - Analyzes impact and provides recommendations
• RemediateAgent - Applies fixes to remediate drift

Usage:
POST to individual endpoints with {"message": "your instruction"}
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