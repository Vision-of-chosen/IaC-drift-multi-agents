#!/usr/bin/env python3
"""
Orchestration Agent for the Terraform Drift Detection & Remediation System.

This agent serves as the central coordinator for the multi-agent system,
managing the workflow and routing user requests to specialized agents.
"""

import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("tools/src")

logger = logging.getLogger(__name__)

from strands import Agent, tool
from strands.agent.state import AgentState
from strands.models.bedrock import BedrockModel
# Try to import our wrapped use_aws first, fall back to original if not available
try:
    from useful_tools.aws_wrapper import use_aws
    logger.info("Using wrapped use_aws tool from useful_tools.aws_wrapper")
except ImportError:
    from strands_tools import file_read, file_write, journal, calculator, use_aws

    logger.warning("Using original use_aws tool from strands_tools")
from strands_tools import file_read, file_write, journal, calculator
from datetime import datetime
# Import additional useful tools for comprehensive drift detection
from useful_tools import cloudtrail_logs
from useful_tools import cloudwatch_logs


from prompts import AgentPrompts
from shared_memory import shared_memory
from config import BEDROCK_REGION
from permission_handlers import create_agent_callback_handler
from typing import Dict, Any, Optional

# Define the use_aws_with_session tool outside the class with the @tool decorator
@tool
def use_aws_with_session(
    service_name: str,
    operation_name: str,
    parameters: Dict[str, Any], 
    region: Optional[str] = None,
    label: Optional[str] = None, 
    profile_name: Optional[str] = None
):
    """Wrapper for use_aws that includes the session key"""
    session_id = shared_memory.get_current_session()
    session_key = f"orchestration_{session_id}" if session_id else None
    # Use the region from environment if not provided
    if not region:
        region = os.environ.get("AWS_REGION", BEDROCK_REGION)
    
    # Log the session key being used
    logger.info(f"Using session key in use_aws_with_session: {session_key}")
    
    # Convert operation name from hyphen format to underscore format
    # For example: 'describe-instances' -> 'describe_instances'
    if '-' in operation_name:
        operation_name = operation_name.replace('-', '_')
    
    # Check if the session key exists in aws_wrapper
    try:
        from useful_tools.aws_wrapper import _boto3_sessions
        available_sessions = list(_boto3_sessions.keys())
        logger.info(f"Available boto3 sessions before AWS call: {available_sessions}")
        
        if session_key and session_key not in available_sessions:
            logger.warning(f"Session key {session_key} not found in available boto3 sessions")
            
            # Try to get user_id from shared memory
            user_id = shared_memory.get("current_user_id", session_id=session_id)
            if user_id:
                logger.info(f"Attempting to recreate boto3 session for user_id: {user_id}")
                
                # Import the function to create and register boto3 session
                try:
                    # Import from parent directory
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from api import create_and_register_boto3_session
                    
                    # Create and register boto3 session
                    create_and_register_boto3_session(user_id, "orchestration", session_id)
                    logger.info(f"Recreated boto3 session for {session_key}")
                except ImportError as e:
                    logger.error(f"Could not import create_and_register_boto3_session: {e}")
    except ImportError:
        logger.warning("Could not import aws_wrapper to check available sessions")
    
    return use_aws(
        service_name=service_name,
        operation_name=operation_name,
        parameters=parameters,
        region=region,
        label=label,
        profile_name=profile_name,
        session_key=session_key
    )

class OrchestrationAgent:
    """Central coordinator for the multi-agent system"""
    
    def __init__(self, model: BedrockModel):
        self.model = model
        self.region = os.environ.get("AWS_REGION", BEDROCK_REGION)
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the orchestration agent instance"""
        agent = Agent(
            model=self.model,
            system_prompt=AgentPrompts.get_prompt("orchestration"),
            name="OrchestrationAgent",
            description="Central coordinator for the Terraform Drift Detection & Remediation System",
            # callback_handler=create_agent_callback_handler("OrchestrationAgent"),
            tools = [
                file_read,
                file_write,
                journal,
                calculator,
                use_aws_with_session
            ]
        )
        
        # Set state after creating agent
        agent.state = AgentState()
        agent.state.shared_memory = shared_memory.data
        agent.state.agent_type = "orchestration" 
        agent.state.aws_region = self.region
        
        return agent
    
    def get_agent(self) -> Agent:
        """Get the agent instance"""
        return self.agent
    
    def update_shared_memory(self) -> None:
        """Update agent state with current shared memory"""
        # Get current session ID
        session_id = shared_memory.get_current_session()
        
        # Create session key
        session_key = f"orchestration_{session_id}" if session_id else None
        
        # Log the session information
        logger.info(f"Updating shared memory for session ID: {session_id}, session key: {session_key}")
        
        # Check if the session key exists in aws_wrapper
        try:
            from useful_tools.aws_wrapper import _boto3_sessions
            available_sessions = list(_boto3_sessions.keys())
            logger.info(f"Available boto3 sessions: {available_sessions}")
            
            if session_key and session_key not in available_sessions:
                logger.warning(f"Session key {session_key} not found in available boto3 sessions")
                
                # Try to get user_id from shared memory and create session
                user_id = shared_memory.get("user_id", session_id=session_id)
                if user_id:
                    logger.info(f"Found user_id {user_id} in shared memory, creating boto3 session")
                    try:
                        from api import create_and_register_boto3_session
                        user_session = create_and_register_boto3_session(user_id, "orchestration", session_id)
                        if user_session:
                            logger.info(f"Successfully created boto3 session for {session_key}")
                    except ImportError:
                        logger.warning("Could not import create_and_register_boto3_session from api")
                else:
                    logger.warning(f"No user_id found in shared memory for session {session_id}")
        except ImportError:
            logger.warning("Could not import aws_wrapper to check for boto3 sessions")
        
        # Get session-specific terraform directory if available
        session_terraform_dir = None
        if session_id:
            session_terraform_dir = shared_memory.get("session_terraform_dir", session_id=session_id)
            if session_terraform_dir:
                logger.info(f"Found session-specific terraform directory: {session_terraform_dir}")
        
        # Create a new state object with updated shared memory
        if hasattr(self.agent, 'state'):
            self.agent.state.shared_memory = shared_memory.data
            
            # Add session key to state if available
            if session_id:
                self.agent.state.aws_session_key = session_key
                logger.info(f"Updated agent state with aws_session_key: {session_key}")
                
                # Update terraform directory if available
                if session_terraform_dir:
                    self.agent.state.terraform_dir = session_terraform_dir
                    logger.info(f"Updated OrchestrationAgent to use session-specific terraform directory: {session_terraform_dir}")
        else:
            self.agent.state = AgentState({
                "shared_memory": shared_memory.data,
                "agent_type": "orchestration",
                "aws_region": self.region,
                "aws_session_key": session_key,
                "terraform_dir": session_terraform_dir if session_terraform_dir else None
            })
            
            if session_key:
                logger.info(f"Created new agent state with aws_session_key: {session_key}")
            
            if session_terraform_dir:
                logger.info(f"Created new agent state with terraform_dir: {session_terraform_dir}")
        
    def _set_shared_memory_wrapper(self, key, value):
        """
        Wrapper function for setting values in shared memory
        
        This wrapper makes it easier for the agent to store data in shared memory
        without needing to use a tool, which might not be available.
        
        Args:
            key: The key to set in shared memory
            value: The value to store
        
        Returns:
            Dict with status and message
        """
        shared_memory.set(key, value)
        return {
            "status": "success",
            "message": f"Successfully stored data in shared memory with key: {key}"
        } 
    def update_agent_status(self, status_info):
        """Update agent status in shared memory"""
        agent_type = self.agent.state.agent_type
        status_key = f"{agent_type}_status"
        
        status_data = {
            "status": status_info,
            "timestamp": datetime.now().isoformat(),
            "agent": agent_type
        }
        
        shared_memory.set(status_key, status_data)

    def generate_report(self):
        """Generate a JSON report using the ReportAgent"""
        self.update_agent_status({
            "action": "generating_report",
            "timestamp": datetime.now().isoformat(),
        })
        
        # Create a prompt for the ReportAgent
        prompt = """
        Generate a JSON report of the terraform drift detection and analysis results.
        Format the report according to the required structure and save it to report.json.
        """
        
        # Run the agent with the report generation prompt
        result = self.agent.run(prompt)
        
        self.update_agent_status({
            "action": "report_generated",
            "timestamp": datetime.now().isoformat(),
        })
        
        # Return the location of the saved report
        report_file = shared_memory.get("drift_report_file", "report.json")
        return f"Report generated and saved to {report_file}"