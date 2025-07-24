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
# Import use_aws directly from useful_tools
from strands_tools import use_aws
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

# Define wrappers for cloudtrail_logs and cloudwatch_logs to include session_key


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
                use_aws,
                cloudtrail_logs,
                cloudwatch_logs,
                file_read,
                file_write,
                journal,
                calculator
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
        
        # Get AWS credentials using the get_aws_credentials method
        aws_credentials = {}
        if session_id:
            aws_credentials = self.get_aws_credentials(session_id)
        
        # Check if the session key exists in aws_wrapper
        try:
            from useful_tools.use_aws import _boto3_sessions
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
                            
                            # Extract and store credentials in shared memory
                            try:
                                credentials = user_session.get_credentials()
                                if credentials:
                                    frozen_creds = credentials.get_frozen_credentials()
                                    
                                    # Update our aws_credentials dictionary with these values
                                    aws_credentials = {
                                        "aws_access_key_id": frozen_creds.access_key,
                                        "aws_secret_access_key": frozen_creds.secret_key,
                                        "aws_region": user_session.region_name
                                    }
                                    
                                    if hasattr(frozen_creds, 'token') and frozen_creds.token:
                                        aws_credentials["aws_session_token"] = frozen_creds.token
                                        
                                    logger.info(f"Extracted AWS credentials from boto3 session")
                            except Exception as e:
                                logger.error(f"Error extracting credentials from boto3 session: {e}")
                    except ImportError:
                        logger.warning("Could not import create_and_register_boto3_session from api")
                else:
                    logger.warning(f"No user_id found in shared memory for session {session_id}")
        except ImportError:
            logger.warning("Could not import use_aws to check for boto3 sessions")
        
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
                
                # Add AWS credentials to state if available
                if aws_credentials:
                    for key, value in aws_credentials.items():
                        setattr(self.agent.state, key, value)
                    logger.info("Updated agent state with AWS credentials from shared memory")
                
                # Update terraform directory if available
                if session_terraform_dir:
                    self.agent.state.terraform_dir = session_terraform_dir
                    logger.info(f"Updated OrchestrationAgent to use session-specific terraform directory: {session_terraform_dir}")
        else:
            state_dict = {
                "shared_memory": shared_memory.data,
                "agent_type": "orchestration",
                "aws_region": aws_credentials.get("aws_region", self.region),
                "aws_session_key": session_key,
                "terraform_dir": session_terraform_dir if session_terraform_dir else None
            }
            
            # Add AWS credentials to state if available
            if aws_credentials:
                for key, value in aws_credentials.items():
                    state_dict[key] = value
                logger.info("Added AWS credentials to agent state from shared memory")
            
            self.agent.state = AgentState(state_dict)
            
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

    def get_aws_credentials(self, session_id=None):
        """
        Get AWS credentials from shared memory for the specified session
        
        Args:
            session_id: Session ID to get credentials for, or None for current session
            
        Returns:
            Dict containing AWS credentials and region
        """
        if session_id is None:
            session_id = shared_memory.get_current_session()
            
        if not session_id:
            logger.warning("No session ID provided or found in shared memory")
            return {}
            
        # Get AWS credentials from shared memory
        aws_access_key_id = None
        aws_secret_access_key = None
        aws_session_token = None
        aws_region = self.region
        
        # First check if user_id is stored in the session
        user_id = shared_memory.get("user_id", session_id=session_id)
        
        if user_id:
            # Get credentials from user-specific storage
            user_credentials_key = f"aws_credentials_{user_id}"
            user_credentials = shared_memory.get(user_credentials_key)
            
            if user_credentials and isinstance(user_credentials, dict):
                aws_access_key_id = user_credentials.get("access_key")
                aws_secret_access_key = user_credentials.get("secret_key")
                
                # Use region from credentials if available
                if "region" in user_credentials:
                    aws_region = user_credentials.get("region")
                
                logger.info(f"Retrieved AWS credentials for user {user_id}")
        
        # If no user-specific credentials, try direct session keys as fallback
        if not aws_access_key_id or not aws_secret_access_key:
            direct_access_key = shared_memory.get("aws_access_key_id", session_id=session_id)
            direct_secret_key = shared_memory.get("aws_secret_access_key", session_id=session_id)
            
            if direct_access_key and direct_secret_key:
                aws_access_key_id = direct_access_key
                aws_secret_access_key = direct_secret_key
                aws_session_token = shared_memory.get("aws_session_token", session_id=session_id)
                logger.info("Retrieved AWS credentials directly from session")
        
        # Try to get region from session if not already set
        session_region = shared_memory.get("aws_region", session_id=session_id)
        if session_region:
            aws_region = session_region
            
        credentials = {
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "aws_region": aws_region
        }
        
        if aws_session_token:
            credentials["aws_session_token"] = aws_session_token
            
        # Check if we have valid credentials
        if not aws_access_key_id or not aws_secret_access_key:
            logger.warning(f"No valid AWS credentials found in shared memory for session {session_id}")
            return {}
            
        logger.info(f"Successfully retrieved AWS credentials from shared memory for session {session_id}")
        return credentials