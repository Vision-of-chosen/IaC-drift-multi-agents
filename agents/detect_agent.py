#!/usr/bin/env python3
"""
Detect Agent for the Terraform Drift Detection & Remediation System.

This agent specializes in detecting drift between Terraform state and
actual AWS infrastructure.
"""

import sys
import os
import json
import logging
# Add root project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add tools directory to path
sys.path.append("tools/src")
# Add useful_tools directory to path
useful_tools_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "useful_tools")
sys.path.append(useful_tools_path)

logger = logging.getLogger(__name__)

from strands import Agent, tool
from strands.agent.state import AgentState
from strands.models.bedrock import BedrockModel
# Import use_aws directly from useful_tools
from useful_tools.use_aws import use_aws
from datetime import datetime
from useful_tools import cloudtrail_logs
from useful_tools import cloudwatch_logs
from useful_tools.terraform_tools import terraform_plan

# Import our custom read_tfstate tool
try:
    # Try to import from useful-tools package
    try:
        from useful_tools.read_tfstate_tool import read_tfstate
        logger.info("Successfully imported read_tfstate from useful_tools package")
    except ImportError as e:
        logger.warning(f"Could not import read_tfstate from useful_tools: {e}")
        
        # Try direct file import
        try:
            import importlib.util
            tool_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "useful_tools/read_tfstate_tool.py"
            )
            
            if os.path.exists(tool_path):
                spec = importlib.util.spec_from_file_location("read_tfstate_tool", tool_path)
                if spec and spec.loader:
                    read_tfstate_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(read_tfstate_module)
                    read_tfstate = read_tfstate_module.read_tfstate
                    logger.info(f"Successfully loaded read_tfstate from useful-tools directory")
                else:
                    logger.error(f"Failed to load read_tfstate tool spec from {tool_path}")
                    read_tfstate = None
            else:
                logger.error(f"read_tfstate tool not found at {tool_path}")
                read_tfstate = None
        except Exception as e2:
            logger.error(f"Error during direct file import of read_tfstate: {e2}")
            read_tfstate = None
except Exception as e:
    logger.error(f"Error importing read_tfstate tool: {e}")
    read_tfstate = None

from prompts import AgentPrompts
from shared_memory import shared_memory
from config import BEDROCK_REGION
from permission_handlers import create_agent_callback_handler
from typing import Dict, Any, Optional

# Define wrappers for cloudtrail_logs and cloudwatch_logs to include session_key
@tool
def cloudtrail_logs_with_session(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    region: Optional[str] = None,
    event_name: Optional[str] = None,
    resource_type: Optional[str] = None,
    max_results: int = 50
):
    """Wrapper for cloudtrail_logs that includes the session key"""
    session_id = shared_memory.get_current_session()
    session_key = f"detection_{session_id}" if session_id else None
    # Use the region from the agent if not provided
    if not region:
        region = os.environ.get("AWS_REGION", BEDROCK_REGION)
        
    return cloudtrail_logs.cloudtrail_logs(
        start_time=start_time,
        end_time=end_time,
        region=region,
        event_name=event_name,
        resource_type=resource_type,
        max_results=max_results,
        session_key=session_key
    )

@tool
def cloudwatch_logs_with_session(
    log_group_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    filter_pattern: Optional[str] = None,
    region: Optional[str] = None,
    max_results: int = 50
):
    """Wrapper for cloudwatch_logs that includes the session key"""
    session_id = shared_memory.get_current_session()
    session_key = f"detection_{session_id}" if session_id else None
    # Use the region from the agent if not provided
    if not region:
        region = os.environ.get("AWS_REGION", BEDROCK_REGION)
        
    return cloudwatch_logs.cloudwatch_logs(
        log_group_name=log_group_name,
        start_time=start_time,
        end_time=end_time,
        filter_pattern=filter_pattern,
        region=region,
        max_results=max_results,
        session_key=session_key
    )

@tool
def aws(service, operation, parameters=None, region=None):
    """
    Simple helper function to execute AWS operations.
    
    Args:
        service: AWS service name (e.g., 's3', 'ec2', 'dynamodb')
        operation: Operation to perform (e.g., 'list_buckets', 'describe_instances')
        parameters: Dictionary of parameters for the operation (default: {})
        region: AWS region (default: from environment or 'us-west-2')
    
    Returns:
        The result of the AWS operation
    """
    if parameters is None:
        parameters = {}
    
    if region is None:
        region = os.environ.get("AWS_REGION", BEDROCK_REGION)
    
    # Get current session ID from shared memory
    session_id = shared_memory.get_current_session()
    session_key = f"detect_{session_id}" if session_id else None
    
    # Get AWS credentials from shared memory
    aws_access_key_id = None
    aws_secret_access_key = None
    aws_session_token = None
    
    if session_id:
        # First check if user_id is stored in the session
        user_id = shared_memory.get("user_id", session_id=session_id)
        
        if user_id:
            # Get credentials from user-specific storage
            user_credentials_key = f"aws_credentials_{user_id}"
            user_credentials = shared_memory.get(user_credentials_key)
            
            if user_credentials and isinstance(user_credentials, dict):
                aws_access_key_id = user_credentials.get("access_key")
                aws_secret_access_key = user_credentials.get("secret_key")
                
                # Use region from credentials if not specified
                if not region and "region" in user_credentials:
                    region = user_credentials.get("region")
                
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
        
        # Try to get region from session if not specified
        if not region:
            session_region = shared_memory.get("aws_region", session_id=session_id)
            if session_region:
                region = session_region
    
    logger.info(f"Executing AWS operation: {service}.{operation} with parameters: {parameters}")
    
    # Create a properly formatted input for use_aws
    tool_input = {
        "toolUseId": f"aws_{service}_{operation}",
        "input": {
            "service_name": service,
            "operation_name": operation,
            "parameters": parameters,
            "region": region,
            "label": f"Execute {service}.{operation}",
            "session_id": session_id,
            "session_key": session_key
        }
    }
    
    # Add AWS credentials to tool input if available
    if aws_access_key_id and aws_secret_access_key:
        tool_input["input"]["aws_access_key_id"] = aws_access_key_id
        tool_input["input"]["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            tool_input["input"]["aws_session_token"] = aws_session_token
        logger.info(f"Using AWS credentials from shared memory for session {session_id}")
    else:
        logger.warning(f"No AWS credentials found in shared memory for session {session_id}")
    
    return use_aws(tool_input)

class DetectAgent:
    """Specialist in detecting Terraform infrastructure drift"""
    
    def __init__(self, model: BedrockModel):
        self.model = model
        self.region = os.environ.get("AWS_REGION", BEDROCK_REGION)
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the detect agent instance"""
        # Create tools list based on availability of read_tfstate
        if read_tfstate:
            tools = [aws, cloudtrail_logs_with_session, cloudwatch_logs_with_session, read_tfstate, terraform_plan]
        else:
            tools = [aws, cloudtrail_logs_with_session, cloudwatch_logs_with_session, terraform_plan]
        
        agent = Agent(
            model=self.model,
            system_prompt=AgentPrompts.get_prompt("detect"),
            name="DetectAgent",
            description="Specialist in detecting Terraform infrastructure drift by comparing state files with actual AWS resources",
            tools=tools,
            # callback_handler=create_agent_callback_handler("DetectAgent")
        )
        agent.state = AgentState()
        agent.state.shared_memory = shared_memory.data
        agent.state.agent_type = "detection"
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
        session_key = f"detection_{session_id}" if session_id else None
        
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
                        user_session = create_and_register_boto3_session(user_id, "detection", session_id)
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
                    logger.info(f"Updated DetectAgent to use session-specific terraform directory: {session_terraform_dir}")
        else:
            state_dict = {
                "shared_memory": shared_memory.data,
                "agent_type": "detection",
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
    
    def read_terraform_state(self, file_path=None) -> dict:
        """
        Read the Terraform state file and store it in shared memory
        
        Args:
            file_path: Path to the Terraform state file, or None to use defaults
        
        Returns:
            The parsed Terraform state data as a dictionary
        """
        # Check if read_tfstate tool is available
        if not read_tfstate or not hasattr(self.agent.tool, 'read_tfstate'):
            logger.warning("read_tfstate tool is not available, reading file directly")
            return self._read_terraform_state_direct(file_path)
        
        try:
            logger.info("Attempting to use read_tfstate tool")
            
            # If no file_path provided, try to use session-specific directory
            if not file_path:
                session_id = shared_memory.get_current_session()
                if session_id:
                    session_terraform_dir = shared_memory.get("session_terraform_dir", session_id=session_id)
                    if session_terraform_dir:
                        potential_state_file = os.path.join(session_terraform_dir, "terraform.tfstate")
                        if os.path.exists(potential_state_file):
                            file_path = potential_state_file
                            logger.info(f"Using session-specific terraform state file: {file_path}")
            
            # Call the read_tfstate tool to get the Terraform state
            inputs = {}
            if file_path:
                inputs["file_path"] = file_path
                
            result = self.agent.tool.read_tfstate(**inputs)
            
            # Check if the result contains tfstate_data
            if "tfstate_data" in result:
                logger.info("Successfully read Terraform state using the read_tfstate tool")
                # Store in shared memory
                shared_memory.set('tfstate_data', result["tfstate_data"])
                logger.info(f"Terraform state stored in shared memory with {len(result['tfstate_data'].get('resources', []))} resources.")
                return result["tfstate_data"]
            elif "error" in result:
                logger.error(f"Error reading Terraform state: {result['error']}")
                return {}
            else:
                logger.warning("read_tfstate tool returned unexpected format, falling back to direct read")
                return self._read_terraform_state_direct(file_path)
        except Exception as e:
            logger.error(f"Error using read_tfstate tool: {e}, falling back to direct read")
            return self._read_terraform_state_direct(file_path)
    
    def _read_terraform_state_direct(self, file_path=None) -> dict:
        """Read Terraform state file directly and store in shared memory"""
        # Try default paths if no file_path provided
        if not file_path:
            # Get session-specific terraform directory if available
            session_id = shared_memory.get_current_session()
            session_terraform_dir = None
            if session_id:
                session_terraform_dir = shared_memory.get("session_terraform_dir", session_id=session_id)
            
            # Create potential paths list based on available directories
            potential_paths = []
            
            # Add paths with session-specific directory if available
            if session_terraform_dir:
                potential_paths.extend([
                    os.path.join(session_terraform_dir, "terraform.tfstate"),
                    os.path.join(session_terraform_dir, ".terraform", "terraform.tfstate")
                ])
            
            # Add default paths as fallback
            potential_paths.extend([
                os.path.join("terraform", "terraform.tfstate"),  # Relative to current directory
                os.path.join(".", "terraform", "terraform.tfstate"),  # Explicit relative path
                os.path.join(".", "terraform.tfstate"),  # In current directory
            ])
            
            logger.info(f"Looking for terraform state file in paths: {potential_paths}")
            
            for path in potential_paths:
                if os.path.exists(path):
                    file_path = path
                    logger.info(f"Found terraform state file at: {file_path}")
                    break
        
        # Return empty dict if we couldn't find any state file
        if not file_path or not os.path.exists(file_path):
            logger.warning("No Terraform state file found.")
            return {}
            
        # Read the file directly
        try:
            with open(file_path, 'r') as f:
                tfstate_data = json.load(f)
                
            # Store in shared memory
            shared_memory.set('tfstate_data', tfstate_data)
            logger.info(f"Terraform state stored in shared memory with {len(tfstate_data.get('resources', []))} resources.")
            
            return tfstate_data
        except Exception as e:
            logger.error(f"Error reading Terraform state file: {e}")
            return {}
    # Thêm phương thức detect_drift tại đây
    def detect_drift(self, resources=None):
        """Detect drift in the specified resources or all resources"""
        result = self.agent.run(f"Detect drift in terraform infrastructure for resources: {resources if resources else 'all'}")
        
        # Cập nhật trạng thái vào shared memory
        self.update_agent_status({
            "action": "detect_drift",
            "resources": resources if resources else "all",
            "completion_time": datetime.now().isoformat(),
            "status": "completed"
        })
        
        return result

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
        """Generate a JSON report of detection results"""
        self.update_agent_status({
            "action": "generating_report",
            "timestamp": datetime.now().isoformat(),
        })
        
        # Create a prompt for the agent to generate the report
        prompt = """
        Generate a JSON report of the terraform drift detection results.
        Format the report according to the required structure and save it to detect_agent_results.json.
        Include all resources that were checked and their drift status.
        """
        
        # Run the agent with the report generation prompt
        result = self.agent.run(prompt)
        
        self.update_agent_status({
            "action": "report_generated",
            "timestamp": datetime.now().isoformat(),
        })
        
        # Return the location of the saved report
        report_file = shared_memory.get("drift_detection_report_file", "detect_agent_results.json")
        return f"Detection report generated and saved to {report_file}"

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
    