#!/usr/bin/env python3
"""
Drift Analyzer Agent for the Terraform Drift Detection & Remediation System.

This agent specializes in analyzing and assessing infrastructure drift impacts.
"""

import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("tools/src")

logger = logging.getLogger(__name__)

from strands import Agent, tool
from strands.agent.state import AgentState
from strands.models.bedrock import BedrockModel
# Try to import our wrapped use_aws first, fall back to original if not available
from strands_tools import use_aws

from datetime import datetime
from useful_tools.aws_documentation import aws_documentation_search
from useful_tools.terraform_documentation import terraform_documentation_search
from useful_tools import cloudtrail_logs
from useful_tools import cloudwatch_logs

from prompts import AgentPrompts
from shared_memory import shared_memory
from config import BEDROCK_REGION
from permission_handlers import create_agent_callback_handler
from typing import Dict, Any, Optional
from useful_tools.aws_documentation import aws_documentation_read

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
    session_key = f"analyzer_{session_id}" if session_id else None
    # Use the region from environment if not provided
    if not region:
        region = os.environ.get("AWS_REGION", BEDROCK_REGION)
    
    # Convert operation name from hyphen format to underscore format
    # For example: 'describe-instances' -> 'describe_instances'
    if '-' in operation_name:
        operation_name = operation_name.replace('-', '_')
        
    return use_aws(
        service_name=service_name,
        operation_name=operation_name,
        parameters=parameters,
        region=region,
        label=label,
        profile_name=profile_name,
        session_key=session_key
    )

class DriftAnalyzerAgent:
    """Specialist in analyzing and assessing infrastructure drift impacts"""
    
    def __init__(self, model: BedrockModel):
        self.model = model
        self.region = os.environ.get("AWS_REGION", BEDROCK_REGION)
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create the drift analyzer agent instance"""
        
        # Define the use_aws_parallel tool wrapper - needed because this is a method
        @tool
        def use_aws_parallel_wrapper(tasks):
            """
            Execute multiple AWS API calls in parallel.
            
            Args:
                tasks: List of dictionaries, each containing service_name, operation_name, and parameters
            """
            return self.use_aws_parallel(tasks)
        
        agent = Agent(
            model=self.model,
            system_prompt=AgentPrompts.get_prompt("analyzer"),
            name="DriftAnalyzerAgent",
            description="Specialist in analyzing Terraform infrastructure drift and providing recommendations",
            tools=[
                use_aws,
                use_aws_parallel_wrapper,
                aws_documentation_search,
                aws_documentation_read
            ],
            # callback_handler=create_agent_callback_handler("DriftAnalyzerAgent")
        )
        
        # Set state after creating agent
        agent.state = AgentState()
        agent.state.shared_memory = shared_memory.data
        agent.state.agent_type = "analyzer"
        
        return agent
    
    def get_agent(self) -> Agent:
        """Get the agent instance"""
        return self.agent
    
    def update_shared_memory(self) -> None:
        """Update agent state with current shared memory"""
        # Create a new state object with updated shared memory
        if hasattr(self.agent, 'state'):
            self.agent.state.shared_memory = shared_memory.data
            
            # Add session key to state if available
            session_id = shared_memory.get_current_session()
            if session_id:
                session_key = f"analyzer_{session_id}"
                self.agent.state.aws_session_key = session_key
                
                # Get session-specific terraform directory if available
                session_terraform_dir = shared_memory.get("session_terraform_dir", session_id=session_id)
                if session_terraform_dir:
                    self.agent.state.terraform_dir = session_terraform_dir
                    logger.info(f"Updated DriftAnalyzerAgent to use session-specific terraform directory: {session_terraform_dir}")
        else:
            session_id = shared_memory.get_current_session()
            session_key = f"analyzer_{session_id}" if session_id else None
            
            # Get session-specific terraform directory if available
            session_terraform_dir = None
            if session_id:
                session_terraform_dir = shared_memory.get("session_terraform_dir", session_id=session_id)
            
            self.agent.state = AgentState({
                "shared_memory": shared_memory.data,
                "agent_type": "analyzer",
                "aws_region": self.region,
                "aws_session_key": session_key,
                "terraform_dir": session_terraform_dir if session_terraform_dir else None
            })
        
    def _set_shared_memory_wrapper(self, key: str, value) -> dict:
        """Wrapper for setting values in shared memory"""
        shared_memory.set(key, value)
        return {"status": "success", "message": f"Value set for key: {key}"} 
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
    
    def _use_aws_task(self, service_name, operation_name, parameters):
        """
        Helper function to execute a single AWS API task.
        This is used by use_aws_parallel to execute individual calls.
        
        Args:
            service_name: AWS service to call (e.g., 's3', 'ec2', 'iam')
            operation_name: Operation to perform (e.g., 'list_buckets', 'describe_instances')
            parameters: Dict of parameters to pass to the operation
            
        Returns:
            Dict containing the result of the AWS operation
        """
        try:
            # Get the current session ID and create a session key
            session_id = shared_memory.get_current_session()
            session_key = f"analyzer_{session_id}" if session_id else None
            
            # Use the existing use_aws tool to make the call
            result = use_aws(
                service_name=service_name,
                operation_name=operation_name,
                parameters=parameters,
                session_key=session_key
            )
            
            return {
                "service": service_name,
                "operation": operation_name,
                "status": "success",
                "result": result,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error executing AWS API call {service_name}.{operation_name}: {e}")
            return {
                "service": service_name,
                "operation": operation_name,
                "status": "error",
                "result": None,
                "error": str(e)
            }

    @tool
    def use_aws_parallel(self, tasks):
        """
        Execute multiple AWS API calls in parallel using concurrent processing.
        
        Args:
            tasks: List of dictionaries, each containing:
                - service_name: AWS service to call (e.g., 's3', 'ec2', 'iam')
                - operation_name: Operation to perform (e.g., 'list_buckets', 'describe_instances')
                - parameters: Dict of parameters to pass to the operation
        
        Returns:
            Dict of results, with keys matching the service_name and operation_name of each task
        """
        import concurrent.futures
        
        results = {}
        
        try:
            logger.info(f"Executing {len(tasks)} AWS tasks in parallel")
            
            # Use ThreadPoolExecutor to run tasks in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Start all tasks and collect the futures
                futures = []
                for task in tasks:
                    service_name = task.get("service_name")
                    operation_name = task.get("operation_name")
                    parameters = task.get("parameters", {})
                    
                    # Validate required parameters
                    if not service_name or not operation_name:
                        logger.error(f"Invalid task, missing service_name or operation_name: {task}")
                        continue
                    
                    # Submit task to executor
                    future = executor.submit(
                        self._use_aws_task,
                        service_name,
                        operation_name,
                        parameters
                    )
                    futures.append((future, service_name, operation_name))
                
                # Collect results as they complete
                for future, service_name, operation_name in futures:
                    try:
                        task_result = future.result()
                        result_key = f"{service_name}.{operation_name}"
                        results[result_key] = task_result
                    except Exception as e:
                        logger.error(f"Exception in parallel task {service_name}.{operation_name}: {e}")
                        results[f"{service_name}.{operation_name}"] = {
                            "service": service_name,
                            "operation": operation_name,
                            "status": "error",
                            "result": None,
                            "error": str(e)
                        }
            
            logger.info(f"Completed {len(results)} parallel AWS tasks")
            return results
            
        except Exception as e:
            logger.error(f"Error executing parallel AWS tasks: {e}")
            return {
                "status": "error",
                "error": str(e),
                "results": results
            }