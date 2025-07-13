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

from strands import Agent
from strands.agent.state import AgentState
from strands.models.bedrock import BedrockModel
from strands_tools import use_aws
from tools.src.strands_tools import cloudtrail_logs
from tools.src.strands_tools import cloudwatch_logs

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


class DetectAgent:
    """Specialist in detecting Terraform infrastructure drift"""
    
    def __init__(self, model: BedrockModel):
        self.model = model
        self.region = os.environ.get("AWS_REGION", BEDROCK_REGION)
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the detect agent instance"""
        # Create a list of available tools
        tools = [use_aws, cloudtrail_logs, cloudwatch_logs]
        
        # Add read_tfstate tool if available
        if read_tfstate:
            tools.append(read_tfstate)
        
        return Agent(
            model=self.model,
            system_prompt=AgentPrompts.get_prompt("detect"),
            name="DetectAgent",
            description="Specialist in detecting Terraform infrastructure drift by comparing state files with actual AWS resources",
            tools=tools,
            state=AgentState({
                "shared_memory": shared_memory.data,
                "agent_type": "detection",
                "aws_region": self.region  # Add AWS region to state so agent knows which region to use
            })
        )
    
    def get_agent(self) -> Agent:
        """Get the agent instance"""
        return self.agent
    
    def update_shared_memory(self) -> None:
        """Update agent state with current shared memory"""
        # Create a new state object with updated shared memory
        self.agent.state = AgentState({
            "shared_memory": shared_memory.data,
            "agent_type": "detection",
            "aws_region": self.region  # Preserve the AWS region in state updates
        })
    
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
            potential_paths = [
                os.path.join("terraform", "terraform.tfstate"),  # Relative to current directory
                os.path.join(".", "terraform", "terraform.tfstate"),  # Explicit relative path
                os.path.join(".", "terraform.tfstate"),  # In current directory
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    file_path = path
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