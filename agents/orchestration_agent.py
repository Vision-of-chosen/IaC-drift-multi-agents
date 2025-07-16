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

from strands import Agent
from strands.agent.state import AgentState
from strands.models.bedrock import BedrockModel

from strands_tools import use_aws

# Import additional useful tools for comprehensive drift detection
from useful_tools import cloudtrail_logs
from useful_tools import cloudwatch_logs
from useful_tools.terraform_tools import terraform_plan, terraform_apply, terraform_import
from useful_tools.terraform_mcp_tool import terraform_run_command, terraform_run_checkov_scan
from useful_tools.aws_documentation import aws_documentation_search
from useful_tools.terraform_documentation import terraform_documentation_search

from prompts import AgentPrompts
from shared_memory import shared_memory
from config import BEDROCK_REGION
from permission_handlers import create_agent_callback_handler

class OrchestrationAgent:
    """Central coordinator for the multi-agent system"""
    
    def __init__(self, model: BedrockModel):
        self.model = model
        self.region = os.environ.get("AWS_REGION", BEDROCK_REGION)
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the orchestration agent instance"""
        return Agent(
            model=self.model,
            system_prompt=AgentPrompts.get_prompt("orchestration"),
            name="OrchestrationAgent",
            description="Central coordinator for the Terraform Drift Detection & Remediation System",
            callback_handler=create_agent_callback_handler("OrchestrationAgent"),
            tools = [
            # Core AWS and state tools
            use_aws,
            
            # Documentation and reference tools
            aws_documentation_search,
            terraform_documentation_search
        ],
            state=AgentState({
                "shared_memory": shared_memory.data,
                "agent_type": "orchestration",
                "aws_region": self.region
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
            "agent_type": "orchestration",
            "aws_region": self.region
        })
        
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