#!/usr/bin/env python3
"""
Remediate Agent for the Terraform Drift Detection & Remediation System.

This agent specializes in automated Terraform infrastructure remediation.
"""

import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("tools/src")

logger = logging.getLogger(__name__)
from datetime import datetime
from strands import Agent
from strands.agent.state import AgentState
from strands.models.bedrock import BedrockModel
from strands_tools import use_aws, file_read, file_write, editor

try:
    from useful_tools.terraform_mcp_tool import (
        terraform_run_command,
        terraform_run_checkov_scan,
        terraform_get_best_practices,
        terraform_get_provider_docs
    )
    TERRAFORM_MCP_TOOLS_AVAILABLE = True
except ImportError:
    logger.warning("terraform_mcp_tool not available, some functionality will be limited")
    TERRAFORM_MCP_TOOLS_AVAILABLE = False

from useful_tools.terraform_tools import (
    terraform_plan,
    terraform_apply,
    terraform_import
)

from prompts import AgentPrompts
from shared_memory import shared_memory
from config import BEDROCK_REGION, TERRAFORM_DIR
from permission_handlers import create_agent_callback_handler
class RemediateAgent:
    """Specialist in automated Terraform infrastructure remediation"""
    
    def __init__(self, model: BedrockModel):
        self.model = model
        self.terraform_dir = TERRAFORM_DIR
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create the remediate agent instance"""
        # Define tools based on availability
        if TERRAFORM_MCP_TOOLS_AVAILABLE:
            logger.info("Terraform MCP tools added to RemediateAgent")
            # Create a combined list of all tools
            tools = [
                use_aws, 
                file_read, 
                file_write, 
                editor,
                terraform_run_command,
                terraform_run_checkov_scan,
                terraform_get_best_practices,
                terraform_get_provider_docs,
                terraform_plan,
                terraform_apply,
                terraform_import
            ]
        else:
            # Basic tools only
            tools = [
                use_aws, 
                file_read, 
                file_write, 
                editor
            ]
        
        agent = Agent(
            model=self.model,
            system_prompt=AgentPrompts.get_prompt("remediate"),
            name="RemediateAgent",
            description="Specialist in automated Terraform infrastructure remediation using AWS best practices",
            # callback_handler=create_agent_callback_handler("RemediateAgent"),
            tools=tools,
        )
        agent.state = AgentState()
        agent.state.shared_memory = shared_memory.data
        agent.state.agent_type = "remediation"
        agent.state.terraform_dir = self.terraform_dir
        return agent
    
    def get_agent(self) -> Agent:
        """Get the agent instance"""
        return self.agent
    
    def update_shared_memory(self) -> None:
        """Update agent state with current shared memory"""
        if hasattr(self.agent, 'state'):
            self.agent.state.shared_memory = shared_memory.data
        else:
            self.agent.state = AgentState({
            "shared_memory": shared_memory.data,
            "agent_type": "remediation",
            "terraform_dir": self.terraform_dir
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