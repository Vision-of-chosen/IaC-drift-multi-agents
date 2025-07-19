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

from strands import Agent
from strands.agent.state import AgentState
from strands.models.bedrock import BedrockModel
from strands_tools import use_aws, retrieve
from datetime import datetime
from useful_tools.aws_documentation import aws_documentation_search
from useful_tools.terraform_documentation import terraform_documentation_search
from useful_tools import cloudtrail_logs
from useful_tools import cloudwatch_logs

from prompts import AgentPrompts
from shared_memory import shared_memory
from config import BEDROCK_REGION
from permission_handlers import create_agent_callback_handler
class DriftAnalyzerAgent:
    """Specialist in analyzing and assessing infrastructure drift impacts"""
    
    def __init__(self, model: BedrockModel):
        self.model = model
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create the drift analyzer agent instance"""
        agent = Agent(
            model=self.model,
            system_prompt=AgentPrompts.get_prompt("analyzer"),
            name="DriftAnalyzerAgent",
            description="Specialist in analyzing drift severity and business impact",
            # callback_handler=create_agent_callback_handler("DriftAnalyzerAgent"),
            tools=[
                use_aws,
                retrieve,
                aws_documentation_search,
                terraform_documentation_search,
                cloudtrail_logs,
                cloudwatch_logs
            ]
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
        if hasattr(self.agent, 'state'):
            self.agent.state.shared_memory = shared_memory.data
        else:
            self.agent.state = AgentState({
            "shared_memory": shared_memory.data,
            "agent_type": "analyzer"
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