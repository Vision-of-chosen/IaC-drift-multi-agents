#!/usr/bin/env python3
"""
Remediate Agent for the Terraform Drift Detection & Remediation System.

This agent specializes in automated remediation of Terraform infrastructure drift
through safe configuration updates and change management.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("tools/src")

from strands import Agent
from strands.agent.state import AgentState
from strands.models.bedrock import BedrockModel
from strands_tools import use_aws, file_read, file_write, editor

from prompts import AgentPrompts
from shared_memory import shared_memory


class RemediateAgent:
    """Specialist in automated remediation of Terraform infrastructure drift"""
    
    def __init__(self, model: BedrockModel):
        self.model = model
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the remediate agent instance"""
        return Agent(
            model=self.model,
            system_prompt=AgentPrompts.get_prompt("remediate"),
            name="RemediateAgent",
            description="Specialist in automated remediation of Terraform infrastructure drift through safe configuration updates",
            tools=[use_aws, file_read, file_write, editor],
            state=AgentState({
                "shared_memory": shared_memory.data,
                "agent_type": "remediation"
            })
        )
    
    def get_agent(self) -> Agent:
        """Get the agent instance"""
        return self.agent
    
    def update_shared_memory(self) -> None:
        """Update agent state with current shared memory"""
        self.agent.state["shared_memory"] = shared_memory.data 