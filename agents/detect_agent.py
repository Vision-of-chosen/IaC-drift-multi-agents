#!/usr/bin/env python3
"""
Detect Agent for the Terraform Drift Detection & Remediation System.

This agent specializes in detecting drift between Terraform state and
actual AWS infrastructure.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("tools/src")

from strands import Agent
from strands.agent.state import AgentState
from strands.models.bedrock import BedrockModel
from strands_tools import use_aws

from prompts import AgentPrompts
from shared_memory import shared_memory


class DetectAgent:
    """Specialist in detecting Terraform infrastructure drift"""
    
    def __init__(self, model: BedrockModel):
        self.model = model
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the detect agent instance"""
        return Agent(
            model=self.model,
            system_prompt=AgentPrompts.get_prompt("detect"),
            name="DetectAgent",
            description="Specialist in detecting Terraform infrastructure drift by comparing state files with actual AWS resources",
            tools=[use_aws],
            state=AgentState({
                "shared_memory": shared_memory.data,
                "agent_type": "detection"
            })
        )
    
    def get_agent(self) -> Agent:
        """Get the agent instance"""
        return self.agent

    # Removed update_shared_memory method - agents access shared memory directly through the global shared_memory object 