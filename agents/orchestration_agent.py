#!/usr/bin/env python3
"""
Orchestration Agent for the Terraform Drift Detection & Remediation System.

This agent serves as the central coordinator that manages workflow between
other specialized agents.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strands import Agent
from strands.agent.state import AgentState
from strands.models.bedrock import BedrockModel

from prompts import AgentPrompts
from shared_memory import shared_memory


class OrchestrationAgent:
    """Central coordinator for Terraform drift detection and remediation operations"""
    
    def __init__(self, model: BedrockModel):
        self.model = model
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the orchestration agent instance"""
        return Agent(
            model=self.model,
            system_prompt=AgentPrompts.get_prompt("orchestration"),
            name="OrchestrationAgent",
            description="Central coordinator for Terraform drift detection and remediation operations",
            state=AgentState({
                "shared_memory": shared_memory.data,
                "agent_type": "orchestration"
            })
        )
    
    def get_agent(self) -> Agent:
        """Get the agent instance"""
        return self.agent
    
    def update_shared_memory(self) -> None:
        """Update agent state with current shared memory"""
        self.agent.state.data["shared_memory"] = shared_memory.data 