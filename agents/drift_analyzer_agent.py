#!/usr/bin/env python3
"""
Drift Analyzer Agent for the Terraform Drift Detection & Remediation System.

This agent specializes in analyzing and assessing infrastructure drift impacts,
providing risk categorization and remediation recommendations.
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


class DriftAnalyzerAgent:
    """Specialist in analyzing infrastructure drift severity and impact"""
    
    def __init__(self, model: BedrockModel):
        self.model = model
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the drift analyzer agent instance"""
        return Agent(
            model=self.model,
            system_prompt=AgentPrompts.get_prompt("analyzer"),
            name="DriftAnalyzerAgent",
            description="Specialist in analyzing infrastructure drift severity, impact, and providing remediation recommendations",
            tools=[use_aws],
            state=AgentState({
                "shared_memory": shared_memory.data,
                "agent_type": "analysis"
            })
        )
    
    def get_agent(self) -> Agent:
        """Get the agent instance"""
        return self.agent
    
    def update_shared_memory(self) -> None:
        """Update agent state with current shared memory"""
        self.agent.state.data["shared_memory"] = shared_memory.data 