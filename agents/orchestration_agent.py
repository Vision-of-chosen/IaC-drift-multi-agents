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

from strands_tools import file_read, file_write, journal, calculator, use_aws
from datetime import datetime
# Import additional useful tools for comprehensive drift detection
from useful_tools import cloudtrail_logs
from useful_tools import cloudwatch_logs


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
        agent = Agent(
            model=self.model,
            system_prompt=AgentPrompts.get_prompt("orchestration"),
            name="OrchestrationAgent",
            description="Central coordinator for the Terraform Drift Detection & Remediation System",
            callback_handler=create_agent_callback_handler("OrchestrationAgent"),
            tools = [
                file_read,
                file_write,
                journal,
                calculator,
                use_aws,
                cloudtrail_logs,
                cloudwatch_logs
            ]
        )
        
        # Set state after creating agent
        agent.state = AgentState()
        agent.state.shared_memory = shared_memory.data
        agent.state.agent_type = "orchestration" 
        agent.state.aws_region = self.region
        
        return agent
    
    def get_agent(self) -> Agent:
        """Get the agent instance"""
        return self.agent
    
    def update_shared_memory(self) -> None:
        """Update agent state with current shared memory"""
        # Create a new state object with updated shared memory
        if hasattr(self.agent, 'state'):
            self.agent.state.shared_memory = shared_memory.data
        else:
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
        """Generate a JSON report using the ReportAgent"""
        self.update_agent_status({
            "action": "generating_report",
            "timestamp": datetime.now().isoformat(),
        })
        
        # Create a prompt for the ReportAgent
        prompt = """
        Generate a JSON report of the terraform drift detection and analysis results.
        Format the report according to the required structure and save it to report.json.
        """
        
        # Run the agent with the report generation prompt
        result = self.agent.run(prompt)
        
        self.update_agent_status({
            "action": "report_generated",
            "timestamp": datetime.now().isoformat(),
        })
        
        # Return the location of the saved report
        report_file = shared_memory.get("drift_report_file", "report.json")
        return f"Report generated and saved to {report_file}"