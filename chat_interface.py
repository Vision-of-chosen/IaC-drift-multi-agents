#!/usr/bin/env python3
"""
Terminal-based chat interface for the Terraform Drift Detection & Remediation System.

This module provides an interactive command-line interface for users to interact
with the multi-agent system for drift detection and remediation.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any

from strands.multiagent.graph import GraphBuilder
from strands.models.bedrock import BedrockModel

from shared_memory import shared_memory
from agents import OrchestrationAgent, DetectAgent, DriftAnalyzerAgent, RemediateAgent

logger = logging.getLogger(__name__)

# Configuration
BEDROCK_MODEL_ID = "apac.anthropic.claude-3-5-sonnet-20240620-v1:0"
BEDROCK_REGION = "ap-southeast-2"
TERRAFORM_DIR = "./terraform"  # Using a relative path that works cross-platform


def create_bedrock_model() -> BedrockModel:
    """Create and configure the Bedrock model"""
    return BedrockModel(
        model_id=BEDROCK_MODEL_ID,
        region_name=BEDROCK_REGION,
    )


def create_terraform_drift_system():
    """Create the complete multi-agent Terraform drift system"""
    
    # Create model
    model = create_bedrock_model()
    
    # Create individual agents
    orchestration_agent = OrchestrationAgent(model)
    detect_agent = DetectAgent(model)
    drift_analyzer_agent = DriftAnalyzerAgent(model)
    remediate_agent = RemediateAgent(model)
    
    # Build the agent graph using GraphBuilder
    builder = GraphBuilder()
    
    # Add agents as nodes
    orchestration_node = builder.add_node(orchestration_agent.get_agent(), "orchestration")
    detect_node = builder.add_node(detect_agent.get_agent(), "detect")
    analyzer_node = builder.add_node(drift_analyzer_agent.get_agent(), "analyzer")
    remediate_node = builder.add_node(remediate_agent.get_agent(), "remediate")
    
    # Define the workflow edges - all agents connect directly to Orchestration Agent
    # Orchestration → DetectAgent
    builder.add_edge(orchestration_node, detect_node)
    
    # Orchestration → DriftAnalyzerAgent
    builder.add_edge(orchestration_node, analyzer_node)
    
    # Orchestration → RemediateAgent
    builder.add_edge(orchestration_node, remediate_node)
    
    # Build the graph
    graph = builder.build()
    
    return graph, {
        'orchestration': orchestration_agent,
        'detect': detect_agent, 
        'analyzer': drift_analyzer_agent,
        'remediate': remediate_agent
    }


class TerraformDriftChatInterface:
    """Terminal-based chat interface for the drift detection system"""
    
    def __init__(self):
        self.graph, self.agents = create_terraform_drift_system()
        self._print_welcome()
    
    def _print_welcome(self):
        """Print welcome message and available commands"""
        print("🚀 Terraform Drift Detection & Remediation System Initialized")
        print("=" * 60)
        print("Available Commands:")
        print("  • detect - Run drift detection")
        print("  • analyze - Analyze detected drift")
        print("  • remediate - Apply drift remediation")
        print("  • status - Check system status")
        print("  • memory - View shared memory")
        print("  • help - Show this help")
        print("  • exit - Exit the system")
        print("=" * 60)
    
    def run(self):
        """Run the interactive chat interface"""
        while True:
            try:
                user_input = input("\n🔧 Terraform Drift System > ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                    
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                    
                if user_input.lower() == 'status':
                    self.show_status()
                    continue
                    
                if user_input.lower() == 'memory':
                    self.show_shared_memory()
                    continue
                
                # Process the request through the multi-agent system
                self.process_request(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Error processing request: {e}")
    
    def process_request(self, user_input: str):
        """Process user request through the multi-agent system"""
        print(f"\n🤖 Processing: {user_input}")
        print("-" * 50)
        
        try:
            # Store the user request in shared memory
            shared_memory.set("user_request", user_input)
            shared_memory.set("workflow_status", "initiated")
            
            # Execute the graph with the user input
            result = self.graph.execute(user_input)
            
            print("\n✅ System Response:")
            print("-" * 50)
            
            # Display results from each agent
            for node_id, node_result in result.results.items():
                agent_results = node_result.get_agent_results()
                for agent_result in agent_results:
                    if hasattr(agent_result, 'message') and agent_result.message:
                        content = ""
                        if hasattr(agent_result.message, 'content'):
                            for block in agent_result.message['content']:
                                if isinstance(block, dict) and 'text' in block:
                                    content += block['text']
                        print(f"\n🤖 {node_id.title()}Agent:")
                        print(content[:500] + "..." if len(content) > 500 else content)
            
            # Update workflow status
            shared_memory.set("workflow_status", "completed")
            
        except Exception as e:
            print(f"❌ Error processing request: {e}")
            shared_memory.set("workflow_status", "failed")
            shared_memory.set("last_error", str(e))
    
    def show_help(self):
        """Show help information"""
        print("\n📖 Terraform Drift Detection & Remediation System Help")
        print("=" * 60)
        print("SYSTEM OVERVIEW:")
        print("This system uses 4 specialized agents to detect and fix Terraform drift:")
        print("  • OrchestrationAgent - Coordinates the entire workflow")
        print("  • DetectAgent - Finds drift between Terraform state and AWS")
        print("  • DriftAnalyzerAgent - Analyzes impact and provides recommendations") 
        print("  • RemediateAgent - Applies fixes to remediate drift")
        print("")
        print("COMMANDS:")
        print("  detect - Start drift detection process")
        print("  analyze - Run analysis on detected drift")
        print("  remediate - Apply recommended fixes")
        print("  status - Show current system status")
        print("  memory - View shared memory contents")
        print("  help - Show this help")
        print("  exit - Exit the system")
        print("")
        print("EXAMPLE USAGE:")
        print("  > detect")
        print("  > analyze high priority drift")
        print("  > remediate security issues only")
        print("=" * 60)
    
    def show_status(self):
        """Show system status"""
        print("\n📊 System Status")
        print("-" * 30)
        print(f"Terraform Directory: {TERRAFORM_DIR}")
        print(f"AWS Region: {BEDROCK_REGION}")
        print(f"Model: {BEDROCK_MODEL_ID}")
        print(f"Agents Active: {len(self.agents)}")
        
        workflow_status = shared_memory.get("workflow_status", "idle")
        print(f"Workflow Status: {workflow_status}")
        
        # Check if terraform directory exists
        if os.path.exists(TERRAFORM_DIR):
            tf_files = list(Path(TERRAFORM_DIR).glob("*.tf"))
            state_files = list(Path(TERRAFORM_DIR).glob("*.tfstate"))
            print(f"Terraform Files: {len(tf_files)} .tf files, {len(state_files)} state files")
        else:
            print(f"⚠️  Terraform directory {TERRAFORM_DIR} not found")
    
    def show_shared_memory(self):
        """Show shared memory contents"""
        print("\n🧠 Shared Memory Contents")
        print("-" * 30)
        if shared_memory.data:
            for key, value in shared_memory.data.items():
                print(f"{key}: {str(value)[:100]}...")
        else:
            print("(Empty)") 