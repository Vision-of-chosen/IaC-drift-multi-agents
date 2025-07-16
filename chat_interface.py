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
from datetime import datetime

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
            shared_memory.set("request_timestamp", datetime.now().isoformat())
            
            # Add request to history
            if "request_history" not in shared_memory.data:
                shared_memory.set("request_history", [])
            
            request_history = shared_memory.get("request_history", [])
            request_history.append({
                "request": user_input,
                "timestamp": datetime.now().isoformat()
            })
            shared_memory.set("request_history", request_history[-5:])  # Keep last 5 requests
            
            # Execute the graph with the user input
            result = self.graph.execute(user_input)
            
            print("\n✅ System Response:")
            print("-" * 50)
            
            # Display results from each agent
            for node_id, node_result in result.results.items():
                agent_results = node_result.get_agent_results()
                for i, agent_result in enumerate(agent_results):
                    # Lấy agent từ danh sách agents
                    agent = self.agents.get(node_id)
                    if agent:
                        # Cập nhật trạng thái agent
                        agent.update_agent_status(f"Processed request: {user_input[:50]}...")
                    
                    # Xử lý nội dung phản hồi
                    content = ""
                    if hasattr(agent_result, 'message') and agent_result.message:
                        # Cách 1: Thử lấy từ message.content là list
                        if hasattr(agent_result.message, 'content'):
                            if isinstance(agent_result.message.content, list):
                                for block in agent_result.message.content:
                                    if isinstance(block, dict) and 'text' in block:
                                        content += block['text']
                            # Cách 2: Nếu content là string
                            elif isinstance(agent_result.message.content, str):
                                content = agent_result.message.content
                        
                        # Cách 3: Nếu message là dict
                        elif isinstance(agent_result.message, dict):
                            if 'content' in agent_result.message:
                                if isinstance(agent_result.message['content'], list):
                                    for block in agent_result.message['content']:
                                        if isinstance(block, dict) and 'text' in block:
                                            content += block['text']
                                elif isinstance(agent_result.message['content'], str):
                                    content = agent_result.message['content']
                    
                    # Cách 4: Nếu tất cả thất bại, thử lấy chuỗi string từ message
                    if not content and hasattr(agent_result, 'message'):
                        content = str(agent_result.message)
                    
                    # In nội dung gỡ lỗi
                    print(f"DEBUG: {node_id} message type: {type(agent_result.message)}")
                    print(f"DEBUG: {node_id} content length: {len(content)}")
                    
                    # Lưu vào shared_memory với nội dung có giá trị
                    shared_memory.set(f"{node_id}_response_{i}", {
                        "content": content if content else agent_result.message if hasattr(agent_result, 'message') else "No extractable content",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Hiển thị nội dung từ agent
                    if content:
                        print(f"\n🤖 {node_id.title()}Agent:")
                        print(content[:500] + "..." if len(content) > 500 else content)
            
            # Update workflow status
            shared_memory.set("workflow_status", "completed")
            shared_memory.set("completion_timestamp", datetime.now().isoformat())
            
        except Exception as e:
            print(f"❌ Error processing request: {e}")
            import traceback
            traceback.print_exc()  # In stack trace để dễ gỡ lỗi
            shared_memory.set("workflow_status", "failed")
            shared_memory.set("last_error", str(e))
            shared_memory.set("error_timestamp", datetime.now().isoformat())
    
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
            # Sắp xếp keys để dễ đọc
            sorted_keys = sorted(shared_memory.data.keys())
            for key in sorted_keys:
                value = shared_memory.data[key]
                
                # Hiển thị chi tiết tùy theo loại key
                if "_response_" in key and isinstance(value, dict):
                    print(f"• {key}:")
                    # Hiển thị nội dung phản hồi với độ dài hợp lý
                    if "content" in value and value["content"]:
                        content = value["content"]
                        if isinstance(content, str) and len(content) > 100:
                            print(f"  - content: {content[:100]}...")
                            print(f"  - content_length: {len(content)} characters")
                        else:
                            print(f"  - content: {content}")
                    
                    # Hiển thị thời gian
                    if "timestamp" in value:
                        print(f"  - timestamp: {value['timestamp']}")
                
                # Hiển thị status của agent
                elif "_status" in key and isinstance(value, dict):
                    print(f"• {key}:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            print(f"  - {sub_key}: {json.dumps(sub_value, indent=2)[:80]}...")
                        else:
                            sub_str = str(sub_value)
                            print(f"  - {sub_key}: {sub_str[:80] + '...' if len(sub_str) > 80 else sub_str}")
                
                # Hiển thị các giá trị khác
                else:
                    value_str = str(value)
                    if len(value_str) > 100:
                        print(f"• {key}: {value_str[:100]}...")
                    else:
                        print(f"• {key}: {value_str}")
        else:
            print("(Empty)") 