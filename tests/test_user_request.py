#!/usr/bin/env python3
"""
Test script to verify shared memory propagation with a user request.
"""

import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared_memory import shared_memory
from strands.models.bedrock import BedrockModel
from agents import OrchestrationAgent, DetectAgent, DriftAnalyzerAgent, RemediateAgent
from config import BEDROCK_REGION


def print_agent_shared_memory(name, agent):
    """Print agent's shared memory contents"""
    print(f"\n=== {name} Shared Memory ===")
    
    if not hasattr(agent.agent, 'state'):
        print(f"❌ {name} has no state attribute")
        return False
    
    if not hasattr(agent.agent.state, 'shared_memory'):
        print(f"❌ {name} state has no shared_memory attribute")
        return False
    
    sm = agent.agent.state.shared_memory
    if not sm:
        print("❌ Shared memory is empty")
        return False
    
    print(f"✅ Shared memory contains {len(sm)} keys")
    for key, value in sm.items():
        value_str = str(value)
        value_preview = value_str[:60] + "..." if len(value_str) > 60 else value_str
        print(f"• {key}: {value_preview}")
    
    # Check if our test user_request is in shared memory
    if "user_request" in sm:
        print(f"✅ Found user_request: {sm['user_request']}")
    else:
        print("❌ user_request not found in shared memory")
    
    return True


def test_user_request_in_shared_memory():
    """Test that a user request properly propagates to agents through shared memory"""
    print("🧪 Testing User Request Propagation Through Shared Memory")
    print("=" * 60)
    
    # 1. Set up a user request in shared memory
    test_request = f"detect drift in terraform infrastructure"
    print(f"Setting user_request: '{test_request}'")
    
    # Store in shared memory
    shared_memory.set("user_request", test_request)
    shared_memory.set("timestamp", datetime.now().isoformat())
    shared_memory.set("workflow_status", "initiated")
    
    # 2. Initialize agents
    try:
        print("\n=== Initializing Agents ===")
        model = BedrockModel(
            model_id="apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name=BEDROCK_REGION
        )
        
        agents = {
            "OrchestrationAgent": OrchestrationAgent(model),
            "DetectAgent": DetectAgent(model),
            "DriftAnalyzerAgent": DriftAnalyzerAgent(model),
            "RemediateAgent": RemediateAgent(model)
        }
        
        print("✅ All agents initialized successfully")
        
        # 3. Test initial shared memory state in each agent
        print("\n=== Initial Shared Memory State ===")
        for name, agent in agents.items():
            print_agent_shared_memory(name, agent)
        
        # 4. Update shared memory with additional info
        print("\n=== Adding Analysis Results to Shared Memory ===")
        analysis_result = {
            "drift_detected": True,
            "resources_affected": 2,
            "severity": "MEDIUM",
            "timestamp": datetime.now().isoformat()
        }
        shared_memory.set("analysis_result", analysis_result)
        print("✅ Added analysis_result to shared memory")
        
        # 5. Update each agent's shared memory and verify
        print("\n=== Updating and Verifying Agent Shared Memory ===")
        for name, agent in agents.items():
            print(f"\n--- Updating {name} ---")
            agent.update_shared_memory()
            print_agent_shared_memory(name, agent)
                
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")


if __name__ == "__main__":
    test_user_request_in_shared_memory()