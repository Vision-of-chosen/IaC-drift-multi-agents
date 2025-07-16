#!/usr/bin/env python3
"""
Test script to verify shared memory functionality in the Terraform Drift Detection & Remediation System.
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


def print_agent_state(name, agent):
    """Print agent state information"""
    print(f"\n=== {name} State ===")
    
    if not hasattr(agent.agent, 'state'):
        print(f"❌ {name} has no state attribute")
        return False
    
    # Check if shared_memory attribute exists
    if not hasattr(agent.agent.state, 'shared_memory'):
        print(f"❌ {name} state has no shared_memory attribute")
        return False
    
    # Print shared_memory attribute type and some content
    sm = agent.agent.state.shared_memory
    print(f"✅ Shared memory type: {type(sm).__name__}")
    
    if isinstance(sm, dict):
        print(f"✅ Shared memory contains {len(sm)} keys")
        if sm:
            print("Sample keys:", list(sm.keys())[:3])
    else:
        print(f"⚠️ Shared memory is not a dictionary: {sm}")
    
    return True


def test_shared_memory_system():
    """Test the shared memory system across all agents"""
    print("🧪 Testing Shared Memory System")
    print("=" * 50)
    
    # 1. Set up test data in shared memory
    test_id = f"test_{datetime.now().strftime('%H%M%S')}"
    print(f"Setting test data with ID: {test_id}")
    
    shared_memory.set(test_id, {
        "timestamp": datetime.now().isoformat(),
        "message": "This is a test entry"
    })
    
    # Print current shared memory contents
    print("\n=== Current Shared Memory Contents ===")
    for key in shared_memory.keys():
        value = shared_memory.get(key)
        value_preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
        print(f"• {key}: {value_preview}")
    
    # 2. Initialize agents
    try:
        print("\n=== Initializing Agents ===")
        model = BedrockModel(
            model_id="apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name=BEDROCK_REGION
        )
        
        orchestration_agent = OrchestrationAgent(model)
        detect_agent = DetectAgent(model)
        drift_analyzer_agent = DriftAnalyzerAgent(model)
        remediate_agent = RemediateAgent(model)
        
        print("✅ All agents initialized successfully")
        
        # 3. Test initial agent states
        agents = {
            "OrchestrationAgent": orchestration_agent,
            "DetectAgent": detect_agent,
            "DriftAnalyzerAgent": drift_analyzer_agent, 
            "RemediateAgent": remediate_agent
        }
        
        print("\n=== Testing Initial Agent States ===")
        for name, agent in agents.items():
            print_agent_state(name, agent)
            
        # 4. Update shared memory
        print("\n=== Updating Shared Memory and Testing Agent Updates ===")
        test_update_id = f"{test_id}_updated"
        shared_memory.set(test_update_id, {
            "timestamp": datetime.now().isoformat(),
            "message": "This is an updated test entry"
        })
        
        # 5. Update each agent and verify
        for name, agent in agents.items():
            print(f"\nUpdating {name} with new shared memory...")
            agent.update_shared_memory()
            
            # Verify the update
            if hasattr(agent.agent, 'state') and hasattr(agent.agent.state, 'shared_memory'):
                sm = agent.agent.state.shared_memory
                if isinstance(sm, dict) and test_update_id in sm:
                    print(f"✅ {name} successfully updated with new shared memory")
                    print(f"  - Found test entry: {test_update_id}")
                else:
                    print(f"❌ {name} failed to update with new shared memory")
            else:
                print(f"❌ {name} has invalid state structure")
                
        # 6. Clean up test data
        print("\n=== Cleaning Up Test Data ===")
        shared_memory.delete(test_id)
        shared_memory.delete(test_update_id)
        print(f"✅ Test data cleaned up")
        
        print("\n=== Test Complete ===")
                
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")


if __name__ == "__main__":
    test_shared_memory_system()