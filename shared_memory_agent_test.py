#!/usr/bin/env python3
import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_memory import shared_memory
from agents import OrchestrationAgent, DetectAgent, DriftAnalyzerAgent, RemediateAgent
from strands.models.bedrock import BedrockModel

def test_agent_shared_memory_flow():
    print("🧪 Testing Shared Memory Data Flow Between Agents...")
    
    # Khởi tạo model và agents
    model = BedrockModel(model_id="apac.anthropic.claude-3-5-sonnet-20240620-v1:0", region_name="ap-southeast-2")
    orchestration_agent = OrchestrationAgent(model)
    detect_agent = DetectAgent(model)
    analyzer_agent = DriftAnalyzerAgent(model)
    remediate_agent = RemediateAgent(model)
    
    # Khởi tạo test data trong shared_memory
    test_timestamp = datetime.now().isoformat()
    shared_memory.set("test_timestamp", test_timestamp)
    shared_memory.set("workflow_status", "initiated")
    
    # Update shared memory cho từng agent
    print("Updating shared memory for all agents...")
    orchestration_agent.update_shared_memory()
    detect_agent.update_shared_memory()
    analyzer_agent.update_shared_memory()
    remediate_agent.update_shared_memory()
    
    # Kiểm tra state của các agent
    agents = {
        "orchestration": orchestration_agent,
        "detect": detect_agent,
        "analyzer": analyzer_agent,
        "remediate": remediate_agent
    }
    
    # Kiểm tra dữ liệu trong state của mỗi agent
    all_pass = True
    for name, agent in agents.items():
        # Truy cập an toàn vào shared_memory của agent
        agent_shared_memory = getattr(agent.agent.state, 'shared_memory', {})
        test_ts = agent_shared_memory.get("test_timestamp") if isinstance(agent_shared_memory, dict) else None
        
        if test_ts == test_timestamp:
            print(f"✅ {name.title()}Agent: Shared memory synchronized correctly")
        else:
            print(f"❌ {name.title()}Agent: Failed to synchronize shared memory")
            all_pass = False
    
    if all_pass:
        print("🎉 All agents successfully synchronized with shared memory!")
    else:
        print("❌ Some agents failed to synchronize with shared memory")
    
    # Xóa test data
    shared_memory.delete("test_timestamp")
    shared_memory.delete("workflow_status")
    
if __name__ == "__main__":
    test_agent_shared_memory_flow()