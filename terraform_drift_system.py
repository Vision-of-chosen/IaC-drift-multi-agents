#!/usr/bin/env python3
"""
Terraform Drift Detection & Remediation System

A multi-agent orchestration system that automatically detects, analyzes, and remediates 
Terraform infrastructure drift through terminal-based chat interaction using strand-agents framework.

Architecture:
- OrchestrationAgent: Central coordinator receiving user requests
- DetectAgent: Infrastructure drift detection specialist  
- DriftAnalyzerAgent: Analysis & assessment specialist
- RemediateAgent: Automated remediation specialist

Features:
- Shared memory across all agents for seamless collaboration
- Terminal-based chat interface
- AWS operations via use_aws tool
- File management via file_read/file_write tools
- Bedrock Claude 3.5 Sonnet model integration

This refactored version uses a modular architecture with separate files for each component:
- prompts.py: Centralized system prompts
- shared_memory.py: Shared memory implementation
- agents/: Individual agent modules
- chat_interface.py: Terminal interface
"""

import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add tools to path
sys.path.append("tools/src")

# Add useful_tools to path
useful_tools_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "useful_tools")
sys.path.append(useful_tools_path)

from chat_interface import TerraformDriftChatInterface

# Configuration
TERRAFORM_DIR = "./terraform"


def main():
    """Main entry point"""
    print("🏗️  Initializing Terraform Drift Detection & Remediation System...")
    
    # Ensure terraform directory exists
    os.makedirs(TERRAFORM_DIR, exist_ok=True)
    
    try:
        # Create and run the chat interface
        chat_interface = TerraformDriftChatInterface()
        chat_interface.run()
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        logger.error(f"System initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 