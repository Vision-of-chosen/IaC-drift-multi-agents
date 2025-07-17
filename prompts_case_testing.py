#!/usr/bin/env python3
"""
Terraform Drift Detection & Remediation System - Case Testing

A multi-agent orchestration system for testing specialized drift detection cases.
This version allows you to select and test different specialized prompt cases:

1. General Drift Detection (Original)
2. Security-Focused Drift Detection  
3. Cost Optimization Drift Detection
4. Multi-Environment Drift Detection
5. Database Security Drift Detection
6. Network Security Drift Detection

Each case implements specialized workflows, tools, and approval processes
optimized for specific infrastructure management scenarios.
"""

import logging
import os
import sys
import importlib
from typing import Dict, Any

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

from permission_handlers import configure_permission_manager, get_permission_status

# Configuration
TERRAFORM_DIR = "./terraform"

# Available prompt cases
PROMPT_CASES = {
    "1": {
        "name": "General Drift Detection (Original)",
        "module": "prompts.prompts",
        "class": "AgentPrompts",
        "description": "Standard drift detection across all AWS resources with general workflows"
    },
    "2": {
        "name": "Security-Focused Drift Detection",
        "module": "prompts.prompts_1", 
        "class": "SecurityDriftPrompts",
        "description": "Security-critical resources with compliance frameworks (CIS, NIST, SOC2, PCI-DSS, GDPR)"
    },
    "3": {
        "name": "Cost Optimization Drift Detection",
        "module": "prompts.prompts_2",
        "class": "CostOptimizationPrompts", 
        "description": "Cost-related drift with financial impact analysis and optimization recommendations"
    },
    "4": {
        "name": "Multi-Environment Drift Detection",
        "module": "prompts.prompts_3",
        "class": "MultiEnvironmentPrompts",
        "description": "Cross-environment consistency between dev/staging/production environments"
    },
    "5": {
        "name": "Database Security Drift Detection", 
        "module": "prompts.prompts_4",
        "class": "DatabaseSecurityPrompts",
        "description": "Database-specific security for RDS, DynamoDB, Aurora with regulatory compliance"
    },
    "6": {
        "name": "Network Security Drift Detection",
        "module": "prompts.prompts_5", 
        "class": "NetworkSecurityPrompts",
        "description": "Network infrastructure security with zero-trust principles and topology analysis"
    }
}

def display_case_menu():
    """Display the available test cases"""
    print("\n" + "="*80)
    print("🧪 TERRAFORM DRIFT DETECTION - CASE TESTING SYSTEM")
    print("="*80)
    print("\nAvailable Test Cases:")
    print("-" * 50)
    
    for key, case in PROMPT_CASES.items():
        print(f"{key}. {case['name']}")
        print(f"   {case['description']}")
        print()
    
    print("-" * 50)

def select_test_case() -> str:
    """Allow user to select a test case"""
    while True:
        display_case_menu()
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice in PROMPT_CASES:
            selected_case = PROMPT_CASES[choice]
            print(f"\n✅ Selected: {selected_case['name']}")
            print(f"📋 Description: {selected_case['description']}")
            
            confirm = input("\nProceed with this selection? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return choice
            else:
                continue
        else:
            print(f"❌ Invalid choice '{choice}'. Please select 1-6.")
            input("Press Enter to try again...")

def load_prompt_case(case_key: str) -> tuple:
    """
    Dynamically load the selected prompt case module and class
    
    Returns:
        tuple: (module, prompts_class, case_info)
    """
    case_info = PROMPT_CASES[case_key]
    
    try:
        # Import the module dynamically
        module = importlib.import_module(case_info["module"])
        
        # Get the prompts class
        prompts_class = getattr(module, case_info["class"])
        
        logger.info(f"Loaded prompt case: {case_info['name']}")
        return module, prompts_class, case_info
        
    except ImportError as e:
        logger.error(f"Failed to import module {case_info['module']}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Failed to find class {case_info['class']} in module {case_info['module']}: {e}")
        raise

def configure_case_specific_permissions(case_key: str) -> Dict[str, Any]:
    """
    Configure permission settings based on the selected case.
    Each case may have different security requirements.
    """
    logger.info(f"Configuring permissions for case {case_key}...")
    
    # Base permissions (common to all cases)
    base_auto_approve = [
        "current_time",
        "file_read", 
        "calculator",
        "aws_documentation_search",
        "terraform_documentation_search",
        "retrieve",
        "read_tfstate",
        "cloudtrail_logs",
        "cloudwatch_logs",
        "terraform_plan",
        "terraform_get_best_practices",
        "terraform_get_provider_docs"
    ]
    
    base_require_approval = [
        "terraform_apply",
        "terraform_run_command", 
        "file_write",
        "editor",
        "shell"
    ]
    
    # Case-specific permission configurations
    case_specific_config = {
        "1": {  # General
            "auto_approve": [],
            "require_approval": ["use_aws", "terraform_run_checkov_scan"]
        },
        "2": {  # Security-Focused
            "auto_approve": ["security_compliance_check", "iam_access_analyzer"],
            "require_approval": ["use_aws", "terraform_run_checkov_scan", "security_validation", "incident_reporter"]
        },
        "3": {  # Cost Optimization
            "auto_approve": ["cost_explorer", "right_sizing_advisor", "cost_calculator"],
            "require_approval": ["use_aws", "budget_approver", "cost_impact_validator"]
        },
        "4": {  # Multi-Environment
            "auto_approve": ["environment_comparator", "deployment_tracker"],
            "require_approval": ["use_aws", "environment_sync", "promotion_workflow"]
        },
        "5": {  # Database Security
            "auto_approve": ["database_security_scanner", "encryption_validator"],
            "require_approval": ["use_aws", "database_security_validator", "encryption_manager"]
        },
        "6": {  # Network Security
            "auto_approve": ["network_topology_analyzer", "security_group_analyzer"],
            "require_approval": ["use_aws", "network_security_validator", "segmentation_tester"]
        }
    }
    
    # Get case-specific configuration
    case_config = case_specific_config.get(case_key, case_specific_config["1"])
    
    # Combine base and case-specific permissions
    auto_approve_tools = base_auto_approve + case_config["auto_approve"]
    require_approval_tools = base_require_approval + case_config["require_approval"]
    
    configure_permission_manager(
        auto_approve_tools=auto_approve_tools,
        require_approval_tools=require_approval_tools
    )
    
    status = get_permission_status()
    logger.info(f"Case {case_key} permissions configured: {len(status['auto_approve_tools'])} auto-approved, {len(status['require_approval_tools'])} require approval")
    
    return status

class CaseTestingChatInterface:
    """
    Modified chat interface that uses the selected prompt case
    """
    
    def __init__(self, prompts_class, case_info):
        self.prompts_class = prompts_class
        self.case_info = case_info
        self.setup_interface()
    
    def setup_interface(self):
        """Setup the chat interface with the selected prompts"""
        try:
            # Import the chat interface module
            from chat_interface import TerraformDriftChatInterface
            
            # Create the chat interface with our prompts_class
            self.chat_interface = TerraformDriftChatInterface(prompts_class=self.prompts_class)
            
            logger.info(f"Chat interface configured for {self.case_info['name']}")
            
        except ImportError as e:
            logger.error(f"Failed to import chat interface: {e}")
            raise
    
    def run(self):
        """Run the chat interface with case-specific prompts"""
        print(f"\n🚀 Starting {self.case_info['name']} Testing Session...")
        print(f"📋 {self.case_info['description']}")
        print("\n" + "="*80)
        
        try:
            self.chat_interface.run()
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted by user. Goodbye!")
        except Exception as e:
            logger.error(f"Error during chat session: {e}")
            print(f"❌ Error occurred: {e}")

def main():
    """Main entry point for case testing"""
    print("🏗️  Initializing Terraform Drift Detection - Case Testing System...")
    
    try:
        # Select test case
        selected_case_key = select_test_case()
        
        # Load the selected prompt case
        print(f"\n📦 Loading prompt case {selected_case_key}...")
        module, prompts_class, case_info = load_prompt_case(selected_case_key)
        print("✅ Prompt case loaded successfully!")
        
        # Configure case-specific permissions
        print("🔐 Configuring case-specific permissions...")
        configure_case_specific_permissions(selected_case_key)
        print("✅ Permissions configured!")
        
        # Ensure terraform directory exists
        os.makedirs(TERRAFORM_DIR, exist_ok=True)
        
        # Create and run the case-specific chat interface
        print("🎯 Initializing case-specific chat interface...")
        case_interface = CaseTestingChatInterface(prompts_class, case_info)
        case_interface.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Testing session cancelled by user. Goodbye!")
    except Exception as e:
        print(f"❌ Failed to initialize case testing system: {e}")
        logger.error(f"Case testing system initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 