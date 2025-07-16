#!/usr/bin/env python3
"""
Permission Configuration Examples for the Terraform Drift Detection & Remediation System

This script demonstrates how to configure permission-based callback handlers
for different scenarios and security requirements.
"""

import logging
from permission_handlers import (
    configure_permission_manager,
    get_permission_status,
    reset_permission_manager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def configure_development_permissions():
    """
    Configure permissions for development environment.
    More permissive for testing and development.
    """
    logger.info("Configuring DEVELOPMENT permission settings...")
    
    auto_approve_tools = [
        # Read-only tools
        "current_time",
        "file_read", 
        "calculator",
        "aws_documentation_search",
        "terraform_documentation_search",
        "retrieve",
        "read_tfstate",
        "cloudtrail_logs",
        "cloudwatch_logs",
        # Development tools
        "terraform_plan",  # Planning is usually safe
        "terraform_get_best_practices",
        "terraform_get_provider_docs"
    ]
    
    require_approval_tools = [
        # Potentially destructive tools
        "terraform_apply",
        "terraform_run_command", 
        "file_write",
        "editor",
        "terraform_run_checkov_scan",
        "use_aws",  # AWS operations need approval
        "shell"
    ]
    
    configure_permission_manager(
        auto_approve_tools=auto_approve_tools,
        require_approval_tools=require_approval_tools
    )
    
    logger.info("Development permissions configured!")
    return get_permission_status()

def configure_production_permissions():
    """
    Configure permissions for production environment.
    Very restrictive - requires approval for almost everything.
    """
    logger.info("Configuring PRODUCTION permission settings...")
    
    auto_approve_tools = [
        # Only completely safe read-only tools
        "current_time",
        "calculator",
        "aws_documentation_search",
        "terraform_documentation_search",
    ]
    
    require_approval_tools = [
        # Everything else requires approval
        "file_read",  # Even reading requires approval in production
        "retrieve",
        "read_tfstate",
        "cloudtrail_logs", 
        "cloudwatch_logs",
        "terraform_plan",
        "terraform_apply",
        "terraform_run_command",
        "file_write",
        "editor", 
        "terraform_run_checkov_scan",
        "use_aws",
        "shell",
        "terraform_get_best_practices",
        "terraform_get_provider_docs"
    ]
    
    configure_permission_manager(
        auto_approve_tools=auto_approve_tools,
        require_approval_tools=require_approval_tools
    )
    
    logger.info("Production permissions configured!")
    return get_permission_status()

def configure_demo_permissions():
    """
    Configure permissions for demo/presentation environment.
    Read-only operations are auto-approved, but any changes require approval.
    """
    logger.info("Configuring DEMO permission settings...")
    
    auto_approve_tools = [
        # Safe read-only and informational tools
        "current_time",
        "file_read",
        "calculator", 
        "aws_documentation_search",
        "terraform_documentation_search",
        "retrieve",
        "read_tfstate",
        "cloudtrail_logs",
        "cloudwatch_logs",
        "terraform_plan",  # Planning is safe for demos
        "terraform_get_best_practices",
        "terraform_get_provider_docs"
    ]
    
    require_approval_tools = [
        # Any modification or execution requires approval
        "terraform_apply",
        "terraform_run_command",
        "file_write", 
        "editor",
        "terraform_run_checkov_scan",
        "use_aws",
        "shell"
    ]
    
    configure_permission_manager(
        auto_approve_tools=auto_approve_tools,
        require_approval_tools=require_approval_tools
    )
    
    logger.info("Demo permissions configured!")
    return get_permission_status()

def configure_security_audit_permissions():
    """
    Configure permissions for security audit environment.
    Only documentation and analysis tools are auto-approved.
    """
    logger.info("Configuring SECURITY AUDIT permission settings...")
    
    auto_approve_tools = [
        # Only documentation and time
        "current_time",
        "calculator",
        "aws_documentation_search", 
        "terraform_documentation_search",
        "terraform_get_best_practices"
    ]
    
    require_approval_tools = [
        # Everything that touches actual infrastructure or files
        "file_read",
        "retrieve", 
        "read_tfstate",
        "cloudtrail_logs",
        "cloudwatch_logs",
        "terraform_plan",
        "terraform_apply",
        "terraform_run_command",
        "file_write",
        "editor",
        "terraform_run_checkov_scan", 
        "use_aws",
        "shell",
        "terraform_get_provider_docs"
    ]
    
    configure_permission_manager(
        auto_approve_tools=auto_approve_tools,
        require_approval_tools=require_approval_tools
    )
    
    logger.info("Security audit permissions configured!")
    return get_permission_status()

def main():
    """
    Main function to demonstrate different permission configurations.
    Uncomment the configuration you want to use.
    """
    print("=" * 60)
    print("Permission Configuration Examples")
    print("=" * 60)
    
    # Choose your configuration:
    
    # For development/testing (more permissive)
    status = configure_development_permissions()
    
    # For production (very restrictive)
    # status = configure_production_permissions()
    
    # For demos/presentations (read-only friendly)
    # status = configure_demo_permissions()
    
    # For security audits (minimal permissions)
    # status = configure_security_audit_permissions()
    
    print("\nCurrent Permission Status:")
    print(f"Auto-approved tools: {len(status['auto_approve_tools'])}")
    print(f"Require approval tools: {len(status['require_approval_tools'])}")
    print(f"Pending approvals: {status['pending_approvals']}")
    
    print("\nAuto-approved tools:")
    for tool in status['auto_approve_tools']:
        print(f"  ✅ {tool}")
    
    print("\nRequire approval tools:")
    for tool in status['require_approval_tools']:
        print(f"  🔐 {tool}")

if __name__ == "__main__":
    main() 