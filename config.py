#!/usr/bin/env python3
"""
Configuration file for the Terraform Drift Detection & Remediation System.

This module centralizes all system configuration constants to enable
easy modification and environment-specific settings.
"""

# Bedrock Model Configuration
BEDROCK_MODEL_ID = "apac.anthropic.claude-3-5-sonnet-20240620-v1:0"
BEDROCK_REGION = "ap-southeast-2"

# Terraform Configuration
TERRAFORM_DIR = "./terraform"  # Using a relative path that works cross-platform

# System Configuration
LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Tools Configuration
TOOLS_DIR = "tools/src"

# Agent Configuration
AGENT_TYPES = {
    "orchestration": "OrchestrationAgent",
    "detect": "DetectAgent",
    "analyzer": "DriftAnalyzerAgent",
    "remediate": "RemediateAgent"
}

# Workflow Configuration
WORKFLOW_EDGES = [
    ("orchestration", "detect"),
    ("detect", "analyzer"),
    ("analyzer", "remediate")
]

# Shared Memory Keys
SHARED_MEMORY_KEYS = {
    "user_request": "user_request",
    "workflow_status": "workflow_status",
    "drift_detection_results": "drift_detection_results",
    "drift_analysis_results": "drift_analysis_results",
    "remediation_results": "remediation_results",
    "last_error": "last_error"
}

# Chat Interface Configuration
CHAT_COMMANDS = {
    "detect": "Run drift detection",
    "analyze": "Analyze detected drift",
    "remediate": "Apply drift remediation",
    "status": "Check system status",
    "memory": "View shared memory",
    "help": "Show help",
    "exit": "Exit the system"
}

# Display Configuration
DISPLAY_TRUNCATE_LENGTH = 500
DISPLAY_MEMORY_PREVIEW_LENGTH = 100 