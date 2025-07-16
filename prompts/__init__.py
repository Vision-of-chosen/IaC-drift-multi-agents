#!/usr/bin/env python3
"""
Prompts package for the Terraform Drift Detection & Remediation System.

This package contains all specialized prompt classes for different drift detection use cases.
"""

# Import all prompt classes for easy access
from .prompts import AgentPrompts
from .prompts_1 import SecurityDriftPrompts
from .prompts_2 import CostOptimizationPrompts
from .prompts_3 import MultiEnvironmentPrompts
from .prompts_4 import DatabaseSecurityPrompts
from .prompts_5 import NetworkSecurityPrompts

__all__ = [
    'AgentPrompts',
    'SecurityDriftPrompts',
    'CostOptimizationPrompts', 
    'MultiEnvironmentPrompts',
    'DatabaseSecurityPrompts',
    'NetworkSecurityPrompts'
] 