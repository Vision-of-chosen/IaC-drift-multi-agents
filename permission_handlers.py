#!/usr/bin/env python3
"""
Permission-based Callback Handlers for the Terraform Drift Detection & Remediation System

This module provides comprehensive callback handlers that ask for user permissions
before allowing agents to execute tools or perform actions. Based on the callback
handler patterns from the multi-agentic system notebook.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Global list to track tool use IDs (prevents duplicate logging)
tool_use_ids = []

class PermissionManager:
    """Manages user permissions for agent actions"""
    
    def __init__(self, auto_approve_tools: List[str] = None, require_approval_tools: List[str] = None):
        """
        Initialize permission manager
        
        Args:
            auto_approve_tools: List of tool names that don't require approval
            require_approval_tools: List of tool names that always require approval
        """
        self.auto_approve_tools = auto_approve_tools or [
            "current_time", "file_read", "calculator", "aws_documentation_search", 
            "terraform_documentation_search"
        ]
        self.require_approval_tools = require_approval_tools or [
            "terraform_apply", "terraform_run_command", "file_write", "editor",
            "terraform_run_checkov_scan", "use_aws", "remediate", "shell"
        ]
        self.pending_approvals = {}
        self.denied_tools = set()  # Track globally denied tools
        self.processed_tool_ids = set()  # Track already processed tool IDs
        
    def get_human_approval(self, function_name: str, parameters: dict, agent_name: str = "Agent") -> bool:
        """
        Prompt the human operator for approval before executing a tool.
        Returns True if approved, False otherwise.
        """
        # Check if this tool has been globally denied
        if function_name in self.denied_tools:
            print(f"🚫 Tool '{function_name}' has been globally denied - blocking automatically")
            return False
            
        banner = "\n" + "="*60 + f"\n🚨  PERMISSION REQUEST - {agent_name}  🚨\n" + "="*60
        body = (
            f"{banner}\n"
            f"Agent: {agent_name}\n"
            f"Function: {function_name}\n\n"
            "Parameters:\n"
            f"{json.dumps(parameters, indent=2)}\n"
            f"{'='*60}\n"
            f"⚠️  This action may modify your infrastructure or files!\n"
            f"Please review the parameters carefully before approving.\n"
            f"{'='*60}"
        )
        
        # Print the approval request
        print(body)
        
        try:
            answer = input("🔐 Do you approve this request? (yes/no/always/never/deny-all): ").strip().lower()
        except KeyboardInterrupt:
            print("\n❌ Request cancelled by user")
            return False
        except EOFError:
            print("\n❌ No input received, denying request")
            return False
        
        # Handle different approval responses
        if answer in ("yes", "y", "approve", "true", "1"):
            print("✅ Request approved")
            return True
        elif answer in ("always", "a"):
            print("✅ Request approved and added to auto-approve list")
            self.auto_approve_tools.append(function_name)
            return True
        elif answer in ("never", "deny-all", "da", "block"):
            print("❌ Request denied and tool blocked for all future attempts")
            self.denied_tools.add(function_name)
            return False
        else:
            print("❌ Request denied")
            return False

    def needs_approval(self, function_name: str) -> bool:
        """Check if a function needs user approval"""
        # If tool is globally denied, it still "needs approval" so we can block it
        if function_name in self.denied_tools:
            return True
        if function_name in self.auto_approve_tools:
            return False
        if function_name in self.require_approval_tools:
            return True
        # Default: require approval for unknown tools
        return True
        
    def clear_denied_tools(self) -> None:
        """Clear the list of globally denied tools"""
        self.denied_tools.clear()
        print("✅ Cleared all globally denied tools")
        
    def get_denied_tools(self) -> set:
        """Get the set of globally denied tools"""
        return self.denied_tools.copy()

# Global permission manager instance
permission_manager = PermissionManager()

def permission_based_callback_handler(**kwargs):
    """
    Enhanced comprehensive callback handler with user permission controls.
    Intercepts tool usage events and asks for user approval before execution.
    """
    global tool_use_ids
    
    # === REASONING EVENTS (Agent's thinking process) ===
    if kwargs.get("reasoning", False):
        if "reasoningText" in kwargs:
            reasoning_text = kwargs['reasoningText']
            logger.info(f"🧠 REASONING: {reasoning_text}")
            
        if "reasoning_signature" in kwargs:
            logger.info(f"🔍 REASONING SIGNATURE: {kwargs['reasoning_signature']}")
    
    # === TEXT GENERATION EVENTS ===
    elif "data" in kwargs:
        # Log streamed text chunks from the model (use print for streaming)
        print(kwargs["data"], end="", flush=True)
        if kwargs.get("complete", False):
            print()  # Add newline when complete
    
    # === TOOL EVENTS WITH PERMISSION CONTROL ===
    elif "current_tool_use" in kwargs:
        tool = kwargs["current_tool_use"]
        tool_use_id = tool["toolUseId"]
        
        # Skip if we've already processed this tool ID
        if tool_use_id in permission_manager.processed_tool_ids:
            return
            
        # Only process if this is a new tool use event
        if tool_use_id not in tool_use_ids:
            tool_name = tool.get('name', 'unknown_tool')
            tool_input = tool.get('input', {})
            
            # Check if tool input is complete (not a partial string)
            # Skip partial tool calls that might be streaming
            if isinstance(tool_input, str) and not tool_input.strip():
                # Empty or whitespace-only input, likely incomplete
                return
            elif isinstance(tool_input, str) and ('{' in tool_input and '}' not in tool_input):
                # Partial JSON string, wait for completion
                return
                
            logger.info(f"\n🔧 REQUESTING TOOL: {tool_name}")
            if "input" in tool:
                logger.info(f"📥 TOOL INPUT: {tool_input}")
            
            # Mark this tool ID as processed to prevent duplicates
            permission_manager.processed_tool_ids.add(tool_use_id)
            
            # Check if this tool needs approval
            if permission_manager.needs_approval(tool_name):
                agent_name = kwargs.get('agent_name', 'Agent')
                
                # Ask for human approval
                approved = permission_manager.get_human_approval(
                    function_name=tool_name,
                    parameters=tool_input,
                    agent_name=agent_name
                )
                
                if not approved:
                    # Log denial and prevent tool execution
                    logger.warning(f"🚫 TOOL DENIED: {tool_name} execution blocked by user")
                    # Store denial in tool metadata (this would need SDK support)
                    tool['_permission_denied'] = True
                    return  # Early return to block execution
                else:
                    logger.info(f"✅ TOOL APPROVED: {tool_name} execution authorized")
            else:
                logger.info(f"🟢 TOOL AUTO-APPROVED: {tool_name}")
            
            tool_use_ids.append(tool_use_id)
    
    # === TOOL RESULTS ===
    elif "tool_result" in kwargs:
        tool_result = kwargs["tool_result"]
        tool_use_id = tool_result.get("toolUseId")
        result_content = tool_result.get("content", [])
        
        logger.info(f"📤 TOOL RESULT: {result_content}")
    
    # === LIFECYCLE EVENTS ===
    elif kwargs.get("init_event_loop", False):
        # logger.info("🔄 Event loop initialized")
        pass
        
    elif kwargs.get("start_event_loop", False):
        # logger.info("▶️ Event loop cycle starting")
        pass
        
    elif kwargs.get("start", False):
        # logger.info("📝 New cycle started")
        pass
        
    elif kwargs.get("complete", False):
        # logger.info("✅ Cycle completed")
        pass
        
    elif kwargs.get("force_stop", False):
        reason = kwargs.get("force_stop_reason", "unknown reason")
        # logger.info(f"🛑 Event loop force-stopped: {reason}")
        pass
    
    # === MESSAGE EVENTS ===
    elif "message" in kwargs:
        message = kwargs["message"]
        role = message.get("role", "unknown")
        # logger.info(f"📬 New message created: {role}")
        pass
    
    # === ERROR EVENTS ===
    elif "error" in kwargs:
        error_info = kwargs["error"]
        logger.error(f"❌ ERROR: {error_info}")

    # === RAW EVENTS (for debugging) ===
    elif "event" in kwargs:
        # Log raw events from the model stream (optional, can be verbose)
        logger.debug(f"🔍 RAW EVENT: {kwargs['event']}")
    
    # === DELTA EVENTS ===
    elif "delta" in kwargs:
        # Raw delta content from the model
        logger.debug(f"📊 DELTA: {kwargs['delta']}")
    
    # === CATCH-ALL FOR DEBUGGING ===
    else:
        # Log any other events we might have missed
        logger.debug(f"❓ OTHER EVENT: {kwargs}")

def create_agent_callback_handler(agent_name: str) -> Callable:
    """
    Create a callback handler specific to an agent.
    This adds the agent name context to the permission requests.
    """
    def agent_specific_callback(**kwargs):
        # Add agent name to kwargs for context
        kwargs['agent_name'] = agent_name
        return permission_based_callback_handler(**kwargs)
    
    return agent_specific_callback

def configure_permission_manager(
    auto_approve_tools: List[str] = None,
    require_approval_tools: List[str] = None
) -> None:
    """
    Configure the global permission manager with custom tool lists.
    
    Args:
        auto_approve_tools: Tools that don't require approval
        require_approval_tools: Tools that always require approval
    """
    global permission_manager
    
    if auto_approve_tools is not None:
        permission_manager.auto_approve_tools = auto_approve_tools
    
    if require_approval_tools is not None:
        permission_manager.require_approval_tools = require_approval_tools

def get_permission_status() -> Dict[str, Any]:
    """Get current permission manager status"""
    return {
        "auto_approve_tools": permission_manager.auto_approve_tools,
        "require_approval_tools": permission_manager.require_approval_tools,
        "denied_tools": list(permission_manager.denied_tools),
        "pending_approvals": len(permission_manager.pending_approvals),
        "processed_tool_ids": len(permission_manager.processed_tool_ids)
    }

def reset_permission_manager() -> None:
    """Reset permission manager to default state"""
    global permission_manager, tool_use_ids
    permission_manager = PermissionManager()
    tool_use_ids = []
    
def clear_denied_tools() -> None:
    """Clear globally denied tools - allows user to try blocked tools again"""
    permission_manager.clear_denied_tools()

# Export the main callback handler and utility functions
__all__ = [
    'permission_based_callback_handler',
    'create_agent_callback_handler',
    'configure_permission_manager',
    'get_permission_status',
    'reset_permission_manager',
    'clear_denied_tools',
    'PermissionManager'
] 