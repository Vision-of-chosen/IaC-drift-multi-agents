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
import asyncio
import threading
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
        self.approval_callback = None  # Callback for sending approval requests to frontend
        self.current_permission_request = None  # Store current permission request for REST API
        
    def set_approval_callback(self, callback: Callable):
        """Set the callback function for sending approval requests to frontend"""
        self.approval_callback = callback

    def get_human_approval_rest(self, function_name: str, parameters: dict, agent_name: str = "Agent", session_id: str = None) -> bool:
        """
        Request approval from the human operator through the REST API.
        Stores the permission request and returns False to pause execution.
        The actual approval will be processed when the user responds.
        """
        # Check if this tool has been globally denied
        if function_name in self.denied_tools:
            logger.info(f"🚫 Tool '{function_name}' has been globally denied - blocking automatically")
            return False
            
        # Generate unique approval request ID
        request_id = f"approval_{int(time.time() * 1000)}_{function_name}"
        
        # Create approval request data
        approval_request = {
            "request_id": request_id,
            "agent_name": agent_name,
            "function_name": function_name,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        # Store the current permission request for REST API access
        self.current_permission_request = approval_request
        
        # Store in pending approvals with a threading event for synchronization
        self.pending_approvals[request_id] = {
            "approved": None,
            "response": None,
            "event": threading.Event(),
            "data": approval_request
        }
        
        logger.info(f"🔐 Permission request created for {function_name} (REST API mode)")
        
        # Wait for response with timeout (60 seconds)
        try:
            response_received = self.pending_approvals[request_id]["event"].wait(timeout=60.0)
            
            if response_received:
                # Get the approval result
                approval_data = self.pending_approvals[request_id]
                approved = approval_data["approved"]
                response = approval_data["response"]
                
                # Clean up
                if request_id in self.pending_approvals:
                    del self.pending_approvals[request_id]
                
                # Clear current request
                if self.current_permission_request and self.current_permission_request.get("request_id") == request_id:
                    self.current_permission_request = None
                
                # Handle different approval responses
                if response in ("always", "a"):
                    logger.info("✅ Request approved and added to auto-approve list")
                    self.auto_approve_tools.append(function_name)
                    return True
                elif response in ("never", "deny-all", "da", "block"):
                    logger.info("❌ Request denied and tool blocked for all future attempts")
                    self.denied_tools.add(function_name)
                    return False
                elif approved:
                    logger.info("✅ Request approved")
                    return True
                else:
                    logger.info("❌ Request denied")
                    return False
                    
            else:
                logger.warning(f"⏰ Permission request for {function_name} timed out - denying")
                if request_id in self.pending_approvals:
                    del self.pending_approvals[request_id]
                if self.current_permission_request and self.current_permission_request.get("request_id") == request_id:
                    self.current_permission_request = None
                return False
                
        except Exception as e:
            logger.error(f"Error waiting for approval: {e}")
            if request_id in self.pending_approvals:
                del self.pending_approvals[request_id]
            if self.current_permission_request and self.current_permission_request.get("request_id") == request_id:
                self.current_permission_request = None
            return False

    async def get_human_approval_async(self, function_name: str, parameters: dict, agent_name: str = "Agent", session_id: str = None) -> bool:
        """
        Request approval from the human operator through the chat API.
        Returns True if approved, False otherwise.
        """
        # Check if this tool has been globally denied
        if function_name in self.denied_tools:
            logger.info(f"🚫 Tool '{function_name}' has been globally denied - blocking automatically")
            return False
            
        # Generate unique approval request ID
        request_id = f"approval_{int(time.time() * 1000)}_{function_name}"
        
        # Create approval request data
        approval_request = {
            "request_id": request_id,
            "agent_name": agent_name,
            "function_name": function_name,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        # Store pending approval
        self.pending_approvals[request_id] = {
            "approved": None,
            "response": None,
            "event": asyncio.Event()
        }
        
        logger.info(f"🔐 Requesting approval for {function_name} from frontend...")
        
        # Send approval request to frontend through callback
        if self.approval_callback:
            try:
                await self.approval_callback(approval_request)
            except Exception as e:
                logger.error(f"Error sending approval request: {e}")
                # Clean up and deny if we can't send the request
                del self.pending_approvals[request_id]
                return False
        else:
            logger.error("No approval callback set - denying request")
            del self.pending_approvals[request_id]
            return False
        
        # Wait for response with timeout (60 seconds)
        try:
            await asyncio.wait_for(self.pending_approvals[request_id]["event"].wait(), timeout=60.0)
            
            # Get the approval result
            approval_data = self.pending_approvals[request_id]
            approved = approval_data["approved"]
            response = approval_data["response"]
            
            # Clean up
            del self.pending_approvals[request_id]
            
            # Handle different approval responses
            if response in ("always", "a"):
                logger.info("✅ Request approved and added to auto-approve list")
                self.auto_approve_tools.append(function_name)
                return True
            elif response in ("never", "deny-all", "da", "block"):
                logger.info("❌ Request denied and tool blocked for all future attempts")
                self.denied_tools.add(function_name)
                return False
            elif approved:
                logger.info("✅ Request approved")
                return True
            else:
                logger.info("❌ Request denied")
                return False
                
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Approval request for {function_name} timed out - denying")
            if request_id in self.pending_approvals:
                del self.pending_approvals[request_id]
            return False
        except Exception as e:
            logger.error(f"Error waiting for approval: {e}")
            if request_id in self.pending_approvals:
                del self.pending_approvals[request_id]
            return False

    def get_human_approval(self, function_name: str, parameters: dict, agent_name: str = "Agent") -> bool:
        """
        Main approval method - now uses REST API approach by default.
        Falls back to console input if no callback is available.
        """
        # For REST API mode, use the synchronous REST approach
        session_id = self.get_current_session_id()
        if session_id:
            return self.get_human_approval_rest(function_name, parameters, agent_name, session_id)
        
        # Fall back to console input
        return self._get_console_approval(function_name, parameters, agent_name)
    
    def get_current_session_id(self) -> Optional[str]:
        """Get the current session ID from shared memory"""
        try:
            from shared_memory import shared_memory
            return shared_memory.get("current_session_id")
        except:
            return None
    
    def _get_console_approval(self, function_name: str, parameters: dict, agent_name: str = "Agent") -> bool:
        """Original console-based approval method"""
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

    def handle_approval_response(self, request_id: str, approved: bool, response: str = None) -> bool:
        """
        Handle approval response from frontend
        
        Args:
            request_id: The approval request ID
            approved: Whether the request was approved
            response: The specific response type (yes/no/always/never/deny-all)
            
        Returns:
            True if the response was processed, False if request_id not found
        """
        if request_id not in self.pending_approvals:
            logger.warning(f"Received approval response for unknown request: {request_id}")
            return False
        
        # Update the pending approval
        approval_data = self.pending_approvals[request_id]
        approval_data["approved"] = approved
        approval_data["response"] = response
        
        # Signal that we have a response (works for both threading.Event and asyncio.Event)
        approval_data["event"].set()
        
        logger.info(f"📬 Processed approval response for {request_id}: {approved} ({response})")
        return True

    def get_current_permission_request(self) -> Optional[Dict[str, Any]]:
        """Get the current permission request for REST API"""
        return self.current_permission_request

    def clear_current_permission_request(self):
        """Clear the current permission request"""
        self.current_permission_request = None

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
        logger.info("✅ Cleared all globally denied tools")
        
    def get_denied_tools(self) -> set:
        """Get the set of globally denied tools"""
        return self.denied_tools.copy()
    
    def get_pending_approvals(self) -> Dict[str, Dict[str, Any]]:
        """Get information about pending approvals"""
        pending_info = {}
        for request_id, data in self.pending_approvals.items():
            pending_info[request_id] = {
                "approved": data["approved"],
                "response": data["response"],
                "waiting": data["approved"] is None,
                "data": data.get("data", {})  # Include the full request data
            }
        return pending_info

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
                session_id = kwargs.get('session_id')  # Get session_id from kwargs
                
                # Use the REST API approval method
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
        logger.info("🔄 Event loop initialized")
        
    elif kwargs.get("start_event_loop", False):
        logger.info("▶️ Event loop cycle starting")
        
    elif kwargs.get("start", False):
        logger.info("📝 New cycle started")
        
    elif kwargs.get("complete", False):
        logger.info("✅ Cycle completed")
        
    elif kwargs.get("force_stop", False):
        reason = kwargs.get("force_stop_reason", "unknown reason")
        logger.info(f"🛑 Event loop force-stopped: {reason}")
    
    # === MESSAGE EVENTS ===
    elif "message" in kwargs:
        message = kwargs["message"]
        role = message.get("role", "unknown")
        logger.info(f"📬 New message created: {role}")
    
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

def create_agent_callback_handler(agent_name: str, session_id: str = None) -> Callable:
    """
    Create a callback handler specific to an agent.
    This adds the agent name and session_id context to the permission requests.
    """
    def agent_specific_callback(**kwargs):
        # Add agent name and session_id to kwargs for context
        kwargs['agent_name'] = agent_name
        if session_id:
            kwargs['session_id'] = session_id
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
        "processed_tool_ids": len(permission_manager.processed_tool_ids),
        "current_permission_request": permission_manager.get_current_permission_request()
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