#!/usr/bin/env python3
"""
AWS wrapper tool for the Terraform Drift Detection & Remediation System.

This module provides a wrapper around the strands_tools use_aws function
to ensure it uses the user-specific boto3 sessions created by the API.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import boto3
from boto3.session import Session as Boto3Session

from strands import tool
from strands_tools import use_aws as original_use_aws
from strands_tools.utils.data_util import convert_datetime_to_str

logger = logging.getLogger(__name__)

# Import shared_memory
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared_memory import shared_memory
    logger.info("Successfully imported shared_memory")
except ImportError as e:
    logger.error(f"Failed to import shared_memory: {e}")
    # Create a dummy shared_memory implementation
    class DummySharedMemory:
        def get(self, key, default=None, session_id=None):
            return default
        def set(self, key, value, session_id=None):
            pass
    shared_memory = DummySharedMemory()

# Store user-specific boto3 sessions
_boto3_sessions: Dict[str, Boto3Session] = {}

def register_boto3_session(session_key: str, session: Boto3Session):
    """
    Register a boto3 session for a specific user and agent.
    
    Args:
        session_key: The key to store the session under (typically "{agent_type}_{session_id}")
        session: The boto3 session to store
    """
    global _boto3_sessions
    _boto3_sessions[session_key] = session
    logger.info(f"Registered boto3 session for {session_key}")
    
    # Also store session information in shared memory
    try:
        # We can't store the actual session object, but we can store the credentials
        credentials = session.get_credentials()
        if credentials:
            frozen_creds = credentials.get_frozen_credentials()
            session_info = {
                "access_key": frozen_creds.access_key,
                "secret_key": frozen_creds.secret_key,
                "region": session.region_name,
                "timestamp": "now"  # Will be converted to ISO format by shared_memory
            }
            # Store in shared memory with a special prefix
            shared_memory.set(f"boto3_session_{session_key}", session_info)
            logger.info(f"Stored boto3 session info in shared memory for {session_key}")
    except Exception as e:
        logger.error(f"Error storing boto3 session info in shared memory: {e}")

def get_boto3_session(session_key: str) -> Optional[Boto3Session]:
    """
    Get a boto3 session for a specific user and agent.
    
    Args:
        session_key: The key to retrieve the session for
        
    Returns:
        The boto3 session if found, None otherwise
    """
    global _boto3_sessions
    
    # First try to get from in-memory cache
    if session_key in _boto3_sessions:
        logger.info(f"Found boto3 session in memory for {session_key}")
        return _boto3_sessions[session_key]
    
    # If not in memory, try to get from shared memory
    try:
        session_info = shared_memory.get(f"boto3_session_{session_key}")
        if session_info:
            logger.info(f"Found boto3 session info in shared memory for {session_key}")
            # Recreate boto3 session from stored credentials
            session = boto3.session.Session(
                aws_access_key_id=session_info.get("access_key"),
                aws_secret_access_key=session_info.get("secret_key"),
                region_name=session_info.get("region")
            )
            # Cache it in memory for future use
            _boto3_sessions[session_key] = session
            return session
    except Exception as e:
        logger.error(f"Error retrieving boto3 session from shared memory: {e}")
    
    return None

def clear_boto3_session(session_key: str):
    """
    Clear a boto3 session for a specific user and agent.
    
    Args:
        session_key: The key to clear the session for
    """
    global _boto3_sessions
    if session_key in _boto3_sessions:
        del _boto3_sessions[session_key]
        logger.info(f"Cleared boto3 session from memory for {session_key}")
    
    # Also clear from shared memory
    try:
        shared_memory.delete(f"boto3_session_{session_key}")
        logger.info(f"Cleared boto3 session from shared memory for {session_key}")
    except Exception as e:
        logger.error(f"Error clearing boto3 session from shared memory: {e}")

@tool
def use_aws(
    service_name: str,
    operation_name: str,
    parameters: Dict[str, Any],
    region: Optional[str] = None,
    label: Optional[str] = None,
    profile_name: Optional[str] = None,
    session_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute AWS service operations using user-specific boto3 sessions.
    
    This wrapper ensures that AWS operations use the correct user-specific
    boto3 session instead of creating a new one from environment variables.
    
    Args:
        service_name: AWS service name (e.g., 's3', 'ec2', 'dynamodb')
        operation_name: Operation to perform (e.g., 'list_buckets')
        parameters: Dictionary of parameters for the operation
        region: AWS region (e.g., 'us-west-2')
        label: Human-readable description of the operation
        profile_name: Optional AWS profile name for credentials
        session_key: Key to identify the boto3 session to use
        
    Returns:
        Dict containing the results of the AWS operation
    """
    # Default region if not provided
    if not region:
        region = os.environ.get("AWS_REGION", "ap-southeast-2")
    
    # Default label if not provided
    if not label:
        label = f"AWS {service_name} {operation_name}"
    
    # Convert operation name from hyphen format to underscore format
    # For example: 'describe-instances' -> 'describe_instances'
    if '-' in operation_name:
        operation_name = operation_name.replace('-', '_')
    
    # Log the session key provided
    logger.info(f"use_aws called with session_key: {session_key}")
    
    # Try to get session_key from agent state if not provided
    if not session_key:
        try:
            # Get the current agent state from the calling context
            from strands.agent.state import get_current_state
            state = get_current_state()
            if state and hasattr(state, 'aws_session_key'):
                session_key = state.aws_session_key
                logger.info(f"Using session key from agent state: {session_key}")
        except Exception as e:
            logger.debug(f"Could not get session key from agent state: {e}")
    
    # Log available session keys for debugging
    global _boto3_sessions
    logger.info(f"Available boto3 sessions in memory: {list(_boto3_sessions.keys())}")
    
    # Try to get user-specific boto3 session if session_key is provided
    user_session = None
    if session_key:
        user_session = get_boto3_session(session_key)
    
    if user_session:
        logger.info(f"Found user-specific boto3 session for {session_key}")
    else:
        if session_key:
            logger.warning(f"No boto3 session found for key: {session_key}")
            
            # Try to get session info from shared memory
            try:
                session_info = shared_memory.get(f"boto3_session_{session_key}")
                if session_info:
                    logger.info(f"Found boto3 session info in shared memory for {session_key}")
                    # Recreate boto3 session from stored credentials
                    user_session = boto3.session.Session(
                        aws_access_key_id=session_info.get("access_key"),
                        aws_secret_access_key=session_info.get("secret_key"),
                        region_name=session_info.get("region")
                    )
                    # Cache it in memory for future use
                    _boto3_sessions[session_key] = user_session
                else:
                    # Try to recreate the session using API
                    try:
                        # Extract session_id and agent_type from session_key
                        parts = session_key.split("_", 1)
                        if len(parts) == 2:
                            agent_type = parts[0]
                            session_id = parts[1]
                            
                            # Get user_id from shared memory
                            shared_memory.set_session(session_id)
                            user_id = shared_memory.get("current_user_id", session_id=session_id)
                            
                            if user_id:
                                logger.info(f"Attempting to recreate boto3 session for user_id: {user_id}")
                                
                                # Import the function to create and register boto3 session
                                import sys
                                import os
                                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                                from api import create_and_register_boto3_session, get_user_boto3_session
                                
                                # Get boto3 session for this user
                                user_session = get_user_boto3_session(user_id)
                                if user_session:
                                    # Register the session
                                    register_boto3_session(session_key, user_session)
                                    logger.info(f"Recreated boto3 session for {session_key}")
                    except Exception as e:
                        logger.error(f"Error recreating boto3 session: {e}")
            except Exception as e:
                logger.error(f"Error retrieving boto3 session from shared memory: {e}")
        else:
            logger.warning("No session_key provided")
    
    # If we have a user-specific session, use it directly
    if user_session:
        try:
            # Create client using the user-specific session
            client = user_session.client(service_name, region_name=region)
            
            # Get the operation method
            operation_method = getattr(client, operation_name)
            
            # Execute the operation
            response = operation_method(**parameters)
            
            # Convert datetime objects to strings for JSON serialization
            response = convert_datetime_to_str(response)
            
            return {
                "status": "success",
                "result": response,
                "message": f"Successfully executed {service_name}.{operation_name}",
                "used_session": session_key
            }
        except Exception as e:
            logger.error(f"Error executing AWS operation with user session: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"Failed to execute {service_name}.{operation_name} with user session",
                "used_session": session_key
            }
    else:
        # Fall back to original use_aws if no user-specific session is available
        logger.warning(f"No user-specific boto3 session found for {session_key}, falling back to default")
        
        # Prepare input for original use_aws
        tool_input = {
            "toolUseId": "use_aws_wrapper",
            "input": {
                "service_name": service_name,
                "operation_name": operation_name,
                "parameters": parameters,
                "region": region,
                "label": label
            }
        }
        
        if profile_name:
            tool_input["input"]["profile_name"] = profile_name
        
        # Call original use_aws
        result = original_use_aws(tool_input)
        
        # Format the response to match our wrapper format
        if result["status"] == "success":
            try:
                # Extract the response from the success message
                response_str = result["content"][0]["text"].replace("Success: ", "")
                response = json.loads(response_str)
                return {
                    "status": "success",
                    "result": response,
                    "message": f"Successfully executed {service_name}.{operation_name}",
                    "used_session": "default"
                }
            except:
                return {
                    "status": "success",
                    "result": result["content"][0]["text"],
                    "message": f"Successfully executed {service_name}.{operation_name}",
                    "used_session": "default"
                }
        else:
            return {
                "status": "error",
                "error": result["content"][0]["text"] if result["content"] else "Unknown error",
                "message": f"Failed to execute {service_name}.{operation_name}",
                "used_session": "default"
            } 