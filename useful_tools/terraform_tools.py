#!/usr/bin/env python3
"""
Terraform Tools MCP Server for Strands Agent.

This tool connects to an MCP server that provides advanced Terraform operations
for infrastructure drift remediation.
"""

import os
import json
import logging
import tempfile
from typing import Dict, List, Optional, Any, Union

from strands import tool
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

logger = logging.getLogger(__name__)

# MCP Server configuration
MCP_SERVER_COMMAND = os.environ.get("TF_TOOLS_MCP_SERVER_COMMAND", "uvx")
MCP_SERVER_ARGS = os.environ.get(
    "TF_TOOLS_MCP_SERVER_ARGS", 
    "awslabs.terraform-mcp-server@latest"
).split()

def get_terraform_tools_mcp_client() -> MCPClient:
    """
    Create and return an MCP client connected to the Terraform tools server.
    """
    try:
        # Connect to the MCP server using stdio transport
        return MCPClient(lambda: stdio_client(
            StdioServerParameters(command=MCP_SERVER_COMMAND, args=MCP_SERVER_ARGS)
        ))
    except Exception as e:
        logger.error(f"Failed to create Terraform tools MCP client: {e}")
        raise

@tool
def terraform_plan(
    terraform_dir: str,
    output_format: str = "json",
    variables: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Generate a Terraform plan to show what changes would be made.
    
    This tool runs 'terraform plan' in the specified directory and returns the output in 
    the desired format. It's essential for understanding what changes need to be made to
    remediate infrastructure drift.
    
    Args:
        terraform_dir: Directory containing Terraform configuration files
        output_format: Output format (json, text) for the plan
        variables: Dictionary of Terraform variables to apply
    
    Returns:
        Dict containing:
            - plan: The Terraform plan output
            - summary: Summary of changes (resources to add/change/destroy)
            - has_changes: Boolean indicating if there are changes
    """
    try:
        # Create MCP client
        mcp_client = get_terraform_tools_mcp_client()
        
        # Prepare the variables if provided
        vars_arg = {}
        if variables:
            vars_arg = {"variables": variables}
            
        logger.info(f"Running Terraform plan in directory: {terraform_dir}")
        
        # Execute the query through the MCP server
        with mcp_client:
            # List available tools from the MCP server
            tools = mcp_client.list_tools_sync()
            tool_names = []
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_names.append(t.tool_spec['name'])
            logger.info(f"Available tools from Terraform tools MCP server: {tool_names}")
            
            # Find the plan tool
            plan_tool = None
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_name = t.tool_spec['name']
                    if tool_name == "terraform_plan" or "plan" in tool_name.lower():
                        plan_tool = t
                        break
                    
            if not plan_tool:
                logger.error("Terraform plan tool not found in MCP server")
                return {
                    "error": "Terraform plan tool not found in MCP server",
                    "plan": "",
                    "summary": "",
                    "has_changes": False
                }
                
            # Execute the plan
            logger.info(f"Using tool: {plan_tool.tool_spec['name']}")
            
            # Prepare arguments
            plan_args = {
                "dir": terraform_dir,
                "output": output_format,
                **vars_arg
            }
            
            # Generate a unique tool use ID
            tool_use_id = f"tf_plan_{terraform_dir.replace('/', '_')}"
            
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name=plan_tool.tool_spec['name'],
                arguments=plan_args
            )
            
            # Process the result
            plan_output = ""
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and 'text' in item:
                        try:
                            # Try to parse JSON from the text content
                            data = json.loads(item['text'])
                            if isinstance(data, dict) and "output" in data:
                                plan_output = data["output"]
                            else:
                                plan_output = data
                        except:
                            # If not JSON, use the text as is
                            plan_output = item['text']
                        break
                
            # Extract summary information
            summary = {}
            has_changes = False
            
            if output_format == "json" and plan_output:
                try:
                    # Try to parse as JSON
                    plan_json = json.loads(plan_output) if isinstance(plan_output, str) else plan_output
                    
                    # Extract resource changes
                    resource_changes = plan_json.get("resource_changes", [])
                    
                    # Count by action
                    action_counts = {"create": 0, "update": 0, "delete": 0, "no-op": 0}
                    
                    for resource in resource_changes:
                        for action in resource.get("change", {}).get("actions", []):
                            if action in action_counts:
                                action_counts[action] += 1
                                
                    summary = {
                        "resources_to_create": action_counts["create"],
                        "resources_to_update": action_counts["update"],
                        "resources_to_delete": action_counts["delete"],
                        "unchanged_resources": action_counts["no-op"],
                        "total_resources": len(resource_changes)
                    }
                    
                    has_changes = (action_counts["create"] > 0 or 
                                action_counts["update"] > 0 or 
                                action_counts["delete"] > 0)
                    
                except Exception as e:
                    logger.error(f"Error parsing Terraform plan JSON: {e}")
                    summary = {"error": f"Could not parse plan JSON: {str(e)}"}
            else:
                # For text format, provide a simple indicator
                has_changes = "No changes" not in plan_output
                summary = {"text_summary": "Plan indicates changes to be made" if has_changes else "No changes to be made"}
                
            return {
                "plan": plan_output,
                "summary": summary,
                "has_changes": has_changes
            }
            
    except Exception as e:
        logger.error(f"Error running Terraform plan: {str(e)}")
        return {
            "error": f"Error running Terraform plan: {str(e)}",
            "plan": "",
            "summary": "",
            "has_changes": False
        }

@tool
def terraform_apply(
    terraform_dir: str,
    auto_approve: bool = False,
    variables: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Apply Terraform configuration to remediate infrastructure drift.
    
    This tool runs 'terraform apply' in the specified directory to create, update, or delete
    resources as needed to match the Terraform configuration. Use with caution as this
    will make actual changes to your infrastructure.
    
    Args:
        terraform_dir: Directory containing Terraform configuration files
        auto_approve: Whether to automatically approve the apply operation
        variables: Dictionary of Terraform variables to apply
    
    Returns:
        Dict containing:
            - output: The output from the Terraform apply operation
            - success: Boolean indicating if the apply was successful
            - changed_resources: Summary of resources that were changed
    """
    try:
        # Create MCP client
        mcp_client = get_terraform_tools_mcp_client()
        
        # Prepare the variables if provided
        vars_arg = {}
        if variables:
            vars_arg = {"variables": variables}
            
        logger.info(f"Running Terraform apply in directory: {terraform_dir}")
        
        # Execute the query through the MCP server
        with mcp_client:
            # List available tools from the MCP server
            tools = mcp_client.list_tools_sync()
            tool_names = []
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_names.append(t.tool_spec['name'])
            logger.info(f"Available tools from Terraform tools MCP server: {tool_names}")
            
            # Find the apply tool
            apply_tool = None
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_name = t.tool_spec['name']
                    if tool_name == "terraform_apply" or "apply" in tool_name.lower():
                        apply_tool = t
                        break
                    
            if not apply_tool:
                logger.error("Terraform apply tool not found in MCP server")
                return {
                    "error": "Terraform apply tool not found in MCP server",
                    "output": "",
                    "success": False,
                    "changed_resources": {}
                }
                
            # Execute the apply
            logger.info(f"Using tool: {apply_tool.tool_spec['name']}")
            
            # Prepare arguments
            apply_args = {
                "dir": terraform_dir,
                "auto_approve": auto_approve,
                **vars_arg
            }
            
            # Generate a unique tool use ID
            tool_use_id = f"tf_apply_{terraform_dir.replace('/', '_')}"
            
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name=apply_tool.tool_spec['name'],
                arguments=apply_args
            )
            
            # Process the result
            output = ""
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and 'text' in item:
                        try:
                            # Try to parse JSON from the text content
                            data = json.loads(item['text'])
                            if isinstance(data, dict) and "output" in data:
                                output = data["output"]
                            else:
                                output = data
                        except:
                            # If not JSON, use the text as is
                            output = item['text']
                        break
                
            success = "Apply complete!" in output if isinstance(output, str) else False
            
            # Try to parse the changed resources from the output
            changed_resources = {"created": [], "updated": [], "destroyed": []}
            
            # Very simple parsing of the text output
            if isinstance(output, str):
                for line in output.split('\n'):
                    if " created" in line:
                        resource = line.split(":")[0].strip() if ":" in line else line
                        changed_resources["created"].append(resource)
                    elif " updated" in line:
                        resource = line.split(":")[0].strip() if ":" in line else line
                        changed_resources["updated"].append(resource)
                    elif " destroyed" in line:
                        resource = line.split(":")[0].strip() if ":" in line else line
                        changed_resources["destroyed"].append(resource)
                    
            return {
                "output": output,
                "success": success,
                "changed_resources": changed_resources
            }
            
    except Exception as e:
        logger.error(f"Error running Terraform apply: {str(e)}")
        return {
            "error": f"Error running Terraform apply: {str(e)}",
            "output": "",
            "success": False,
            "changed_resources": {}
        }

@tool
def terraform_import(
    terraform_dir: str,
    resource_address: str,
    resource_id: str
) -> Dict[str, Any]:
    """
    Import existing infrastructure into Terraform state.
    
    This tool runs 'terraform import' to bring existing resources under Terraform management,
    which is useful when reconciling drift where resources exist but aren't in the Terraform state.
    
    Args:
        terraform_dir: Directory containing Terraform configuration files
        resource_address: Terraform resource address (e.g., aws_s3_bucket.example)
        resource_id: Actual resource ID in the cloud provider (e.g., bucket-name)
    
    Returns:
        Dict containing:
            - output: The output from the Terraform import operation
            - success: Boolean indicating if the import was successful
    """
    try:
        # Create MCP client
        mcp_client = get_terraform_tools_mcp_client()
            
        logger.info(f"Running Terraform import in directory: {terraform_dir}")
        
        # Execute the query through the MCP server
        with mcp_client:
            # List available tools from the MCP server
            tools = mcp_client.list_tools_sync()
            tool_names = []
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_names.append(t.tool_spec['name'])
            logger.info(f"Available tools from Terraform tools MCP server: {tool_names}")
            
            # Find the import tool
            import_tool = None
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_name = t.tool_spec['name']
                    if tool_name == "terraform_import" or "import" in tool_name.lower():
                        import_tool = t
                        break
                    
            if not import_tool:
                logger.error("Terraform import tool not found in MCP server")
                return {
                    "error": "Terraform import tool not found in MCP server",
                    "output": "",
                    "success": False
                }
                
            # Execute the import
            logger.info(f"Using tool: {import_tool.tool_spec['name']}")
            
            # Prepare arguments
            import_args = {
                "dir": terraform_dir,
                "resource_address": resource_address,
                "resource_id": resource_id
            }
            
            # Generate a unique tool use ID
            tool_use_id = f"tf_import_{resource_address.replace('.', '_')}"
            
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name=import_tool.tool_spec['name'],
                arguments=import_args
            )
            
            # Process the result
            output = ""
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and 'text' in item:
                        try:
                            # Try to parse JSON from the text content
                            data = json.loads(item['text'])
                            if isinstance(data, dict) and "output" in data:
                                output = data["output"]
                            else:
                                output = data
                        except:
                            # If not JSON, use the text as is
                            output = item['text']
                        break
                
            # Check if import was successful
            success = "Import successful" in output or "Import complete!" in output if isinstance(output, str) else False
                    
            return {
                "output": output,
                "success": success
            }
            
    except Exception as e:
        logger.error(f"Error running Terraform import: {str(e)}")
        return {
            "error": f"Error running Terraform import: {str(e)}",
            "output": "",
            "success": False
        } 