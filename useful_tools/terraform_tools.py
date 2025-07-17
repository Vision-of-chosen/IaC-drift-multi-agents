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

# Default AWS region for all Terraform operations
DEFAULT_AWS_REGION = "us-east-2"

def ensure_region_set(variables: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Ensure that the aws_region variable is set to ap-southeast-2.
    
    Args:
        variables: Optional dictionary of Terraform variables
        
    Returns:
        Dictionary with aws_region set to ap-southeast-2
    """
    if variables is None:
        variables = {}
    
    # Always ensure aws_region is set to ap-southeast-2
    variables["aws_region"] = DEFAULT_AWS_REGION
    
    # Validate the variables before returning
    if "aws_region" not in variables:
        logger.error("Failed to set aws_region variable")
        variables["aws_region"] = DEFAULT_AWS_REGION
    
    if variables["aws_region"] != DEFAULT_AWS_REGION:
        logger.warning(f"aws_region was set to unexpected value: {variables['aws_region']}, forcing to {DEFAULT_AWS_REGION}")
        variables["aws_region"] = DEFAULT_AWS_REGION
    
    # Additional validation to prevent truncation issues
    region_value = variables["aws_region"]
    if len(region_value) < 10:  # ap-southeast-2 is 13 characters
        logger.error(f"aws_region value appears truncated: '{region_value}', resetting to {DEFAULT_AWS_REGION}")
        variables["aws_region"] = DEFAULT_AWS_REGION
    
    logger.info(f"Terraform operation will use AWS region: {variables['aws_region']} (length: {len(variables['aws_region'])})")
    
    # Log all variables for debugging
    logger.debug(f"All Terraform variables being set: {variables}")
    
    return variables

def get_terraform_tools_mcp_client(variables: Optional[Dict[str, str]] = None) -> MCPClient:
    """
    Create and return an MCP client connected to the Terraform tools server.
    
    Args:
        variables: Optional variables to set as environment variables for the MCP server
    """
    try:
        # Set up environment variables for the MCP server to avoid serialization issues
        env = os.environ.copy()
        if variables:
            for key, value in variables.items():
                # Set TF_VAR_ prefixed environment variables that Terraform will automatically pick up
                env_key = f"TF_VAR_{key}"
                env[env_key] = str(value)
                logger.debug(f"Setting environment variable {env_key}={value}")
        
        # Connect to the MCP server using stdio transport
        return MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command=MCP_SERVER_COMMAND, 
                args=MCP_SERVER_ARGS,
                env=env  # Pass the modified environment
            )
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
    Run Terraform plan to analyze infrastructure changes.
    
    This tool runs 'terraform plan' to show what changes would be made to infrastructure.
    It automatically runs 'terraform init' if the directory is not initialized.
    The AWS region is automatically set to ap-southeast-2.
    
    Args:
        terraform_dir: Directory containing Terraform configuration files
        output_format: Output format ('json' or 'human'). Default is 'json'
        variables: Optional dictionary of Terraform variables to pass (aws_region will be set to ap-southeast-2)
    
    Returns:
        Dict containing:
            - plan: The plan output in the specified format
            - summary: A summary of planned changes
            - has_changes: Boolean indicating if there are changes to apply
            - init_output: Output from terraform init if it was run
    """

    try:
        # Ensure aws_region is set to ap-southeast-2
        variables = ensure_region_set(variables)
        
        # Create MCP client with variables set as environment variables
        mcp_client = get_terraform_tools_mcp_client(variables)
            
        logger.info(f"Running Terraform plan in directory: {terraform_dir} with region: {variables['aws_region']}")
        
        # Execute through the MCP server
        with mcp_client:
            # List available tools from the MCP server
            tools = mcp_client.list_tools_sync()
            tool_names = []
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_names.append(t.tool_spec['name'])
            logger.info(f"Available tools from Terraform tools MCP server: {tool_names}")
            
            # Check if terraform directory is initialized by looking for .terraform folder
            terraform_init_dir = os.path.join(terraform_dir, ".terraform")
            init_output = ""
            
            if not os.path.exists(terraform_init_dir):
                logger.info(f"Terraform directory {terraform_dir} not initialized. Running 'terraform init' first...")
                
                # Find ExecuteTerraformCommand tool
                command_tool = None
                for t in tools:
                    if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                        tool_name = t.tool_spec['name']
                        if tool_name == "ExecuteTerraformCommand":
                            command_tool = t
                            break
                
                if command_tool:
                    try:
                        # Run terraform init
                        # Validate variables before sending to MCP server
                        validated_variables = {}
                        if variables:
                            for key, value in variables.items():
                                if isinstance(value, str) and len(value) > 0:
                                    validated_variables[key] = value
                                    logger.debug(f"Validated variable {key}: '{value}' (length: {len(value)})")
                                else:
                                    logger.warning(f"Skipping invalid variable {key}: '{value}'")
                        
                        # Use environment variables instead of passing variables directly to avoid serialization issues
                        init_args = {
                            "command": "init",
                            "working_directory": terraform_dir,
                            "strip_ansi": True
                        }
                        
                        logger.info(f"Sending init args to MCP server: {init_args}")
                        logger.info(f"Variables will be passed via TF_VAR_ environment variables: {validated_variables}")
                        
                        init_tool_use_id = f"tf_init_{terraform_dir.replace('/', '_')}"
                        init_result = mcp_client.call_tool_sync(
                            tool_use_id=init_tool_use_id,
                            name=command_tool.tool_spec['name'],
                            arguments=init_args
                        )
                        
                        # Process init result
                        if isinstance(init_result, dict) and "content" in init_result:
                            for item in init_result["content"]:
                                if isinstance(item, dict) and 'text' in item:
                                    try:
                                        # Try to parse JSON from the text content
                                        data = json.loads(item['text'])
                                        if isinstance(data, dict) and "output" in data:
                                            init_output = data["output"]
                                        else:
                                            init_output = data
                                    except:
                                        # If not JSON, use the text as is
                                        init_output = item['text']
                                    break
                        
                        logger.info(f"Terraform init completed for {terraform_dir}")
                        
                    except Exception as e:
                        logger.error(f"Failed to run terraform init: {e}")
                        return {
                            "error": f"Failed to initialize Terraform directory: {str(e)}",
                            "plan": "",
                            "summary": "",
                            "has_changes": False,
                            "init_output": f"Init failed: {str(e)}"
                        }
                else:
                    logger.warning("ExecuteTerraformCommand tool not found, attempting plan anyway...")
            
            # Find the ExecuteTerraformCommand tool for plan
            command_tool = None
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_name = t.tool_spec['name']
                    if tool_name == "ExecuteTerraformCommand":
                        command_tool = t
                        break
                    
            if not command_tool:
                logger.error("ExecuteTerraformCommand tool not found in MCP server")
                return {
                    "error": "ExecuteTerraformCommand tool not found in MCP server",
                    "plan": "",
                    "summary": "",
                    "has_changes": False,
                    "init_output": init_output
                }
                
            # Execute the plan
            logger.info(f"Using tool: {command_tool.tool_spec['name']}")
            
            # Prepare arguments
            # Validate variables before sending to MCP server
            validated_variables = {}
            if variables:
                for key, value in variables.items():
                    if isinstance(value, str) and len(value) > 0:
                        validated_variables[key] = value
                        logger.debug(f"Validated variable {key}: '{value}' (length: {len(value)})")
                    else:
                        logger.warning(f"Skipping invalid variable {key}: '{value}'")
            
            # Use environment variables instead of passing variables directly to avoid serialization issues
            plan_args = {
                "command": "plan",
                "working_directory": terraform_dir,
                "strip_ansi": True
            }
            
            logger.info(f"Sending plan args to MCP server: {plan_args}")
            logger.info(f"Variables will be passed via TF_VAR_ environment variables: {validated_variables}")
            
            # Add output format if specified
            if output_format == "json":
                plan_args["output_format"] = "json"
            
            # Generate a unique tool use ID
            tool_use_id = f"tf_plan_{terraform_dir.replace('/', '_')}"
            
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name=command_tool.tool_spec['name'],
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
                            elif isinstance(data, dict) and "plan" in data:
                                plan_output = data["plan"]
                            else:
                                plan_output = data
                        except:
                            # If not JSON, use the text as is
                            plan_output = item['text']
                        break
            
            # Analyze plan for changes
            has_changes = False
            summary = "No changes detected"
            
            if isinstance(plan_output, str):
                # Look for indicators of changes in the plan output
                change_indicators = [
                    "will be created",
                    "will be updated",
                    "will be destroyed",
                    "will be replaced",
                    "Plan:",
                    "# ",  # Resource changes often start with #
                    "+ ",  # Additions
                    "- ",  # Deletions
                    "~ "   # Updates
                ]
                
                for indicator in change_indicators:
                    if indicator in plan_output:
                        has_changes = True
                        break
                
                # Try to extract summary from plan output
                lines = plan_output.split('\n')
                for line in lines:
                    if "Plan:" in line:
                        summary = line.strip()
                        break
                
                if has_changes and summary == "No changes detected":
                    summary = "Infrastructure changes detected in plan"
                    
            return {
                "plan": plan_output,
                "summary": summary,
                "has_changes": has_changes,
                "init_output": init_output,
                "region": variables["aws_region"]
            }
            
    except Exception as e:
        logger.error(f"Error running Terraform plan: {str(e)}")
        return {
            "error": f"Error running Terraform plan: {str(e)}",
            "plan": "",
            "summary": "",
            "has_changes": False,
            "init_output": "",
            "region": DEFAULT_AWS_REGION
        }


@tool
def terraform_apply(
    terraform_dir: str,
    auto_approve: bool = False,
    variables: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Apply Terraform changes to infrastructure.
    
    This tool runs 'terraform apply' to make the changes shown in a plan.
    It requires explicit approval unless auto_approve is set to True.
    The AWS region is automatically set to ap-southeast-2.
    
    Args:
        terraform_dir: Directory containing Terraform configuration files
        auto_approve: Whether to auto-approve the apply (default: False)
        variables: Optional dictionary of Terraform variables to pass (aws_region will be set to ap-southeast-2)
    
    Returns:
        Dict containing:
            - output: The apply output
            - success: Boolean indicating if apply succeeded
            - changed_resources: Information about resources that were changed
    """

    try:
        # Ensure aws_region is set to ap-southeast-2
        variables = ensure_region_set(variables)
        
        # Create MCP client with variables set as environment variables
        mcp_client = get_terraform_tools_mcp_client(variables)
            
        logger.info(f"Running Terraform apply in directory: {terraform_dir} with region: {variables['aws_region']}")
        
        # Execute the query through the MCP server
        with mcp_client:
            # List available tools from the MCP server
            tools = mcp_client.list_tools_sync()
            tool_names = []
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_names.append(t.tool_spec['name'])
            logger.info(f"Available tools from Terraform tools MCP server: {tool_names}")
            
            # Find the ExecuteTerraformCommand tool
            command_tool = None
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_name = t.tool_spec['name']
                    if tool_name == "ExecuteTerraformCommand":
                        command_tool = t
                        break
                    
            if not command_tool:
                logger.error("ExecuteTerraformCommand tool not found in MCP server")
                return {
                    "error": "ExecuteTerraformCommand tool not found in MCP server",
                    "output": "",
                    "success": False,
                    "changed_resources": {},
                    "region": variables["aws_region"]
                }
                
            # Execute the apply
            logger.info(f"Using tool: {command_tool.tool_spec['name']}")
            
            # Prepare arguments
            # Validate variables before sending to MCP server
            validated_variables = {}
            if variables:
                for key, value in variables.items():
                    if isinstance(value, str) and len(value) > 0:
                        validated_variables[key] = value
                        logger.debug(f"Validated variable {key}: '{value}' (length: {len(value)})")
                    else:
                        logger.warning(f"Skipping invalid variable {key}: '{value}'")
            
            # Use environment variables instead of passing variables directly to avoid serialization issues
            apply_args = {
                "command": "apply",
                "working_directory": terraform_dir,
                "auto_approve": auto_approve,
                "strip_ansi": True
            }
            
            logger.info(f"Sending apply args to MCP server: {apply_args}")
            logger.info(f"Variables will be passed via TF_VAR_ environment variables: {validated_variables}")
            
            # Generate a unique tool use ID
            tool_use_id = f"tf_apply_{terraform_dir.replace('/', '_')}"
            
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name=command_tool.tool_spec['name'],
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
                
            # Check if apply was successful
            success = "Apply complete!" in output or "apply completed successfully" in output if isinstance(output, str) else False
            
            # Try to extract changed resources info
            changed_resources = {}
            if isinstance(output, str) and success:
                lines = output.split('\n')
                for line in lines:
                    if "Apply complete!" in line:
                        # Extract resource counts if available
                        if "resources:" in line:
                            changed_resources["summary"] = line.strip()
                        break
                    
            return {
                "output": output,
                "success": success,
                "changed_resources": changed_resources,
                "region": variables["aws_region"]
            }
            
    except Exception as e:
        logger.error(f"Error running Terraform apply: {str(e)}")
        return {
            "error": f"Error running Terraform apply: {str(e)}",
            "output": "",
            "success": False,
            "changed_resources": {},
            "region": DEFAULT_AWS_REGION
        }


@tool
def terraform_import(
    terraform_dir: str,
    resource_address: str,
    resource_id: str,
    variables: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Import existing infrastructure into Terraform state.
    
    This tool runs 'terraform import' to bring existing resources under Terraform management,
    which is useful when reconciling drift where resources exist but aren't in the Terraform state.
    The AWS region is automatically set to ap-southeast-2.
    
    Args:
        terraform_dir: Directory containing Terraform configuration files
        resource_address: Terraform resource address (e.g., aws_s3_bucket.example)
        resource_id: Actual resource ID in the cloud provider (e.g., bucket-name)
        variables: Optional dictionary of Terraform variables to pass (aws_region will be set to ap-southeast-2)
    
    Returns:
        Dict containing:
            - output: The output from the Terraform import operation
            - success: Boolean indicating if the import was successful
    """
    try:
        # Ensure aws_region is set to ap-southeast-2
        variables = ensure_region_set(variables)
        
        # Create MCP client with variables set as environment variables
        mcp_client = get_terraform_tools_mcp_client(variables)
            
        logger.info(f"Running Terraform import in directory: {terraform_dir} with region: {variables['aws_region']}")
        
        # Execute the query through the MCP server
        with mcp_client:
            # List available tools from the MCP server
            tools = mcp_client.list_tools_sync()
            tool_names = []
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_names.append(t.tool_spec['name'])
            logger.info(f"Available tools from Terraform tools MCP server: {tool_names}")
            
            # Find the ExecuteTerraformCommand tool
            command_tool = None
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_name = t.tool_spec['name']
                    if tool_name == "ExecuteTerraformCommand":
                        command_tool = t
                        break
                    
            if not command_tool:
                logger.error("ExecuteTerraformCommand tool not found in MCP server")
                return {
                    "error": "ExecuteTerraformCommand tool not found in MCP server",
                    "output": "",
                    "success": False,
                    "region": variables["aws_region"]
                }
                
            # Execute the import
            logger.info(f"Using tool: {command_tool.tool_spec['name']}")
            
            # Prepare arguments
            # Validate variables before sending to MCP server
            validated_variables = {}
            if variables:
                for key, value in variables.items():
                    if isinstance(value, str) and len(value) > 0:
                        validated_variables[key] = value
                        logger.debug(f"Validated variable {key}: '{value}' (length: {len(value)})")
                    else:
                        logger.warning(f"Skipping invalid variable {key}: '{value}'")
            
            # Use environment variables instead of passing variables directly to avoid serialization issues
            import_args = {
                "command": "import",
                "working_directory": terraform_dir,
                "resource_address": resource_address,
                "resource_id": resource_id,
                "strip_ansi": True
            }
            
            logger.info(f"Sending import args to MCP server: {import_args}")
            logger.info(f"Variables will be passed via TF_VAR_ environment variables: {validated_variables}")
            
            # Generate a unique tool use ID
            tool_use_id = f"tf_import_{resource_address.replace('.', '_')}"
            
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name=command_tool.tool_spec['name'],
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
                "success": success,
                "region": variables["aws_region"]
            }
            
    except Exception as e:
        logger.error(f"Error running Terraform import: {str(e)}")
        return {
            "error": f"Error running Terraform import: {str(e)}",
            "output": "",
            "success": False,
            "region": DEFAULT_AWS_REGION
        } 