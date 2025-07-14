#!/usr/bin/env python3
"""
AWS Terraform MCP Server Tool for Strands Agent.

This tool connects to a local MCP server that provides Terraform operations
for infrastructure drift remediation, including commands and best practices.
"""

import os
import json
import logging
import subprocess
from typing import Dict, List, Optional, Any, Union, Literal

from strands import tool
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

logger = logging.getLogger(__name__)

# MCP Server configuration - using local path to the terraform-mcp-server
MCP_SERVER_PATH = os.environ.get("TERRAFORM_MCP_SERVER_PATH", 
                                "./IaC-drift-multi-agents/mcp/src/terraform-mcp-server")

def get_terraform_mcp_client() -> MCPClient:
    """
    Create and return an MCP client connected to the local Terraform MCP server.
    """
    try:
        # Build the command to run the local MCP server
        command = "uvx"
        args = ["awslabs.terraform-mcp-server@latest"]
        
        # Add the path to python path environment variable
        env = {"PYTHONPATH": MCP_SERVER_PATH}
        
        logger.info(f"Creating Terraform MCP client with command: {command} {' '.join(args)}")
        logger.info(f"Using PYTHONPATH: {env['PYTHONPATH']}")
        
        # Connect to the MCP server using stdio transport
        return MCPClient(lambda: stdio_client(
            StdioServerParameters(command=command, args=args, env=env)
        ))
    except Exception as e:
        logger.error(f"Failed to create Terraform MCP client: {e}")
        raise

@tool
def terraform_run_command(
    command: str,
    working_directory: str,
    variables: Optional[Dict[str, str]] = None,
    aws_region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a Terraform command using the local Terraform MCP server.
    
    This tool executes Terraform commands like init, plan, validate, apply, and destroy
    using the local MCP server. It provides a consistent interface for Terraform operations
    and captures the output for analysis.
    
    Args:
        command: Terraform command to execute (init, plan, validate, apply, destroy)
        working_directory: Directory containing Terraform files
        variables: Optional dictionary of Terraform variables to pass
        aws_region: Optional AWS region to use
    
    Returns:
        Dict containing:
            - output: The command output
            - success: Boolean indicating if the command succeeded
            - command_info: Details about the executed command
    """
    try:
        # Validate the command
        valid_commands = ["init", "plan", "validate", "apply", "destroy"]
        if command not in valid_commands:
            return {
                "error": f"Invalid command '{command}'. Valid commands are: {', '.join(valid_commands)}",
                "output": "",
                "success": False,
                "command_info": {}
            }
            
        logger.info(f"Running Terraform command '{command}' in directory: {working_directory}")
        
        # Create MCP client
        mcp_client = get_terraform_mcp_client()
        
        # Execute the command through the MCP server
        with mcp_client:
            # List available tools
            tools = mcp_client.list_tools_sync()
            tool_names = []
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_names.append(t.tool_spec['name'])
            logger.info(f"Available tools from Terraform MCP server: {tool_names}")
            
            # Find the command execution tool
            command_tool = None
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_name = t.tool_spec['name']
                    if tool_name == "ExecuteTerraformCommand" or "terraform" in tool_name.lower() and "command" in tool_name.lower():
                        command_tool = t
                        break
            
            if not command_tool:
                logger.error("ExecuteTerraformCommand tool not found in MCP server")
                return {
                    "error": "ExecuteTerraformCommand tool not found in MCP server",
                    "output": "",
                    "success": False,
                    "command_info": {}
                }
            
            # Prepare arguments
            args = {
                "command": command,
                "working_directory": working_directory,
                "strip_ansi": True
            }
            
            if variables:
                args["variables"] = variables
                
            if aws_region:
                args["aws_region"] = aws_region
            
            # Execute the command
            tool_use_id = f"tf_cmd_{command}_{working_directory.replace('/', '_')}"
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name=command_tool.tool_spec['name'],
                arguments=args
            )
            
            # Process the result
            output_text = ""
            exit_code = 1
            
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and 'text' in item:
                        try:
                            # Try to parse JSON from the text content
                            data = json.loads(item['text'])
                            output_text = data.get("output", "")
                            exit_code = data.get("exit_code", 1)
                        except:
                            # If not JSON, use the text as is
                            output_text = item['text']
                        break
            
            success = exit_code == 0
            
            return {
                "output": output_text,
                "success": success,
                "command_info": {
                    "command": command,
                    "working_directory": working_directory,
                    "variables": variables,
                    "aws_region": aws_region,
                    "exit_code": exit_code
                }
            }
            
    except Exception as e:
        logger.error(f"Error running Terraform command: {str(e)}")
        return {
            "error": f"Error running Terraform command: {str(e)}",
            "output": "",
            "success": False,
            "command_info": {
                "command": command,
                "working_directory": working_directory,
            }
        }

@tool
def terraform_run_checkov_scan(
    working_directory: str,
    check_ids: Optional[List[str]] = None,
    skip_check_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run a Checkov security scan on Terraform code.
    
    This tool uses the Checkov security scanner to analyze Terraform code for security
    vulnerabilities and compliance issues. It identifies security risks in your
    infrastructure code before deployment.
    
    Args:
        working_directory: Directory containing Terraform files
        check_ids: Optional list of specific check IDs to run
        skip_check_ids: Optional list of check IDs to skip
    
    Returns:
        Dict containing:
            - results: The scan results with identified issues
            - summary: Summary of passed and failed checks
            - success: Boolean indicating if the scan succeeded
    """
    try:
        logger.info(f"Running Checkov security scan in directory: {working_directory}")
        
        # Create MCP client
        mcp_client = get_terraform_mcp_client()
        
        # Execute the scan through the MCP server
        with mcp_client:
            # List available tools
            tools = mcp_client.list_tools_sync()
            tool_names = []
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_names.append(t.tool_spec['name'])
            logger.info(f"Available tools from Terraform MCP server: {tool_names}")
            
            # Find the checkov scan tool
            scan_tool = None
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_name = t.tool_spec['name']
                    if tool_name == "RunCheckovScan" or "checkov" in tool_name.lower():
                        scan_tool = t
                        break
            
            if not scan_tool:
                logger.error("RunCheckovScan tool not found in MCP server")
                return {
                    "error": "RunCheckovScan tool not found in MCP server",
                    "results": {},
                    "summary": {},
                    "success": False
                }
            
            # Prepare arguments
            args = {
                "working_directory": working_directory,
                "framework": "terraform",
                "output_format": "json"
            }
            
            if check_ids:
                args["check_ids"] = json.dumps(check_ids)  # Convert list to JSON string
                
            if skip_check_ids:
                args["skip_check_ids"] = json.dumps(skip_check_ids)  # Convert list to JSON string
            
            # Execute the scan
            tool_use_id = f"tf_checkov_{working_directory.replace('/', '_')}"
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name=scan_tool.tool_spec['name'],
                arguments=args
            )
            
            # Process the result
            scan_results = {}
            
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and 'text' in item:
                        try:
                            # Try to parse JSON from the text content
                            scan_results = json.loads(item['text'])
                        except:
                            # If not JSON, use the text as is
                            scan_results = {"output": item['text']}
                        break
            
            # Extract summary statistics
            success = not scan_results.get("has_failures", True)
            
            summary = {
                "passed_checks": len(scan_results.get("passed_checks", [])),
                "failed_checks": len(scan_results.get("failed_checks", [])),
                "skipped_checks": len(scan_results.get("skipped_checks", [])),
                "resource_count": scan_results.get("resource_count", 0)
            }
            
            return {
                "results": scan_results,
                "summary": summary,
                "success": success
            }
            
    except Exception as e:
        logger.error(f"Error running Checkov scan: {str(e)}")
        return {
            "error": f"Error running Checkov scan: {str(e)}",
            "results": {},
            "summary": {},
            "success": False
        }

@tool
def terraform_get_best_practices() -> Dict[str, Any]:
    """
    Get AWS Terraform best practices guidance.
    
    This tool retrieves AWS-specific best practices for Terraform from the MCP server.
    It provides guidance on creating secure, efficient, and maintainable Terraform
    configurations for AWS resources.
    
    Returns:
        Dict containing:
            - content: Best practices content in Markdown format
            - sections: Key sections of the best practices guide
    """
    try:
        logger.info("Retrieving Terraform AWS best practices")
        
        # Create MCP client
        mcp_client = get_terraform_mcp_client()
        
        # Get the best practices resource
        with mcp_client:
            # Find a tool that can retrieve best practices
            tools = mcp_client.list_tools_sync()
            best_practices_tool = None
            
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_name = t.tool_spec['name']
                    if "best" in tool_name.lower() and "practices" in tool_name.lower():
                        best_practices_tool = t
                        break
            
            if best_practices_tool:
                # Use the tool to get best practices
                tool_use_id = "tf_best_practices"
                result = mcp_client.call_tool_sync(
                    tool_use_id=tool_use_id,
                    name=best_practices_tool.tool_spec['name'],
                    arguments={}
                )
                
                content = ""
                if isinstance(result, dict) and "content" in result:
                    for item in result["content"]:
                        if isinstance(item, dict) and 'text' in item:
                            content = item['text']
                            break
            else:
                # Try to get the resource directly
                try:
                    # Use call_tool_sync with a generic resource getter instead of get_resource
                    resource_tool = None
                    for t in tools:
                        if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                            tool_name = t.tool_spec['name']
                            if "get" in tool_name.lower() and "resource" in tool_name.lower():
                                resource_tool = t
                                break
                    
                    if resource_tool:
                        resource_result = mcp_client.call_tool_sync(
                            tool_use_id="get_resource",
                            name=resource_tool.tool_spec['name'],
                            arguments={"uri": "terraform://aws_best_practices"}
                        )
                        
                        if isinstance(resource_result, dict) and "content" in resource_result:
                            for item in resource_result["content"]:
                                if isinstance(item, dict) and 'text' in item:
                                    content = item['text']
                                    break
                    else:
                        # Try a direct call to a known tool name as fallback
                        try:
                            resource_result = mcp_client.call_tool_sync(
                                tool_use_id="get_aws_best_practices",
                                name="GetResource",
                                arguments={"uri": "terraform://aws_best_practices"}
                            )
                            
                            if isinstance(resource_result, dict) and "content" in resource_result:
                                for item in resource_result["content"]:
                                    if isinstance(item, dict) and 'text' in item:
                                        content = item['text']
                                        break
                        except:
                            raise Exception("No resource getter tool found")
                except:
                    return {
                        "error": "Could not find best practices tool or resource",
                        "content": "",
                        "sections": []
                    }
            
            # Extract sections with simple parsing
            sections = []
            current_section = None
            
            for line in content.split("\n"):
                if line.startswith("## "):
                    if current_section:
                        sections.append(current_section)
                    current_section = {
                        "title": line[3:].strip(),
                        "content": []
                    }
                elif current_section:
                    current_section["content"].append(line)
            
            if current_section:
                sections.append(current_section)
            
            # Convert section content lists to strings
            for section in sections:
                section["content"] = "\n".join(section["content"])
            
            return {
                "content": content,
                "sections": sections
            }
            
    except Exception as e:
        logger.error(f"Error getting Terraform best practices: {str(e)}")
        return {
            "error": f"Error getting Terraform best practices: {str(e)}",
            "content": "",
            "sections": []
        }

@tool
def terraform_get_provider_docs(
    asset_name: str,
    asset_type: str = "resource"
) -> Dict[str, Any]:
    """
    Get documentation for AWS or AWSCC provider resources.
    
    This tool retrieves documentation for AWS or AWSCC provider resources and data sources.
    It provides detailed information on resource configurations, arguments, and attributes.
    
    Args:
        asset_name: Name of the resource (e.g., 'aws_s3_bucket', 'awscc_lambda_function')
        asset_type: Type of documentation to search - 'resource', 'data_source', or 'both'
    
    Returns:
        Dict containing:
            - results: List of documentation entries
            - provider_type: Whether this is an AWS or AWSCC provider resource
    """
    try:
        logger.info(f"Getting provider documentation for {asset_name}, type: {asset_type}")
        
        # Create MCP client
        mcp_client = get_terraform_mcp_client()
        
        # Determine which provider to search
        is_awscc = asset_name.startswith("awscc_")
        
        # Execute the search
        with mcp_client:
            # Find the appropriate provider docs tool
            tools = mcp_client.list_tools_sync()
            aws_provider_tool = None
            awscc_provider_tool = None
            
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_name = t.tool_spec['name']
                    if "aws" in tool_name.lower() and "provider" in tool_name.lower() and not "awscc" in tool_name.lower():
                        aws_provider_tool = t
                    if "awscc" in tool_name.lower() and "provider" in tool_name.lower():
                        awscc_provider_tool = t
            
            if is_awscc and not awscc_provider_tool:
                return {
                    "error": "AWSCC provider documentation tool not found",
                    "results": [],
                    "provider_type": "AWSCC"
                }
            
            if not is_awscc and not aws_provider_tool:
                return {
                    "error": "AWS provider documentation tool not found",
                    "results": [],
                    "provider_type": "AWS"
                }
            
            # Choose the appropriate tool
            provider_tool = awscc_provider_tool if is_awscc else aws_provider_tool
            provider_type = "AWSCC" if is_awscc else "AWS"
            
            # Check if provider tool was found
            if not provider_tool:
                return {
                    "error": f"{provider_type} provider documentation tool not found",
                    "results": [],
                    "provider_type": provider_type
                }
            
            # Execute the tool
            tool_use_id = f"tf_provider_docs_{asset_name}"
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name=provider_tool.tool_spec['name'],
                arguments={
                    "asset_name": asset_name,
                    "asset_type": asset_type
                }
            )
            
            # Process the result
            docs_results = []
            
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and 'text' in item:
                        try:
                            # Try to parse JSON from the text content
                            docs_results = json.loads(item['text'])
                        except:
                            # If not JSON, use the text as is
                            docs_results = [{"content": item['text']}]
                        break
            
            return {
                "results": docs_results,
                "provider_type": provider_type
            }
            
    except Exception as e:
        logger.error(f"Error getting provider documentation: {str(e)}")
        return {
            "error": f"Error getting provider documentation: {str(e)}",
            "results": [],
            "provider_type": "unknown"
        } 