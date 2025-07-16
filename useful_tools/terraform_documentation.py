#!/usr/bin/env python3
"""
Terraform Documentation Tool for Strands Agent.

This tool connects to an MCP server that provides access to Terraform documentation
for drift analysis and remediation.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Literal

from strands import tool
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

logger = logging.getLogger(__name__)

# MCP Server configuration
MCP_SERVER_COMMAND = os.environ.get("TF_MCP_SERVER_COMMAND", "uvx")
MCP_SERVER_ARGS = os.environ.get(
    "TF_MCP_SERVER_ARGS", 
    "awslabs.terraform-mcp-server@latest"
).split()

def get_terraform_mcp_client() -> MCPClient:
    """
    Create and return an MCP client connected to the Terraform MCP server.
    """
    try:
        # Connect to the MCP server using stdio transport
        return MCPClient(lambda: stdio_client(
            StdioServerParameters(command=MCP_SERVER_COMMAND, args=MCP_SERVER_ARGS)
        ))
    except Exception as e:
        logger.error(f"Failed to create Terraform MCP client: {e}")
        raise

@tool
def terraform_documentation_search(
    asset_name: str,
    asset_type: Literal["resource", "data_source", "both"] = "resource",
    provider: Literal["aws", "awscc"] = "aws"
) -> Dict[str, Any]:
    """
    Search Terraform documentation for provider resources and configuration options.
    
    This tool queries the Terraform MCP server to retrieve official Terraform
    documentation about providers, resources, and their configuration attributes. It's useful
    for understanding the correct resource definitions, identifying drift issues, and
    determining the appropriate remediation steps.
    
    Args:
        asset_name: Name of the resource or data source (e.g., 'aws_s3_bucket', 's3_bucket')
        asset_type: Type of documentation to search - 'resource' (default), 'data_source', or 'both'
        provider: Provider to search - 'aws' (default) or 'awscc'
    
    Returns:
        Dict containing:
            - results: List of documentation entries
            - summary: Brief summary of the documentation
    """
    try:
        # Create MCP client
        mcp_client = get_terraform_mcp_client()
            
        logger.info(f"Querying Terraform documentation for {provider} {asset_type}: {asset_name}")
        
        # Execute the query through the MCP server
        with mcp_client:
            # List available tools from the MCP server
            tools = mcp_client.list_tools_sync()
            tool_names = []
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_names.append(t.tool_spec['name'])
            logger.info(f"Available tools from Terraform MCP server: {tool_names}")
            
            # Select the appropriate search tool based on provider
            search_tool_name = "SearchAwsProviderDocs" if provider == "aws" else "SearchAwsccProviderDocs"
            
            # Generate a unique tool use ID
            tool_use_id = f"tf_docs_search_{provider}_{asset_name}"
            
            # Call the tool
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name=search_tool_name,
                arguments={
                    "asset_name": asset_name,
                    "asset_type": asset_type
                }
            )
            
            # Process the result - extract content from the tool result
            if isinstance(result, dict) and "status" in result and result["status"] == 'error':
                error_message = "Unknown error"
                if isinstance(result, dict) and "content" in result and result["content"]:
                    for item in result["content"]:
                        if isinstance(item, dict) and 'text' in item:
                            error_message = item['text']
                            break
                
                logger.error(f"Error from MCP server: {error_message}")
                return {
                    "error": error_message,
                    "results": [],
                    "summary": ""
                }
            
            # Extract results from the tool response
            docs_results = []
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and 'text' in item:
                        try:
                            # Try to parse JSON from the text content
                            import json
                            parsed_results = json.loads(item['text'])
                            if isinstance(parsed_results, list):
                                docs_results.extend(parsed_results)
                            else:
                                docs_results.append(parsed_results)
                        except Exception as e:
                            logger.error(f"Error parsing MCP result: {e}")
                            # If not JSON, use the text as is
                            docs_results.append({"description": item['text']})
                
            # Format the results for the agent
            formatted_results = []
            for doc in docs_results:
                if not isinstance(doc, dict):
                    continue
                
                formatted_doc = {
                    "name": doc.get("name", "Unnamed Resource") if isinstance(doc, dict) else "Unnamed Resource",
                    "description": doc.get("description", "No description available") if isinstance(doc, dict) else str(doc),
                    "url": doc.get("url", "") if isinstance(doc, dict) else "",
                    "provider": provider,
                    "type": asset_type,
                    "arguments": [],
                    "attributes": []
                }
                
                # Add arguments if available
                if isinstance(doc, dict) and "arguments" in doc and isinstance(doc["arguments"], list):
                    formatted_doc["arguments"] = []
                    for arg in doc["arguments"]:
                        if isinstance(arg, dict):
                            formatted_doc["arguments"].append({
                                "name": arg.get("name", ""),
                                "description": arg.get("description", ""),
                                "required": arg.get("required", False)
                            })
                
                # Add attributes if available
                if isinstance(doc, dict) and "attributes" in doc and isinstance(doc["attributes"], list):
                    formatted_doc["attributes"] = []
                    for attr in doc["attributes"]:
                        if isinstance(attr, dict):
                            formatted_doc["attributes"].append({
                                "name": attr.get("name", ""),
                                "description": attr.get("description", "")
                })
                
                # Add examples if available
                if isinstance(doc, dict) and "examples" in doc and isinstance(doc["examples"], list):
                    formatted_doc["examples"] = doc["examples"]
                
                formatted_results.append(formatted_doc)
                
            # Generate a summary
            summary = f"Found {len(formatted_results)} Terraform documentation entries for {provider} {asset_type}: {asset_name}"
                
            return {
                "results": formatted_results,
                "summary": summary
            }
            
    except Exception as e:
        logger.error(f"Error using Terraform MCP server: {str(e)}")
        return {
            "error": f"Error using Terraform MCP server: {str(e)}",
            "results": [],
            "summary": ""
        }

@tool
def run_terraform_checkov_scan(
    working_directory: str,
    check_ids: Optional[List[str]] = None,
    skip_check_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run a Checkov security scan on Terraform files to identify potential security issues.
    
    This tool uses the Terraform MCP server to execute a Checkov security scan on the
    specified Terraform files. It's useful for identifying security and compliance issues
    in Terraform code as part of drift analysis and remediation.
    
    Args:
        working_directory: Directory containing Terraform files to scan
        check_ids: Optional list of specific check IDs to run
        skip_check_ids: Optional list of check IDs to skip
    
    Returns:
        Dict containing:
            - results: List of security findings
            - summary: Brief summary of the scan results
            - passed_checks: Number of passed checks
            - failed_checks: Number of failed checks
            - skipped_checks: Number of skipped checks
    """
    try:
        # Create MCP client
        mcp_client = get_terraform_mcp_client()
        
        logger.info(f"Running Checkov scan on Terraform files in: {working_directory}")
        
        # Execute the scan through the MCP server
        with mcp_client:
            # Generate a unique tool use ID
            tool_use_id = f"tf_checkov_scan_{os.path.basename(working_directory)}"
            
            # Call the RunCheckovScan tool
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name="RunCheckovScan",
                arguments={
                    "working_directory": working_directory,
                    "framework": "terraform",
                    "check_ids": check_ids,
                    "skip_check_ids": skip_check_ids,
                    "output_format": "json"
                }
            )
            
            # Process the result
            if isinstance(result, dict) and "status" in result and result["status"] == 'error':
                error_message = "Unknown error"
                if isinstance(result, dict) and "content" in result and result["content"]:
                    for item in result["content"]:
                        if isinstance(item, dict) and 'text' in item:
                            error_message = item['text']
                            break
                
                logger.error(f"Error from MCP server: {error_message}")
                return {
                    "error": error_message,
                    "results": [],
                    "summary": "",
                    "passed_checks": 0,
                    "failed_checks": 0,
                    "skipped_checks": 0
                }
            
            # Extract scan results
            scan_results = {}
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and 'text' in item:
                        try:
                            # Try to parse JSON from the text content
                            import json
                            scan_results = json.loads(item['text'])
                            break
                        except Exception as e:
                            logger.error(f"Error parsing Checkov scan results: {e}")
                            scan_results = {"results": item['text']}
            
            # Format the results for the agent
            formatted_results = []
            passed_checks = scan_results.get("passed_checks", 0)
            failed_checks = scan_results.get("failed_checks", 0)
            skipped_checks = scan_results.get("skipped_checks", 0)
            
            # Process failed checks
            if "failed_checks" in scan_results and isinstance(scan_results["failed_checks"], list):
                for check in scan_results["failed_checks"]:
                    if isinstance(check, dict):
                        formatted_results.append({
                            "id": check.get("check_id", "Unknown"),
                            "name": check.get("check_name", "Unknown Check"),
                            "file": check.get("file_path", "Unknown"),
                            "resource": check.get("resource", "Unknown"),
                            "guideline": check.get("guideline", ""),
                            "severity": check.get("severity", "UNKNOWN"),
                            "status": "FAILED"
                        })
            
            # Generate a summary
            summary = f"Checkov scan completed with {passed_checks} passed, {failed_checks} failed, and {skipped_checks} skipped checks"
            
            return {
                "results": formatted_results,
                "summary": summary,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "skipped_checks": skipped_checks
            }
            
    except Exception as e:
        logger.error(f"Error running Checkov scan: {str(e)}")
        return {
            "error": f"Error running Checkov scan: {str(e)}",
            "results": [],
            "summary": "",
            "passed_checks": 0,
            "failed_checks": 0,
            "skipped_checks": 0
        }

@tool
def get_terraform_best_practices() -> Dict[str, Any]:
    """
    Retrieve AWS Terraform best practices from the Terraform MCP server.
    
    This tool connects to the Terraform MCP server to retrieve best practices for
    working with Terraform on AWS. It's useful for understanding how to properly
    configure resources and avoid common issues that could lead to drift.
    
    Returns:
        Dict containing:
            - content: The best practices content
            - summary: Brief summary of the best practices
    """
    try:
        # Create MCP client
        mcp_client = get_terraform_mcp_client()
        
        logger.info("Retrieving Terraform best practices")
        
        # Execute the query through the MCP server
        with mcp_client:
            # Generate a unique tool use ID
            tool_use_id = "tf_best_practices"
            
            # Call the tool to get best practices
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name="GetTerraformBestPractices",
                arguments={}
            )
            
            # Process the result
            if isinstance(result, dict) and "status" in result and result["status"] == 'error':
                error_message = "Unknown error"
                if isinstance(result, dict) and "content" in result and result["content"]:
                    for item in result["content"]:
                        if isinstance(item, dict) and 'text' in item:
                            error_message = item['text']
                            break
                
                logger.error(f"Error from MCP server: {error_message}")
                return {
                    "error": error_message,
                    "content": "",
                    "summary": ""
                }
            
            # Extract content
            content = ""
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and 'text' in item:
                        content = item['text']
                        break
            
            return {
                "content": content,
                "summary": "Retrieved AWS Terraform best practices"
            }
            
    except Exception as e:
        logger.error(f"Error retrieving Terraform best practices: {str(e)}")
        return {
            "error": f"Error retrieving Terraform best practices: {str(e)}",
            "content": "",
            "summary": ""
        } 