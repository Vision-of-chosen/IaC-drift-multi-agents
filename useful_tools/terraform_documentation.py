#!/usr/bin/env python3
"""
Terraform Documentation MCP Server Tool for Strands Agent.

This tool connects to an MCP server that provides access to Terraform documentation
for drift analysis and remediation.
"""

import os
import logging
from typing import Dict, List, Optional, Any

from strands import tool
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

logger = logging.getLogger(__name__)

# MCP Server configuration
MCP_SERVER_COMMAND = os.environ.get("TF_MCP_SERVER_COMMAND", "uvx")
MCP_SERVER_ARGS = os.environ.get(
    "TF_MCP_SERVER_ARGS", 
    "awslabs.terraform-documentation-mcp-server@latest"
).split()

def get_terraform_docs_mcp_client() -> MCPClient:
    """
    Create and return an MCP client connected to the Terraform documentation server.
    """
    try:
        # Connect to the MCP server using stdio transport
        return MCPClient(lambda: stdio_client(
            StdioServerParameters(command=MCP_SERVER_COMMAND, args=MCP_SERVER_ARGS)
        ))
    except Exception as e:
        logger.error(f"Failed to create Terraform docs MCP client: {e}")
        raise

@tool
def terraform_documentation_search(
    provider: str,
    resource_type: Optional[str] = None,
    topic: Optional[str] = None,
    attribute: Optional[str] = None,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Search Terraform documentation for provider resources and configuration options.
    
    This tool queries the Terraform documentation MCP server to retrieve official Terraform
    documentation about providers, resources, and their configuration attributes. It's useful
    for understanding the correct resource definitions, identifying drift issues, and
    determining the appropriate remediation steps.
    
    Args:
        provider: Terraform provider name (e.g., 'aws', 'azure', 'google')
        resource_type: Optional specific resource type (e.g., 'aws_s3_bucket', 'aws_instance')
        topic: Optional topic to search for (e.g., 'security', 'permissions')
        attribute: Optional specific attribute to search for (e.g., 'acl', 'instance_type')
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Dict containing:
            - results: List of documentation entries
            - summary: Brief summary of the documentation
    """
    try:
        # Create MCP client
        mcp_client = get_terraform_docs_mcp_client()
        
        # Prepare the query
        query = f"{provider}"
        if resource_type:
            query += f" {resource_type}"
        if attribute:
            query += f" {attribute}"
        if topic:
            query += f" {topic}"
            
        logger.info(f"Querying Terraform documentation with: {query}")
        
        # Execute the query through the MCP server
        with mcp_client:
            # List available tools from the MCP server
            tools = mcp_client.list_tools_sync()
            tool_names = []
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_names.append(t.tool_spec['name'])
            logger.info(f"Available tools from Terraform docs MCP server: {tool_names}")
            
            # Find the search tool
            search_tool = None
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_name = t.tool_spec['name']
                    if tool_name == "search" or "search" in tool_name.lower() or "documentation" in tool_name.lower():
                        search_tool = t
                        break
                    
            if not search_tool:
                logger.error("No search tool found in Terraform documentation MCP server")
                return {
                    "error": "No search tool found in Terraform documentation MCP server",
                    "results": [],
                    "summary": ""
                }
                
            # Execute the search
            logger.info(f"Using tool: {search_tool.tool_spec['name']}")
            
            # Generate a unique tool use ID
            tool_use_id = f"tf_docs_search_{provider}"
            
            # Call the tool using call_tool_sync instead of execute_tool_sync
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name=search_tool.tool_spec['name'],
                arguments={"query": query, "max_results": max_results}
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
            results_data = {}
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and 'text' in item:
                        try:
                            # Try to parse JSON from the text content
                            import json
                            results_data = json.loads(item['text'])
                            break
                        except:
                            # If not JSON, use the text as is
                            results_data = {"results": [{"title": "Documentation", "excerpt": item['text']}]}
                            break
                
            # Format the results
            formatted_results = []
            for doc in results_data.get("results", []):
                formatted_results.append({
                    "title": doc.get("title", "Untitled"),
                    "url": doc.get("url", ""),
                    "excerpt": doc.get("excerpt", "No excerpt available")[:500],
                    "provider": provider,
                    "resourceType": resource_type,
                    "relevance_score": doc.get("score", 0)
                })
                
            # Generate a summary
            summary = f"Found {len(formatted_results)} Terraform documentation entries for {provider}"
            if resource_type:
                summary += f" {resource_type}"
            if attribute:
                summary += f" attribute '{attribute}'"
                
            return {
                "results": formatted_results,
                "summary": summary
            }
            
    except Exception as e:
        logger.error(f"Error using Terraform documentation MCP server: {str(e)}")
        return {
            "error": f"Error using Terraform documentation MCP server: {str(e)}",
            "results": [],
            "summary": ""
        } 