#!/usr/bin/env python3
"""
AWS Documentation Tool for Strands Agent.

This tool connects to the AWS Documentation MCP server that provides access 
to AWS documentation for drift analysis and remediation.
"""

import os
import logging
import re
from typing import Dict, List, Optional, Any

from strands import tool
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

logger = logging.getLogger(__name__)

# MCP Server configuration using the PyPI package
AWS_MCP_SERVER_COMMAND = "uvx"
AWS_MCP_SERVER_ARGS = ["awslabs.aws-documentation-mcp-server@latest"]

def get_aws_docs_mcp_client() -> MCPClient:
    """
    Create and return an MCP client connected to the AWS documentation MCP server.
    """
    try:
        # Connect to the MCP server using stdio transport
        return MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command=AWS_MCP_SERVER_COMMAND, 
                args=AWS_MCP_SERVER_ARGS
            )
        ))
    except Exception as e:
        logger.error(f"Failed to create AWS docs MCP client: {e}")
        raise

@tool
def aws_documentation_search(
    service: str,
    resource_type: Optional[str] = None,
    topic: Optional[str] = None,
    api_action: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search AWS documentation for relevant information about AWS services and resources.
    
    This tool queries the AWS documentation MCP server to retrieve official AWS documentation
    about services, resources, APIs, and best practices. It's useful for understanding
    the expected configuration of resources, identifying drift issues, and determining
    the appropriate remediation steps.
    
    Args:
        service: AWS service name (e.g., 's3', 'ec2', 'iam')
        resource_type: Optional specific resource type (e.g., 'bucket', 'instance', 'role')
        topic: Optional topic to search for (e.g., 'security', 'permissions', 'configuration')
        api_action: Optional API action name (e.g., 'CreateBucket', 'RunInstances')
    
    Returns:
        Dict containing:
            - results: List of documentation entries
            - summary: Brief summary of the documentation
    """
    try:
        # Create MCP client
        mcp_client = get_aws_docs_mcp_client()
        
        # Prepare the search phrase
        search_phrase = f"{service}"
        if resource_type:
            search_phrase += f" {resource_type}"
        if topic:
            search_phrase += f" {topic}"
        if api_action:
            search_phrase += f" {api_action}"
            
        logger.info(f"Querying AWS documentation with: {search_phrase}")
        
        # Execute the query through the MCP server
        with mcp_client:
            # List available tools from the MCP server
            tools = mcp_client.list_tools_sync()
            tool_names = []
            for t in tools:
                if hasattr(t, 'tool_spec') and isinstance(t.tool_spec, dict) and 'name' in t.tool_spec:
                    tool_names.append(t.tool_spec['name'])
            logger.info(f"Available tools from AWS docs MCP server: {tool_names}")
            
            # Generate a unique tool use ID
            tool_use_id = f"aws_docs_search_{service}"
            
            # Call the search_documentation tool
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name="search_documentation",
                arguments={
                    "search_phrase": search_phrase,
                    "limit": 10
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
            
            # Extract search results from the tool response
            search_results = []
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and 'text' in item:
                        try:
                            # Try to parse JSON from the text content
                            import json
                            parsed_results = json.loads(item['text'])
                            if isinstance(parsed_results, list):
                                search_results.extend(parsed_results)
                            else:
                                search_results.append(parsed_results)
                        except Exception as e:
                            logger.error(f"Error parsing MCP result: {e}")
                            # If not JSON, use the text as is
                            search_results.append({"title": "Documentation", "context": item['text']})
            
            # Format the results
            formatted_results = []
            for doc in search_results:
                if not isinstance(doc, dict):
                    continue
                    
                # Extract result information
                title = doc.get("title", "Untitled")
                url = doc.get("url", "")
                context = doc.get("context", "No context available")
                
                # Create result entry
                result_entry = {
                    "title": title,
                    "url": url,
                    "excerpt": context[:500] if context else "No excerpt available",
                    "service": service,
                    "relevance_score": 1.0 / (doc.get("rank_order", 1) or 1)  # Convert rank to score (lower rank = higher score)
                }
                
                formatted_results.append(result_entry)
                
            # Generate a summary
            summary = f"Found {len(formatted_results)} AWS documentation entries for {service}"
            if resource_type:
                summary += f" {resource_type}"
            if topic:
                summary += f" related to {topic}"
                
            return {
                "results": formatted_results,
                "summary": summary
            }
            
    except Exception as e:
        logger.error(f"Error using AWS documentation MCP server: {str(e)}")
        return {
            "error": f"Error using AWS documentation MCP server: {str(e)}",
            "results": [],
            "summary": ""
        }

@tool
def aws_documentation_read(
    url: str
) -> Dict[str, Any]:
    """
    Read and extract content from a specific AWS documentation page.
    
    This tool connects to the AWS documentation MCP server to retrieve and parse
    the content of a specific AWS documentation page. It's useful for getting
    detailed information about AWS services, resources, and configurations.
    
    Args:
        url: URL of the AWS documentation page to read (must be from docs.aws.amazon.com)
    
    Returns:
        Dict containing:
            - content: The extracted content from the documentation page
            - title: The title of the documentation page
            - url: The URL of the documentation page
            - summary: Brief summary of the documentation
    """
    try:
        # Validate URL
        if not re.match(r'^https?://docs\.aws\.amazon\.com/', url):
            return {
                "error": "Invalid URL. URL must be from the docs.aws.amazon.com domain",
                "content": "",
                "title": "",
                "url": url,
                "summary": ""
            }
            
        # Create MCP client
        mcp_client = get_aws_docs_mcp_client()
        
        logger.info(f"Reading AWS documentation from: {url}")
        
        # Execute the query through the MCP server
        with mcp_client:
            # Generate a unique tool use ID
            tool_use_id = f"aws_docs_read_{url.split('/')[-1]}"
            
            # Call the read_documentation tool
            result = mcp_client.call_tool_sync(
                tool_use_id=tool_use_id,
                name="read_documentation",
                arguments={
                    "url": url,
                    "max_length": 10000,
                    "start_index": 0
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
                    "content": "",
                    "title": "",
                    "url": url,
                    "summary": ""
                }
            
            # Extract content from the tool response
            content = ""
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and 'text' in item:
                        content = item['text']
                        break
            
            # Extract title from content (usually the first heading)
            title = url.split('/')[-1].replace('.html', '').replace('-', ' ').title()
            if content:
                # Try to find a markdown heading at the beginning
                match = re.search(r'^#\s+(.+?)$', content, re.MULTILINE)
                if match:
                    title = match.group(1)
            
            return {
                "content": content,
                "title": title,
                "url": url,
                "summary": f"Retrieved AWS documentation from {url}"
            }
            
    except Exception as e:
        logger.error(f"Error reading AWS documentation: {str(e)}")
        return {
            "error": f"Error reading AWS documentation: {str(e)}",
            "content": "",
            "title": "",
            "url": url,
            "summary": ""
        } 