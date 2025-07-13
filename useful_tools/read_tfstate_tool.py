"""
Terraform State File Reader Tool for Strands Agent.

This tool reads Terraform state files (.tfstate) and parses their content into Python dictionaries.
It's designed to work with the IaC drift detection system to analyze infrastructure drift.
"""

import json
import os
import logging
from typing import Any, Dict, cast

from strands.types.tools import ToolResult, ToolUse, ToolResultContent
from strands.tools.decorator import tool
from strands_tools.utils import console_util

# Configure logging
logger = logging.getLogger(__name__)

# Note: Using @tool decorator automatically extracts the tool specification
# from the function signature, docstring, and type hints.

@tool
def read_tfstate(file_path: str = "") -> dict:
    """
    Read and parse Terraform state files (.tfstate) into a structured Python dictionary.

    Features:
    1. Automatic location of .tfstate files
    2. Full parsing of state file structure
    3. Resource extraction and analysis
    4. Complete Terraform resource inventory

    This tool is essential for drift detection, as it provides the expected infrastructure state.
    
    How It Works:
    ------------
    1. Looks for the Terraform state file in the provided path or default locations
    2. Reads the file and parses its JSON content
    3. Validates that the content matches expected Terraform state format
    4. Returns the parsed data as a structured dictionary
    
    Args:
        file_path: Optional path to the Terraform state file.
            If not provided, will look in default locations.
        
    Returns:
        dict: A dictionary containing:
            - tfstate_data: The complete parsed Terraform state
            - summary: Basic information about the state file
            - message: Success message
        
        If an error occurs, returns a dictionary with an "error" key.
    """
    console = console_util.create()
    logger.info(f"read_tfstate tool called with file_path={file_path}")
    
    try:
        # If no file path is provided, try default locations
        if not file_path or file_path.strip() == "":
            potential_paths = [
                os.path.join(".", "terraform", "terraform.tfstate"),  # Relative path
                os.path.join(".", "terraform.tfstate"),               # Root directory
                os.path.abspath(os.path.join(".", "terraform", "terraform.tfstate"))  # Absolute path
            ]
            
            logger.info(f"No file_path provided, trying default paths: {potential_paths}")
            for path in potential_paths:
                if os.path.exists(path):
                    file_path = path
                    logger.info(f"Found state file at: {file_path}")
                    break
            
            if not file_path:
                logger.error("Could not find terraform.tfstate file in default locations")
                return {
                    "error": "Could not find terraform.tfstate file in default locations. Please specify a file path."
                }
        
        # Check if the file exists
        if not os.path.exists(file_path):
            logger.error(f"Terraform state file not found at: {file_path}")
            return {
                "error": f"Terraform state file not found at: {file_path}"
            }
        
        # Read and parse the Terraform state file
        with open(file_path, 'r') as f:
            tfstate_data = json.load(f)
        
        # Basic validation that this is a Terraform state file
        if not isinstance(tfstate_data, dict) or 'version' not in tfstate_data or 'resources' not in tfstate_data:
            logger.error(f"The file at {file_path} does not appear to be a valid Terraform state file")
            return {
                "error": f"The file at {file_path} does not appear to be a valid Terraform state file."
            }
        
        # Get summary info
        resource_count = len(tfstate_data.get('resources', []))
        terraform_version = tfstate_data.get('terraform_version', 'unknown')
        
        logger.info(f"Successfully read Terraform state file with {resource_count} resources, version {terraform_version}")
        # Return the parsed data along with summary info
        return {
            "tfstate_data": tfstate_data,
            "summary": {
                "file_path": file_path,
                "terraform_version": terraform_version,
                "resource_count": resource_count
            },
            "message": f"Successfully read Terraform state file from: {file_path}"
        }
        
    except json.JSONDecodeError:
        logger.error(f"The file at {file_path} is not valid JSON")
        return {
            "error": f"The file at {file_path} is not valid JSON."
        }
    except Exception as e:
        logger.error(f"Error reading Terraform state file: {str(e)}")
        return {
            "error": f"Error reading Terraform state file: {str(e)}"
        }