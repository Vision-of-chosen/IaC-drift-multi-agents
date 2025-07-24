#!/usr/bin/env python3
"""
Report Agent for the Terraform Drift Detection & Remediation System.

This agent specializes in generating structured reports from drift analysis data.
"""

import os
import sys
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("tools/src")

logger = logging.getLogger(__name__)

from strands import Agent
from strands.agent.state import AgentState
from strands.models.bedrock import BedrockModel
from strands_tools import file_read, file_write, journal, calculator

from prompts import AgentPrompts
from shared_memory import shared_memory
from config import BEDROCK_REGION
from permission_handlers import create_agent_callback_handler

class ReportAgent:
    """Specialist in generating structured reports from drift analysis results"""
    
    def __init__(self, model: BedrockModel):
        self.model = model
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create the report agent instance"""
        agent = Agent(
            model=self.model,
            system_prompt=AgentPrompts.get_prompt("report"),
            name="ReportAgent",
            description="Specialist in generating structured reports from drift analysis",
            # callback_handler=create_agent_callback_handler("ReportAgent"),
            tools=[
                file_read,      # Read configuration and previous reports
                file_write,     # Write JSON report to file
                journal,     # Create structured reports
                calculator      # Calculate metrics for the report
            ]
        )
        
        # Set state after creating agent
        agent.state = AgentState()
        agent.state.shared_memory = shared_memory.data
        agent.state.agent_type = "report"
        
        return agent
    
    def get_agent(self) -> Agent:
        """Get the agent instance"""
        return self.agent
    
    def update_shared_memory(self) -> None:
        """Update agent state with current shared memory"""
        if hasattr(self.agent, 'state'):
            self.agent.state.shared_memory = shared_memory.data
        else:
            self.agent.state = AgentState({
                "shared_memory": shared_memory.data,
                "agent_type": "report"
            })
    
    def generate_json_report(self, output_file: str = "report.json") -> dict:
        """Generate a structured JSON report from drift analysis results and save to file
        
        Args:
            output_file: The filename to save the JSON report (default: report.json)
            
        Returns:
            The generated report as a dictionary
        """
        self.update_shared_memory()
        
        # Get data from shared memory
        drift_detection_results = shared_memory.get("drift_detection_results", {})
        drift_analysis_results = shared_memory.get("drift_analysis_results", {})
        
        if not drift_detection_results:
            logger.warning("No drift detection results found in shared memory")
            report = self._create_empty_report()
        else:
            # Generate the report based on the data
            report = self._generate_structured_report(drift_detection_results, drift_analysis_results)
        
        # Save report to file
        try:
            self.agent.tool.file_write(
                path=output_file,
                content=json.dumps(report, ensure_ascii=False, indent=2)
            )
            logger.info(f"JSON report saved to {output_file}")
            
            # Store report in shared memory for access by other agents
            shared_memory.set("drift_json_report", report)
            shared_memory.set("drift_report_file", output_file)
            
            # Update agent status
            self.update_agent_status({
                "action": "generate_json_report",
                "completion_time": datetime.now().isoformat(),
                "status": "completed",
                "report_file": output_file
            })
        except Exception as e:
            logger.error(f"Error writing report to {output_file}: {e}")
            self.update_agent_status({
                "action": "generate_json_report",
                "completion_time": datetime.now().isoformat(),
                "status": "error",
                "error_message": str(e)
            })
        
        return report
    
    def _generate_structured_report(self, detection_results: Dict[str, Any], 
                                  analysis_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a structured report based on detection and analysis results"""
        # Extract summary information
        summary = detection_results.get("summary", {})
        total_resources = summary.get("total_resources", 0)
        drift_count = sum(summary.get(k, 0) for k in ["new", "changed", "deleted", "unsupported"])
        
        # Generate scan ID if not already in shared memory
        scan_id = shared_memory.get("scan_id", f"scan-{uuid.uuid4().hex[:6]}")
        if "scan_id" not in shared_memory.data:
            shared_memory.set("scan_id", scan_id)
        
        # Determine risk level
        risk_level = self._calculate_risk_level(summary.get("critical_issues", 0), drift_count)
        
        # Get current time for creation timestamp
        current_time = datetime.now().isoformat() + "Z"
        start_time = datetime.now()
        
        # Get user info from shared memory, default to system
        user_id = shared_memory.get("current_user_id", "system")
        
        # Create report structure directly at root level (not in scanDetails)
        report = {
            "id": scan_id,
            "fileName": shared_memory.get("terraform_filename", "terraform-plan"),
            "scanDate": current_time,
            "status": "completed",
            "totalResources": total_resources,
            "driftCount": drift_count,
            "riskLevel": risk_level,
            "duration": "00:00:00",  # Will be updated at the end
            "createdBy": user_id,
            "createdOn": current_time,
            "modifiedBy": user_id,
            "drifts": self._format_drift_items(detection_results.get("drift_detection_results", []),
                                             analysis_results)
        }
        
        # Calculate actual duration in HH:MM:SS format
        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()
        hours, remainder = divmod(int(duration_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        report["duration"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        return report
    
    def _calculate_risk_level(self, critical_issues: int, drift_count: int) -> str:
        """Calculate overall risk level based on critical issues and drift count"""
        if critical_issues > 0:
            return "high"
        elif drift_count > 10:
            return "high"
        elif drift_count > 5:
            return "medium"
        elif drift_count > 0:
            return "low"
        return "none"
    
    def _format_drift_items(self, drift_items: List[Dict[str, Any]],
                           analysis_results: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Format drift items into the required structure for the report"""
        formatted_drifts = []
        
        # Get analyzed drifts for lookup if available
        analyzed_drifts = {}
        if analysis_results and "drift_analysis_results" in analysis_results:
            for item in analysis_results["drift_analysis_results"]:
                resource_id = item.get("resource_id")
                if resource_id:
                    analyzed_drifts[resource_id] = item
        
        # Process each drift item
        for idx, item in enumerate(drift_items):
            resource_id = item.get("resource_id", f"unknown-{idx}")
            resource_type = item.get("resource_type", "unknown")
            
            # Extract resource name from ID
            resource_name = self._extract_resource_name(resource_id, resource_type)
            
            # Find the corresponding analysis if available
            analysis = analyzed_drifts.get(resource_id, {})
            
            # Create drift entry with unique ID
            drift_id = f"drift-{uuid.uuid4().hex[:6]}"
            
            # Convert state to JSON strings
            before_state = self._extract_config_state(item.get("expected_config", {}))
            after_state = self._extract_config_state(item.get("actual_config", {}))
            
            drift_entry = {
                "driftCode": drift_id,
                "resourceType": resource_type,
                "resourceName": resource_name,
                "riskLevel": analysis.get("severity", item.get("severity", "medium")).lower(),
                "beforeStateJson": json.dumps(before_state),
                "afterStateJson": json.dumps(after_state),
                "aiExplanation": analysis.get("impact_assessment", 
                                           f"Detected changes in {resource_type} configuration."),
                "aiAction": self._format_remediation(
                    analysis.get("remediation_steps", []),
                    item.get("recommended_action", "")
                )
            }
            
            formatted_drifts.append(drift_entry)
            
        return formatted_drifts
    
    def _extract_resource_name(self, resource_id: str, resource_type: str) -> str:
        """Extract a human-readable resource name from the resource ID"""
        # Handle common AWS resource ID formats
        if '/' in resource_id:
            return resource_id.split('/')[-1]
        elif ':' in resource_id:
            parts = resource_id.split(':')
            return parts[-1].split('/')[-1]
        elif '.' in resource_id and resource_type in resource_id:
            # Handle terraform style IDs like aws_s3_bucket.my_bucket
            return resource_id.split('.')[-1]
        return resource_id
    
    def _extract_config_state(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the most relevant configuration details for the report"""
        # Extract the most important configuration details
        # This is a simplified version - expand based on resource types
        relevant_config = {}
        
        # Common security-related attributes to check for
        security_attrs = [
            "encryption", "versioning", "public_access_block", "access",
            "security_groups", "iam_instance_profile", "policy", "kms_key_id"
        ]
        
        for attr in security_attrs:
            if attr in config:
                relevant_config[attr] = config[attr]
        
        # Include any non-security attributes if we don't have much yet
        if len(relevant_config) < 3:
            for key, value in config.items():
                # Skip complex nested structures unless they're important
                if not isinstance(value, (dict, list)) or key in ["tags", "attributes"]:
                    if isinstance(value, dict) and len(value) > 5:
                        # For large dicts, just indicate presence
                        relevant_config[key] = f"{len(value)} key-value pairs"
                    else:
                        relevant_config[key] = value
                if len(relevant_config) >= 5:  # Limit to 5 attributes for readability
                    break
        
        return relevant_config
    
    def _format_remediation(self, steps: List[str], fallback: str) -> str:
        """Format remediation steps into a numbered list"""
        if not steps and fallback:
            steps = [fallback]
        
        if not steps:
            return "1. Review the detected changes\n2. Update Terraform code to reflect the changes\n3. Run terraform plan to verify changes"
        
        formatted = ""
        for i, step in enumerate(steps, 1):
            formatted += f"{i}. {step}\n"
        
        return formatted.rstrip()
    
    def _create_empty_report(self) -> Dict[str, Any]:
        """Create an empty report when no drift data is available"""
        scan_id = shared_memory.get("scan_id", f"scan-{uuid.uuid4().hex[:6]}")
        if "scan_id" not in shared_memory.data:
            shared_memory.set("scan_id", scan_id)

        # Get current timestamp for report
        current_time = datetime.now().isoformat() + "Z"
            
        # Get information from shared memory 
        user_request = shared_memory.get("user_request", "")
        workflow_status = shared_memory.get("workflow_status", "completed")
        user_id = shared_memory.get("current_user_id", "system")

        # Create report structure
        report = {
            "id": scan_id,
            "fileName": shared_memory.get("terraform_filename", "terraform-plan"),
            "scanDate": current_time,
            "status": workflow_status,
            "totalResources": 0,
            "driftCount": 0,
            "riskLevel": "none",
            "duration": "00:00:00",
            "createdBy": user_id,
            "createdOn": current_time,
            "modifiedBy": user_id,
            "drifts": [],
            "systemInfo": {
                "lastRequest": user_request,
                "reportGeneratedAt": current_time,
                "message": "No drift detection results available. Run drift detection first to get detailed results."
            }
        }
        return report
        
    def update_agent_status(self, status_info):
        """Update agent status in shared memory"""
        agent_type = self.agent.state.agent_type
        status_key = f"{agent_type}_status"
        
        status_data = {
            "status": status_info,
            "timestamp": datetime.now().isoformat(),
            "agent": agent_type
        }
        
        shared_memory.set(status_key, status_data) 