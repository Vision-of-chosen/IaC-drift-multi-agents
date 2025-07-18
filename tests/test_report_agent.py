#!/usr/bin/env python3
"""
Test script to verify the ReportAgent functionality.
This script creates a sample drift detection result and uses the ReportAgent to generate a report.
"""

import os
import sys
import json
from datetime import datetime

# Add the parent directory to the path to be able to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared_memory import shared_memory
from agents.report_agent import ReportAgent
from strands.models.bedrock import BedrockModel
from config import BEDROCK_REGION, BEDROCK_MODEL_ID

def create_sample_drift_data():
    """Create sample drift detection and analysis data for testing"""
    # Sample drift detection results
    drift_detection_results = {
        "drift_detection_results": [
            {
                "resource_type": "aws_s3_bucket",
                "resource_id": "aws_s3_bucket.app-storage-bucket",
                "drift_type": "changed",
                "expected_config": {
                    "versioning": "Disabled",
                    "encryption": "None",
                    "public_access_block": "false"
                },
                "actual_config": {
                    "versioning": "Enabled",
                    "encryption": "AES256",
                    "public_access_block": "true"
                },
                "drift_details": {
                    "versioning": {
                        "expected": "Disabled",
                        "actual": "Enabled"
                    },
                    "encryption": {
                        "expected": "None",
                        "actual": "AES256"
                    },
                    "public_access_block": {
                        "expected": "false",
                        "actual": "true"
                    }
                },
                "severity": "high",
                "source": "Modified by AWS Console",
                "recommended_action": "Update Terraform code to reflect new configuration",
                "dependencies": []
            }
        ],
        "summary": {
            "total_resources": 45,
            "total_drifts": 1,
            "new": 0,
            "changed": 1,
            "deleted": 0,
            "unsupported": 0,
            "critical_issues": 1
        }
    }
    
    # Sample drift analysis results
    drift_analysis_results = {
        "drift_analysis_results": [
            {
                "resource_type": "aws_s3_bucket",
                "resource_id": "aws_s3_bucket.app-storage-bucket",
                "severity": "high",
                "impact_assessment": "Detected a critical security configuration change on an S3 bucket. Versioning and encryption enabled is positive, but needs to be checked to see if the change was intentional.",
                "remediation_steps": [
                    "Confirm with DevOps team about this change",
                    "Update Terraform code to reflect new configuration",
                    "Run terraform plan to synchronize state"
                ]
            }
        ]
    }
    
    return drift_detection_results, drift_analysis_results

def main():
    print("🧪 Testing ReportAgent functionality...")
    
    # Create sample data
    drift_detection_results, drift_analysis_results = create_sample_drift_data()
    
    # Store in shared memory
    shared_memory.set("drift_detection_results", drift_detection_results)
    shared_memory.set("drift_analysis_results", drift_analysis_results)
    shared_memory.set("terraform_filename", "production-infrastructure.tfplan")
    
    # Create Bedrock model
    model = BedrockModel(
        model_id=BEDROCK_MODEL_ID,
        region_name=BEDROCK_REGION,
    )
    
    # Create ReportAgent
    report_agent = ReportAgent(model)
    
    # Generate report
    print("📊 Generating report...")
    report = report_agent.generate_json_report("report.json")
    
    # Print report summary
    print("\n📝 Report Summary:")
    print("-" * 50)
    scan_details = report.get("scanDetails", {})
    print(f"Scan ID: {scan_details.get('id')}")
    print(f"File: {scan_details.get('fileName')}")
    print(f"Date: {scan_details.get('scanDate')}")
    print(f"Status: {scan_details.get('status')}")
    print(f"Total Resources: {scan_details.get('totalResources')}")
    print(f"Drift Count: {scan_details.get('driftCount')}")
    print(f"Risk Level: {scan_details.get('riskLevel')}")
    print(f"Total Drifts: {len(report.get('drifts', []))}")
    
    print("\n✅ Report saved to report.json")
    
    # Display the actual JSON report
    print("\n📋 JSON Report Content:")
    print("-" * 50)
    print(json.dumps(report, indent=2))
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 