#!/usr/bin/env python3
"""
Demo script to create intentional infrastructure drift for testing the 
Terraform Drift Detection & Remediation System.

This script makes manual changes to AWS resources that were created by Terraform,
creating drift that can be detected and remediated by the system.

⚠️  WARNING: This script makes actual changes to AWS resources.
    Only run this in test/development environments!
"""

import boto3
import json
import time
import sys
from typing import Dict, Any, Optional

def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🔧 {text}")
    print("="*60)

def print_status(text: str):
    """Print status message"""
    print(f"📋 {text}")

def print_success(text: str):
    """Print success message"""
    print(f"✅ {text}")

def print_warning(text: str):
    """Print warning message"""
    print(f"⚠️  {text}")

def print_error(text: str):
    """Print error message"""
    print(f"❌ {text}")

def get_terraform_outputs() -> Dict[str, Any]:
    """Get outputs from Terraform state"""
    import subprocess
    import os
    
    if not os.path.exists("terraform"):
        print_error("Terraform directory not found!")
        return {}
    
    try:
        # Change to terraform directory and get outputs
        result = subprocess.run(
            ["terraform", "output", "-json"],
            cwd="terraform",
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to get Terraform outputs: {e}")
        return {}
    except json.JSONDecodeError as e:
        print_error(f"Failed to parse Terraform outputs: {e}")
        return {}

def create_s3_drift(bucket_name: str, region: str = "ap-southeast-2"):
    """Create drift in S3 bucket configuration"""
    print_status(f"Creating S3 drift for bucket: {bucket_name}")
    
    try:
        s3_client = boto3.client('s3', region_name=region)
        
        # Add tags that weren't in Terraform
        s3_client.put_bucket_tagging(
            Bucket=bucket_name,
            Tagging={
                'TagSet': [
                    {'Key': 'DriftTest', 'Value': 'true'},
                    {'Key': 'ModifiedOutsideTerraform', 'Value': 'yes'},
                    {'Key': 'Owner', 'Value': 'manual-change'}
                ]
            }
        )
        print_success("Added manual tags to S3 bucket")
        
        # Enable static website hosting (if not already configured in Terraform)
        try:
            s3_client.put_bucket_website(
                Bucket=bucket_name,
                WebsiteConfiguration={
                    'IndexDocument': {'Suffix': 'index.html'},
                    'ErrorDocument': {'Key': 'error.html'}
                }
            )
            print_success("Enabled static website hosting")
        except Exception as e:
            print_warning(f"Could not enable website hosting: {e}")
            
    except Exception as e:
        print_error(f"Failed to create S3 drift: {e}")

def create_security_group_drift(sg_id: str, region: str = "ap-southeast-2"):
    """Create drift in security group configuration"""
    print_status(f"Creating Security Group drift for: {sg_id}")
    
    try:
        ec2_client = boto3.client('ec2', region_name=region)
        
        # Add a new ingress rule that wasn't in Terraform
        ec2_client.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 443,
                    'ToPort': 443,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'Manual HTTPS access'}]
                }
            ]
        )
        print_success("Added manual HTTPS rule to security group")
        
        # Modify the description of existing SSH rule (this creates drift)
        # Note: This might fail if the rule doesn't exist exactly as expected
        try:
            # Get current rules first
            response = ec2_client.describe_security_groups(GroupIds=[sg_id])
            sg = response['SecurityGroups'][0]
            
            # Find SSH rule and modify it
            for rule in sg['IpPermissions']:
                if rule.get('FromPort') == 22 and rule.get('ToPort') == 22:
                    print_success("Found SSH rule - drift detection will notice description differences")
                    break
        except Exception as e:
            print_warning(f"Could not modify SSH rule: {e}")
            
    except Exception as e:
        print_error(f"Failed to create Security Group drift: {e}")

def create_iam_drift(role_name: str, region: str = "ap-southeast-2"):
    """Create drift in IAM role configuration"""
    print_status(f"Creating IAM drift for role: {role_name}")
    
    try:
        iam_client = boto3.client('iam', region_name=region)
        
        # Add an additional inline policy
        additional_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": "arn:aws:logs:*:*:*"
                }
            ]
        }
        
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName='ManuallyAddedLoggingPolicy',
            PolicyDocument=json.dumps(additional_policy)
        )
        print_success("Added manual inline policy to IAM role")
        
        # Add tags to the role
        iam_client.tag_role(
            RoleName=role_name,
            Tags=[
                {'Key': 'DriftTest', 'Value': 'true'},
                {'Key': 'ManuallyAdded', 'Value': 'yes'}
            ]
        )
        print_success("Added manual tags to IAM role")
        
    except Exception as e:
        print_error(f"Failed to create IAM drift: {e}")

def create_cloudwatch_drift(log_group_name: str, region: str = "ap-southeast-2"):
    """Create drift in CloudWatch log group configuration"""
    print_status(f"Creating CloudWatch drift for log group: {log_group_name}")
    
    try:
        cloudwatch_client = boto3.client('logs', region_name=region)
        
        # Change retention period (if different from Terraform)
        cloudwatch_client.put_retention_policy(
            logGroupName=log_group_name,
            retentionInDays=14  # Different from Terraform's 7 days
        )
        print_success("Changed log retention period manually")
        
        # Add tags to log group
        cloudwatch_client.tag_log_group(
            logGroupName=log_group_name,
            tags={
                'DriftTest': 'true',
                'ManualChange': 'retention-modified'
            }
        )
        print_success("Added manual tags to log group")
        
    except Exception as e:
        print_error(f"Failed to create CloudWatch drift: {e}")

def main():
    """Main function to create intentional drift"""
    print_header("Terraform Drift Creation Demo")
    
    print_warning("This script will create intentional drift in your AWS infrastructure!")
    print_warning("Only run this in test/development environments.")
    
    response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Aborted.")
        sys.exit(0)
    
    # Get Terraform outputs
    print_status("Getting Terraform outputs...")
    outputs = get_terraform_outputs()
    
    if not outputs:
        print_error("Could not get Terraform outputs. Make sure you've applied your Terraform configuration.")
        sys.exit(1)
    
    # Extract resource identifiers
    try:
        bucket_name = outputs.get('s3_bucket_name', {}).get('value')
        sg_id = outputs.get('security_group_id', {}).get('value')
        role_name = outputs.get('iam_role_name', {}).get('value')
        log_group_name = outputs.get('cloudwatch_log_group_name', {}).get('value')
        
        print_success(f"Found resources:")
        print(f"  S3 Bucket: {bucket_name}")
        print(f"  Security Group: {sg_id}")
        print(f"  IAM Role: {role_name}")
        print(f"  Log Group: {log_group_name}")
        
    except Exception as e:
        print_error(f"Failed to parse Terraform outputs: {e}")
        sys.exit(1)
    
    # Create drift in each resource type
    if bucket_name:
        create_s3_drift(bucket_name)
        time.sleep(2)
    
    if sg_id:
        create_security_group_drift(sg_id)
        time.sleep(2)
    
    if role_name:
        create_iam_drift(role_name)
        time.sleep(2)
    
    if log_group_name:
        create_cloudwatch_drift(log_group_name)
        time.sleep(2)
    
    print_header("Drift Creation Complete!")
    print_success("Infrastructure drift has been created successfully.")
    print("\nNext steps:")
    print("1. Run the Terraform drift detection system:")
    print("   python terraform_drift_system.py")
    print("\n2. Use the 'detect' command to find the drift")
    print("3. Use the 'analyze' command to assess the impact")
    print("4. Use the 'remediate' command to fix the drift")
    print("\n⚠️  The drift detection system should now find differences between")
    print("   your Terraform state and the actual AWS resources!")

if __name__ == "__main__":
    main() 