#!/usr/bin/env python3
"""
Notification Agent for the Terraform Drift Detection & Remediation System.

This agent specializes in monitoring AWS infrastructure changes and sending
email notifications to alert about potential drift using AWS-native services
like EventBridge, SNS, and AWS Config for improved cost-effectiveness.
"""

import sys
import os
import json
import logging
import boto3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from botocore.exceptions import ClientError

# Add root project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add tools directory to path
sys.path.append("tools/src")
# Add useful_tools directory to path
useful_tools_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "useful_tools")
sys.path.append(useful_tools_path)

logger = logging.getLogger(__name__)

from strands import Agent
from strands.agent.state import AgentState
from strands.models.bedrock import BedrockModel
from strands_tools import use_aws
from useful_tools import cloudtrail_logs
from useful_tools import cloudwatch_logs
from shared_memory import shared_memory
from config import BEDROCK_REGION

class NotificationAgent:
    """
    Agent responsible for monitoring AWS infrastructure changes and sending email notifications
    about potential drift to specified recipients using AWS-native services.
    """
    
    def __init__(self, model: BedrockModel):
        """Initialize the notification agent with a language model."""
        self.model = model
        self.agent = self._create_agent()
        self.recipient_email = "nguyenxuanlamnghean123@gmail.com"  # Default recipient
        self.aws_region = os.environ.get("AWS_REGION", "ap-southeast-2")
        self.sns_topic_arn = None
        self.eventbridge_rule_name = "DriftDetectionRule"
        self.config_recorder_name = "DriftConfigRecorder"
        
    def _create_agent(self) -> Agent:
        """Create and configure the notification agent."""
        agent = Agent(
            name="NotificationAgent",
            model=self.model,
            description="Monitors AWS infrastructure changes and sends email notifications about potential drift using AWS-native services",
            tools=[
                use_aws.use_aws,
                self._set_shared_memory_wrapper,
                self._setup_eventbridge_rule,
                self._setup_sns_topic,
                self._setup_aws_config,
                self._send_test_notification,
                self._check_notification_status
            ],
            state=AgentState({
                "shared_memory": shared_memory.data,
                "agent_type": "notification",
                "last_notification_time": datetime.now().isoformat()
            })
        )
        return agent
    
    def get_agent(self) -> Agent:
        """Get the configured agent instance."""
        return self.agent
    
    def update_shared_memory(self) -> None:
        """Update agent state with current shared memory."""
        if hasattr(self.agent, 'state'):
            self.agent.state.shared_memory = shared_memory.data
        else:
            self.agent.state = AgentState({
                "shared_memory": shared_memory.data,
                "agent_type": "notification",
                "last_notification_time": datetime.now().isoformat()
            })
    
    def update_agent_status(self, status_info):
        """Update agent status in shared memory."""
        agent_type = self.agent.state.agent_type
        status_key = f"{agent_type}_status"
        
        status_data = {
            "status": status_info,
            "timestamp": datetime.now().isoformat(),
            "agent": agent_type
        }
        
        shared_memory.set(status_key, status_data)
    
    def _set_shared_memory_wrapper(self, key: str, value) -> dict:
        """Wrapper for setting values in shared memory."""
        shared_memory.set(key, value)
        return {"status": "success", "message": f"Value set for key: {key}"}
    
    def _setup_sns_topic(self, topic_name: str = "DriftNotificationTopic") -> Dict[str, Any]:
        """
        Set up an SNS topic for drift notifications and subscribe the recipient email.
        
        Args:
            topic_name: Name of the SNS topic to create
            
        Returns:
            Dict with status and topic ARN
        """
        try:
            # Create SNS client
            sns_client = boto3.client('sns', region_name=self.aws_region)
            
            # Create SNS topic
            response = sns_client.create_topic(Name=topic_name)
            topic_arn = response['TopicArn']
            
            # Store topic ARN
            self.sns_topic_arn = topic_arn
            shared_memory.set("notification_sns_topic_arn", topic_arn)
            
            # Subscribe email to topic
            subscription_response = sns_client.subscribe(
                TopicArn=topic_arn,
                Protocol='email',
                Endpoint=self.recipient_email
            )
            
            # Add topic tags
            sns_client.tag_resource(
                ResourceArn=topic_arn,
                Tags=[
                    {
                        'Key': 'Purpose',
                        'Value': 'DriftDetection'
                    },
                    {
                        'Key': 'CreatedBy',
                        'Value': 'NotificationAgent'
                    }
                ]
            )
            
            logger.info(f"SNS topic created: {topic_arn}")
            logger.info(f"Email subscription created: {subscription_response['SubscriptionArn']}")
            
            return {
                "status": "success",
                "message": f"SNS topic created and email {self.recipient_email} subscribed",
                "topic_arn": topic_arn,
                "subscription_arn": subscription_response.get('SubscriptionArn', 'pending confirmation')
            }
            
        except ClientError as e:
            error_msg = f"Error setting up SNS topic: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def _setup_eventbridge_rule(self, 
                               rule_name: Optional[str] = None,
                               resource_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Set up EventBridge rule to monitor AWS API calls for infrastructure changes.
        
        Args:
            rule_name: Name of the EventBridge rule
            resource_types: List of AWS resource types to monitor
            
        Returns:
            Dict with status and rule ARN
        """
        try:
            # Use provided rule name or default
            rule_name = rule_name or self.eventbridge_rule_name
            
            # Default resource types if none provided
            if not resource_types:
                resource_types = [
                    "AWS::EC2::Instance", 
                    "AWS::S3::Bucket", 
                    "AWS::IAM::Role",
                    "AWS::Lambda::Function",
                    "AWS::DynamoDB::Table"
                ]
            
            # Create EventBridge client
            events_client = boto3.client('events', region_name=self.aws_region)
            
            # Create event pattern for AWS API calls related to the specified resource types
            # This pattern will match any API call that creates, updates, or deletes resources
            event_pattern = {
                "source": ["aws.ec2", "aws.s3", "aws.iam", "aws.lambda", "aws.dynamodb"],
                "detail-type": ["AWS API Call via CloudTrail"],
                "detail": {
                    "eventSource": ["ec2.amazonaws.com", "s3.amazonaws.com", "iam.amazonaws.com", 
                                   "lambda.amazonaws.com", "dynamodb.amazonaws.com"],
                    "eventName": [{
                        "prefix": "Create"
                    }, {
                        "prefix": "Update"
                    }, {
                        "prefix": "Delete"
                    }, {
                        "prefix": "Modify"
                    }, {
                        "prefix": "Put"
                    }]
                }
            }
            
            # Create the EventBridge rule
            rule_response = events_client.put_rule(
                Name=rule_name,
                EventPattern=json.dumps(event_pattern),
                State='ENABLED',
                Description=f'Detect infrastructure changes for drift notification'
            )
            
            rule_arn = rule_response['RuleArn']
            
            # Make sure SNS topic exists
            if not self.sns_topic_arn:
                sns_setup = self._setup_sns_topic()
                if sns_setup["status"] != "success":
                    return {
                        "status": "error",
                        "message": f"Failed to set up SNS topic: {sns_setup['message']}"
                    }
            
            # Add SNS as target for the rule
            target_response = events_client.put_targets(
                Rule=rule_name,
                Targets=[
                    {
                        'Id': 'DriftNotificationTarget',
                        'Arn': self.sns_topic_arn,
                        'InputTransformer': {
                            'InputPathsMap': {
                                'eventName': '$.detail.eventName',
                                'eventSource': '$.detail.eventSource',
                                'awsRegion': '$.detail.awsRegion',
                                'eventTime': '$.detail.eventTime',
                                'requestParameters': '$.detail.requestParameters',
                                'responseElements': '$.detail.responseElements',
                                'sourceIPAddress': '$.detail.sourceIPAddress'
                            },
                            'InputTemplate': """
"AWS Infrastructure Change Detected - Potential Drift"

Event: <eventName>
Service: <eventSource>
Region: <awsRegion>
Time: <eventTime>
Source IP: <sourceIPAddress>

Request Parameters:
<requestParameters>

Response Elements:
<responseElements>

This change may indicate infrastructure drift. Please check your Terraform state.
"""
                        }
                    }
                ]
            )
            
            # Store rule info in shared memory
            shared_memory.set("notification_eventbridge_rule", {
                "rule_name": rule_name,
                "rule_arn": rule_arn,
                "resource_types": resource_types,
                "creation_time": datetime.now().isoformat()
            })
            
            logger.info(f"EventBridge rule created: {rule_arn}")
            
            return {
                "status": "success",
                "message": f"EventBridge rule '{rule_name}' created successfully",
                "rule_arn": rule_arn,
                "resource_types": resource_types
            }
            
        except ClientError as e:
            error_msg = f"Error setting up EventBridge rule: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def _setup_aws_config(self, 
                         recorder_name: Optional[str] = None,
                         include_global_resources: bool = True) -> Dict[str, Any]:
        """
        Set up AWS Config for drift detection.
        
        Args:
            recorder_name: Name of the Config recorder
            include_global_resources: Whether to include global resources like IAM
            
        Returns:
            Dict with status and recorder details
        """
        try:
            # Use provided recorder name or default
            recorder_name = recorder_name or self.config_recorder_name
            
            # Create Config client
            config_client = boto3.client('config', region_name=self.aws_region)
            
            # Check if config recorder already exists
            existing_recorders = config_client.describe_configuration_recorders()
            
            if existing_recorders['ConfigurationRecorders']:
                recorder_exists = any(recorder['name'] == recorder_name 
                                     for recorder in existing_recorders['ConfigurationRecorders'])
                
                if recorder_exists:
                    logger.info(f"AWS Config recorder '{recorder_name}' already exists")
                    
                    # Make sure it's enabled
                    config_client.start_configuration_recorder(
                        ConfigurationRecorderName=recorder_name
                    )
                    
                    return {
                        "status": "success",
                        "message": f"AWS Config recorder '{recorder_name}' already exists and is enabled",
                        "recorder_name": recorder_name
                    }
            
            # Create IAM role for AWS Config
            iam_client = boto3.client('iam', region_name=self.aws_region)
            
            # Create role with trust relationship
            try:
                role_response = iam_client.create_role(
                    RoleName='AWSConfigRole',
                    AssumeRolePolicyDocument=json.dumps({
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {
                                    "Service": "config.amazonaws.com"
                                },
                                "Action": "sts:AssumeRole"
                            }
                        ]
                    }),
                    Description='IAM role for AWS Config'
                )
                
                role_arn = role_response['Role']['Arn']
                
                # Attach AWS managed policy for Config
                iam_client.attach_role_policy(
                    RoleName='AWSConfigRole',
                    PolicyArn='arn:aws:iam::aws:policy/service-role/AWS_ConfigRole'
                )
                
            except ClientError as e:
                if 'EntityAlreadyExists' in str(e):
                    # Role already exists, get its ARN
                    role_response = iam_client.get_role(RoleName='AWSConfigRole')
                    role_arn = role_response['Role']['Arn']
                else:
                    raise e
            
            # Create S3 bucket for Config recordings if needed
            s3_client = boto3.client('s3', region_name=self.aws_region)
            bucket_name = f"aws-config-recordings-{self.aws_region}-{datetime.now().strftime('%Y%m%d')}"
            
            try:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={
                        'LocationConstraint': self.aws_region
                    }
                )
                
                # Add bucket policy
                bucket_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Sid": "AWSConfigBucketPermissionsCheck",
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "config.amazonaws.com"
                            },
                            "Action": "s3:GetBucketAcl",
                            "Resource": f"arn:aws:s3:::{bucket_name}"
                        },
                        {
                            "Sid": "AWSConfigBucketDelivery",
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "config.amazonaws.com"
                            },
                            "Action": "s3:PutObject",
                            "Resource": f"arn:aws:s3:::{bucket_name}/AWSLogs/*"
                        }
                    ]
                }
                
                s3_client.put_bucket_policy(
                    Bucket=bucket_name,
                    Policy=json.dumps(bucket_policy)
                )
                
            except ClientError as e:
                if 'BucketAlreadyExists' in str(e) or 'BucketAlreadyOwnedByYou' in str(e):
                    # Try another bucket name with random suffix
                    import uuid
                    bucket_name = f"aws-config-recordings-{self.aws_region}-{uuid.uuid4().hex[:8]}"
                    
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={
                            'LocationConstraint': self.aws_region
                        }
                    )
                else:
                    # Use default bucket if creation fails
                    bucket_name = f"config-bucket-{self.aws_region}"
            
            # Create delivery channel
            delivery_channel_response = config_client.put_delivery_channel(
                DeliveryChannel={
                    'name': 'default',
                    's3BucketName': bucket_name,
                    'configSnapshotDeliveryProperties': {
                        'deliveryFrequency': 'Six_Hours'
                    }
                }
            )
            
            # Create configuration recorder
            recorder_response = config_client.put_configuration_recorder(
                ConfigurationRecorder={
                    'name': recorder_name,
                    'roleARN': role_arn,
                    'recordingGroup': {
                        'allSupported': True,
                        'includeGlobalResourceTypes': include_global_resources
                    }
                }
            )
            
            # Start the configuration recorder
            config_client.start_configuration_recorder(
                ConfigurationRecorderName=recorder_name
            )
            
            # Store config info in shared memory
            shared_memory.set("notification_aws_config", {
                "recorder_name": recorder_name,
                "role_arn": role_arn,
                "bucket_name": bucket_name,
                "include_global_resources": include_global_resources,
                "creation_time": datetime.now().isoformat()
            })
            
            logger.info(f"AWS Config recorder '{recorder_name}' created and started")
            
            return {
                "status": "success",
                "message": f"AWS Config recorder '{recorder_name}' created and started successfully",
                "recorder_name": recorder_name,
                "role_arn": role_arn,
                "bucket_name": bucket_name
            }
            
        except ClientError as e:
            error_msg = f"Error setting up AWS Config: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def _send_test_notification(self, subject: Optional[str] = None, message: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a test notification through SNS.
        
        Args:
            subject: Custom subject for the notification
            message: Custom message for the notification
            
        Returns:
            Dict with status and message ID
        """
        try:
            # Make sure SNS topic exists
            if not self.sns_topic_arn:
                sns_setup = self._setup_sns_topic()
                if sns_setup["status"] != "success":
                    return {
                        "status": "error",
                        "message": f"Failed to set up SNS topic: {sns_setup['message']}"
                    }
            
            # Create SNS client
            sns_client = boto3.client('sns', region_name=self.aws_region)
            
            # Default subject and message if not provided
            if not subject:
                subject = "AWS Drift Monitoring - Test Notification"
                
            if not message:
                message = f"""
This is a test notification from the AWS Drift Monitoring System.

Monitoring is active and working correctly.
Recipient: {self.recipient_email}
Time: {datetime.now().isoformat()}

The system is configured to notify you of any infrastructure changes that might indicate drift.
"""
            
            # Send the notification
            response = sns_client.publish(
                TopicArn=self.sns_topic_arn,
                Message=message,
                Subject=subject
            )
            
            message_id = response['MessageId']
            
            # Log success and update shared memory
            logger.info(f"Test notification sent successfully, Message ID: {message_id}")
            shared_memory.set("last_notification_sent", datetime.now().isoformat())
            shared_memory.set("last_notification_subject", subject)
            
            return {
                "status": "success",
                "message": "Test notification sent successfully",
                "message_id": message_id,
                "recipient": self.recipient_email
            }
            
        except ClientError as e:
            error_msg = f"Error sending test notification: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def _check_notification_status(self) -> Dict[str, Any]:
        """
        Check the status of the notification setup.
        
        Returns:
            Dict with status details
        """
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "components": {}
            }
            
            # Check SNS topic
            if self.sns_topic_arn:
                sns_client = boto3.client('sns', region_name=self.aws_region)
                
                try:
                    # Get topic attributes
                    topic_attributes = sns_client.get_topic_attributes(
                        TopicArn=self.sns_topic_arn
                    )
                    
                    # Get subscriptions
                    subscriptions = sns_client.list_subscriptions_by_topic(
                        TopicArn=self.sns_topic_arn
                    )
                    
                    status["components"]["sns"] = {
                        "status": "active",
                        "topic_arn": self.sns_topic_arn,
                        "subscriptions": [
                            {
                                "endpoint": sub["Endpoint"],
                                "protocol": sub["Protocol"],
                                "status": sub.get("SubscriptionArn", "PendingConfirmation")
                            }
                            for sub in subscriptions.get("Subscriptions", [])
                        ]
                    }
                    
                except ClientError:
                    status["components"]["sns"] = {
                        "status": "error",
                        "message": "SNS topic exists but could not retrieve details"
                    }
            else:
                status["components"]["sns"] = {
                    "status": "not_configured",
                    "message": "SNS topic not set up"
                }
            
            # Check EventBridge rule
            events_client = boto3.client('events', region_name=self.aws_region)
            
            try:
                rule_info = events_client.describe_rule(
                    Name=self.eventbridge_rule_name
                )
                
                # Get targets
                targets = events_client.list_targets_by_rule(
                    Rule=self.eventbridge_rule_name
                )
                
                status["components"]["eventbridge"] = {
                    "status": "active" if rule_info.get("State") == "ENABLED" else "disabled",
                    "rule_name": self.eventbridge_rule_name,
                    "rule_arn": rule_info.get("Arn"),
                    "targets": [
                        {
                            "id": target["Id"],
                            "arn": target["Arn"]
                        }
                        for target in targets.get("Targets", [])
                    ]
                }
                
            except ClientError:
                status["components"]["eventbridge"] = {
                    "status": "not_configured",
                    "message": "EventBridge rule not set up"
                }
            
            # Check AWS Config
            config_client = boto3.client('config', region_name=self.aws_region)
            
            try:
                config_recorders = config_client.describe_configuration_recorders()
                recorder_status = config_client.describe_configuration_recorder_status()
                
                if config_recorders["ConfigurationRecorders"]:
                    recorder = next(
                        (r for r in config_recorders["ConfigurationRecorders"] 
                         if r["name"] == self.config_recorder_name),
                        config_recorders["ConfigurationRecorders"][0]
                    )
                    
                    recorder_status_info = next(
                        (s for s in recorder_status["ConfigurationRecordersStatus"] 
                         if s["name"] == recorder["name"]),
                        None
                    )
                    
                    status["components"]["config"] = {
                        "status": "active" if recorder_status_info and recorder_status_info.get("recording") else "disabled",
                        "recorder_name": recorder["name"],
                        "role_arn": recorder.get("roleARN"),
                        "last_status": recorder_status_info.get("lastStatus") if recorder_status_info else "Unknown"
                    }
                else:
                    status["components"]["config"] = {
                        "status": "not_configured",
                        "message": "AWS Config recorder not set up"
                    }
                    
            except ClientError:
                status["components"]["config"] = {
                    "status": "error",
                    "message": "Could not retrieve AWS Config status"
                }
            
            # Determine overall status
            components_status = [comp["status"] for comp in status["components"].values()]
            
            if all(s == "active" for s in components_status):
                status["overall_status"] = "healthy"
            elif any(s == "error" for s in components_status):
                status["overall_status"] = "error"
            elif any(s == "not_configured" for s in components_status):
                status["overall_status"] = "incomplete"
            else:
                status["overall_status"] = "degraded"
            
            # Store status in shared memory
            shared_memory.set("notification_system_status", status)
            
            return {
                "status": "success",
                "notification_system": status
            }
            
        except Exception as e:
            error_msg = f"Error checking notification status: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def set_recipient_email(self, email: str) -> Dict[str, Any]:
        """
        Set the recipient email address for notifications.
        
        Args:
            email: Email address to send notifications to
            
        Returns:
            Dict with status and message
        """
        try:
            # Basic email validation
            if '@' not in email or '.' not in email:
                return {
                    "status": "error",
                    "message": f"Invalid email format: {email}"
                }
            
            # Update recipient email
            self.recipient_email = email
            shared_memory.set("notification_recipient_email", email)
            
            # If SNS topic exists, update subscription
            if self.sns_topic_arn:
                sns_client = boto3.client('sns', region_name=self.aws_region)
                
                # List existing subscriptions
                subscriptions = sns_client.list_subscriptions_by_topic(
                    TopicArn=self.sns_topic_arn
                )
                
                # Check if email is already subscribed
                email_already_subscribed = any(
                    sub["Endpoint"] == email and sub["Protocol"] == "email"
                    for sub in subscriptions.get("Subscriptions", [])
                )
                
                if not email_already_subscribed:
                    # Subscribe new email
                    subscription_response = sns_client.subscribe(
                        TopicArn=self.sns_topic_arn,
                        Protocol='email',
                        Endpoint=email
                    )
                    
                    logger.info(f"New email subscription created: {subscription_response['SubscriptionArn']}")
            
            return {
                "status": "success",
                "message": f"Notification recipient email set to {email}"
            }
            
        except Exception as e:
            error_msg = f"Error setting recipient email: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def start_continuous_monitoring(self, interval_minutes: int = 15) -> Dict[str, Any]:
        """
        Start continuous monitoring of AWS infrastructure changes using AWS-native services.
        
        Args:
            interval_minutes: Not used with event-driven architecture, kept for API compatibility
            
        Returns:
            Dict with status and message
        """
        try:
            # Set up components in order
            
            # 1. Set up SNS topic first
            sns_result = self._setup_sns_topic()
            if sns_result["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Failed to set up SNS topic: {sns_result['message']}",
                    "details": sns_result
                }
            
            # 2. Set up EventBridge rule
            eventbridge_result = self._setup_eventbridge_rule()
            if eventbridge_result["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Failed to set up EventBridge rule: {eventbridge_result['message']}",
                    "sns_setup": sns_result,
                    "eventbridge_setup": eventbridge_result
                }
            
            # 3. Set up AWS Config (optional, can continue if this fails)
            config_result = self._setup_aws_config()
            
            # 4. Send test notification
            test_notification_result = self._send_test_notification()
            
            # Store monitoring configuration in shared memory
            shared_memory.set("notification_monitoring_active", True)
            shared_memory.set("notification_monitoring_start_time", datetime.now().isoformat())
            
            # Check notification system status
            status_result = self._check_notification_status()
            
            return {
                "status": "success",
                "message": "AWS-native drift notification system set up successfully",
                "components": {
                    "sns": sns_result,
                    "eventbridge": eventbridge_result,
                    "config": config_result,
                    "test_notification": test_notification_result,
                    "system_status": status_result.get("notification_system", {})
                }
            }
            
        except Exception as e:
            error_msg = f"Error starting continuous monitoring: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }