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
    
    def set_recipient_email(self, email: str) -> Dict[str, Any]:
        """
        Set the recipient email address for notifications.
        
        Args:
            email: Email address to send notifications to
            
        Returns:
            Dict with status and message
        """
        try:
            # Validate email format (basic validation)
            if '@' not in email or '.' not in email:
                return {
                    "status": "error",
                    "message": f"Invalid email format: {email}"
                }
            
            # Set the recipient email
            self.recipient_email = email
            
            # Update shared memory
            shared_memory.set("notification_recipient_email", email)
            
            logger.info(f"Notification recipient email set to: {email}")
            
            return {
                "status": "success",
                "message": f"Recipient email set to: {email}"
            }
        except Exception as e:
            error_msg = f"Error setting recipient email: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
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
    
    def _setup_sns_topic(self, topic_name: str = "DriftAlertTopic") -> Dict[str, Any]:
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
    
    def _create_lambda_function(self) -> Dict[str, Any]:
        """
        Create a Lambda function to process EventBridge events and send notifications.
        
        Returns:
            Dict with status and function ARN
        """
        try:
            # Create Lambda client
            lambda_client = boto3.client('lambda', region_name=self.aws_region)
            
            # Create IAM role for Lambda
            iam_client = boto3.client('iam', region_name=self.aws_region)
            
            # Define role name
            role_name = 'DriftDetectionLambdaRole'
            
            try:
                # Try to get the role if it already exists
                role_response = iam_client.get_role(RoleName=role_name)
                role_arn = role_response['Role']['Arn']
                logger.info(f"Using existing IAM role: {role_arn}")
            except ClientError:
                # Create the role if it doesn't exist
                logger.info(f"Creating new IAM role: {role_name}")
                
                # Create trust relationship policy document
                trust_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "lambda.amazonaws.com"
                            },
                            "Action": "sts:AssumeRole"
                        }
                    ]
                }
                
                # Create the role
                role_response = iam_client.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(trust_policy),
                    Description='Role for Drift Detection Lambda function'
                )
                
                role_arn = role_response['Role']['Arn']
                
                # Attach policies to the role
                iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
                )
                
                # Create and attach SNS publish policy
                sns_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": "sns:Publish",
                            "Resource": self.sns_topic_arn
                        }
                    ]
                }
                
                sns_policy_response = iam_client.create_policy(
                    PolicyName='DriftDetectionSNSPublishPolicy',
                    PolicyDocument=json.dumps(sns_policy),
                    Description='Allow Lambda to publish to SNS topic'
                )
                
                iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=sns_policy_response['Policy']['Arn']
                )
                
                # Wait for role to propagate
                import time
                time.sleep(10)
            
            # Define Lambda function code with single quotes
            lambda_code = '''
import json
import logging
import re
import boto3
from datetime import datetime, timedelta
import textwrap

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sns = boto3.client('sns')
SNS_TOPIC_ARN = "{sns_topic_arn}"

def is_drift_event(event_name):
    return re.search(r'(Put|Delete|Modify|Create|Start|Stop|Run|Attach|Authorize|Terminate)', event_name, re.IGNORECASE)

IGNORED_SOURCES = [
    "logs.amazonaws.com",
    "cloudtrail.amazonaws.com"
]

def lambda_handler(event, context):
    detail = event.get("detail", {{}})
    event_name = detail.get("eventName", "unknown")
    source = detail.get("eventSource", "")
    user = detail.get("userIdentity", {{}}).get("arn", "unknown")
    
    event_time_utc = detail.get("eventTime", datetime.utcnow().isoformat())
    try:
        event_time_vn = datetime.strptime(event_time_utc, "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=7)
    except ValueError:
        try:
            event_time_vn = datetime.strptime(event_time_utc.split('.')[0], "%Y-%m-%dT%H:%M:%S") + timedelta(hours=7)
        except:
            event_time_vn = datetime.utcnow() + timedelta(hours=7)
            
    formatted_time = event_time_vn.strftime("%d/%m/%Y %H:%M:%S")

    if is_drift_event(event_name) and source not in IGNORED_SOURCES:
        logger.info(f"🚨 [DRIFT] Event: {{event_name}} by {{user}}")
        logger.info("➡️ Chi tiết:")
        logger.info(json.dumps(detail, indent=2))

        message = f"""
Xin chào Quản trị viên,

Hệ thống vừa ghi nhận một thay đổi cấu hình trong môi trường AWS mà rất có thể gây ra tình trạng **Drift** – tức là cấu hình thực tế không còn đồng nhất với cấu hình chuẩn được quản lý bằng Terraform hoặc CloudFormation.

🔹 THÔNG TIN CHI TIẾT:: 
    - Sự kiện                   : {{event_name}}
    - Tài nguyên liên quan      : IAM User
    - Thời gian ghi nhận        : {{formatted_time}}
    - Tài khoản thực hiện       : {{user}}
    - Nguồn                     : {{source}}

Những thay đổi này có thể ảnh hưởng đến độ ổn định, bảo mật hoặc khả năng khôi phục cấu hình hạ tầng.

🔹 KHUYẾN NGHỊ:
    - Kiểm tra lại các thay đổi đã diễn ra.
    - So sánh với cấu hình chuẩn trong hệ thống quản lý hạ tầng (Infrastructure as Code).
    - Thực hiện kiểm tra drift (drift detection) để xác định mức độ sai lệch.
    - Liên hệ quản trị viên nếu thay đổi không được cho phép hoặc chưa được xác minh.

🔹 HỖ TRỢ & TƯ VẤN:  
    Để tìm hiểu thêm về cách sử dụng Terraform, kiểm tra Drift, và giải quyết các sự cố tương tự:

    Truy cập: 👉 http://destroydrift.raiijino.buzz

Tại đây, bạn có thể đặt câu hỏi, tham khảo tài liệu và nhận hỗ trợ hệ thống AI GeniDetect.

Trân trọng,  
GeniDetect
"""
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"[CẢNH BÁO DRIFT] {{event_name}}",
            Message=message
        )
        return {{
            'statusCode': 200,
            'body': json.dumps('Notification sent successfully')
        }}
    else:
        logger.info(f"✅ Event '{{event_name}}' được bỏ qua (không gây drift hoặc thuộc source bị loại)")
        return {{
            'statusCode': 200,
            'body': json.dumps('Event ignored - not a drift event or from ignored source')
        }}
'''.format(sns_topic_arn=self.sns_topic_arn)
            
            # Create a temporary zip file for the Lambda function
            import tempfile
            import zipfile
            
            with tempfile.NamedTemporaryFile(suffix='.zip') as temp_zip:
                with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
                    zipf.writestr('lambda_function.py', lambda_code)
                
                # Read the zip file
                with open(temp_zip.name, 'rb') as f:
                    zip_bytes = f.read()
            
            # Create or update Lambda function
            function_name = 'DriftDetectionFunction'
            
            try:
                # Try to get the function if it exists
                lambda_client.get_function(FunctionName=function_name)
                
                # Update the function if it exists
                update_response = lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=zip_bytes,
                    Publish=True
                )
                
                function_arn = update_response['FunctionArn']
                logger.info(f"Updated Lambda function: {function_arn}")
                
            except ClientError:
                # Create the function if it doesn't exist
                create_response = lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime='python3.9',
                    Role=role_arn,
                    Handler='lambda_function.lambda_handler',
                    Code={'ZipFile': zip_bytes},
                    Description='Process EventBridge events and send drift notifications',
                    Timeout=30,
                    MemorySize=128,
                    Publish=True
                )
                
                function_arn = create_response['FunctionArn']
                logger.info(f"Created Lambda function: {function_arn}")
            
            # Store Lambda function ARN
            shared_memory.set("notification_lambda_function_arn", function_arn)
            
            return {
                "status": "success",
                "message": f"Lambda function created/updated successfully",
                "function_arn": function_arn
            }
            
        except Exception as e:
            error_msg = f"Error creating Lambda function: {str(e)}"
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
            
            # Create Lambda function for processing events
            lambda_result = self._create_lambda_function()
            if lambda_result["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Failed to create Lambda function: {lambda_result['message']}"
                }
            
            lambda_function_arn = lambda_result["function_arn"]
            
            # Add Lambda as target for the rule
            lambda_client = boto3.client('lambda', region_name=self.aws_region)
            
            # Add permission for EventBridge to invoke the Lambda function
            try:
                lambda_client.add_permission(
                    FunctionName='DriftDetectionFunction',
                    StatementId=f'EventBridge-{rule_name}',
                    Action='lambda:InvokeFunction',
                    Principal='events.amazonaws.com',
                    SourceArn=rule_arn
                )
            except ClientError as e:
                # If the permission already exists, continue
                if 'ResourceConflictException' not in str(e):
                    raise e
            
            # Add Lambda as target for the rule
            target_response = events_client.put_targets(
                Rule=rule_name,
                Targets=[
                    {
                        'Id': 'DriftDetectionLambdaTarget',
                        'Arn': lambda_function_arn
                    }
                ]
            )
            
            # Create a separate rule for S3 events
            s3_rule_name = f"{rule_name}-S3Events"
            s3_event_pattern = {
                "source": ["aws.s3"],
                "detail-type": ["Object Created", "Object Deleted", "Object Restored", "Object Tagging"]
            }
            
            s3_rule_response = events_client.put_rule(
                Name=s3_rule_name,
                EventPattern=json.dumps(s3_event_pattern),
                State='ENABLED',
                Description=f'Detect S3 object changes for drift notification'
            )
            
            s3_rule_arn = s3_rule_response['RuleArn']
            
            # Add permission for EventBridge to invoke the Lambda function for S3 events
            try:
                lambda_client.add_permission(
                    FunctionName='DriftDetectionFunction',
                    StatementId=f'EventBridge-{s3_rule_name}',
                    Action='lambda:InvokeFunction',
                    Principal='events.amazonaws.com',
                    SourceArn=s3_rule_arn
                )
            except ClientError as e:
                # If the permission already exists, continue
                if 'ResourceConflictException' not in str(e):
                    raise e
            
            # Add Lambda as target for the S3 rule
            s3_target_response = events_client.put_targets(
                Rule=s3_rule_name,
                Targets=[
                    {
                        'Id': 'S3DriftDetectionLambdaTarget',
                        'Arn': lambda_function_arn
                    }
                ]
            )
            
            # Enable S3 to send events to EventBridge for all buckets
            s3_client = boto3.client('s3', region_name=self.aws_region)
            
            # Get list of all buckets
            try:
                response = s3_client.list_buckets()
                buckets = [bucket['Name'] for bucket in response['Buckets']]
                
                # Enable EventBridge notifications for each bucket
                for bucket in buckets:
                    try:
                        s3_client.put_bucket_notification_configuration(
                            Bucket=bucket,
                            NotificationConfiguration={
                                'EventBridgeConfiguration': {}
                            }
                        )
                        logger.info(f"Enabled EventBridge notifications for bucket: {bucket}")
                    except ClientError as e:
                        logger.warning(f"Could not enable EventBridge for bucket {bucket}: {e}")
            except ClientError as e:
                logger.warning(f"Could not list buckets: {e}")
            
            # Store rule info in shared memory
            shared_memory.set("notification_eventbridge_rule", {
                "rule_name": rule_name,
                "rule_arn": rule_arn,
                "s3_rule_name": s3_rule_name,
                "s3_rule_arn": s3_rule_arn,
                "lambda_function_arn": lambda_function_arn,
                "resource_types": resource_types,
                "creation_time": datetime.now().isoformat()
            })
            
            logger.info(f"EventBridge rule created: {rule_arn}")
            
            return {
                "status": "success",
                "message": f"EventBridge rules and Lambda function created successfully",
                "rule_arn": rule_arn,
                "s3_rule_arn": s3_rule_arn,
                "lambda_function_arn": lambda_function_arn,
                "resource_types": resource_types
            }
            
        except ClientError as e:
            error_msg = f"Error setting up EventBridge rule: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def _setup_aws_config(self) -> Dict[str, Any]:
        """
        Set up AWS Config to record configuration changes.
        
        Returns:
            Dict with status and message
        """
        try:
            # Create Config client
            config_client = boto3.client('config', region_name=self.aws_region)
            
            # Define role name for Config
            role_name = 'DriftConfigRole'
            
            try:
                # Try to get the role if it already exists
                iam_client = boto3.client('iam', region_name=self.aws_region)
                role_response = iam_client.get_role(RoleName=role_name)
                role_arn = role_response['Role']['Arn']
                logger.info(f"Using existing IAM role: {role_arn}")
            except ClientError:
                # Create the role if it doesn't exist
                logger.info(f"Creating new IAM role: {role_name}")
                
                # Create trust relationship policy document
                trust_policy = {
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
                }
                
                # Create the role
                role_response = iam_client.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(trust_policy),
                    Description='Role for AWS Config to record configuration changes'
                )
                
                role_arn = role_response['Role']['Arn']
                
                # Attach policies to the role
                iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn='arn:aws:iam::aws:policy/service-role/AWSConfigRole'
                )
                
                # Wait for role to propagate
                import time
                time.sleep(10)
            
            # Create Config recorder
            recorder_name = self.config_recorder_name
            
            try:
                # Try to get the recorder if it already exists
                recorders = config_client.describe_configuration_recorders()
                existing_recorder = next(
                    (r for r in recorders.get("ConfigurationRecorders", []) if r["name"] == recorder_name),
                    None
                )
                
                if existing_recorder:
                    logger.info(f"Config recorder '{recorder_name}' already exists.")
                else:
                    raise ClientError({"Error": {"Code": "NoSuchConfigurationRecorderException"}}, "describe_configuration_recorders")
            except ClientError:
                # Create the recorder if it doesn't exist
                logger.info(f"Creating new Config recorder: {recorder_name}")
                
                # Define recorder configuration
                recorder_config = {
                    "name": recorder_name,
                    "roleARN": role_arn,
                    "recordingGroup": {
                        "allSupported": True,
                        "includeGlobalResourceTypes": True,
                        "resourceTypes": resource_types
                    }
                }
                
                config_client.put_configuration_recorder(
                    ConfigurationRecorder=recorder_config
                )
                
                # Start recording
                config_client.start_configuration_recorder(
                    ConfigurationRecorderName=recorder_name
                )
                
                logger.info(f"Config recorder '{recorder_name}' created and started.")
            
            # Store recorder ARN in shared memory
            shared_memory.set("notification_config_recorder_arn", role_arn)
            
            return {
                "status": "success",
                "message": f"AWS Config recorder '{recorder_name}' set up successfully."
            }
            
        except ClientError as e:
            error_msg = f"Error setting up AWS Config: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            }
    
    def _send_test_notification(self, subject: str = "Test Drift Notification", message: str = "This is a test notification from the Drift Detection System.") -> Dict[str, Any]:
        """
        Send a test notification to verify the notification system.
        
        Args:
            subject: The subject line for the notification email
            message: The message body for the notification email
            
        Returns:
            Dict with status and message
        """
        try:
            # Create SNS client
            sns_client = boto3.client('sns', region_name=self.aws_region)
            
            # Publish a test message to the SNS topic
            response = sns_client.publish(
                TopicArn=self.sns_topic_arn,
                Subject=subject,
                Message=message
            )
            
            logger.info(f"Test notification sent. Message ID: {response['MessageId']}")
            
            return {
                "status": "success",
                "message": "Test notification sent successfully."
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
        Check the status of the notification system components.
        
        Returns:
            Dict with status and overall status
        """
        try:
            # Create clients for status checks
            sns_client = boto3.client('sns', region_name=self.aws_region)
            events_client = boto3.client('events', region_name=self.aws_region)
            config_client = boto3.client('config', region_name=self.aws_region)
            
            status = {"components": {}}
            
            # Check SNS
            try:
                sns_client.get_topic_attributes(TopicArn=self.sns_topic_arn)
                status["components"]["sns"] = {
                    "status": "active",
                    "topic_arn": self.sns_topic_arn,
                    "subscription_arn": shared_memory.get("notification_sns_subscription_arn", "N/A")
                }
            except ClientError:
                status["components"]["sns"] = {
                    "status": "not_configured",
                    "message": "SNS topic not set up"
                }
            
            # Check EventBridge
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
            try:
                recorders = config_client.describe_configuration_recorders()
                recorder_status = config_client.describe_configuration_recorder_status()
                
                if recorders["ConfigurationRecorders"]:
                    recorder = next(
                        (r for r in recorders["ConfigurationRecorders"] 
                         if r["name"] == self.config_recorder_name),
                        recorders["ConfigurationRecorders"][0]
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
    
    def start_continuous_monitoring(self, interval_minutes: int = 1) -> Dict[str, Any]:
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
            config_result = {"status": "skipped", "message": "AWS Config setup skipped"}
            setup_aws_config = shared_memory.get("notification_config", {}).get("setup_aws_config", False)
            
            if setup_aws_config:
                try:
                    config_result = self._setup_aws_config()
                    logger.info("AWS Config setup result: %s", config_result)
                except Exception as config_error:
                    logger.warning(f"AWS Config setup failed but continuing: {str(config_error)}")
                    config_result = {
                        "status": "error",
                        "message": f"AWS Config setup failed but continuing: {str(config_error)}"
                    }
            
            # 4. Send test notification
            test_notification_result = {"status": "skipped", "message": "Test notification skipped"}
            try:
                if self.sns_topic_arn:
                    test_notification_result = self._send_test_notification()
            except Exception as notification_error:
                logger.warning(f"Test notification failed but continuing: {str(notification_error)}")
                test_notification_result = {
                    "status": "error",
                    "message": f"Test notification failed but continuing: {str(notification_error)}"
                }
            
            # Store monitoring configuration in shared memory
            shared_memory.set("notification_monitoring_active", True)
            shared_memory.set("notification_monitoring_start_time", datetime.now().isoformat())
            
            # Check notification system status
            status_result = {"status": "skipped", "message": "Status check skipped"}
            try:
                status_result = self._check_notification_status()
            except Exception as status_error:
                logger.warning(f"Status check failed but continuing: {str(status_error)}")
                status_result = {
                    "status": "error", 
                    "message": f"Status check failed but continuing: {str(status_error)}"
                }
            
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