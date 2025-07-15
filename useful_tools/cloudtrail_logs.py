#!/usr/bin/env python3
"""
CloudTrail logs analysis tool for Strands Agent.

This tool helps fetch and analyze AWS CloudTrail logs for infrastructure drift detection.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
from strands import tool

logger = logging.getLogger(__name__)

@tool
def cloudtrail_logs(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    region: Optional[str] = None,
    event_name: Optional[str] = None,
    resource_type: Optional[str] = None,
    max_results: int = 50
) -> Dict:
    """
    Fetch and analyze AWS CloudTrail logs for infrastructure changes.
    
    This tool queries CloudTrail logs to detect infrastructure changes that might indicate drift,
    focusing on resource creation, modification, or deletion events that could affect
    infrastructure as code consistency.
    
    Args:
        start_time: Start time for log search (ISO format: YYYY-MM-DDTHH:MM:SS). Defaults to 24h ago.
        end_time: End time for log search (ISO format: YYYY-MM-DDTHH:MM:SS). Defaults to now.
        region: AWS region to search. Defaults to AWS_REGION env variable or 'us-east-1'.
        event_name: Specific CloudTrail event name to filter (e.g., 'CreateBucket', 'RunInstances').
        resource_type: AWS resource type to filter for (e.g., 'AWS::S3::Bucket').
        max_results: Maximum number of results to return (default: 50).
    
    Returns:
        Dict containing:
            - events: List of CloudTrail events matching the criteria
            - summary: Analysis of infrastructure changes detected
            - potentially_drifted_resources: Resources that may have drifted
    """
    try:
        # Set up time range
        if not end_time:
            end_time_dt = datetime.utcnow()
        else:
            end_time_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            
        if not start_time:
            start_time_dt = end_time_dt - timedelta(hours=24)
        else:
            start_time_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        
        # Get AWS region
        if not region:
            region = os.environ.get('AWS_REGION', 'us-east-1')
            
        # Initialize CloudTrail client
        cloudtrail = boto3.client('cloudtrail', region_name=region)
        
        # Build lookup attributes
        lookup_attributes = []
        if event_name:
            lookup_attributes.append({
                'AttributeKey': 'EventName',
                'AttributeValue': event_name
            })
        
        # Query CloudTrail
        kwargs = {
            'StartTime': start_time_dt,
            'EndTime': end_time_dt,
            'MaxResults': max_results
        }
        
        if lookup_attributes:
            kwargs['LookupAttributes'] = lookup_attributes
            
        response = cloudtrail.lookup_events(**kwargs)
        
        # Process and analyze events
        events = response.get('Events', [])
        processed_events = []
        
        # Drift-relevant event types for common AWS resources
        drift_relevant_prefixes = [
            'Create', 'Update', 'Delete', 'Modify', 'Put', 'Run', 'Start', 'Stop', 
            'Terminate', 'Associate', 'Disassociate', 'Attach', 'Detach'
        ]
        
        potentially_drifted_resources = {}
        
        for event in events:
            event_data = json.loads(event.get('CloudTrailEvent', '{}'))
            event_name = event_data.get('eventName', '')
            event_source = event_data.get('eventSource', '').replace('.amazonaws.com', '')
            
            # Check if this is a drift-relevant event
            is_drift_relevant = False
            if event_name:  # Add check to ensure event_name is not None
                is_drift_relevant = any(event_name.startswith(prefix) for prefix in drift_relevant_prefixes)
            
            # Extract resource information
            resources = []
            if 'requestParameters' in event_data:
                if resource_type and 'resourceType' in event_data.get('requestParameters', {}):
                    if event_data['requestParameters']['resourceType'] != resource_type:
                        continue
                        
                # Extract resource IDs based on common patterns
                request_params = event_data.get('requestParameters', {})
                for key in ['bucketName', 'instanceId', 'functionName', 'tableName', 'roleArn', 
                           'roleSessionName', 'stackName', 'clusterName', 'repositoryName']:
                    if key in request_params and request_params[key]:
                        resources.append(f"{key}: {request_params[key]}")
                        
                        if is_drift_relevant:
                            resource_id = request_params[key]
                            if event_source not in potentially_drifted_resources:
                                potentially_drifted_resources[event_source] = []
                            potentially_drifted_resources[event_source].append({
                                'resourceId': resource_id,
                                'eventName': event_name,
                                'eventTime': event.get('EventTime', '').isoformat() if isinstance(event.get('EventTime'), datetime) else event.get('EventTime', '')
                            })
            
            processed_event = {
                'eventId': event.get('EventId'),
                'eventName': event_name,
                'eventTime': event.get('EventTime', '').isoformat() if isinstance(event.get('EventTime'), datetime) else event.get('EventTime', ''),
                'username': event.get('Username', 'Unknown'),
                'resources': resources,
                'eventSource': event_source,
                'isDriftRelevant': is_drift_relevant
            }
            
            processed_events.append(processed_event)
        
        # Create summary of potential drift
        drift_summary = {
            'totalEvents': len(processed_events),
            'driftRelevantEvents': sum(1 for e in processed_events if e.get('isDriftRelevant', False)),
            'affectedResourceTypes': list(potentially_drifted_resources.keys()),
            'timeRange': f"{start_time_dt.isoformat()} to {end_time_dt.isoformat()}"
        }
        
        return {
            'events': processed_events[:max_results],
            'summary': drift_summary,
            'potentially_drifted_resources': potentially_drifted_resources
        }
        
    except ClientError as e:
        logger.error(f"AWS ClientError: {str(e)}")
        return {
            'error': f"AWS ClientError: {str(e)}",
            'events': [],
            'summary': {},
            'potentially_drifted_resources': {}
        }
    except Exception as e:
        logger.error(f"Error fetching CloudTrail logs: {str(e)}")
        return {
            'error': f"Error fetching CloudTrail logs: {str(e)}",
            'events': [],
            'summary': {},
            'potentially_drifted_resources': {}
        } 