#!/usr/bin/env python3
"""
CloudWatch logs analysis tool for Strands Agent.

This tool helps fetch and analyze AWS CloudWatch logs for infrastructure drift detection.
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
def cloudwatch_logs(
    log_group_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    filter_pattern: Optional[str] = None,
    region: Optional[str] = None,
    max_results: int = 50
) -> Dict:
    """
    Fetch and analyze AWS CloudWatch logs for infrastructure events.
    
    This tool queries CloudWatch logs to find entries that might indicate infrastructure changes
    or drift events, specifically focusing on patterns that suggest resource modifications.
    
    Args:
        log_group_name: Name of the CloudWatch log group to search.
        start_time: Start time for log search (ISO format: YYYY-MM-DDTHH:MM:SS). Defaults to 24h ago.
        end_time: End time for log search (ISO format: YYYY-MM-DDTHH:MM:SS). Defaults to now.
        filter_pattern: CloudWatch logs filter pattern. Default focuses on common drift indicators.
        region: AWS region to search. Defaults to AWS_REGION env variable or 'us-east-1'.
        max_results: Maximum number of results to return (default: 50).
    
    Returns:
        Dict containing:
            - events: List of CloudWatch log events matching the criteria
            - summary: Analysis of detected events
            - drift_indicators: Events strongly indicating possible infrastructure drift
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
            
        # Convert to milliseconds since epoch for CloudWatch API
        start_time_ms = int(start_time_dt.timestamp() * 1000)
        end_time_ms = int(end_time_dt.timestamp() * 1000)
        
        # Get AWS region
        if not region:
            region = os.environ.get('AWS_REGION', 'us-east-1')
            
        # Default filter pattern for infrastructure changes if none provided
        if not filter_pattern:
            filter_pattern = '? "CREATE_" ? "UPDATE_" ? "DELETE_" ? "MODIFY_" ? "ERROR" ? "FAIL" ? "CREATE_FAILED" ? "UPDATE_FAILED"'
            
        # Initialize CloudWatch Logs client
        logs_client = boto3.client('logs', region_name=region)
        
        # Query CloudWatch Logs
        response = logs_client.filter_log_events(
            logGroupName=log_group_name,
            startTime=start_time_ms,
            endTime=end_time_ms,
            filterPattern=filter_pattern,
            limit=max_results
        )
        
        # Process and analyze events
        events = response.get('events', [])
        processed_events = []
        
        # Keywords indicating potential infrastructure drift
        drift_keywords = [
            'CREATE_', 'UPDATE_', 'DELETE_', 'MODIFY_', 'FAILED', 'ERROR',
            'CREATE_FAILED', 'UPDATE_FAILED', 'DELETE_FAILED', 'ROLLBACK'
        ]
        
        drift_indicators = []
        
        for event in events:
            message = event.get('message', '')
            
            # Check if this message indicates potential drift
            drift_related = False
            matching_keywords = []
            
            for keyword in drift_keywords:
                if keyword in message:
                    drift_related = True
                    matching_keywords.append(keyword)
            
            # Extract timestamp
            timestamp = datetime.fromtimestamp(event.get('timestamp', 0) / 1000)
            
            processed_event = {
                'eventId': event.get('eventId'),
                'timestamp': timestamp.isoformat(),
                'message': message[:500] + ('...' if len(message) > 500 else ''),
                'logStream': event.get('logStreamName', ''),
                'isDriftRelated': drift_related,
                'matchingKeywords': matching_keywords
            }
            
            processed_events.append(processed_event)
            
            if drift_related:
                drift_indicators.append(processed_event)
        
        # Create summary
        event_summary = {
            'totalEvents': len(processed_events),
            'driftRelatedEvents': len(drift_indicators),
            'logGroup': log_group_name,
            'timeRange': f"{start_time_dt.isoformat()} to {end_time_dt.isoformat()}",
            'filterPattern': filter_pattern
        }
        
        # Analyze most common drift indicators
        keyword_counts = {}
        for event in drift_indicators:
            for keyword in event['matchingKeywords']:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                
        event_summary['keywordFrequency'] = sorted(
            [{'keyword': k, 'count': v} for k, v in keyword_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        )
        
        return {
            'events': processed_events,
            'summary': event_summary,
            'drift_indicators': drift_indicators[:min(10, len(drift_indicators))]
        }
        
    except ClientError as e:
        logger.error(f"AWS ClientError: {str(e)}")
        return {
            'error': f"AWS ClientError: {str(e)}",
            'events': [],
            'summary': {},
            'drift_indicators': []
        }
    except Exception as e:
        logger.error(f"Error fetching CloudWatch logs: {str(e)}")
        return {
            'error': f"Error fetching CloudWatch logs: {str(e)}",
            'events': [],
            'summary': {},
            'drift_indicators': []
        } 