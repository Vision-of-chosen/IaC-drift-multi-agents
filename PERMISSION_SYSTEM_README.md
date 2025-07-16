# Permission-Based Callback Handlers for Multi-Agent System

This document explains the permission-based callback handler system that provides user authorization controls for agent actions in the Terraform Drift Detection & Remediation System.

## Overview

The permission system is inspired by the comprehensive callback handlers from the multi-agentic system notebook, enhanced with user permission controls. It intercepts tool usage events and asks for user approval before allowing agents to execute potentially dangerous operations.

## Features

- 🔐 **User Permission Controls**: Ask for human approval before executing tools
- 📝 **Comprehensive Logging**: Log all agent activities, reasoning, and tool usage
- ⚙️ **Configurable Security Levels**: Different permission profiles for different environments
- 🎯 **Agent-Specific Context**: Know which agent is requesting permissions
- 🔄 **Dynamic Configuration**: Change permission settings at runtime
- 📊 **Activity Monitoring**: Track all agent interactions and decisions

## Components

### 1. Permission Manager (`PermissionManager`)

The core component that manages which tools require approval:

```python
from permission_handlers import PermissionManager

# Create with custom settings
pm = PermissionManager(
    auto_approve_tools=["file_read", "current_time"],
    require_approval_tools=["terraform_apply", "use_aws"]
)
```

### 2. Callback Handler (`permission_based_callback_handler`)

The main callback handler that intercepts all agent events and enforces permissions:

```python
from permission_handlers import permission_based_callback_handler

# Used automatically when agents are created
agent = Agent(
    callback_handler=permission_based_callback_handler,
    # ... other parameters
)
```

### 3. Agent-Specific Handlers (`create_agent_callback_handler`)

Creates callback handlers with agent context:

```python
from permission_handlers import create_agent_callback_handler

# Create handler for specific agent
handler = create_agent_callback_handler("DetectAgent")

agent = Agent(
    callback_handler=handler,
    # ... other parameters
)
```

## Permission Levels

### Development Mode (Permissive)
```python
auto_approve_tools = [
    "current_time", "file_read", "calculator",
    "aws_documentation_search", "terraform_documentation_search",
    "retrieve", "read_tfstate", "cloudtrail_logs", "cloudwatch_logs",
    "terraform_plan", "terraform_get_best_practices"
]

require_approval_tools = [
    "terraform_apply", "terraform_run_command", "file_write",
    "editor", "use_aws", "shell"
]
```

### Production Mode (Restrictive)
```python
auto_approve_tools = [
    "current_time", "calculator", 
    "aws_documentation_search", "terraform_documentation_search"
]

require_approval_tools = [
    # Almost everything requires approval
    "file_read", "retrieve", "read_tfstate", "terraform_plan",
    "terraform_apply", "terraform_run_command", "file_write",
    "editor", "use_aws", "shell"
]
```

### Demo Mode (Read-Only Friendly)
```python
auto_approve_tools = [
    "current_time", "file_read", "calculator",
    "aws_documentation_search", "terraform_documentation_search",
    "retrieve", "read_tfstate", "cloudtrail_logs", "cloudwatch_logs",
    "terraform_plan"  # Safe for demos
]

require_approval_tools = [
    "terraform_apply", "terraform_run_command", "file_write",
    "editor", "use_aws", "shell"
]
```

## Usage Examples

### Basic Setup

1. **Configure permissions** in your main application:

```python
from permission_handlers import configure_permission_manager

# Configure for your environment
configure_permission_manager(
    auto_approve_tools=["file_read", "current_time"],
    require_approval_tools=["terraform_apply", "use_aws"]
)
```

2. **Agents automatically use the permission system** (already integrated):

```python
from agents import DetectAgent, OrchestrationAgent

# Agents are automatically configured with permission handlers
detect_agent = DetectAgent(model)
orchestration_agent = OrchestrationAgent(model)
```

3. **Run your system** - permissions will be enforced automatically.

### User Interaction Flow

When an agent tries to use a restricted tool, the user sees:

```
============================================================
🚨  PERMISSION REQUEST - DetectAgent  🚨
============================================================
Agent: DetectAgent
Function: use_aws

Parameters:
{
  "service": "ec2",
  "operation": "describe_instances",
  "region": "us-west-2"
}
============================================================
⚠️  This action may modify your infrastructure or files!
Please review the parameters carefully before approving.
============================================================
🔐 Do you approve this request? (yes/no/always/never): 
```

### Response Options

- **`yes`** or **`y`**: Approve this single request
- **`no`** or **`n`**: Deny this request  
- **`always`** or **`a`**: Approve and auto-approve this tool going forward
- **`never`**: Deny and always deny this tool going forward

### Environment-Specific Configuration

Use the provided configuration examples:

```bash
# Run with development permissions
python permission_config_example.py

# Or import specific configurations
from permission_config_example import configure_development_permissions
configure_development_permissions()
```

## Logging and Monitoring

The system provides comprehensive logging of all activities:

```
🧠 REASONING: Agent is analyzing the drift detection results...
🔧 REQUESTING TOOL: use_aws
📥 TOOL INPUT: {"service": "ec2", "operation": "describe_instances"}
✅ TOOL APPROVED: use_aws execution authorized
📤 TOOL RESULT: [{"InstanceId": "i-1234567890abcdef0", ...}]
```

### Log Levels

- **INFO**: Tool requests, approvals, agent lifecycle
- **WARNING**: Denied tools, permission violations  
- **ERROR**: System errors, callback failures
- **DEBUG**: Raw events, deltas, detailed traces

## Configuration Files

### Main System Configuration

The main system (`terraform_drift_system.py`) includes a default configuration:

```python
def configure_system_permissions():
    # Configure for development environment
    auto_approve_tools = [
        "current_time", "file_read", "calculator",
        # ... safe tools
    ]
    
    require_approval_tools = [
        "terraform_apply", "use_aws", "file_write",
        # ... potentially dangerous tools
    ]
```

### Environment-Specific Configurations

Use `permission_config_example.py` for different environments:

```python
# Development
configure_development_permissions()

# Production  
configure_production_permissions()

# Demo
configure_demo_permissions()

# Security Audit
configure_security_audit_permissions()
```

## Integration with Existing Agents

All agents in the system are already configured with permission handlers:

### OrchestrationAgent
```python
# Already includes:
callback_handler=create_agent_callback_handler("OrchestrationAgent")
```

### DetectAgent
```python
# Already includes:
callback_handler=create_agent_callback_handler("DetectAgent")
```

### DriftAnalyzerAgent
```python
# Already includes:
callback_handler=create_agent_callback_handler("DriftAnalyzerAgent")
```

### RemediateAgent
```python
# Already includes:
callback_handler=create_agent_callback_handler("RemediateAgent")
```

## Security Considerations

### Tool Classification

**Safe Tools** (typically auto-approved):
- `current_time`, `calculator`
- `aws_documentation_search`, `terraform_documentation_search`
- `file_read` (in most environments)
- `terraform_plan` (planning is usually safe)

**Dangerous Tools** (require approval):
- `terraform_apply` (modifies infrastructure)
- `use_aws` (can modify AWS resources)
- `file_write`, `editor` (modify local files)
- `shell` (execute system commands)

**Context-Dependent Tools**:
- `terraform_run_command` (depends on the command)
- `read_tfstate` (may expose sensitive information)

### Best Practices

1. **Start Restrictive**: Begin with production-level permissions and relax as needed
2. **Review Regularly**: Check permission logs for patterns
3. **Environment Separation**: Use different permission levels for dev/staging/prod
4. **Monitor Always**: Keep logging enabled to track all agent activities
5. **Test Permissions**: Verify permission settings before deployment

## Troubleshooting

### Common Issues

1. **Permission Denied Loop**: Tool keeps asking for permission
   - Check if tool is in `require_approval_tools` list
   - Use `always` response to auto-approve going forward

2. **Agent Blocked**: Agent can't complete task
   - Review which tools the agent needs
   - Add necessary tools to `auto_approve_tools` if safe

3. **Too Many Prompts**: Getting overwhelmed with permission requests
   - Use environment-specific configurations
   - Add commonly used safe tools to auto-approve list

### Debug Information

Get current permission status:

```python
from permission_handlers import get_permission_status

status = get_permission_status()
print(f"Auto-approved: {status['auto_approve_tools']}")
print(f"Require approval: {status['require_approval_tools']}")
```

Reset permissions:

```python
from permission_handlers import reset_permission_manager
reset_permission_manager()
```

## Advanced Usage

### Custom Permission Policies

Create custom permission policies for specific use cases:

```python
from permission_handlers import PermissionManager, configure_permission_manager

def configure_custom_permissions():
    # Custom policy for specific workflow
    auto_approve_tools = ["custom_tool_1", "custom_tool_2"]
    require_approval_tools = ["dangerous_custom_tool"]
    
    configure_permission_manager(
        auto_approve_tools=auto_approve_tools,
        require_approval_tools=require_approval_tools
    )
```

### Runtime Permission Changes

Modify permissions during system operation:

```python
from permission_handlers import permission_manager

# Add tool to auto-approve list
permission_manager.auto_approve_tools.append("new_safe_tool")

# Add tool to require approval list  
permission_manager.require_approval_tools.append("new_dangerous_tool")
```

### Integration with External Systems

The permission system can be extended to integrate with external authorization systems, audit logs, or monitoring platforms by customizing the callback handlers.

## Summary

The permission-based callback handler system provides comprehensive security controls for your multi-agent Terraform drift detection system. It ensures that:

- ✅ All agent actions are logged and monitored
- ✅ Dangerous operations require explicit user approval
- ✅ Different security levels can be configured for different environments
- ✅ Users have full visibility into what agents are doing
- ✅ The system can be customized for specific security requirements

This provides a robust foundation for secure automation of infrastructure drift detection and remediation while maintaining human oversight of critical operations. 