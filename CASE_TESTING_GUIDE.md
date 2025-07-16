# Terraform Drift Detection - Case Testing Guide

## Overview
The `prompts_case_testing.py` file allows you to test different specialized drift detection cases with tailored workflows, tools, and approval processes.

## Available Test Cases

### 1. General Drift Detection (Original)
- **File**: `prompts/prompts.py` 
- **Use Case**: Standard drift detection across all AWS resources
- **Best For**: General infrastructure management and learning the system

### 2. Security-Focused Drift Detection  
- **File**: `prompts/prompts_1.py`
- **Use Case**: Security-critical resources with compliance frameworks
- **Focus**: IAM policies, security groups, encryption, VPC security
- **Compliance**: CIS, NIST, SOC2, PCI-DSS, GDPR
- **Best For**: Security teams, compliance audits, security incident response

### 3. Cost Optimization Drift Detection
- **File**: `prompts/prompts_2.py` 
- **Use Case**: Cost-related drift with financial impact analysis
- **Focus**: Instance types, storage classes, reserved instances, unused resources
- **Features**: ROI calculations, rightsizing, utilization analysis
- **Best For**: FinOps teams, cost optimization initiatives, budget management

### 4. Multi-Environment Drift Detection
- **File**: `prompts/prompts_3.py`
- **Use Case**: Cross-environment consistency between dev/staging/production
- **Focus**: Environment comparison, deployment readiness, promotion workflows
- **Best For**: DevOps teams, deployment pipelines, environment management

### 5. Database Security Drift Detection
- **File**: `prompts/prompts_4.py`
- **Use Case**: Database-specific security configurations
- **Focus**: RDS, DynamoDB, Aurora encryption, access controls, backup security
- **Compliance**: GDPR, HIPAA, PCI-DSS for database security
- **Best For**: Database administrators, data protection teams, regulatory compliance

### 6. Network Security Drift Detection
- **File**: `prompts/prompts_5.py`
- **Use Case**: Network infrastructure security with zero-trust principles  
- **Focus**: VPC, security groups, NACLs, routing, network segmentation
- **Features**: Topology analysis, attack surface assessment, zero-trust compliance
- **Best For**: Network security teams, infrastructure architects, security assessments

## How to Use

### 1. Run the Testing System
```bash
python prompts_case_testing.py
```

### 2. Select Your Test Case
The system will display a menu with all available cases. Enter the number (1-6) for your desired case.

### 3. Confirm Selection
Review the case description and confirm your selection.

### 4. Start Testing
The system will:
- Load the specialized prompts
- Configure case-specific permissions  
- Initialize the chat interface with specialized agents
- Start the testing session

## Case-Specific Features

### Specialized Tools Per Case
Each case includes specialized tools relevant to its domain:

- **Security**: `security_compliance_check`, `iam_access_analyzer`
- **Cost**: `cost_explorer`, `right_sizing_advisor`, `cost_calculator`  
- **Multi-Env**: `environment_comparator`, `deployment_tracker`
- **Database**: `database_security_scanner`, `encryption_validator`
- **Network**: `network_topology_analyzer`, `security_group_analyzer`

### Approval Workflows
Each case has tailored approval requirements:

- **Security**: Security team approval for all IAM/encryption changes
- **Cost**: Budget approval for cost-impacting changes
- **Multi-Env**: Environment-specific approval workflows
- **Database**: DBA approval for database security changes
- **Network**: Network team approval for topology changes

### Output Formats
Each case generates specialized reports optimized for its use case:

- **Security**: Compliance violation reports, security incident documentation
- **Cost**: Cost impact analysis, optimization recommendations with ROI
- **Multi-Env**: Environment comparison matrices, promotion readiness
- **Database**: Data sensitivity assessments, regulatory compliance reports  
- **Network**: Network topology security, attack surface analysis

## Testing Scenarios

### Example Test Commands

1. **Security Case**: "Detect security drift in my IAM policies and security groups"
2. **Cost Case**: "Find cost optimization opportunities in my EC2 instances"  
3. **Multi-Env Case**: "Compare configurations between dev and production"
4. **Database Case**: "Check encryption and access controls on my RDS instances"
5. **Network Case**: "Analyze network security and segmentation in my VPC"

## Tips for Effective Testing

1. **Start with General Case**: If new to the system, begin with Case 1 to understand basic workflows
2. **Use Appropriate Cases**: Match the case to your specific testing needs
3. **Review Permissions**: Each case has different permission requirements - review before testing
4. **Test Incrementally**: Start with read-only operations before testing remediation
5. **Document Results**: Each case generates detailed reports - save them for review

## Integration with Existing System

- The testing system uses the same infrastructure as `terraform_drift_system.py`
- All existing tools and permissions are maintained
- Shared memory and agent coordination remain the same
- Only prompts and workflows are specialized per case

## Testing the Integration

### Verify Prompt Integration
Before using the case testing system, you can verify the integration works correctly:

```bash
python test_prompts_integration.py
```

This test script will:
- ✅ Verify all prompt files can be loaded
- ✅ Test that agents accept custom prompt classes
- ✅ Confirm the chat interface passes prompts correctly

### Integration Status
The system has been modified to support dynamic prompt loading:

- **✅ All Agents Updated**: OrchestrationAgent, DetectAgent, DriftAnalyzerAgent, RemediateAgent
- **✅ Chat Interface Updated**: TerraformDriftChatInterface accepts `prompts_class` parameter
- **✅ Case Testing System**: CaseTestingChatInterface properly passes selected prompts
- **✅ Backward Compatibility**: Original system still works with default prompts

## Troubleshooting

- **Import Errors**: Ensure all prompt files (prompts_1.py through prompts_5.py) exist
- **Permission Issues**: Check case-specific permission configurations
- **Tool Errors**: Some specialized tools may be simulated - check tool availability
- **Prompt Not Applied**: Run `python test_prompts_integration.py` to verify integration
- **Agent Initialization Errors**: Check that all agents in `/agents` folder have been updated

### Debug Information
When running the case testing system, you'll see confirmation of which prompts are being used:
```
🚀 Terraform Drift Detection & Remediation System Initialized
📋 Using Specialized Prompts: SecurityDriftPrompts
============================================================
```

This testing system allows you to explore different specialized workflows and understand how each case optimizes for specific infrastructure management scenarios. 