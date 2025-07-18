# Terraform Drift Detection & Remediation System

An advanced multi-agent system powered by AWS Bedrock Claude that automatica## 🔒 Security Features

1. **AWS Security**
   - IAM role integration
   - Secure credential handling
   - Resource tagging

2. **API Security**
   - Request validation
   - Rate limiting
   - Session management

## � Monitoring & Loggingzes, and remediates infrastructure drift in AWS environments. The system provides a FastAPI-based web interface for step-by-step infrastructure management.

## 🌟 Key Featuresraform Drift Detection & Remediation System

An advanced multi-agent system powered by AWS Bedrock Claude that automatically detects, analyzes, and remediates infrastructure drift in AWS environments. The system provides a FastAPI-based web interface for step-by-step infrastructure management.

## � Key Features

- **Smart Multi-Agent Architecture**
  - Orchestration Agent coordinates the workflow
  - Detection Agent identifies infrastructure drift
  - Analyzer Agent assesses impact and severity
  - Remediation Agent implements fixes
  - Report Agent generates detailed documentation

- **API-First Design**
  - RESTful endpoints for each workflow step
  - Session management for workflow state
  - Real-time progress updates
  - Background task processing

- **AWS Integration**
  - AWS Bedrock Claude for intelligent analysis
  - Terraform state management
  - AWS infrastructure querying
  - Secure remediation execution

## 🚀 Quick Start

1. **Prerequisites**
   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Configure AWS credentials
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export AWS_REGION=your_region
   ```

2. **Start the API Server**
   ```bash
   # Run the FastAPI server
   uvicorn api:app --reload
   ```

3. **Access the API**
   - API documentation: http://localhost:8000/docs
   - Swagger UI: http://localhost:8000/redoc

## � API Endpoints

The system exposes the following RESTful endpoints:

### Detection Endpoints
- `POST /detect` - Start drift detection process
  ```json
  {
    "terraform_dir": "string",
    "resource_types": ["string"]
  }
  ```

### Analysis Endpoints
- `GET /status` - Check current system status
- `POST /analyze` - Analyze detected drift
  ```json
  {
    "severity_level": "string",
    "resource_ids": ["string"]
  }
  ```

### Remediation Endpoints
- `POST /remediate` - Execute drift remediation
  ```json
  {
    "resource_ids": ["string"],
    "auto_approve": boolean
  }
  ```

### Reporting Endpoints
- `GET /report` - Generate detailed reports
- `GET /report/{report_id}` - Get specific report

## 🏗️ Project Structure

```
IaC-drift-multi-agents/
├── api.py                    # FastAPI application
├── config.py                # Configuration settings
├── shared_memory.py         # Shared state management
├── agents/                  # AI agents
│   ├── orchestration_agent.py
│   ├── detect_agent.py
│   ├── drift_analyzer_agent.py
│   ├── remediate_agent.py
│   └── report_agent.py
├── useful_tools/           # Utility functions
│   ├── terraform_tools.py
│   ├── aws_documentation.py
│   └── terraform_mcp_tool.py
└── terraform/              # Infrastructure
    ├── main.tf
    └── variables.tf
```

## ⚙️ Configuration

### Environment Variables
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=your_region

# Bedrock Configuration
BEDROCK_MODEL_ID=anthropic.claude-v2
BEDROCK_REGION=us-east-1

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

## 🔒 Security Features

1. **Permission Management**
   - Role-based access control
   - Resource-level permissions
   - Audit logging

2. **AWS Security**
   - IAM role integration
   - Secure credential handling
   - Resource tagging

3. **API Security**
   - Request validation
   - Rate limiting
   - Session management

## � Monitoring & Logging

### System Monitoring
- API endpoint metrics
- Agent performance tracking
- Resource usage statistics

### Logging
- Request/response logging
- Error tracking
- Audit trail

### Status Tracking
- Background task status
- Agent operation progress
- Drift detection results

## � Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with:
   - Detailed description
   - Test cases
   - Documentation updates

## 📚 Documentation

- [Architecture Details](README_ARCHITECTURE.md)
- [System Overview](drift_system_README.md)
- API Documentation (available at /docs endpoint)

## 🆘 Support

For issues and feature requests:
1. Check existing GitHub issues
2. Review the documentation
3. Create a new issue with detailed information

## 🤝 Contributing

This system is built on the strand-agents framework. For contributions:

1. Fork the repository
2. Create feature branches
3. Test thoroughly with non-production infrastructure
4. Submit pull requests with detailed descriptions

## 📜 License

This project follows the same license as the strand-agents framework.

---

**⚠️ Important Note**: Always test in non-production environments before applying changes to production infrastructure. This system makes actual changes to AWS resources and Terraform configurations. 