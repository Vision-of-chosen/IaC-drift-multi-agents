# Terraform Drift Detection & Remediation System

A comprehensive multi-agent system that automatically detects, analyzes, and remediates Terraform infrastructure drift using AWS Bedrock-powered Claude 3.5 Sonnet agents, accessed through both a REST API and terminal-based chat interface.

## 🚀 System Overview

This system uses specialized AI agents to detect, analyze, and fix discrepancies between your Terraform configurations and actual cloud infrastructure. It provides both real-time detection and notification capabilities, supporting multi-user environments with isolated credentials and sessions.

[![Terraform Drift System](https://img.shields.io/badge/Terraform-Drift%20Detection-blue)](https://github.com/yourusername/terraform-drift-system)
[![API Status](https://img.shields.io/badge/API-Online-brightgreen)](https://destroydrift.raiijino.buzz)

## 🏗️ Architecture Overview

The system implements a multi-agent architecture with shared memory for agent communication:

![System Architecture](docs/architecture.png)

> **Note**: The architecture diagram can be generated from the mermaid code in `docs/architecture_diagram.md`

### Agent Ecosystem

1. **OrchestrationAgent** (Central Coordinator)
   - Routes user requests to specialized agents
   - Coordinates the overall workflow
   - Manages system state and multi-agent interactions
   - Acts as the primary interface between user and specialized agents

2. **DetectAgent** (Drift Detection Specialist)
   - Parses Terraform state files
   - Queries actual AWS infrastructure
   - Identifies discrepancies between expected and actual state
   - Generates comprehensive drift detection reports

3. **DriftAnalyzerAgent** (Analysis & Assessment)
   - Analyzes severity and impact of detected drift
   - Categorizes drift types (security, performance, compliance)
   - Provides risk assessments
   - Recommends remediation strategies

4. **RemediateAgent** (Automated Remediation)
   - Generates corrected Terraform configurations
   - Creates/updates .tf files with proper configurations
   - Executes Terraform plans and applies changes
   - Validates successful remediation

5. **ReportAgent** (Reporting Specialist)
   - Generates structured JSON reports of drift findings
   - Formats analysis for human readability
   - Provides standardized output for integration with other systems
   - Creates session-specific reports with detailed drift information

6. **NotificationAgent** (Alerting & Monitoring)
   - Sets up AWS EventBridge and SNS notifications
   - Monitors real-time infrastructure changes
   - Sends email alerts on detected drift
   - Creates notification rules for proactive drift detection

## ⚙️ Key Features

- **Multi-Agent Architecture**: Specialized agents for different aspects of drift management
- **API-First Design**: Complete REST API for integration with other systems
- **Session Isolation**: Supports multiple users with isolated sessions and credentials
- **Real-time Notifications**: Email alerts for infrastructure changes
- **Comprehensive Reporting**: Detailed drift reports in standardized JSON format
- **Interactive Chat**: Terminal and API-based chat interfaces for natural language interaction
- **AWS Integration**: Deep integration with AWS services for accurate detection
- **Security-First**: Proper credential management and session isolation

## 🔌 API Endpoints

The system provides a comprehensive API for integration:

### Core Endpoints

- `POST /chat` - Chat with the system (main interaction point)
- `POST /parallel-agents` - Execute multiple agents in parallel
- `POST /start-session` - Start a new isolated session
- `GET /status` - Get current session status
- `GET /shared-memory` - View shared memory contents
- `GET /system-status` - Check overall system health

### Report & Analysis Endpoints

- `GET /report` - Get existing or generate new drift report
- `POST /generate-report` - Generate fresh drift report
- `GET /conversation` - Retrieve conversation history
- `GET /journal` - Get daily activity journal

### AWS Integration Endpoints

- `POST /set-aws-credentials` - Set AWS credentials for a session
- `GET /aws-credentials-status` - Check AWS credential status
- `POST /upload-terraform` - Upload and process Terraform files
- `GET /terraform-status` - Check Terraform file status
- `GET /aws-resources/{user_id}` - List AWS resources for a user

### Notification Endpoints

- `POST /setup-notifications` - Configure drift notification system
- `GET /notification-status` - Check notification system status
- `POST /run-notification-check` - Send test notification

## 🔧 Session Management

The system supports multi-user operation through session management:

- **Session IDs**: Each interaction gets a unique session ID
- **X-Session-ID Header**: For maintaining continuity across API calls
- **Isolated Memory**: Each session has private memory space
- **User-Specific AWS Credentials**: Credentials are tied to user and session

## 🚀 Getting Started

### Prerequisites

1. **AWS Credentials**:
   ```bash
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export AWS_REGION=ap-southeast-2
   ```

2. **Python Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Terraform CLI**:
   ```bash
   # macOS
   brew install terraform
   
   # Ubuntu/Debian
   sudo apt-get install terraform
   ```

### Quick Start

1. **Launch the API server**:
   ```bash
   python api.py
   ```

2. **For terminal interface**:
   ```bash
   python chat_interface.py
   ```

3. **Access the API documentation**:
   ```
   http://localhost:8000/docs
   ```

4. **Set up your Terraform directory**:
   - Place your Terraform files in `./terraform/`
   - Or upload via the API

## 🤖 Example Usage

### Using the Chat Interface

```bash
🔧 Terraform Drift System > detect drift in s3 buckets
🤖 OrchestrationAgent: Routing to DetectAgent to scan S3 resources...
🤖 DetectAgent: Scanning S3 buckets for drift...
✅ Detected 2 drifted S3 buckets. Use 'analyze' for details.
```

### Using the API

```bash
# Start a session
curl -X POST "http://localhost:8000/start-session" -H "Content-Type: application/json"
# Response: {"session_id": "49958daf-843f-4574-8724-ff064f32d664", ...}

# Set credentials
curl -X POST "http://localhost:8000/set-aws-credentials" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: 49958daf-843f-4574-8724-ff064f32d664" \
  -d '{"aws_access_key_id": "YOUR_KEY", "aws_secret_access_key": "YOUR_SECRET", "aws_region": "ap-southeast-2"}'

# Chat with system
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: 49958daf-843f-4574-8724-ff064f32d664" \
  -d '{"message": "detect drift in my s3 buckets"}'

# Get drift report
curl -X GET "http://localhost:8000/report" \
  -H "X-Session-ID: 49958daf-843f-4574-8724-ff064f32d664"
```

## 📊 Report Format

The system generates standardized JSON reports with the following structure:

```json
[
  {
    "id": "scan-4ec010",
    "fileName": "terraform-plan",
    "scanDate": "2025-07-22T00:24:00.487670Z",
    "status": "completed",
    "totalResources": 4,
    "driftCount": 2,
    "riskLevel": "medium",
    "duration": "00:01:35",
    "createdBy": "user123",
    "createdOn": "2025-07-22T00:24:00.496013",
    "modifiedBy": "system",
    "drifts": [
      {
        "driftCode": "drift-a8f920",
        "resourceType": "aws_s3_bucket",
        "resourceName": "my-bucket",
        "riskLevel": "high",
        "beforeStateJson": "{\"encryption\":true}",
        "afterStateJson": "{\"encryption\":false}",
        "aiExplanation": "S3 bucket encryption was disabled",
        "aiAction": "1. Update Terraform to enable encryption\n2. Run terraform apply"
      }
    ]
  }
]
```

## 🛠️ Advanced Features

### Email Notifications

Set up real-time infrastructure change notifications:

```bash
curl -X POST "http://localhost:8000/setup-notifications" \
  -H "Content-Type: application/json" \
  -d '{
    "recipient_emails": ["your-email@example.com"],
    "resource_types": ["AWS::S3::Bucket", "AWS::EC2::Instance"],
    "setup_aws_config": true
  }'
```

### Multi-User Support

The system supports multiple users with isolated sessions:

```bash
# User 1
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: session1" \
  -H "X-User-ID: user1" \
  -d '{"message": "detect drift"}'

# User 2
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: session2" \
  -H "X-User-ID: user2" \
  -d '{"message": "detect drift"}'
```

## 🔒 Security Considerations

- AWS credentials are stored in isolated memory spaces per session
- No permanent storage of credentials
- Each user gets isolated AWS sessions
- Supports temporary credentials via session tokens
- Permission-based access control for sensitive operations

## 🐛 Troubleshooting

### Common Issues

1. **AWS Credential Issues**:
   ```bash
   curl -X GET "http://localhost:8000/test-aws-connection?user_id=your_user_id"
   ```

2. **Session Management**:
   ```bash
   curl -X GET "http://localhost:8000/current-session-info"
   ```

3. **API Debugging**:
   ```bash
   curl -X GET "http://localhost:8000/debug-boto3-sessions?session_id=your_session_id"
   ```

## 📚 Additional Resources

- **API Documentation**: Access Swagger UI at `/docs` endpoint
- **Architecture Document**: See `README_ARCHITECTURE.md` for detailed design
- **Sample Reports**: Explore example report files in the repository

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**⚠️ Important**: This system makes actual changes to AWS resources and Terraform configurations. Always test in non-production environments first. 
