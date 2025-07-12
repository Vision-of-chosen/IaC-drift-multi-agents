# Terraform Drift Detection & Remediation System

A multi-agent orchestration system that automatically detects, analyzes, and remediates Terraform infrastructure drift through terminal-based chat interaction using the strand-agents framework.

## 🏗️ Architecture Overview

The system uses four specialized AI agents working together through a shared memory architecture:

### Agent Ecosystem

1. **OrchestrationAgent** (Central Coordinator)
   - Receives user requests from ChatBot UI
   - Routes requests to appropriate specialized agents
   - Manages workflow between DetectAgent, DriftAnalyzerAgent, and RemediateAgent
   - Coordinates shared memory and data flow
   - Provides status updates to user

2. **DetectAgent** (Drift Detection Specialist)
   - Parses current Terraform state files
   - Queries actual AWS infrastructure using use-aws tool
   - Compares planned vs actual resource configurations
   - Generates drift detection reports
   - Stores findings in shared memory

3. **DriftAnalyzerAgent** (Analysis & Assessment)
   - Analyzes drift severity and impact assessment
   - Categorizes drift types (configuration, resource state, security implications)
   - Generates remediation recommendations
   - Provides analysis summary to shared memory

4. **RemediateAgent** (Automated Remediation)
   - Generates corrected Terraform configurations
   - Creates new .tf files using strand-tools file writer
   - Generates and reviews Terraform plans
   - Applies approved changes using AWS tools
   - Updates shared memory with remediation results

## 🚀 Getting Started

### Prerequisites

1. **AWS Credentials**: Ensure your AWS credentials are configured
   ```bash
   aws configure
   # or use environment variables
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export AWS_REGION=ap-southeast-2
   ```

2. **Strands Agent SDK**: Install the required components
   ```bash
   # Install from the AI-backend directory
   pip install -e ./sdk-python
   pip install -e ./tools
   
   # Or install additional dependencies
   pip install -r requirements.txt
   ```

3. **Terraform**: Install Terraform CLI
   ```bash
   # macOS
   brew install terraform
   
   # Ubuntu/Debian
   sudo apt-get install terraform
   
   # Or download from: https://www.terraform.io/downloads.html
   ```

### Installation

1. **Clone and setup**:
   ```bash
   cd AI-backend
   python terraform_drift_system.py
   ```

2. **Configure your Terraform directory**:
   - Place your Terraform files in `./terraform/`
   - Ensure you have `.tf` and `.tfstate` files
   - The system will automatically detect and analyze these files

## 🎯 Usage

### Interactive Terminal Interface

Start the system:
```bash
python terraform_drift_system.py
```

### Available Commands

- `detect` - Run drift detection process
- `analyze` - Analyze detected drift 
- `remediate` - Apply drift remediation
- `status` - Check system status
- `memory` - View shared memory
- `help` - Show help information
- `exit` - Exit the system

### Example Usage Scenarios

1. **Basic Drift Detection**:
   ```
   > detect
   ```

2. **Targeted Analysis**:
   ```
   > analyze high priority security drift
   ```

3. **Selective Remediation**:
   ```
   > remediate only critical issues
   ```

4. **Full Workflow**:
   ```
   > detect and analyze all infrastructure drift
   > remediate approved changes only
   ```

## 🔧 Configuration

### Model Configuration

The system uses **Anthropic Claude 3.5 Sonnet** via AWS Bedrock:
- **Model**: `apac.anthropic.claude-3-5-sonnet-20240620-v1:0`
- **Region**: `ap-southeast-2`
- **Temperature**: `0.1` (for consistent, deterministic responses)

### Directory Structure

```
AI-backend/
├── terraform_drift_system.py    # Main system file
├── requirements.txt              # Python dependencies
├── README.md                    # This file
└── terraform/                  # Your Terraform files
    ├── main.tf                 # Your infrastructure definitions
    ├── variables.tf            # Variable definitions
    ├── outputs.tf              # Output definitions
    └── terraform.tfstate       # Current state file
```

## 🛠️ System Workflow

### 1. User Interaction Flow
```
User Request → ChatBot UI → Orchestration Agent → Specialized Agents
```

### 2. Agent Collaboration Pattern
```
DetectAgent → [Shared Memory] → DriftAnalyzerAgent → [Shared Memory] → RemediateAgent
```

### 3. Tool Integration

- **strand-tools**: AWS operations, file writing, state parsing
- **use_aws**: Advanced AWS analysis, resource querying
- **file_read/file_write**: File system operations
- **editor**: Code modification and generation

## 🔍 Key Features

### Shared Memory Architecture
- **Cross-Agent Data Sharing**: All agents access common memory space
- **State Persistence**: Maintain workflow state across agent interactions
- **Context Preservation**: Retain conversation context and intermediate results

### Tool Ecosystem Integration
- **strand-tools**: Primary toolkit for AWS operations and file management
- **AWS Integration**: Direct access to AWS APIs for resource state querying
- **File System Access**: Direct access to ./terraform directory structure

### Automation Capabilities
- **Automated Drift Detection**: Continuous monitoring of infrastructure state
- **Intelligent Analysis**: Context-aware drift impact assessment
- **Remediation Planning**: Automated generation of corrective Terraform code
- **Safe Execution**: Controlled application of infrastructure changes

## 📊 Expected Outputs

1. **Drift Detection Reports**: Detailed infrastructure state comparisons
2. **Analysis Summaries**: Risk assessments and remediation priorities
3. **Remediation Plans**: Generated Terraform configurations
4. **Execution Reports**: Applied changes and their outcomes
5. **Workflow Status**: Real-time progress updates through terminal chat

## 🔒 Security & Safety

### Safety Features
- **User Confirmation**: Required for mutative operations
- **Plan Review**: All changes reviewed before application
- **Rollback Procedures**: Automatic backup and restore capabilities
- **Error Handling**: Comprehensive error catching and reporting

### Best Practices
- Always review generated Terraform plans before applying
- Test in non-production environments first
- Keep backups of your original Terraform state files
- Monitor AWS costs during remediation operations

## 🧪 Example Terraform Setup

Create a simple test setup in `./terraform/`:

```hcl
# terraform/main.tf
provider "aws" {
  region = "ap-southeast-2"
}

resource "aws_s3_bucket" "test_bucket" {
  bucket = "my-terraform-test-bucket-${random_string.suffix.result}"
}

resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# terraform/outputs.tf
output "bucket_name" {
  value = aws_s3_bucket.test_bucket.id
}
```

Initialize and apply:
```bash
cd terraform
terraform init
terraform plan
terraform apply
```

## 🎭 Success Metrics

- **Accurate drift detection** across all AWS resources
- **Comprehensive analysis** of drift implications
- **Safe and effective remediation** execution
- **Seamless agent collaboration** through shared memory
- **Intuitive terminal-based user experience**

## 🐛 Troubleshooting

### Common Issues

1. **AWS Credentials**:
   ```bash
   aws sts get-caller-identity  # Test AWS access
   ```

2. **Terraform State Issues**:
   ```bash
   terraform refresh  # Sync state with actual resources
   ```

3. **Agent Communication**:
   - Check shared memory contents with `memory` command
   - Use `status` command to verify system health

### Debug Mode

Enable detailed logging:
```bash
export STRANDS_LOG_LEVEL=DEBUG
python terraform_drift_system.py
```

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