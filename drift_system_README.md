# Terraform Drift Detection & Remediation System

A multi-agent orchestration system that automatically detects, analyzes, and remediates Terraform infrastructure drift through terminal-based chat interaction using the strand-agents framework.

## Architecture

The system uses a modular architecture with 4 specialized agents:

- **OrchestrationAgent**: Central coordinator receiving user requests
- **DetectAgent**: Infrastructure drift detection specialist  
- **DriftAnalyzerAgent**: Analysis & assessment specialist
- **RemediateAgent**: Automated remediation specialist

## Features

- ✅ Shared memory across all agents for seamless collaboration
- ✅ Terminal-based chat interface
- ✅ AWS operations via use_aws tool
- ✅ File management via file_read/file_write tools
- ✅ Bedrock Claude 3.5 Sonnet model integration
- ✅ Modular architecture for easy maintenance

## File Structure

```
.
├── terraform_drift_system.py      # Main entry point
├── config.py                      # System configuration
├── prompts.py                     # Centralized system prompts
├── shared_memory.py               # Shared memory implementation
├── chat_interface.py              # Terminal interface
├── agents/                        # Agent modules
│   ├── __init__.py
│   ├── orchestration_agent.py
│   ├── detect_agent.py
│   ├── drift_analyzer_agent.py
│   └── remediate_agent.py
├── demo_drift_creation.py         # Demo script for creating drift
└── terraform/                     # Terraform configuration directory
    ├── main.tf
    ├── outputs.tf
    └── variables.tf
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure AWS Credentials**:
   ```bash
   aws configure
   ```

3. **Run the System**:
   ```bash
   python terraform_drift_system.py
   ```

4. **Available Commands**:
   - `detect` - Run drift detection
   - `analyze` - Analyze detected drift
   - `remediate` - Apply drift remediation
   - `status` - Check system status
   - `memory` - View shared memory
   - `help` - Show help
   - `exit` - Exit the system

## System Components

### 1. Main Entry Point (`terraform_drift_system.py`)
- Initializes the system
- Sets up logging
- Launches the chat interface

### 2. Configuration (`config.py`)
- Centralizes all system constants
- Bedrock model configuration
- Terraform directory paths
- Agent workflow definitions

### 3. Prompts (`prompts.py`)
- Contains all agent system prompts
- Centralized prompt management
- Easy prompt customization

### 4. Shared Memory (`shared_memory.py`)
- Cross-agent data sharing
- Workflow state management
- Result persistence

### 5. Chat Interface (`chat_interface.py`)
- Terminal-based user interaction
- Command processing
- System status display

### 6. Agents (`agents/`)
Each agent is in its own module:
- **OrchestrationAgent**: Coordinates workflow
- **DetectAgent**: Detects infrastructure drift
- **DriftAnalyzerAgent**: Analyzes drift impact
- **RemediateAgent**: Applies fixes

## Usage Examples

### Basic Drift Detection
```
🔧 Terraform Drift System > detect
```

### Analyze Specific Drift
```
🔧 Terraform Drift System > analyze high priority drift
```

### Remediate Security Issues
```
🔧 Terraform Drift System > remediate security issues only
```

### Check System Status
```
🔧 Terraform Drift System > status
```

## Configuration

Edit `config.py` to modify:
- **Bedrock Model**: Change `BEDROCK_MODEL_ID` and `BEDROCK_REGION`
- **Terraform Directory**: Update `TERRAFORM_DIR`
- **Agent Workflow**: Modify `WORKFLOW_EDGES`

## Creating Intentional Drift for Testing

Use the demo script to create intentional drift:
```bash
python demo_drift_creation.py
```

⚠️ **Warning**: Only run this in test/development environments!

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure all dependencies are installed
   - Check Python path configuration

2. **AWS Access Issues**:
   - Verify AWS credentials are configured
   - Check IAM permissions

3. **Terraform Directory Not Found**:
   - Ensure `TERRAFORM_DIR` exists
   - Check directory permissions

### Debug Mode

Enable debug logging by modifying `config.py`:
```python
LOGGING_LEVEL = "DEBUG"
```

## Development

### Adding New Agents

1. Create new agent file in `agents/`
2. Add system prompt to `prompts.py`
3. Update `agents/__init__.py`
4. Configure workflow in `config.py`

### Modifying Prompts

Edit `prompts.py` to customize agent behavior:
```python
class AgentPrompts:
    NEW_AGENT = """Your new agent prompt here..."""
```

### Extending Functionality

- Add new tools to agent initialization
- Extend shared memory with new keys
- Add new chat commands to the interface

## Security Considerations

- ⚠️ The system makes actual changes to AWS resources
- ⚠️ Always review changes before applying
- ⚠️ Test in non-production environments first
- ⚠️ Ensure proper IAM permissions and restrictions

## License

This project is part of the AI-backend system. See the main repository for license information. 