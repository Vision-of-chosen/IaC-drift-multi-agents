# Terraform Drift Detection & Remediation System Architecture

This document explains the architecture of the Terraform Drift Detection & Remediation System, which uses a multi-agent approach to detect, analyze, and remediate infrastructure drift.

## System Architecture Overview

The system uses a hub-and-spoke architecture with four specialized AI agents:

1. **OrchestrationAgent** (Central Hub)
   - Receives user requests from the Chat Interface
   - Directly coordinates all specialized agents
   - Manages shared memory and data flow
   - Provides status updates to the user

2. **DetectAgent** (Independent Specialist)
   - Specialized in drift detection
   - Parses Terraform state files
   - Queries actual AWS infrastructure
   - Compares planned vs actual configurations
   - Reports findings to shared memory

3. **DriftAnalyzerAgent** (Independent Specialist)
   - Specialized in drift analysis
   - Assesses severity and impact of detected drift
   - Categorizes drift types
   - Generates remediation recommendations
   - Reports analysis to shared memory

4. **RemediateAgent** (Independent Specialist)
   - Specialized in drift remediation
   - Generates corrected Terraform configurations
   - Creates and updates .tf files
   - Executes Terraform plans and applies changes
   - Reports remediation results to shared memory

## Key Architectural Features

### Independent Agent Design

Each specialized agent operates independently and is directly coordinated by the Orchestration Agent. This design provides several benefits:

- **Flexibility**: Agents can be activated in any order based on user needs
- **Resilience**: Failure in one agent doesn't block the entire system
- **Parallel Processing**: Multiple agents can work simultaneously when appropriate
- **Modularity**: New agents can be added without modifying existing agent relationships

### Shared Memory Architecture

All agents communicate through a central shared memory system:

- **Data Persistence**: Information persists across agent activations
- **Context Sharing**: All agents have access to the same context
- **Workflow State**: System maintains state information throughout the process
- **Asynchronous Communication**: Agents don't need direct connections to share data

## Workflow Patterns

The system supports multiple workflow patterns:

1. **Sequential Processing**:
   ```
   User → Orchestration → Detect → Orchestration → Analyze → Orchestration → Remediate
   ```

2. **Direct Access**:
   ```
   User → Orchestration → [Detect|Analyze|Remediate]
   ```

3. **Partial Workflows**:
   ```
   User → Orchestration → Detect → Orchestration → Analyze
   ```

## Agent Communication

Agents communicate through two primary mechanisms:

1. **Direct Coordination**: The Orchestration Agent directly activates specialized agents with specific instructions.

2. **Shared Memory**: Agents read and write to shared memory to exchange information:
   - DetectAgent stores detection results in `drift_detection_results`
   - DriftAnalyzerAgent stores analysis in `drift_analysis_results`
   - RemediateAgent stores remediation outcomes in `remediation_results`

## System Diagram

The system architecture follows this pattern:

```
User ↔ Chat Interface ↔ Orchestration Agent
                           ↓
                      Shared Memory
                       ↙    ↓    ↘
                DetectAgent  AnalyzerAgent  RemediateAgent
                     ↓           ↓              ↓
                AWS Resources  Documentation  Terraform Files
```

## Benefits of This Architecture

- **Flexibility**: Supports various workflow patterns based on user needs
- **Scalability**: Easy to add new specialized agents
- **Resilience**: No single point of failure in the agent network
- **Maintainability**: Clear separation of concerns between agents
- **Extensibility**: Easy to enhance individual agent capabilities 