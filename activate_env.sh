#!/bin/bash
# Activation script for IaC Drift Detection System

echo "🔧 Activating IaC Drift Detection System environment..."

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Load environment variables
if [[ -f ".env" ]]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded from .env"
else
    echo "⚠️  No .env file found. Please create one with your AWS credentials."
fi

echo "✅ Environment activated!"
echo "💡 You can now run: python terraform_drift_system.py"
echo "💡 To deactivate: deactivate"
