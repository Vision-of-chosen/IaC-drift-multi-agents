#!/bin/bash

# Terraform Drift Detection & Remediation System Setup Script
# This script helps you quickly set up and test the drift detection system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    Terraform Drift Detection System                 ║"
echo "║                          Setup Script                               ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

print_status "Starting setup process..."

# Check Python version
print_status "Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 is required but not found. Please install Python 3.8+"
    exit 1
fi

# Check pip
print_status "Checking pip installation..."
if command_exists pip3; then
    print_success "pip3 found"
elif command_exists pip; then
    print_success "pip found"
else
    print_error "pip is required but not found. Please install pip"
    exit 1
fi

# Check AWS CLI
print_status "Checking AWS CLI installation..."
if command_exists aws; then
    AWS_VERSION=$(aws --version 2>&1 | cut -d/ -f2 | cut -d' ' -f1)
    print_success "AWS CLI $AWS_VERSION found"
    
    # Check AWS credentials
    print_status "Checking AWS credentials..."
    if aws sts get-caller-identity >/dev/null 2>&1; then
        ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
        REGION=$(aws configure get region || echo "ap-southeast-2")
        print_success "AWS credentials configured for account $ACCOUNT_ID in region $REGION"
    else
        print_warning "AWS credentials not configured. Please run 'aws configure'"
        echo "You can set up credentials later with:"
        echo "  aws configure"
        echo "Or export environment variables:"
        echo "  export AWS_ACCESS_KEY_ID=your_key"
        echo "  export AWS_SECRET_ACCESS_KEY=your_secret"
        echo "  export AWS_REGION=ap-southeast-2"
    fi
else
    print_warning "AWS CLI not found. Installing via pip..."
    pip3 install awscli
fi

# Check Terraform
print_status "Checking Terraform installation..."
if command_exists terraform; then
    TERRAFORM_VERSION=$(terraform version | head -n1 | cut -d' ' -f2)
    print_success "Terraform $TERRAFORM_VERSION found"
else
    print_warning "Terraform not found. Please install Terraform:"
    echo "  https://www.terraform.io/downloads.html"
    echo "Or use package manager:"
    echo "  # macOS: brew install terraform"
    echo "  # Ubuntu: sudo apt-get install terraform"
fi

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    print_success "Python dependencies installed"
else
    print_warning "requirements.txt not found, installing basic dependencies..."
    pip3 install boto3 botocore requests rich pydantic
fi

# Install Strands Agent SDK
print_status "Installing Strands Agent SDK..."
if [ -d "sdk-python" ]; then
    pip3 install -e ./sdk-python
    print_success "Strands Agent SDK installed"
else
    print_error "sdk-python directory not found. Please ensure you're in the AI-backend directory"
    exit 1
fi

# Install Strands Tools
print_status "Installing Strands Tools..."
if [ -d "tools" ]; then
    pip3 install -e ./tools
    print_success "Strands Tools installed"
else
    print_error "tools directory not found. Please ensure you're in the AI-backend directory"
    exit 1
fi

# Create terraform directory if it doesn't exist
print_status "Setting up Terraform directory..."
if [ ! -d "terraform" ]; then
    mkdir -p terraform
    print_success "Created terraform directory"
fi

# Initialize Terraform if terraform files exist
if [ -f "terraform/main.tf" ]; then
    print_status "Initializing Terraform configuration..."
    cd terraform
    if terraform init; then
        print_success "Terraform initialized successfully"
        
        # Validate configuration
        if terraform validate; then
            print_success "Terraform configuration is valid"
        else
            print_warning "Terraform configuration validation failed"
        fi
    else
        print_warning "Terraform initialization failed"
    fi
    cd ..
fi

# Create example .env file
print_status "Creating example environment file..."
cat > .env.example << EOF
# AWS Configuration
AWS_REGION=ap-southeast-2
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here

# Optional: AWS Profile
# AWS_PROFILE=your_profile_name

# Strands Configuration
STRANDS_LOG_LEVEL=INFO

# Optional: Bypass tool consent for automation
# BYPASS_TOOL_CONSENT=false
EOF
print_success "Created .env.example file"

# Final status
echo ""
print_success "Setup completed successfully!"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Configure AWS credentials (if not already done):"
echo "   aws configure"
echo ""
echo "2. Copy and customize environment file:"
echo "   cp .env.example .env"
echo "   # Edit .env with your settings"
echo ""
echo "3. Test the Terraform configuration:"
echo "   cd terraform"
echo "   terraform plan"
echo ""
echo "4. Run the drift detection system:"
echo "   python terraform_drift_system.py"
echo ""
echo -e "${GREEN}🚀 Your Terraform Drift Detection System is ready!${NC}"
echo ""
echo "For more information, see README.md" 