#!/usr/bin/env python3
"""
Simple runner script for Terraform Drift Detection Case Testing

This script provides a quick way to run the case testing system with
additional validation and setup checks.
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if required files exist"""
    required_files = [
        "prompts/prompts.py",
        "prompts/prompts_1.py", 
        "prompts/prompts_2.py",
        "prompts/prompts_3.py",
        "prompts/prompts_4.py",
        "prompts/prompts_5.py",
        "prompts/__init__.py",
        "prompts_case_testing.py",
        "chat_interface.py",
        "permission_handlers.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present before running.")
        return False
    
    print("✅ All required files found!")
    return True

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible!")
    return True

def main():
    """Main runner function"""
    print("🚀 Terraform Drift Detection - Case Testing Runner")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check required files
    if not check_requirements():
        sys.exit(1)
    
    print("\n📋 Available Test Cases:")
    print("1. General Drift Detection (Original)")
    print("2. Security-Focused Drift Detection")
    print("3. Cost Optimization Drift Detection") 
    print("4. Multi-Environment Drift Detection")
    print("5. Database Security Drift Detection")
    print("6. Network Security Drift Detection")
    
    print("\n🧪 Options:")
    print("Run 'python test_prompts_integration.py' first to verify integration")
    
    print("\n🎯 Starting Case Testing System...")
    print("-" * 60)
    
    try:
        # Run the case testing system
        subprocess.run([sys.executable, "prompts_case_testing.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running case testing system: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Testing interrupted by user. Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 