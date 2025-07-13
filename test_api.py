#!/usr/bin/env python3
"""
Example client for the Terraform Drift Detection API

This demonstrates how to integrate the API into a chatbot application
with proper error handling, session management, and user interface.
"""

import requests
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime

class TerraformDriftChatbot:
    """Example chatbot client for the Terraform Drift Detection API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
        self.user_id = None
        self.conversation_history = []
        
    def connect(self, user_id: str = None) -> bool:
        """Connect to the API and verify it's working"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self.user_id = user_id or f"user_{int(time.time())}"
                print(f"✅ Connected to Terraform Drift Detection API")
                print(f"👤 User ID: {self.user_id}")
                return True
            else:
                print(f"❌ Failed to connect: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def send_message(self, message: str) -> Optional[str]:
        """Send a message to the chatbot and get response"""
        if not message.strip():
            return "Please enter a message."
        
        try:
            payload = {
                "message": message,
                "user_id": self.user_id,
                "session_id": self.session_id
            }
            
            print(f"🤖 Processing: {message}")
            
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=6000  # Allow for longer processing times
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Update session ID
                if data.get('session_id'):
                    self.session_id = data['session_id']
                
                # Store conversation history
                self.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "user_message": message,
                    "bot_response": data.get('response', ''),
                    "status": data.get('status', ''),
                    "session_id": self.session_id
                })
                
                return data.get('response', 'No response received')
            
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                print(f"❌ {error_msg}")
                return error_msg
                
        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. The system might be processing a complex request."
        except requests.exceptions.ConnectionError:
            return "🔌 Connection error. Please check if the API server is running."
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get system info: {response.status_code}"}
        except Exception as e:
            return {"error": f"Error getting system info: {str(e)}"}
    
    def get_available_commands(self) -> Dict[str, str]:
        """Get available commands"""
        try:
            response = requests.get(f"{self.base_url}/commands", timeout=10)
            if response.status_code == 200:
                return response.json().get("commands", {})
            else:
                return {"error": f"Failed to get commands: {response.status_code}"}
        except Exception as e:
            return {"error": f"Error getting commands: {str(e)}"}
    
    def show_help(self):
        """Show help information"""
        print("\n" + "="*60)
        print("🔧 Terraform Drift Detection & Remediation System")
        print("="*60)
        
        commands = self.get_available_commands()
        if "error" not in commands:
            print("\n📋 Available Commands:")
            for cmd, desc in commands.items():
                print(f"  • {cmd:<10} - {desc}")
        else:
            print(f"\n❌ Could not load commands: {commands['error']}")
        
        print("\n💡 Example Usage:")
        print("  • 'detect' - Run drift detection on your infrastructure")
        print("  • 'analyze critical drift' - Analyze critical drift issues")
        print("  • 'remediate security issues' - Fix security-related drift")
        print("  • 'status' - Check current system status")
        print("  • 'help' - Show this help message")
        
        print("\n⚠️  Remember: This system makes actual changes to AWS resources!")
        print("="*60)
    
    def show_system_status(self):
        """Show current system status"""
        print("\n📊 System Status:")
        print("-" * 30)
        
        info = self.get_system_info()
        if "error" not in info:
            print(f"Status: {info.get('status', 'unknown')}")
            print(f"Terraform Directory: {info.get('terraform_dir', 'unknown')}")
            print(f"AWS Region: {info.get('aws_region', 'unknown')}")
            print(f"Active Agents: {info.get('agents_count', 0)}")
            print(f"Workflow Status: {info.get('workflow_status', 'unknown')}")
            print(f"Terraform Files: {info.get('tf_files_count', 0)} .tf files")
            print(f"State Files: {info.get('state_files_count', 0)} .tfstate files")
        else:
            print(f"❌ {info['error']}")
    
    def run_interactive_session(self):
        """Run an interactive chat session"""
        print("\n🚀 Starting Terraform Drift Detection Chatbot")
        print("Type 'help' for available commands, 'quit' to exit")
        
        if not self.connect():
            print("❌ Failed to connect to API. Please ensure the server is running.")
            return
        
        self.show_help()
        
        while True:
            try:
                user_input = input(f"\n🔧 {self.user_id} > ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if user_input.lower() == 'status':
                    self.show_system_status()
                    continue
                
                if user_input.lower() == 'history':
                    self.show_conversation_history()
                    continue
                
                # Send message to API
                response = self.send_message(user_input)
                
                if response:
                    print(f"\n🤖 System Response:")
                    print("-" * 50)
                    print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_conversation_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print("📝 No conversation history available.")
            return
        
        print("\n📝 Conversation History:")
        print("-" * 50)
        
        for i, entry in enumerate(self.conversation_history[-5:], 1):  # Show last 5 entries
            timestamp = entry['timestamp'][:19]  # Remove microseconds
            print(f"{i}. [{timestamp}]")
            print(f"   User: {entry['user_message']}")
            print(f"   Bot: {entry['bot_response'][:100]}...")
            print(f"   Status: {entry['status']}")
            print()

def main():
    """Main function to run the example client"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Terraform Drift Detection Chatbot Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--user-id", help="User ID for the session")
    parser.add_argument("--message", help="Single message to send")
    parser.add_argument("--batch-file", help="File containing batch commands")
    
    args = parser.parse_args()
    
    chatbot = TerraformDriftChatbot(args.url)
    
    if args.message:
        # Single message mode
        if chatbot.connect(args.user_id):
            response = chatbot.send_message(args.message)
            print(f"\n🤖 Response: {response}")
    
    elif args.batch_file:
        # Batch processing mode
        try:
            with open(args.batch_file, 'r') as f:
                commands = [line.strip() for line in f if line.strip()]
            
            if chatbot.connect(args.user_id):
                print(f"🔄 Processing {len(commands)} commands from {args.batch_file}")
                
                for i, cmd in enumerate(commands, 1):
                    print(f"\n[{i}/{len(commands)}] {cmd}")
                    response = chatbot.send_message(cmd)
                    print(f"Response: {response[:200]}..." if len(response) > 200 else response)
                    time.sleep(2)  # Small delay between commands
                
        except FileNotFoundError:
            print(f"❌ Batch file not found: {args.batch_file}")
        except Exception as e:
            print(f"❌ Error processing batch file: {e}")
    
    else:
        # Interactive mode
        chatbot.run_interactive_session()

if __name__ == "__main__":
    main() 