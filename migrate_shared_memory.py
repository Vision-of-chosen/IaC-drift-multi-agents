#!/usr/bin/env python3
"""
Shared Memory Migration Tool

This script helps migrate data between in-memory storage and database storage.
It can:
1. Export all in-memory data to a database
2. Import data from a database to memory
3. Backup database to a JSON file
4. Restore database from a JSON file
"""

import os
import sys
import json
import argparse
import sqlite3
from datetime import datetime
from typing import Dict, Any

# Import the SharedMemory class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_memory import SharedMemory

def export_memory_to_db(memory_path: str = None, db_path: str = "shared_memory.db"):
    """
    Export in-memory data to a SQLite database.
    
    Args:
        memory_path: Path to a JSON file containing memory data (optional)
        db_path: Path to the SQLite database file
    """
    # Create memory instance (in-memory mode)
    memory = SharedMemory(use_db=False)
    
    # Load data from JSON file if provided
    if memory_path and os.path.exists(memory_path):
        with open(memory_path, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                memory.data[key] = value
        print(f"Loaded {len(memory.data)} items from {memory_path}")
    
    # Create database instance
    db_memory = SharedMemory(use_db=True, db_path=db_path)
    
    # Copy data to database
    count = 0
    for key, value in memory.data.items():
        db_memory.set(key, value)
        count += 1
    
    print(f"Exported {count} items to database at {db_path}")
    return count

def import_db_to_memory(db_path: str = "shared_memory.db", output_path: str = "memory_export.json"):
    """
    Import data from a SQLite database to memory and save to a JSON file.
    
    Args:
        db_path: Path to the SQLite database file
        output_path: Path to save the JSON file
    """
    # Create database instance
    db_memory = SharedMemory(use_db=True, db_path=db_path)
    
    # Get all data
    data = db_memory.get_all()
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, default=str, indent=2)
    
    print(f"Imported {len(data)} items from database at {db_path}")
    print(f"Saved to {output_path}")
    return len(data)

def backup_db(db_path: str = "shared_memory.db", backup_path: str = None):
    """
    Backup a SQLite database to a JSON file.
    
    Args:
        db_path: Path to the SQLite database file
        backup_path: Path to save the backup file
    """
    if not backup_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"shared_memory_backup_{timestamp}.json"
    
    # Create database instance
    db_memory = SharedMemory(use_db=True, db_path=db_path)
    
    # Get all data
    data = db_memory.get_all()
    
    # Save to JSON file
    with open(backup_path, 'w') as f:
        json.dump(data, f, default=str, indent=2)
    
    print(f"Backed up {len(data)} items from database at {db_path}")
    print(f"Saved to {backup_path}")
    return len(data)

def restore_db(backup_path: str, db_path: str = "shared_memory.db", merge: bool = False):
    """
    Restore a SQLite database from a JSON file.
    
    Args:
        backup_path: Path to the backup file
        db_path: Path to the SQLite database file
        merge: Whether to merge with existing data or replace
    """
    if not os.path.exists(backup_path):
        print(f"Backup file {backup_path} not found")
        return 0
    
    # Load data from JSON file
    with open(backup_path, 'r') as f:
        data = json.load(f)
    
    # Create database instance
    db_memory = SharedMemory(use_db=True, db_path=db_path)
    
    # Clear database if not merging
    if not merge:
        db_memory.clear()
    
    # Copy data to database
    count = 0
    for key, value in data.items():
        db_memory.set(key, value)
        count += 1
    
    print(f"Restored {count} items to database at {db_path}")
    return count

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Shared Memory Migration Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export in-memory data to a database")
    export_parser.add_argument("--memory-path", help="Path to a JSON file containing memory data")
    export_parser.add_argument("--db-path", default="shared_memory.db", help="Path to the SQLite database file")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import data from a database to memory")
    import_parser.add_argument("--db-path", default="shared_memory.db", help="Path to the SQLite database file")
    import_parser.add_argument("--output-path", default="memory_export.json", help="Path to save the JSON file")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Backup a database to a JSON file")
    backup_parser.add_argument("--db-path", default="shared_memory.db", help="Path to the SQLite database file")
    backup_parser.add_argument("--backup-path", help="Path to save the backup file")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore a database from a JSON file")
    restore_parser.add_argument("backup_path", help="Path to the backup file")
    restore_parser.add_argument("--db-path", default="shared_memory.db", help="Path to the SQLite database file")
    restore_parser.add_argument("--merge", action="store_true", help="Merge with existing data instead of replacing")
    
    args = parser.parse_args()
    
    if args.command == "export":
        export_memory_to_db(args.memory_path, args.db_path)
    elif args.command == "import":
        import_db_to_memory(args.db_path, args.output_path)
    elif args.command == "backup":
        backup_db(args.db_path, args.backup_path)
    elif args.command == "restore":
        restore_db(args.backup_path, args.db_path, args.merge)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 