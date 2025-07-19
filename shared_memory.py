#!/usr/bin/env python3
"""
Shared Memory Module for Agent Communication

This module provides a shared memory implementation that can be used for
inter-agent communication. It supports both in-memory and database-backed storage,
as well as session-based namespacing.
"""

import os
import json
import sqlite3
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading

class SharedMemory:
    """
    A shared memory implementation for inter-agent communication.
    Supports both in-memory and database-backed storage, with session-based namespacing.
    """
    
    def __init__(self, use_db: bool = False, db_path: str = "shared_memory.db"):
        """
        Initialize the shared memory.
        
        Args:
            use_db: Whether to use database storage (True) or in-memory storage (False)
            db_path: Path to the SQLite database file
        """
        self.data: Dict[str, Any] = {}
        self.use_db = use_db
        self.db_path = db_path
        self._lock = threading.Lock()
        self._current_session_id = None
        
        if use_db:
            self._init_db()
    
    def _init_db(self):
        """Initialize the database with required tables"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # Create table for shared memory with session_id column
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS shared_memory (
            key TEXT,
            session_id TEXT,
            value TEXT,
            value_type TEXT,
            timestamp TEXT,
            PRIMARY KEY (key, session_id)
        )
        ''')
        
        # Create index on session_id for faster lookups
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_session_id ON shared_memory(session_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_db_connection(self):
        """Get a connection to the SQLite database"""
        return sqlite3.connect(self.db_path)
    
    def _execute_db_query(self, query, params=None):
        """Execute a database query with proper connection management"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                return cursor
    
    def _serialize_value(self, value: Any) -> tuple:
        """
        Serialize a value for storage in the database.
        
        Returns:
            Tuple of (serialized_value, value_type)
        """
        if value is None:
            return "null", "null"
        elif isinstance(value, (str, int, float, bool)):
            return str(value), type(value).__name__
        else:
            # For complex types like dict, list, etc.
            return json.dumps(value), "json"
    
    def _deserialize_value(self, value_str: str, value_type: str) -> Any:
        """
        Deserialize a value from the database.
        
        Args:
            value_str: The serialized value
            value_type: The type of the value
            
        Returns:
            The deserialized value
        """
        if value_type == "null":
            return None
        elif value_type == "str":
            return value_str
        elif value_type == "int":
            return int(value_str)
        elif value_type == "float":
            return float(value_str)
        elif value_type == "bool":
            return value_str.lower() == "true"
        elif value_type == "json":
            return json.loads(value_str)
        else:
            # Default fallback
            return value_str
    
    def set_session(self, session_id: str) -> None:
        """
        Set the current session ID for subsequent operations.
        
        Args:
            session_id: The session ID to use
        """
        self._current_session_id = session_id
    
    def clear_session(self) -> None:
        """Clear the current session ID."""
        self._current_session_id = None
    
    def get_session(self) -> Optional[str]:
        """
        Get the current session ID.
        
        Returns:
            The current session ID or None if not set
        """
        return self._current_session_id
    
    def _get_key_with_session(self, key: str, session_id: Optional[str] = None) -> str:
        """
        Get a key with session ID prefix for in-memory storage.
        
        Args:
            key: The original key
            session_id: Optional session ID (uses current session if not provided)
            
        Returns:
            Key with session prefix if session is set, otherwise original key
        """
        session = session_id or self._current_session_id
        if session:
            return f"{session}:{key}"
        return key
    
    def set(self, key: str, value: Any, session_id: Optional[str] = None) -> None:
        """Set a value in shared memory."""
        # Use provided session_id directly instead of current session
        # This avoids race conditions with set_session
        session = session_id or self._current_session_id
        
        if self.use_db:
            with self._lock:
                # Use context manager for database connection
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    serialized_value, value_type = self._serialize_value(value)
                    timestamp = datetime.now().isoformat()
                    
                    cursor.execute(
                        "INSERT OR REPLACE INTO shared_memory (key, session_id, value, value_type, timestamp) VALUES (?, ?, ?, ?, ?)",
                        (key, session or "", serialized_value, value_type, timestamp)
                    )
                    conn.commit()
        else:
            with self._lock:
                prefixed_key = self._get_key_with_session(key, session)
                self.data[prefixed_key] = value
    
    def get(self, key: str, default: Any = None, session_id: Optional[str] = None) -> Any:
        """
        Get a value from shared memory.
        
        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist
            session_id: Optional session ID (uses current session if not provided)
            
        Returns:
            The value associated with the key, or the default value
        """
        # Use provided session_id or current session_id
        session = session_id or self._current_session_id
        
        if self.use_db:
            with self._lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                if session:
                    # Try to get session-specific value first
                    cursor.execute(
                        "SELECT value, value_type FROM shared_memory WHERE key = ? AND session_id = ?", 
                        (key, session)
                    )
                    result = cursor.fetchone()
                    
                    if not result:
                        # If not found, try global value (empty session_id)
                        cursor.execute(
                            "SELECT value, value_type FROM shared_memory WHERE key = ? AND session_id = ?", 
                            (key, "")
                        )
                        result = cursor.fetchone()
                else:
                    # No session specified, get global value
                    cursor.execute(
                        "SELECT value, value_type FROM shared_memory WHERE key = ? AND session_id = ?", 
                        (key, "")
                    )
                    result = cursor.fetchone()
                
                conn.close()
                
                if result:
                    value_str, value_type = result
                    return self._deserialize_value(value_str, value_type)
                return default
        else:
            with self._lock:
                # Try session-specific key first
                prefixed_key = self._get_key_with_session(key, session)
                if prefixed_key in self.data:
                    return self.data[prefixed_key]
                
                # If session-specific key not found and we're using a session,
                # try the global key
                if session and key in self.data:
                    return self.data[key]
                
                return default
    
    def delete(self, key: str, session_id: Optional[str] = None) -> bool:
        """
        Delete a key from shared memory.
        
        Args:
            key: The key to delete
            session_id: Optional session ID (uses current session if not provided)
            
        Returns:
            True if the key was deleted, False otherwise
        """
        # Use provided session_id or current session_id
        session = session_id or self._current_session_id
        
        if self.use_db:
            with self._lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                if session:
                    cursor.execute(
                        "DELETE FROM shared_memory WHERE key = ? AND session_id = ?", 
                        (key, session)
                    )
                else:
                    cursor.execute(
                        "DELETE FROM shared_memory WHERE key = ? AND session_id = ?", 
                        (key, "")
                    )
                
                deleted = cursor.rowcount > 0
                
                conn.commit()
                conn.close()
                
                return deleted
        else:
            with self._lock:
                prefixed_key = self._get_key_with_session(key, session)
                if prefixed_key in self.data:
                    del self.data[prefixed_key]
                    return True
                return False
    
    def keys(self, session_id: Optional[str] = None, include_global: bool = True) -> List[str]:
        """
        Get all keys in shared memory for a specific session.
        
        Args:
            session_id: Optional session ID (uses current session if not provided)
            include_global: Whether to include global keys (not session-specific)
            
        Returns:
            List of all keys for the session
        """
        # Use provided session_id or current session_id
        session = session_id or self._current_session_id
        
        if self.use_db:
            with self._lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                if session and include_global:
                    # Get both session-specific and global keys
                    cursor.execute(
                        "SELECT key FROM shared_memory WHERE session_id = ? OR session_id = ?", 
                        (session, "")
                    )
                elif session:
                    # Get only session-specific keys
                    cursor.execute(
                        "SELECT key FROM shared_memory WHERE session_id = ?", 
                        (session,)
                    )
                else:
                    # Get only global keys
                    cursor.execute(
                        "SELECT key FROM shared_memory WHERE session_id = ?", 
                        ("",)
                    )
                
                keys = [row[0] for row in cursor.fetchall()]
                
                conn.close()
                
                return keys
        else:
            with self._lock:
                if not session:
                    # Return only global keys (no session prefix)
                    return [k for k in self.data.keys() if ":" not in k]
                
                prefix = f"{session}:"
                prefix_len = len(prefix)
                
                # Get session-specific keys (remove prefix)
                session_keys = [k[prefix_len:] for k in self.data.keys() if k.startswith(prefix)]
                
                if include_global:
                    # Add global keys (no session prefix)
                    global_keys = [k for k in self.data.keys() if ":" not in k]
                    return session_keys + global_keys
                
                return session_keys
    
    def clear_session_data(self, session_id: Optional[str] = None) -> int:
        """
        Clear all data for a specific session.
        
        Args:
            session_id: Optional session ID (uses current session if not provided)
            
        Returns:
            Number of items deleted
        """
        # Use provided session_id or current session_id
        session = session_id or self._current_session_id
        
        if not session:
            return 0
        
        if self.use_db:
            with self._lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute(
                    "DELETE FROM shared_memory WHERE session_id = ?", 
                    (session,)
                )
                
                deleted = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                return deleted
        else:
            with self._lock:
                prefix = f"{session}:"
                keys_to_delete = [k for k in self.data.keys() if k.startswith(prefix)]
                
                for key in keys_to_delete:
                    del self.data[key]
                
                return len(keys_to_delete)
    
    def clear(self) -> None:
        """Clear all data from shared memory."""
        if self.use_db:
            with self._lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM shared_memory")
                
                conn.commit()
                conn.close()
        else:
            with self._lock:
                self.data.clear()
    
    def size(self, session_id: Optional[str] = None) -> int:
        """
        Get the number of items in shared memory.
        
        Args:
            session_id: Optional session ID (uses current session if not provided)
            
        Returns:
            Number of items
        """
        # Use provided session_id or current session_id
        session = session_id or self._current_session_id
        
        if self.use_db:
            with self._lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                if session:
                    cursor.execute(
                        "SELECT COUNT(*) FROM shared_memory WHERE session_id = ?", 
                        (session,)
                    )
                else:
                    cursor.execute("SELECT COUNT(*) FROM shared_memory")
                
                count = cursor.fetchone()[0]
                
                conn.close()
                
                return count
        else:
            with self._lock:
                if not session:
                    return len(self.data)
                
                prefix = f"{session}:"
                return len([k for k in self.data.keys() if k.startswith(prefix)])
    
    def items(self, session_id: Optional[str] = None, include_global: bool = True) -> List[tuple]:
        """
        Get all key-value pairs in shared memory for a specific session.
        
        Args:
            session_id: Optional session ID (uses current session if not provided)
            include_global: Whether to include global items (not session-specific)
            
        Returns:
            List of (key, value) tuples
        """
        # Use provided session_id or current session_id
        session = session_id or self._current_session_id
        
        if self.use_db:
            with self._lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                if session and include_global:
                    # Get both session-specific and global items
                    cursor.execute(
                        "SELECT key, value, value_type FROM shared_memory WHERE session_id = ? OR session_id = ?", 
                        (session, "")
                    )
                elif session:
                    # Get only session-specific items
                    cursor.execute(
                        "SELECT key, value, value_type FROM shared_memory WHERE session_id = ?", 
                        (session,)
                    )
                else:
                    # Get only global items
                    cursor.execute(
                        "SELECT key, value, value_type FROM shared_memory WHERE session_id = ?", 
                        ("",)
                    )
                
                items = []
                
                for key, value_str, value_type in cursor.fetchall():
                    value = self._deserialize_value(value_str, value_type)
                    items.append((key, value))
                
                conn.close()
                
                return items
        else:
            with self._lock:
                if not session:
                    # Return only global items (no session prefix)
                    return [(k, v) for k, v in self.data.items() if ":" not in k]
                
                prefix = f"{session}:"
                prefix_len = len(prefix)
                
                # Get session-specific items (remove prefix from keys)
                session_items = [(k[prefix_len:], v) for k, v in self.data.items() if k.startswith(prefix)]
                
                if include_global:
                    # Add global items (no session prefix)
                    global_items = [(k, v) for k, v in self.data.items() if ":" not in k]
                    return session_items + global_items
                
                return session_items
    
    def contains(self, key: str, session_id: Optional[str] = None) -> bool:
        """
        Check if a key exists in shared memory.
        
        Args:
            key: The key to check
            session_id: Optional session ID (uses current session if not provided)
            
        Returns:
            True if the key exists, False otherwise
        """
        # Use provided session_id or current session_id
        session = session_id or self._current_session_id
        
        if self.use_db:
            with self._lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                if session:
                    # Check session-specific key first
                    cursor.execute(
                        "SELECT 1 FROM shared_memory WHERE key = ? AND session_id = ?", 
                        (key, session)
                    )
                    exists = cursor.fetchone() is not None
                    
                    if not exists:
                        # If not found, check global key
                        cursor.execute(
                            "SELECT 1 FROM shared_memory WHERE key = ? AND session_id = ?", 
                            (key, "")
                        )
                        exists = cursor.fetchone() is not None
                else:
                    # Check global key
                    cursor.execute(
                        "SELECT 1 FROM shared_memory WHERE key = ? AND session_id = ?", 
                        (key, "")
                    )
                    exists = cursor.fetchone() is not None
                
                conn.close()
                
                return exists
        else:
            with self._lock:
                # Check session-specific key first
                prefixed_key = self._get_key_with_session(key, session)
                if prefixed_key in self.data:
                    return True
                
                # If session-specific key not found and we're using a session,
                # check the global key
                if session and key in self.data:
                    return True
                
                return False
    
    def get_all(self, session_id: Optional[str] = None, include_global: bool = True) -> Dict[str, Any]:
        """
        Get all data from shared memory for a specific session.
        
        Args:
            session_id: Optional session ID (uses current session if not provided)
            include_global: Whether to include global data (not session-specific)
            
        Returns:
            Dictionary of all key-value pairs
        """
        items = self.items(session_id, include_global)
        return dict(items)
    
    def get_all_sessions(self) -> List[str]:
        """
        Get all session IDs in shared memory.
        
        Returns:
            List of all session IDs
        """
        if self.use_db:
            with self._lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute("SELECT DISTINCT session_id FROM shared_memory WHERE session_id != ''")
                sessions = [row[0] for row in cursor.fetchall()]
                
                conn.close()
                
                return sessions
        else:
            with self._lock:
                # Extract session IDs from prefixed keys
                sessions = set()
                for key in self.data.keys():
                    if ":" in key:
                        session_id = key.split(":", 1)[0]
                        sessions.add(session_id)
                
                return list(sessions)

# Create a singleton instance
# By default, use in-memory storage
# To use database storage, set USE_DB_STORAGE environment variable to "true"
use_db = os.environ.get("USE_DB_STORAGE", "false").lower() == "true"
db_path = os.environ.get("DB_STORAGE_PATH", "shared_memory.db")

shared_memory = SharedMemory(use_db=use_db, db_path=db_path) 