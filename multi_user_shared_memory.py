#!/usr/bin/env python3
"""
Multi-user shared memory implementation for the Terraform Drift Detection & Remediation System.

This module provides a user-scoped shared memory system that allows different users
to have isolated memory spaces while enabling agents to share information within
each user's context.
"""

import logging
from typing import Any, Dict, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class MultiUserSharedMemory:
    """Multi-user shared memory implementation with user and conversation scoping"""
    
    def __init__(self):
        self.data: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
    
    def _get_user_key(self, user_id: str, conversation_id: Optional[int] = None) -> str:
        """Generate a scoped key for user data"""
        if conversation_id:
            return f"user_{user_id}_conv_{conversation_id}"
        return f"user_{user_id}_global"
    
    def _ensure_user_space(self, user_key: str) -> None:
        """Ensure user memory space exists"""
        if user_key not in self.data:
            self.data[user_key] = {}
    
    def set(self, user_id: str, key: str, value: Any, conversation_id: Optional[int] = None) -> None:
        """Store data in user-scoped shared memory"""
        with self._lock:
            user_key = self._get_user_key(user_id, conversation_id)
            self._ensure_user_space(user_key)
            self.data[user_key][key] = value
            logger.info(f"MultiUserSharedMemory: Set {key} for user {user_id[:8]}... (conv: {conversation_id})")
    
    def get(self, user_id: str, key: str, default: Any = None, conversation_id: Optional[int] = None) -> Any:
        """Retrieve data from user-scoped shared memory"""
        with self._lock:
            user_key = self._get_user_key(user_id, conversation_id)
            return self.data.get(user_key, {}).get(key, default)
    
    def update(self, user_id: str, updates: Dict[str, Any], conversation_id: Optional[int] = None) -> None:
        """Update multiple keys in user-scoped shared memory"""
        with self._lock:
            user_key = self._get_user_key(user_id, conversation_id)
            self._ensure_user_space(user_key)
            self.data[user_key].update(updates)
            logger.info(f"MultiUserSharedMemory: Updated {list(updates.keys())} for user {user_id[:8]}... (conv: {conversation_id})")
    
    def clear_user(self, user_id: str, conversation_id: Optional[int] = None) -> None:
        """Clear memory for a specific user or conversation"""
        with self._lock:
            user_key = self._get_user_key(user_id, conversation_id)
            if user_key in self.data:
                self.data[user_key].clear()
                logger.info(f"MultiUserSharedMemory: Cleared data for user {user_id[:8]}... (conv: {conversation_id})")
    
    def clear_all(self) -> None:
        """Clear all shared memory (admin function)"""
        with self._lock:
            self.data.clear()
            logger.info("MultiUserSharedMemory: Cleared all data")
    
    def keys(self, user_id: str, conversation_id: Optional[int] = None) -> list:
        """Get all keys for a specific user/conversation"""
        with self._lock:
            user_key = self._get_user_key(user_id, conversation_id)
            return list(self.data.get(user_key, {}).keys())
    
    def has_key(self, user_id: str, key: str, conversation_id: Optional[int] = None) -> bool:
        """Check if key exists in user-scoped shared memory"""
        with self._lock:
            user_key = self._get_user_key(user_id, conversation_id)
            return key in self.data.get(user_key, {})
    
    def delete(self, user_id: str, key: str, conversation_id: Optional[int] = None) -> bool:
        """Delete a key from user-scoped shared memory"""
        with self._lock:
            user_key = self._get_user_key(user_id, conversation_id)
            user_data = self.data.get(user_key, {})
            if key in user_data:
                del user_data[key]
                logger.info(f"MultiUserSharedMemory: Deleted {key} for user {user_id[:8]}... (conv: {conversation_id})")
                return True
            return False
    
    def size(self, user_id: str = None, conversation_id: Optional[int] = None) -> int:
        """Get the number of items in shared memory"""
        with self._lock:
            if user_id:
                user_key = self._get_user_key(user_id, conversation_id)
                return len(self.data.get(user_key, {}))
            else:
                # Return total size across all users
                return sum(len(user_data) for user_data in self.data.values())
    
    def get_user_data(self, user_id: str, conversation_id: Optional[int] = None) -> Dict[str, Any]:
        """Get all data for a specific user/conversation"""
        with self._lock:
            user_key = self._get_user_key(user_id, conversation_id)
            return self.data.get(user_key, {}).copy()
    
    def get_all_users(self) -> list:
        """Get list of all users who have data in memory"""
        with self._lock:
            users = set()
            for user_key in self.data.keys():
                if user_key.startswith("user_"):
                    # Extract user_id from key format: user_AKIAXXXXXXXXXXXXXXXX_...
                    parts = user_key.split("_")
                    if len(parts) >= 2:
                        users.add(parts[1])
            return list(users)
    
    def get_user_conversations(self, user_id: str) -> list:
        """Get list of conversation IDs for a user that have data in memory"""
        with self._lock:
            conversations = []
            user_prefix = f"user_{user_id}_conv_"
            for user_key in self.data.keys():
                if user_key.startswith(user_prefix):
                    # Extract conversation ID from key
                    conv_id = user_key.replace(user_prefix, "")
                    if conv_id.isdigit():
                        conversations.append(int(conv_id))
            return sorted(conversations)
    
    def cleanup_empty_spaces(self) -> None:
        """Remove empty user memory spaces"""
        with self._lock:
            empty_keys = [key for key, data in self.data.items() if not data]
            for key in empty_keys:
                del self.data[key]
            if empty_keys:
                logger.info(f"MultiUserSharedMemory: Cleaned up {len(empty_keys)} empty memory spaces")


class UserScopedMemoryAdapter:
    """Adapter to make user-scoped memory work like the original shared memory for agents"""
    
    def __init__(self, multi_user_memory: MultiUserSharedMemory, user_id: str, conversation_id: Optional[int] = None):
        self.multi_user_memory = multi_user_memory
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get user's data (for backward compatibility)"""
        return self.multi_user_memory.get_user_data(self.user_id, self.conversation_id)
    
    def set(self, key: str, value: Any) -> None:
        """Store data in user-scoped shared memory"""
        self.multi_user_memory.set(self.user_id, key, value, self.conversation_id)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve data from user-scoped shared memory"""
        return self.multi_user_memory.get(self.user_id, key, default, self.conversation_id)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple keys in user-scoped shared memory"""
        self.multi_user_memory.update(self.user_id, updates, self.conversation_id)
    
    def clear(self) -> None:
        """Clear user's shared memory"""
        self.multi_user_memory.clear_user(self.user_id, self.conversation_id)
    
    def keys(self) -> list:
        """Get all keys for this user"""
        return self.multi_user_memory.keys(self.user_id, self.conversation_id)
    
    def has_key(self, key: str) -> bool:
        """Check if key exists in user's shared memory"""
        return self.multi_user_memory.has_key(self.user_id, key, self.conversation_id)
    
    def delete(self, key: str) -> bool:
        """Delete a key from user's shared memory"""
        return self.multi_user_memory.delete(self.user_id, key, self.conversation_id)
    
    def size(self) -> int:
        """Get the number of items in user's shared memory"""
        return self.multi_user_memory.size(self.user_id, self.conversation_id)


# Global multi-user shared memory instance
multi_user_shared_memory = MultiUserSharedMemory()


def get_user_memory(user_id: str, conversation_id: Optional[int] = None) -> UserScopedMemoryAdapter:
    """Get a user-scoped memory adapter"""
    return UserScopedMemoryAdapter(multi_user_shared_memory, user_id, conversation_id)


def cleanup_memory():
    """Cleanup empty memory spaces"""
    multi_user_shared_memory.cleanup_empty_spaces() 