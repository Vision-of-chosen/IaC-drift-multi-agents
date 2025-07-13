#!/usr/bin/env python3
"""
Shared memory implementation for the Terraform Drift Detection & Remediation System.

This module provides a centralized data store that allows different agents to
share information and coordinate their work.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class SharedMemory:
    """Shared memory implementation for cross-agent collaboration"""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
    
    def set(self, key: str, value: Any) -> None:
        """Store data in shared memory"""
        self.data[key] = value
        logger.info(f"SharedMemory: Set {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve data from shared memory"""
        return self.data.get(key, default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple keys in shared memory"""
        self.data.update(updates)
        logger.info(f"SharedMemory: Updated {list(updates.keys())}")
    
    def clear(self) -> None:
        """Clear all shared memory"""
        self.data.clear()
        logger.info("SharedMemory: Cleared all data")
    
    def keys(self) -> list:
        """Get all keys in shared memory"""
        return list(self.data.keys())
    
    def has_key(self, key: str) -> bool:
        """Check if key exists in shared memory"""
        return key in self.data
    
    def delete(self, key: str) -> bool:
        """Delete a key from shared memory"""
        if key in self.data:
            del self.data[key]
            logger.info(f"SharedMemory: Deleted {key}")
            return True
        return False
    
    def size(self) -> int:
        """Get the number of items in shared memory"""
        return len(self.data)


# Global shared memory instance
shared_memory = SharedMemory() 