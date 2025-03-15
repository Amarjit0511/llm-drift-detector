"""
Storage management for LLM Drift Detector.

This module provides interfaces and implementations for storing
drift detection data, samples, and metrics in various backends.
"""

import logging
from typing import Dict, List, Any, Optional, Type, Union

from .base import BaseStorage, StorageError

__all__ = [
    "BaseStorage",
    "StorageError",
    "get_storage",
    "get_storage_types"
]

logger = logging.getLogger(__name__)

# Storage registry
_storage_types: Dict[str, Type[BaseStorage]] = {}

def register_storage(storage_class: Type[BaseStorage]) -> None:
    """
    Register a storage class.
    
    Args:
        storage_class: Storage class to register
    """
    storage_type = getattr(storage_class, "storage_type", None)
    if not storage_type:
        logger.warning(f"Cannot register storage class without storage_type: {storage_class.__name__}")
        return
    
    _storage_types[storage_type.lower()] = storage_class
    logger.debug(f"Registered storage type: {storage_type}")

def get_storage(storage_type: str, **kwargs) -> Optional[BaseStorage]:
    """
    Get a storage instance of the specified type.
    
    Args:
        storage_type: Type of storage to get ('local', 'redis', 'sql')
        **kwargs: Additional arguments for storage initialization
        
    Returns:
        Optional[BaseStorage]: Storage instance or None if type not found
    """
    # Import storage implementations to register them
    from . import local, redis, sql
    
    storage_class = _storage_types.get(storage_type.lower())
    if not storage_class:
        logger.warning(f"Storage type not found: {storage_type}")
        return None
    
    try:
        return storage_class(**kwargs)
    except Exception as e:
        logger.error(f"Error creating storage '{storage_type}': {str(e)}")
        return None

def get_storage_types() -> Dict[str, Type[BaseStorage]]:
    """
    Get all registered storage types.
    
    Returns:
        Dict[str, Type[BaseStorage]]: Dictionary of storage types
    """
    # Import storage implementations to register them
    from . import local, redis, sql
    
    return _storage_types.copy()

def create_storage_from_config() -> BaseStorage:
    """
    Create a storage instance based on the configuration.
    
    Returns:
        BaseStorage: Storage instance
    """
    from ..config import get_config
    config = get_config()
    
    storage_type = config.get("data.storage.type", "local")
    
    # Get storage-specific configuration
    storage_config = {}
    if storage_type in config.get("data.storage", {}):
        storage_config = config.get(f"data.storage.{storage_type}", {})
    
    storage = get_storage(storage_type, **storage_config)
    if not storage:
        logger.warning(f"Failed to create storage of type '{storage_type}', falling back to local storage")
        storage = get_storage("local")
    
    return storage