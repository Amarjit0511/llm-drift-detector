"""
Plugin system for extending LLM Drift Detector functionality.

This module provides a flexible plugin architecture that allows
users to add custom providers, detectors, storage backends, and
other components without modifying the core code.
"""

import os
import sys
import logging
import importlib.util
from typing import Dict, List, Any, Optional, Type, TypeVar, Callable
import inspect

from .base import Plugin, PluginMount, PluginRegistry

__all__ = [
    "Plugin",
    "PluginMount",
    "PluginRegistry",
    "load_plugins",
    "get_plugins",
    "get_plugin",
    "register_plugin"
]

logger = logging.getLogger(__name__)

# Plugin registry
_plugin_registry = PluginRegistry()

T = TypeVar('T', bound=Plugin)

def load_plugins(directory: Optional[str] = None) -> Dict[str, Type[Plugin]]:
    """
    Load plugins from a directory.
    
    Args:
        directory: Optional directory path to load plugins from,
                  defaults to the directory specified in config
        
    Returns:
        Dict[str, Type[Plugin]]: Dictionary of loaded plugin classes
    """
    if directory is None:
        from ..config import get_config
        config = get_config()
        directory = config.get("plugins.directory", "./plugins/")
        
        # Check if plugins are enabled
        if not config.get("plugins.enabled", True):
            logger.info("Plugins are disabled in configuration")
            return {}
    
    logger.info(f"Loading plugins from {directory}")
    
    if not os.path.isdir(directory):
        logger.warning(f"Plugin directory does not exist: {directory}")
        return {}
    
    # Get allowlist and blocklist
    from ..config import get_config
    config = get_config()
    allowlist = config.get("plugins.allowlist", [])
    blocklist = config.get("plugins.blocklist", [])
    
    # Find Python files in the directory
    loaded_plugins = {}
    
    for filename in os.listdir(directory):
        if not filename.endswith(".py") or filename.startswith("_"):
            continue
        
        module_name = filename[:-3]  # Remove .py extension
        
        # Check allowlist and blocklist
        if allowlist and module_name not in allowlist:
            logger.debug(f"Skipping plugin '{module_name}' (not in allowlist)")
            continue
        
        if module_name in blocklist:
            logger.debug(f"Skipping plugin '{module_name}' (in blocklist)")
            continue
        
        # Import the module
        try:
            module_path = os.path.join(directory, filename)
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                logger.warning(f"Failed to load plugin '{module_name}': Invalid module specification")
                continue
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Plugin) and 
                    obj is not Plugin and
                    not getattr(obj, '_abstract', False)):
                    
                    # Register the plugin
                    _plugin_registry.register(obj)
                    loaded_plugins[obj.plugin_name] = obj
                    logger.info(f"Loaded plugin: {obj.plugin_name} ({obj.__module__}.{obj.__name__})")
            
        except Exception as e:
            logger.warning(f"Error loading plugin '{module_name}': {str(e)}")
    
    return loaded_plugins

def get_plugins(plugin_type: Optional[Type[T]] = None) -> Dict[str, Type[Plugin]]:
    """
    Get all registered plugins, optionally filtered by type.
    
    Args:
        plugin_type: Optional plugin type to filter by
        
    Returns:
        Dict[str, Type[Plugin]]: Dictionary of plugin classes
    """
    if plugin_type is not None:
        return _plugin_registry.get_plugins_by_type(plugin_type)
    else:
        return _plugin_registry.get_all_plugins()

def get_plugin(name: str) -> Optional[Type[Plugin]]:
    """
    Get a plugin by name.
    
    Args:
        name: Name of the plugin
        
    Returns:
        Optional[Type[Plugin]]: Plugin class if found, None otherwise
    """
    return _plugin_registry.get_plugin(name)

def register_plugin(plugin_class: Type[Plugin]) -> None:
    """
    Register a plugin programmatically.
    
    Args:
        plugin_class: Plugin class to register
    """
    _plugin_registry.register(plugin_class)
    logger.info(f"Registered plugin: {plugin_class.plugin_name} ({plugin_class.__module__}.{plugin_class.__name__})")