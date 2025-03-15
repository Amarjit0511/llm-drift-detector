"""
Base classes for the plugin system.

This module defines the core classes and interfaces for the
plugin architecture, including the Plugin base class and
the plugin registry.
"""

import logging
import inspect
from typing import Dict, List, Any, Optional, Type, TypeVar, ClassVar, Set, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='Plugin')

class Plugin(ABC):
    """
    Base class for all plugins.
    
    All plugins must inherit from this class and define required
    class attributes and methods.
    """
    
    # Class attributes to be defined by subclasses
    plugin_name: ClassVar[str] = ""
    plugin_type: ClassVar[str] = ""
    plugin_description: ClassVar[str] = ""
    plugin_version: ClassVar[str] = "0.1.0"
    
    # Internal attributes
    _abstract: ClassVar[bool] = False
    
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """
        Get plugin metadata.
        
        Returns:
            Dict[str, Any]: Dictionary with plugin metadata
        """
        return {
            "name": cls.plugin_name,
            "type": cls.plugin_type,
            "description": cls.plugin_description,
            "version": cls.plugin_version,
            "class": cls.__name__,
            "module": cls.__module__
        }
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate the plugin configuration.
        
        Returns:
            bool: True if plugin is valid, False otherwise
        """
        # Check required class attributes
        if not cls.plugin_name:
            logger.warning(f"Plugin {cls.__name__} missing required 'plugin_name' attribute")
            return False
        
        if not cls.plugin_type:
            logger.warning(f"Plugin {cls.__name__} missing required 'plugin_type' attribute")
            return False
        
        return True


class PluginMount(type):
    """
    Metaclass for plugin types.
    
    This metaclass automatically registers plugin subclasses
    with the plugin registry when they are defined.
    """
    
    def __new__(mcs, name, bases, attrs):
        # Create the class
        cls = super().__new__(mcs, name, bases, attrs)
        
        # Mark base classes as abstract
        if name != 'Plugin' and 'plugin_type' in attrs and attrs.get('_abstract', False):
            cls._abstract = True
        
        # Register non-abstract subclasses of Plugin
        if (not getattr(cls, '_abstract', False) and 
            name != 'Plugin' and 
            isinstance(cls, PluginMount) and
            issubclass(cls, Plugin)):
            
            # Import here to avoid circular import
            from . import _plugin_registry
            _plugin_registry.register(cls)
        
        return cls


class PluginRegistry:
    """
    Registry for plugins.
    
    This class manages the registration and lookup of plugins.
    """
    
    def __init__(self):
        """Initialize an empty plugin registry."""
        self._plugins: Dict[str, Type[Plugin]] = {}
        self._plugins_by_type: Dict[str, Dict[str, Type[Plugin]]] = {}
    
    def register(self, plugin_class: Type[Plugin]) -> bool:
        """
        Register a plugin class.
        
        Args:
            plugin_class: Plugin class to register
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        # Validate the plugin
        if not plugin_class.validate():
            return False
        
        plugin_name = plugin_class.plugin_name
        plugin_type = plugin_class.plugin_type
        
        # Check for duplicate plugin name
        if plugin_name in self._plugins:
            existing = self._plugins[plugin_name]
            logger.warning(f"Plugin name conflict: '{plugin_name}' already registered by {existing.__module__}.{existing.__name__}")
            return False
        
        # Register the plugin
        self._plugins[plugin_name] = plugin_class
        
        # Register by type
        if plugin_type not in self._plugins_by_type:
            self._plugins_by_type[plugin_type] = {}
        
        self._plugins_by_type[plugin_type][plugin_name] = plugin_class
        
        return True
    
    def get_plugin(self, name: str) -> Optional[Type[Plugin]]:
        """
        Get a plugin by name.
        
        Args:
            name: Name of the plugin
            
        Returns:
            Optional[Type[Plugin]]: Plugin class if found, None otherwise
        """
        return self._plugins.get(name)
    
    def get_all_plugins(self) -> Dict[str, Type[Plugin]]:
        """
        Get all registered plugins.
        
        Returns:
            Dict[str, Type[Plugin]]: Dictionary of all plugin classes
        """
        return self._plugins.copy()
    
    def get_plugins_by_type(self, plugin_type: Union[str, Type]) -> Dict[str, Type[Plugin]]:
        """
        Get plugins by type.
        
        Args:
            plugin_type: Type name or plugin type class
            
        Returns:
            Dict[str, Type[Plugin]]: Dictionary of plugin classes of the specified type
        """
        # If plugin_type is a class, get its plugin_type attribute
        if inspect.isclass(plugin_type):
            if hasattr(plugin_type, 'plugin_type'):
                plugin_type = plugin_type.plugin_type
            else:
                return {}
        
        return self._plugins_by_type.get(plugin_type, {}).copy()
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a plugin by name.
        
        Args:
            name: Name of the plugin
            
        Returns:
            bool: True if plugin was unregistered, False otherwise
        """
        if name not in self._plugins:
            return False
        
        plugin_class = self._plugins[name]
        plugin_type = plugin_class.plugin_type
        
        # Remove from main registry
        del self._plugins[name]
        
        # Remove from type registry
        if plugin_type in self._plugins_by_type and name in self._plugins_by_type[plugin_type]:
            del self._plugins_by_type[plugin_type][name]
            
            # Clean up empty type dict
            if not self._plugins_by_type[plugin_type]:
                del self._plugins_by_type[plugin_type]
        
        return True
    
    def clear(self) -> None:
        """Clear all registered plugins."""
        self._plugins.clear()
        self._plugins_by_type.clear()
    
    def get_plugin_types(self) -> Set[str]:
        """
        Get all registered plugin types.
        
        Returns:
            Set[str]: Set of plugin type names
        """
        return set(self._plugins_by_type.keys())
    
    def count(self) -> int:
        """
        Get the number of registered plugins.
        
        Returns:
            int: Number of registered plugins
        """
        return len(self._plugins)