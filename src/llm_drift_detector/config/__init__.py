"""
Configuration management for LLM Drift Detector.

This module provides utilities for loading, validating, and accessing
configuration settings throughout the application.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from pydantic import ValidationError

from .default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


class Config:
    """
    Configuration manager for LLM Drift Detector.
    
    Handles loading configuration from various sources with fallback
    to default values when settings are not specified.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to a YAML or JSON configuration file
        """
        self._config = DEFAULT_CONFIG.copy()
        self._config_path = config_path
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to a YAML or JSON configuration file
            
        Raises:
            ConfigurationError: If the file cannot be loaded or has invalid format
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            if config_path.suffix.lower() in ('.yaml', '.yml'):
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
            else:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )
            
            # Update configuration with values from file
            self._update_nested_dict(self._config, file_config)
            logger.info(f"Loaded configuration from {config_path}")
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Error parsing configuration file: {str(e)}")
    
    def load_from_env(self, prefix: str = "LLM_DRIFT_") -> None:
        """
        Load configuration from environment variables.
        
        Environment variables should be prefixed with the specified prefix
        and use double underscores to represent nested keys.
        
        Example:
            LLM_DRIFT_PROVIDERS__OPENAI__API_KEY=sk-123456
            
        Args:
            prefix: Prefix for environment variables
        """
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and split into parts
                config_key = key[len(prefix):]
                parts = config_key.split('__')
                
                # Build nested dictionary
                current = env_config
                for part in parts[:-1]:
                    part = part.lower()
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the value
                current[parts[-1].lower()] = self._parse_env_value(value)
        
        # Update configuration with values from environment
        if env_config:
            self._update_nested_dict(self._config, env_config)
            logger.info("Loaded configuration from environment variables")
    
    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value to appropriate type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Parsed value with appropriate type
        """
        # Convert to lowercase for boolean values
        value_lower = value.lower()
        
        # Check for boolean values
        if value_lower in ('true', 'yes', '1'):
            return True
        elif value_lower in ('false', 'no', '0'):
            return False
        
        # Check for numeric values
        try:
            # Try to convert to int
            return int(value)
        except ValueError:
            try:
                # Try to convert to float
                return float(value)
            except ValueError:
                # Keep as string
                return value
    
    def _update_nested_dict(self, target: Dict, source: Dict) -> None:
        """
        Update nested dictionary recursively.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # Recursively update nested dictionaries
                self._update_nested_dict(target[key], value)
            else:
                # Update or add value
                target[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Keys can be nested using dots, e.g., 'providers.openai.api_key'.
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            default: Default value to return if key is not found
            
        Returns:
            Configuration value or default if not found
        """
        parts = key.split('.')
        current = self._config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Keys can be nested using dots, e.g., 'providers.openai.api_key'.
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            value: Value to set
        """
        parts = key.split('.')
        current = self._config
        
        # Navigate to the correct nested level
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the value
        current[parts[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export the full configuration as a dictionary.
        
        Returns:
            Dictionary with all configuration values
        """
        return self._config.copy()
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            file_path: Path to save the configuration file
            
        Raises:
            ConfigurationError: If the file cannot be saved
        """
        file_path = Path(file_path)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(file_path.parent, exist_ok=True)
            
            if file_path.suffix.lower() in ('.yaml', '.yml'):
                with open(file_path, 'w') as f:
                    yaml.dump(self._config, f, default_flow_style=False)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'w') as f:
                    json.dump(self._config, f, indent=2)
            else:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {file_path.suffix}"
                )
            
            logger.info(f"Saved configuration to {file_path}")
            
        except (IOError, yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Error saving configuration file: {str(e)}")


# Global configuration instance
_global_config = Config()

def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Config: Global configuration instance
    """
    return _global_config

def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from a file and update the global configuration.
    
    Args:
        config_path: Path to a YAML or JSON configuration file
        
    Returns:
        Config: Updated global configuration instance
    """
    global _global_config
    
    if config_path:
        _global_config = Config(config_path)
    
    # Also check for environment variables
    _global_config.load_from_env()
    
    return _global_config