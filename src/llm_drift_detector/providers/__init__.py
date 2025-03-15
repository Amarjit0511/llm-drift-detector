"""
LLM provider integrations for drift detection.

This module provides interfaces and implementations for connecting
to various LLM providers like OpenAI, vLLM, HuggingFace, etc.
"""

import logging
from typing import Dict, List, Any, Optional, Type, Union

from .base import BaseProvider, ProviderSample, ProviderError

__all__ = [
    "BaseProvider",
    "ProviderSample",
    "ProviderError",
    "get_provider",
    "get_all_providers",
    "register_provider"
]

logger = logging.getLogger(__name__)

# Provider registry
_providers: Dict[str, Type[BaseProvider]] = {}

def get_provider(provider_type: str) -> Optional[Type[BaseProvider]]:
    """
    Get a provider class by type.
    
    Args:
        provider_type: Provider type identifier (e.g., 'openai', 'vllm')
        
    Returns:
        Optional[Type[BaseProvider]]: Provider class if found, None otherwise
    """
    # Import providers here to ensure they are registered
    from . import openai, azure, anthropic, vllm, huggingface, custom
    
    return _providers.get(provider_type.lower())

def get_all_providers() -> Dict[str, Type[BaseProvider]]:
    """
    Get all registered providers.
    
    Returns:
        Dict[str, Type[BaseProvider]]: Dictionary of provider types to provider classes
    """
    # Import providers here to ensure they are registered
    from . import openai, azure, anthropic, vllm, huggingface, custom
    
    return _providers.copy()

def register_provider(provider_class: Type[BaseProvider]) -> None:
    """
    Register a provider class.
    
    Args:
        provider_class: Provider class to register
    """
    provider_type = getattr(provider_class, "provider_type", None)
    if not provider_type:
        logger.warning(f"Cannot register provider class without provider_type: {provider_class.__name__}")
        return
    
    _providers[provider_type.lower()] = provider_class
    logger.debug(f"Registered provider: {provider_type}")

def create_provider(
    provider_type: str,
    model_name: str,
    **kwargs
) -> Optional[BaseProvider]:
    """
    Create a provider instance.
    
    Args:
        provider_type: Provider type identifier (e.g., 'openai', 'vllm')
        model_name: Name of the model to use
        **kwargs: Additional provider-specific arguments
        
    Returns:
        Optional[BaseProvider]: Provider instance if successful, None otherwise
    """
    provider_class = get_provider(provider_type)
    if not provider_class:
        logger.warning(f"Provider type not found: {provider_type}")
        return None
    
    try:
        return provider_class(model_name=model_name, **kwargs)
    except Exception as e:
        logger.error(f"Error creating provider '{provider_type}': {str(e)}")
        return None