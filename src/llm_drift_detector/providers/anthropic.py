"""
Anthropic provider integration.

This module provides integration with Anthropic's API for accessing
Claude models like Claude 2, Claude Instant, and Claude 3.
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json
import numpy as np

from .base import BaseProvider, ProviderSample, ProviderError

logger = logging.getLogger(__name__)

class AnthropicProvider(BaseProvider):
    """
    Provider for Anthropic Claude API.
    
    This provider enables access to Anthropic's Claude models
    for high-quality text generation with strong instruction following.
    """
    
    provider_type = "anthropic"
    
    def __init__(
        self,
        model_name: str,
        api_key_env_var: str = "ANTHROPIC_API_KEY",
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout_seconds: int = 60,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize the Anthropic provider.
        
        Args:
            model_name: Name of the model (e.g., "claude-2", "claude-instant-1")
            api_key_env_var: Environment variable containing the API key
            api_base: Optional custom API base URL
            api_version: Optional API version
            timeout_seconds: Timeout for API requests in seconds
            max_retries: Maximum number of retries for failed requests
            **kwargs: Additional keyword arguments
        """
        super().__init__(model_name=model_name, **kwargs)
        
        # API configuration
        self.api_key = os.environ.get(api_key_env_var)
        if not self.api_key:
            raise ProviderError(f"Anthropic API key not found in environment variable: {api_key_env_var}")
        
        self.api_base = api_base or "https://api.anthropic.com"
        self.api_version = api_version or "v1"
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        # Initialize client
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """
        Initialize the Anthropic client.
        
        This method initializes the Anthropic client with the provided configuration.
        """
        try:
            # Import required modules
            import anthropic
            
            # Configure client
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout_seconds
            )
            
            # Store constants for convenience
            self._constants = anthropic
            
            logger.info(f"Initialized Anthropic client for model {self.model_name}")
            
        except ImportError:
            raise ProviderError("anthropic is not installed. Please install it with 'pip install anthropic'")
        except Exception as e:
            raise ProviderError(f"Failed to initialize Anthropic client: {str(e)}")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text from the model.
        
        Args:
            prompt: The input prompt to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Optional list of stop sequences
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple[str, Dict[str, Any]]: Generated text and metadata
        """
        if not self._client:
            raise ProviderError("Anthropic client is not initialized")
        
        start_time = time.time()
        error = None
        
        try:
            # Check if system message is provided
            system = kwargs.pop("system", None)
            
            # Prepare messages for Claude 3 models
            if self.model_name.startswith("claude-3"):
                # Use the messages API for Claude 3
                messages = []
                
                # Add system message if provided
                if system:
                    messages.append({
                        "role": "system",
                        "content": system
                    })
                
                # Add user message
                messages.append({
                    "role": "user",
                    "content": prompt
                })
                
                # Prepare parameters
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
                
                if stop:
                    params["stop_sequences"] = stop
                
                # Update with any additional parameters
                params.update(kwargs)
                
                # Run in thread to avoid blocking event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: self._client.messages.create(**params)
                    )
                    response = future.result()
                
                # Extract text
                text = response.content[0].text
                
                # Prepare metadata
                metadata = {
                    "model": response.model,
                    "finish_reason": response.stop_reason,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                    "provider": "anthropic"
                }
                
            else:
                # Use the legacy completions API for Claude 2 and Claude Instant
                # Prepare parameters
                params = {
                    "model": self.model_name,
                    "prompt": self._constants.HUMAN_PROMPT + prompt + self._constants.AI_PROMPT,
                    "max_tokens_to_sample": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
                
                if system:
                    params["system"] = system
                
                if stop:
                    params["stop_sequences"] = stop
                
                # Update with any additional parameters
                params.update(kwargs)
                
                # Run in thread to avoid blocking event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: self._client.completions.create(**params)
                    )
                    response = future.result()
                
                # Extract text
                text = response.completion
                
                # Prepare metadata
                metadata = {
                    "model": self.model_name,
                    "finish_reason": response.stop_reason,
                    "provider": "anthropic"
                }
                
                # Approximate token counts since the API doesn't return them
                input_tokens = len(params["prompt"].split()) // 2
                output_tokens = len(text.split())
                total_tokens = input_tokens + output_tokens
                
                metadata.update({
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                })
            
            return text, metadata
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating text with Anthropic: {error_msg}")
            raise ProviderError(f"Anthropic generation error: {error_msg}")
        
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.debug(f"Anthropic generation took {elapsed_time:.2f}s")
    
    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate text for multiple prompts.
        
        Anthropic's API doesn't support batch processing, so we process each prompt individually.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Optional list of stop sequences
            **kwargs: Additional generation parameters
            
        Returns:
            List[Tuple[str, Dict[str, Any]]]: List of generated texts and metadata
        """
        results = []
        
        # Process prompts in parallel with concurrency control
        semaphore = asyncio.Semaphore(3)  # Limit concurrent requests to Anthropic API
        
        async def process_prompt(prompt):
            async with semaphore:
                try:
                    return await self.generate(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        **kwargs
                    )
                except Exception as e:
                    logger.error(f"Error in batch generation for prompt: {prompt[:50]}...: {str(e)}")
                    return ("", {"error": str(e), "provider": "anthropic"})
        
        # Create tasks for all prompts
        tasks = [process_prompt(prompt) for prompt in prompts]
        
        # Execute tasks and gather results
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def get_embedding(
        self,
        text: str,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get embedding for text.
        
        Note: Anthropic does not currently provide a dedicated embedding API.
        This method raises an error as embeddings are not supported.
        
        Args:
            text: Text to get embedding for
            **kwargs: Additional parameters
            
        Raises:
            ProviderError: Always raised as embeddings are not supported
        """
        # Check if Anthropic has released an embedding API
        # As of the time of writing, they don't provide this
        raise ProviderError("Embedding generation is not supported by Anthropic provider")
    
    def cleanup(self):
        """
        Clean up resources used by the provider.
        """
        # No specific cleanup needed for Anthropic
        self._client = None


# Register the provider
from . import register_provider
register_provider(AnthropicProvider)