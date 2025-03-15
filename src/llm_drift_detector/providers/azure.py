"""
Azure OpenAI provider integration.

This module provides integration with Azure OpenAI services,
allowing access to Azure-hosted OpenAI models like GPT-3.5, GPT-4,
and embedding models.
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

class AzureProvider(BaseProvider):
    """
    Provider for Azure OpenAI services.
    
    This provider enables access to OpenAI models deployed on Azure,
    including GPT models and embedding models.
    """
    
    provider_type = "azure"
    
    def __init__(
        self,
        model_name: str,
        api_key_env_var: str = "AZURE_OPENAI_API_KEY",
        endpoint_env_var: str = "AZURE_OPENAI_ENDPOINT",
        api_version: str = "2023-07-01-preview",
        deployment_name: Optional[str] = None,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        embedding_deployment: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Azure OpenAI provider.
        
        Args:
            model_name: Name of the model (e.g., "gpt-3.5-turbo", "gpt-4")
            api_key_env_var: Environment variable containing the API key
            endpoint_env_var: Environment variable containing the endpoint URL
            api_version: Azure OpenAI API version
            deployment_name: Azure deployment name (defaults to model_name if not provided)
            timeout_seconds: Timeout for API requests in seconds
            max_retries: Maximum number of retries for failed requests
            embedding_deployment: Optional different deployment for embeddings
            **kwargs: Additional keyword arguments
        """
        super().__init__(model_name=model_name, **kwargs)
        
        # API configuration
        self.api_key = os.environ.get(api_key_env_var)
        if not self.api_key:
            raise ProviderError(f"Azure OpenAI API key not found in environment variable: {api_key_env_var}")
        
        self.endpoint = os.environ.get(endpoint_env_var)
        if not self.endpoint:
            raise ProviderError(f"Azure OpenAI endpoint not found in environment variable: {endpoint_env_var}")
        
        self.api_version = api_version
        self.deployment_name = deployment_name or model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        # Set embedding deployment (default to ada embedding if not specified)
        self.embedding_deployment = embedding_deployment or "text-embedding-ada-002"
        
        # Initialize client
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """
        Initialize the Azure OpenAI client.
        
        This method initializes the Azure OpenAI client with the provided configuration.
        """
        try:
            import openai
            
            # Configure client
            client_args = {
                "api_key": self.api_key,
                "azure_endpoint": self.endpoint,
                "api_version": self.api_version,
                "timeout": self.timeout_seconds,
                "max_retries": self.max_retries,
            }
            
            # Initialize client
            self._client = openai.AzureOpenAI(**client_args)
            
            logger.info(f"Initialized Azure OpenAI client for deployment {self.deployment_name}")
            
        except ImportError:
            raise ProviderError("openai is not installed. Please install it with 'pip install openai'")
        except Exception as e:
            raise ProviderError(f"Failed to initialize Azure OpenAI client: {str(e)}")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
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
            raise ProviderError("Azure OpenAI client is not initialized")
        
        start_time = time.time()
        error = None
        
        try:
            # Check if system message is provided
            system_message = kwargs.pop("system_message", None)
            
            # Prepare messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            # Prepare parameters
            params = {
                "model": self.deployment_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
            if stop:
                params["stop"] = stop
            
            # Update with any additional parameters
            params.update(kwargs)
            
            # Run in thread to avoid blocking event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: self._client.chat.completions.create(**params)
                )
                response = future.result()
            
            # Extract text
            if not response.choices:
                raise ProviderError("Empty response from Azure OpenAI API")
            
            text = response.choices[0].message.content
            
            # Prepare metadata
            metadata = {
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "provider": "azure",
                "deployment": self.deployment_name
            }
            
            return text, metadata
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating text with Azure OpenAI: {error_msg}")
            raise ProviderError(f"Azure OpenAI generation error: {error_msg}")
        
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.debug(f"Azure OpenAI generation took {elapsed_time:.2f}s")
    
    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate text for multiple prompts.
        
        Azure OpenAI's API doesn't support batch processing for chat completions,
        so we process each prompt individually.
        
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
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
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
                    return ("", {"error": str(e), "provider": "azure"})
        
        # Create tasks for all prompts
        tasks = [process_prompt(prompt) for prompt in prompts]
        
        # Execute tasks and gather results
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def get_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get embedding for text.
        
        Args:
            text: Text to get embedding for
            model: Optional model to use for embedding (overrides self.embedding_deployment)
            **kwargs: Additional parameters
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Embedding vector and metadata
        """
        if not self._client:
            raise ProviderError("Azure OpenAI client is not initialized")
        
        start_time = time.time()
        
        try:
            # Use specified model or default embedding deployment
            embedding_model = model or self.embedding_deployment
            
            # Prepare parameters
            params = {
                "model": embedding_model,
                "input": text
            }
            
            # Update with any additional parameters
            params.update(kwargs)
            
            # Run in thread to avoid blocking event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: self._client.embeddings.create(**params)
                )
                response = future.result()
            
            # Extract embedding
            if not response.data:
                raise ProviderError("Empty embedding response from Azure OpenAI API")
            
            embedding = np.array(response.data[0].embedding)
            
            # Prepare metadata
            metadata = {
                "model": embedding_model,
                "dimensions": len(embedding),
                "provider": "azure",
                "deployment": embedding_model,
                "input_tokens": response.usage.total_tokens
            }
            
            return embedding, metadata
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error getting embedding with Azure OpenAI: {error_msg}")
            raise ProviderError(f"Azure OpenAI embedding error: {error_msg}")
        
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.debug(f"Azure OpenAI embedding took {elapsed_time:.2f}s")
    
    async def get_batch_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get embeddings for multiple texts.
        
        Azure OpenAI's API supports batch processing for embeddings, so we
        process all texts in a single request when possible.
        
        Args:
            texts: List of texts to get embeddings for
            model: Optional model to use for embedding
            **kwargs: Additional parameters
            
        Returns:
            List[Tuple[np.ndarray, Dict[str, Any]]]: List of embedding vectors and metadata
        """
        if not self._client:
            raise ProviderError("Azure OpenAI client is not initialized")
        
        # Check for empty input
        if not texts:
            return []
        
        start_time = time.time()
        
        try:
            # Use specified model or default embedding deployment
            embedding_model = model or self.embedding_deployment
            
            # Prepare parameters
            params = {
                "model": embedding_model,
                "input": texts
            }
            
            # Update with any additional parameters
            params.update(kwargs)
            
            # Run in thread to avoid blocking event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: self._client.embeddings.create(**params)
                )
                response = future.result()
            
            # Extract embeddings and match with input texts
            results = []
            
            # Sort by index to ensure correct order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            
            for i, data in enumerate(sorted_data):
                embedding = np.array(data.embedding)
                metadata = {
                    "model": embedding_model,
                    "dimensions": len(embedding),
                    "provider": "azure",
                    "deployment": embedding_model,
                    "input_tokens": response.usage.total_tokens // len(texts)  # Approximate tokens per text
                }
                results.append((embedding, metadata))
            
            return results
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error getting batch embeddings with Azure OpenAI: {error_msg}")
            
            # If batch request fails, fall back to individual requests
            logger.info("Falling back to individual embedding requests")
            
            results = []
            for text in texts:
                try:
                    embedding, metadata = await self.get_embedding(text, model, **kwargs)
                    results.append((embedding, metadata))
                except Exception as e:
                    logger.error(f"Error getting embedding for text: {text[:50]}...: {str(e)}")
                    results.append((np.array([]), {"error": str(e), "provider": "azure"}))
            
            return results
        
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.debug(f"Azure OpenAI batch embedding took {elapsed_time:.2f}s for {len(texts)} texts")


# Register the provider
from . import register_provider
register_provider(AzureProvider)