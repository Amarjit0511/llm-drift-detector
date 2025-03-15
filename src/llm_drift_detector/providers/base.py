"""
Base interfaces for LLM providers.

This module provides abstract base classes and common functionality
for all LLM provider implementations, ensuring a consistent interface
across different providers.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import asyncio
import numpy as np
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

class ProviderError(Exception):
    """Exception raised for provider-specific errors."""
    pass


@dataclass
class ProviderSample:
    """
    Data class representing a single LLM generation sample.
    
    This class stores both inputs and outputs from an LLM,
    along with metadata and performance metrics.
    """
    # Basic information
    timestamp: datetime
    provider_name: str
    model_name: str
    
    # Input and output
    prompt: str
    response: str
    
    # Performance metrics
    response_time: Optional[float] = None
    token_count: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    
    # Additional information
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    This class defines the interface that all provider implementations
    must follow, ensuring consistent behavior across different providers.
    """
    
    # Class attribute to identify the provider type
    provider_type: str = "base"
    
    def __init__(
        self,
        model_name: str,
        **kwargs
    ):
        """
        Initialize the base provider.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional provider-specific arguments
        """
        self.model_name = model_name
        self.initialized = False
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def get_embedding(
        self,
        text: str,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get embedding for text.
        
        Args:
            text: Text to get embedding for
            **kwargs: Additional parameters
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Embedding vector and metadata
        """
        pass
    
    async def get_batch_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get embeddings for multiple texts.
        
        Default implementation processes each text individually.
        Providers should override this method if they support batch processing.
        
        Args:
            texts: List of texts to get embeddings for
            **kwargs: Additional parameters
            
        Returns:
            List[Tuple[np.ndarray, Dict[str, Any]]]: List of embedding vectors and metadata
        """
        results = []
        
        # Process texts in parallel with concurrency control
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def process_text(text):
            async with semaphore:
                try:
                    return await self.get_embedding(text, **kwargs)
                except Exception as e:
                    logger.error(f"Error getting embedding for text: {text[:50]}...: {str(e)}")
                    return (np.array([]), {"error": str(e), "provider": self.provider_type})
        
        # Create tasks for all texts
        tasks = [process_text(text) for text in texts]
        
        # Execute tasks and gather results
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def sample(
        self,
        prompt: str,
        **kwargs
    ) -> ProviderSample:
        """
        Generate a complete sample including metadata and timing.
        
        This method wraps around generate() to provide a standardized
        sample format with timing information.
        
        Args:
            prompt: The input prompt to generate from
            **kwargs: Additional generation parameters
            
        Returns:
            ProviderSample: Complete sample with response and metadata
        """
        start_time = datetime.now()
        generation_start = None
        response_time = None
        error = None
        response_text = ""
        metadata = {}
        
        try:
            generation_start = datetime.now()
            response_text, metadata = await self.generate(prompt, **kwargs)
            response_time = (datetime.now() - generation_start).total_seconds()
        except Exception as e:
            error = str(e)
            logger.error(f"Error in provider sample: {error}")
        
        # Create sample
        sample = ProviderSample(
            timestamp=start_time,
            provider_name=self.provider_type,
            model_name=self.model_name,
            prompt=prompt,
            response=response_text,
            response_time=response_time,
            token_count=metadata.get("total_tokens"),
            input_tokens=metadata.get("input_tokens"),
            output_tokens=metadata.get("output_tokens"),
            finish_reason=metadata.get("finish_reason"),
            error=error,
            metadata=metadata
        )
        
        return sample
    
    async def sample_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[ProviderSample]:
        """
        Generate multiple complete samples including metadata and timing.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List[ProviderSample]: List of complete samples
        """
        # Generate batch responses
        start_time = datetime.now()
        results = await self.generate_batch(prompts, **kwargs)
        
        # Create samples
        samples = []
        for i, (response_text, metadata) in enumerate(results):
            prompt = prompts[i]
            error = metadata.get("error")
            
            sample = ProviderSample(
                timestamp=start_time,
                provider_name=self.provider_type,
                model_name=self.model_name,
                prompt=prompt,
                response=response_text,
                response_time=metadata.get("response_time"),
                token_count=metadata.get("total_tokens"),
                input_tokens=metadata.get("input_tokens"),
                output_tokens=metadata.get("output_tokens"),
                finish_reason=metadata.get("finish_reason"),
                error=error,
                metadata=metadata
            )
            
            samples.append(sample)
        
        return samples
    
    def cleanup(self):
        """
        Clean up resources used by the provider.
        
        This method should be implemented by providers that need to
        release resources (e.g., close connections, free memory).
        """
        pass
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.cleanup()