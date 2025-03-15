"""
vLLM provider integration for high-throughput LLM inference.

This module provides integration with vLLM, a high-throughput and
memory-efficient inference engine for LLMs that supports local
deployment of models.
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

class VLLMProvider(BaseProvider):
    """
    Provider for vLLM inference engine.
    
    This provider enables high-performance inference with local models
    using the vLLM library, which optimizes GPU memory usage and throughput.
    """
    
    provider_type = "vllm"
    
    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_num_batched_tokens: int = 4096,
        host: str = "localhost",
        port: int = 8000,
        use_existing_server: bool = False,
        **kwargs
    ):
        """
        Initialize the vLLM provider.
        
        Args:
            model_name: Name of the model (e.g., "llama-2-7b")
            model_path: Optional path to model weights (if not using HF Hub)
            tensor_parallel_size: Number of GPUs to use for inference
            gpu_memory_utilization: Fraction of GPU memory to use
            max_num_batched_tokens: Maximum number of tokens to batch
            host: Hostname for vLLM server
            port: Port for vLLM server
            use_existing_server: Whether to use an existing vLLM server instead of starting a new one
            **kwargs: Additional keyword arguments
        """
        super().__init__(model_name=model_name, **kwargs)
        
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.host = host
        self.port = port
        self.use_existing_server = use_existing_server
        
        # vLLM client and server
        self._vllm_client = None
        self._vllm_server = None
        
        # Initialize vLLM
        if not use_existing_server:
            self._start_server()
        
        # Initialize client
        self._init_client()
    
    def _start_server(self):
        """
        Start the vLLM server as a subprocess.
        
        This method starts a new vLLM server subprocess if use_existing_server is False.
        """
        try:
            # Import required modules
            import subprocess
            import sys
            
            # Check if vLLM is installed
            try:
                import vllm
                logger.info(f"Using vLLM version {vllm.__version__}")
            except ImportError:
                raise ProviderError("vLLM is not installed. Please install it with 'pip install vllm'")
            
            # Prepare command
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.api_server",
                "--model", self.model_name
            ]
            
            # Add optional arguments
            if self.model_path:
                cmd.extend(["--model-path", self.model_path])
            
            cmd.extend([
                "--tensor-parallel-size", str(self.tensor_parallel_size),
                "--gpu-memory-utilization", str(self.gpu_memory_utilization),
                "--max-num-batched-tokens", str(self.max_num_batched_tokens),
                "--host", self.host,
                "--port", str(self.port)
            ])
            
            # Start the server
            logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")
            self._vllm_server = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            time.sleep(10)  # Give it some time to initialize
            logger.info("vLLM server started")
            
        except Exception as e:
            raise ProviderError(f"Failed to start vLLM server: {str(e)}")
    
    def _init_client(self):
        """
        Initialize the vLLM client.
        
        This method sets up the HTTP client to communicate with the vLLM server.
        """
        try:
            import requests
            
            # Define the client as a simple class that makes HTTP requests
            class VLLMClient:
                def __init__(self, host, port):
                    self.base_url = f"http://{host}:{port}/v1"
                
                def generate(self, prompt, **kwargs):
                    completion_url = f"{self.base_url}/completions"
                    
                    # Prepare request payload
                    payload = {
                        "prompt": prompt,
                        "max_tokens": kwargs.get("max_tokens", 512),
                        "temperature": kwargs.get("temperature", 0.7),
                        "top_p": kwargs.get("top_p", 1.0),
                        "stop": kwargs.get("stop", None)
                    }
                    
                    # Make request
                    response = requests.post(completion_url, json=payload)
                    
                    if response.status_code != 200:
                        raise ProviderError(f"vLLM API error: {response.status_code} - {response.text}")
                    
                    return response.json()
            
            # Create client
            self._vllm_client = VLLMClient(self.host, self.port)
            
            # Test connection
            try:
                response = self._vllm_client.generate("Hello, world!", max_tokens=10)
                logger.debug(f"vLLM test response: {response}")
            except Exception as e:
                logger.warning(f"vLLM server test failed: {str(e)}")
            
        except ImportError as e:
            raise ProviderError(f"Required module not found: {str(e)}")
        except Exception as e:
            raise ProviderError(f"Failed to initialize vLLM client: {str(e)}")
    
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
        if not self._vllm_client:
            raise ProviderError("vLLM client is not initialized")
        
        start_time = time.time()
        error = None
        
        try:
            # Prepare parameters
            params = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
            if stop:
                params["stop"] = stop
            
            # Update with any additional parameters
            params.update(kwargs)
            
            # Make the request
            response = self._vllm_client.generate(prompt, **params)
            
            # Process response
            if "choices" not in response or not response["choices"]:
                raise ProviderError("Invalid response from vLLM server")
            
            text = response["choices"][0]["text"]
            
            # Extract metadata
            metadata = {
                "model": self.model_name,
                "finish_reason": response["choices"][0].get("finish_reason", "unknown"),
                "provider": "vllm"
            }
            
            # Add token counts if available
            if "usage" in response:
                metadata["input_tokens"] = response["usage"].get("prompt_tokens", 0)
                metadata["output_tokens"] = response["usage"].get("completion_tokens", 0)
                metadata["total_tokens"] = response["usage"].get("total_tokens", 0)
            
            return text, metadata
            
        except Exception as e:
            error = str(e)
            logger.error(f"Error generating text with vLLM: {error}")
            raise ProviderError(f"vLLM generation error: {error}")
        
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.debug(f"vLLM generation took {elapsed_time:.2f}s")
    
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
        results = []
        
        # Process prompts in batches
        # vLLM supports batching natively on the server side,
        # but the API processes one prompt at a time
        for prompt in prompts:
            try:
                text, metadata = await self.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    **kwargs
                )
                results.append((text, metadata))
            except Exception as e:
                logger.error(f"Error in batch generation for prompt: {prompt[:50]}...: {str(e)}")
                # Add error result
                results.append(("", {"error": str(e), "provider": "vllm"}))
        
        return results
    
    async def get_embedding(
        self,
        text: str,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get embedding for text.
        
        Note: vLLM does not natively support embeddings, so this method is not implemented.
        
        Args:
            text: Text to get embedding for
            **kwargs: Additional parameters
            
        Raises:
            ProviderError: Always raised as this method is not implemented
        """
        raise ProviderError("Embedding generation is not supported by vLLM provider")
    
    def cleanup(self):
        """
        Clean up resources used by the provider.
        
        This method stops the vLLM server if it was started by this provider.
        """
        if self._vllm_server:
            logger.info("Stopping vLLM server")
            self._vllm_server.terminate()
            try:
                self._vllm_server.wait(timeout=10)
            except:
                self._vllm_server.kill()
            self._vllm_server = None
        
        self._vllm_client = None
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


# Register the provider
from . import register_provider
register_provider(VLLMProvider)