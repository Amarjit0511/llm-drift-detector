"""
HuggingFace provider integration for local and API-based models.

This module provides integration with HuggingFace models, both
through the Inference API and for local model inference using the
transformers library.
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import numpy as np

from .base import BaseProvider, ProviderSample, ProviderError

logger = logging.getLogger(__name__)

class HuggingFaceProvider(BaseProvider):
    """
    Provider for HuggingFace models.
    
    This provider supports both the HuggingFace Inference API and
    local inference using the transformers library.
    """
    
    provider_type = "huggingface"
    
    def __init__(
        self,
        model_name: str,
        token_env_var: Optional[str] = "HF_API_TOKEN",
        use_auth_token: bool = True,
        use_api: bool = False,
        device: str = "cuda",
        batch_size: int = 1,
        **kwargs
    ):
        """
        Initialize the HuggingFace provider.
        
        Args:
            model_name: Name of the model on HuggingFace Hub
            token_env_var: Environment variable containing the API token
            use_auth_token: Whether to use authentication token for private models
            use_api: Whether to use the HuggingFace Inference API
            device: Device to use for local inference (cuda, cpu)
            batch_size: Batch size for local inference
            **kwargs: Additional keyword arguments
        """
        super().__init__(model_name=model_name, **kwargs)
        
        self.use_api = use_api
        self.device = device
        self.batch_size = batch_size
        
        # Get API token if specified
        self.api_token = None
        if token_env_var and use_auth_token:
            self.api_token = os.environ.get(token_env_var)
        
        # Model and tokenizer for local inference
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._embedding_model = None
        
        # Initialize based on mode
        if use_api:
            self._init_api_client()
        else:
            self._init_local_model()
    
    def _init_api_client(self):
        """
        Initialize the HuggingFace Inference API client.
        """
        try:
            # Import required modules
            from huggingface_hub import InferenceClient
            
            # Initialize the client
            self._client = InferenceClient(
                token=self.api_token
            )
            
            logger.info(f"Initialized HuggingFace Inference API client for {self.model_name}")
            
        except ImportError:
            raise ProviderError("huggingface_hub is not installed. Please install it with 'pip install huggingface_hub'")
        except Exception as e:
            raise ProviderError(f"Failed to initialize HuggingFace API client: {str(e)}")
    
    def _init_local_model(self):
        """
        Initialize the local model using the transformers library.
        """
        try:
            # Import required modules
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, pipeline
            
            # Check if we should use GPU
            use_gpu = self.device == "cuda" and torch.cuda.is_available()
            
            if not use_gpu and self.device == "cuda":
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            
            # Initialize tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=self.api_token
            )
            
            # Check model type and load appropriate model
            # Try to determine if it's a text generation model
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    use_auth_token=self.api_token,
                    device_map=self.device if use_gpu else None,
                    torch_dtype=torch.float16 if use_gpu else torch.float32
                )
                
                # Create text generation pipeline
                self._pipeline = pipeline(
                    "text-generation",
                    model=self._model,
                    tokenizer=self._tokenizer,
                    device=0 if use_gpu else -1
                )
                
                logger.info(f"Initialized local HuggingFace model {self.model_name} for text generation")
                
            except Exception as e:
                # It might be an embedding model
                logger.warning(f"Failed to load as generation model, trying as embedding model: {str(e)}")
                
                try:
                    self._embedding_model = AutoModel.from_pretrained(
                        self.model_name,
                        use_auth_token=self.api_token,
                        device_map=self.device if use_gpu else None
                    )
                    
                    logger.info(f"Initialized local HuggingFace model {self.model_name} for embeddings")
                    
                except Exception as e2:
                    raise ProviderError(f"Failed to load model as either generation or embedding model: {str(e2)}")
            
        except ImportError as e:
            raise ProviderError(f"Required module not found: {str(e)}")
        except Exception as e:
            raise ProviderError(f"Failed to initialize local HuggingFace model: {str(e)}")
    
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
        start_time = time.time()
        error = None
        
        try:
            if self.use_api:
                return await self._generate_api(prompt, max_tokens, temperature, top_p, stop, **kwargs)
            else:
                return await self._generate_local(prompt, max_tokens, temperature, top_p, stop, **kwargs)
                
        except Exception as e:
            error = str(e)
            logger.error(f"Error generating text with HuggingFace: {error}")
            raise ProviderError(f"HuggingFace generation error: {error}")
        
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.debug(f"HuggingFace generation took {elapsed_time:.2f}s")
    
    async def _generate_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text using the HuggingFace Inference API.
        """
        # Prepare parameters
        params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "return_full_text": False
        }
        
        # Update with any additional parameters
        params.update(kwargs)
        
        # Call the API
        response = self._client.text_generation(
            prompt,
            model=self.model_name,
            **params
        )
        
        # Prepare metadata
        metadata = {
            "model": self.model_name,
            "provider": "huggingface",
            "api": True
        }
        
        # Post-process to handle stop sequences
        text = response
        if stop:
            for stop_seq in stop:
                if stop_seq in text:
                    text = text[:text.find(stop_seq)]
        
        return text, metadata
    
    async def _generate_local(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text using the local model.
        """
        if self._pipeline is None:
            raise ProviderError("Local model is not initialized or is not a text generation model")
        
        # Prepare generation parameters
        params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id
        }
        
        # Update with any additional parameters
        params.update(kwargs)
        
        # Run in separate thread to avoid blocking
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                self._pipeline,
                prompt,
                **params
            )
            response = future.result()
        
        # Extract generated text
        if isinstance(response, list) and len(response) > 0:
            if "generated_text" in response[0]:
                text = response[0]["generated_text"]
            else:
                text = response[0]
        else:
            text = str(response)
        
        # Remove the prompt if it's included in the output
        if text.startswith(prompt):
            text = text[len(prompt):]
        
        # Post-process to handle stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in text:
                    text = text[:text.find(stop_seq)]
        
        # Count tokens
        input_tokens = len(self._tokenizer.encode(prompt))
        output_tokens = len(self._tokenizer.encode(text))
        
        # Prepare metadata
        metadata = {
            "model": self.model_name,
            "provider": "huggingface",
            "api": False,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
        
        return text, metadata
    
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
        
        # Process API requests individually
        if self.use_api:
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
                    results.append(("", {"error": str(e), "provider": "huggingface"}))
            
            return results
        
        # For local models, process in batches
        if self._pipeline is None:
            raise ProviderError("Local model is not initialized or is not a text generation model")
        
        # Prepare generation parameters
        params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id
        }
        
        # Update with any additional parameters
        params.update(kwargs)
        
        # Process in batches
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i+self.batch_size]
            
            try:
                # Run in separate thread to avoid blocking
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self._pipeline,
                        batch_prompts,
                        **params
                    )
                    batch_responses = future.result()
                
                # Process each response in the batch
                for j, response in enumerate(batch_responses):
                    prompt = batch_prompts[j]
                    
                    try:
                        # Extract generated text
                        if isinstance(response, dict) and "generated_text" in response:
                            text = response["generated_text"]
                        else:
                            text = str(response)
                        
                        # Remove the prompt if it's included in the output
                        if text.startswith(prompt):
                            text = text[len(prompt):]
                        
                        # Post-process to handle stop sequences
                        if stop:
                            for stop_seq in stop:
                                if stop_seq in text:
                                    text = text[:text.find(stop_seq)]
                        
                        # Count tokens
                        input_tokens = len(self._tokenizer.encode(prompt))
                        output_tokens = len(self._tokenizer.encode(text))
                        
                        # Prepare metadata
                        metadata = {
                            "model": self.model_name,
                            "provider": "huggingface",
                            "api": False,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens
                        }
                        
                        results.append((text, metadata))
                        
                    except Exception as e:
                        logger.error(f"Error processing batch response: {str(e)}")
                        results.append(("", {"error": str(e), "provider": "huggingface"}))
                
            except Exception as e:
                logger.error(f"Error in batch generation: {str(e)}")
                # Add error results for all prompts in the batch
                for _ in batch_prompts:
                    results.append(("", {"error": str(e), "provider": "huggingface"}))
        
        return results
    
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
            
        Raises:
            ProviderError: If embeddings are not supported by the model
        """
        start_time = time.time()
        
        try:
            if self.use_api:
                return await self._get_embedding_api(text, **kwargs)
            else:
                return await self._get_embedding_local(text, **kwargs)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error getting embedding with HuggingFace: {error_msg}")
            raise ProviderError(f"HuggingFace embedding error: {error_msg}")
        
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.debug(f"HuggingFace embedding took {elapsed_time:.2f}s")
    
    async def _get_embedding_api(
        self,
        text: str,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get embedding using the HuggingFace Inference API.
        """
        # Call the API
        response = self._client.feature_extraction(
            text,
            model=self.model_name
        )
        
        # Process response
        if isinstance(response, list):
            # Average token embeddings if we got token-level embeddings
            embedding = np.mean(response, axis=0)
        else:
            embedding = np.array(response)
        
        # Prepare metadata
        metadata = {
            "model": self.model_name,
            "provider": "huggingface",
            "api": True,
            "dimensions": embedding.shape[0]
        }
        
        return embedding, metadata
    
    async def _get_embedding_local(
        self,
        text: str,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get embedding using the local model.
        """
        if self._embedding_model is None and self._model is None:
            raise ProviderError("Local model is not initialized or does not support embeddings")
        
        # Import torch
        import torch
        
        # Use embedding model if available, otherwise use the text generation model
        model = self._embedding_model if self._embedding_model is not None else self._model
        
        # Encode the input
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run in separate thread to avoid blocking
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                lambda: model(**inputs, output_hidden_states=True)
            )
            with torch.no_grad():
                outputs = future.result()
        
        # Extract embeddings - different models output different formats
        if hasattr(outputs, "last_hidden_state"):
            # For encoder models like BERT
            hidden_states = outputs.last_hidden_state
            # Average token embeddings (ignoring padding)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                # Mask padded values
                masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                sum_embeddings = torch.sum(masked_hidden, dim=1)
                sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
                embedding = sum_embeddings / sum_mask
            else:
                # Simple average if no mask
                embedding = torch.mean(hidden_states, dim=1)
        elif hasattr(outputs, "hidden_states"):
            # For decoder models, use the last layer hidden state
            hidden_states = outputs.hidden_states[-1]
            # Use the last token's embedding for generation models
            embedding = hidden_states[:, -1, :]
        else:
            raise ProviderError("Unsupported model format for embedding extraction")
        
        # Convert to numpy array
        embedding = embedding.cpu().numpy().squeeze()
        
        # Prepare metadata
        metadata = {
            "model": self.model_name,
            "provider": "huggingface",
            "api": False,
            "dimensions": embedding.shape[0]
        }
        
        return embedding, metadata
    
    async def get_batch_embeddings(
        self,
        texts: List[str],
        **kwargs
    ) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to get embeddings for
            **kwargs: Additional parameters
            
        Returns:
            List[Tuple[np.ndarray, Dict[str, Any]]]: List of embedding vectors and metadata
        """
        results = []
        
        # Process API requests individually
        if self.use_api:
            for text in texts:
                try:
                    embedding, metadata = await self.get_embedding(text, **kwargs)
                    results.append((embedding, metadata))
                except Exception as e:
                    logger.error(f"Error getting embedding for text: {text[:50]}...: {str(e)}")
                    results.append((np.array([]), {"error": str(e), "provider": "huggingface"}))
            
            return results
        
        # For local models, batch process
        if self._embedding_model is None and self._model is None:
            raise ProviderError("Local model is not initialized or does not support embeddings")
        
        # Import torch
        import torch
        
        # Use embedding model if available, otherwise use the text generation model
        model = self._embedding_model if self._embedding_model is not None else self._model
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            try:
                # Encode the inputs
                inputs = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move to same device as model
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Run in separate thread to avoid blocking
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: model(**inputs, output_hidden_states=True)
                    )
                    with torch.no_grad():
                        outputs = future.result()
                
                # Extract embeddings - different models output different formats
                if hasattr(outputs, "last_hidden_state"):
                    # For encoder models like BERT
                    hidden_states = outputs.last_hidden_state
                    # Average token embeddings (ignoring padding)
                    attention_mask = inputs.get("attention_mask", None)
                    if attention_mask is not None:
                        # Mask padded values
                        masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                        sum_embeddings = torch.sum(masked_hidden, dim=1)
                        sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
                        embeddings = sum_embeddings / sum_mask
                    else:
                        # Simple average if no mask
                        embeddings = torch.mean(hidden_states, dim=1)
                elif hasattr(outputs, "hidden_states"):
                    # For decoder models, use the last layer hidden state
                    hidden_states = outputs.hidden_states[-1]
                    # Use the last token's embedding for generation models
                    embeddings = hidden_states[:, -1, :]
                else:
                    raise ProviderError("Unsupported model format for embedding extraction")
                
                # Convert to numpy array
                embeddings = embeddings.cpu().numpy()
                
                # Create results for each text
                for j, embedding in enumerate(embeddings):
                    metadata = {
                        "model": self.model_name,
                        "provider": "huggingface",
                        "api": False,
                        "dimensions": embedding.shape[0]
                    }
                    results.append((embedding, metadata))
                
            except Exception as e:
                logger.error(f"Error in batch embedding: {str(e)}")
                # Add error results for all texts in the batch
                for _ in batch_texts:
                    results.append((np.array([]), {"error": str(e), "provider": "huggingface"}))
        
        return results
    
    def cleanup(self):
        """
        Clean up resources used by the provider.
        """
        # Clear model and tokenizer to free up memory
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._embedding_model = None
        
        # Force garbage collection
        try:
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


# Register the provider
from . import register_provider
register_provider(HuggingFaceProvider)