"""
Data collection functionality for LLM Drift Detector.

This module handles collecting and storing LLM inputs and outputs
from various providers for later analysis.
"""

import os
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import json
import pandas as pd
from pydantic import BaseModel, Field
import aiohttp
import asyncio
from pathlib import Path
import numpy as np

from ..config import Config, get_config

logger = logging.getLogger(__name__)

class LLMSample(BaseModel):
    """Individual LLM input/output sample."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    provider_name: str
    model_name: str
    prompt: str
    response: str
    response_time: Optional[float] = None
    token_count: Optional[int] = None
    total_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            np.ndarray: lambda arr: arr.tolist()
        }


class SampleBatch(BaseModel):
    """Batch of LLM samples."""
    
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    samples: List[LLMSample] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert batch to pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame representation of the batch
        """
        samples_dict = [sample.dict() for sample in self.samples]
        return pd.DataFrame(samples_dict)
    
    def save(self, directory: Union[str, Path]) -> Path:
        """
        Save batch to a file.
        
        Args:
            directory: Directory to save the file
            
        Returns:
            Path: Path to the saved file
        """
        directory = Path(directory)
        os.makedirs(directory, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_samples_{timestamp}_{self.batch_id}.json"
        file_path = directory / filename
        
        with open(file_path, 'w') as f:
            json.dump(self.dict(), f, default=str)
        
        logger.debug(f"Saved batch {self.batch_id} to {file_path}")
        return file_path


class DataCollector:
    """
    Collects and manages LLM samples.
    
    This class provides a provider-agnostic way to collect and store
    LLM inputs and outputs for drift detection.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data collector.
        
        Args:
            config: Configuration object, uses global config if None
        """
        self.config = config or get_config()
        self.storage_path = Path(self.config.get("data.collection.storage_path", "./data/collected/"))
        self.batch_size = self.config.get("data.collection.batch_size", 100)
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Current batch of samples
        self.current_batch = SampleBatch()
        
        # Async session for making HTTP requests
        self._session = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp session.
        
        Returns:
            aiohttp.ClientSession: Session for making HTTP requests
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    def add_sample(self, sample: Union[LLMSample, Dict[str, Any]]) -> LLMSample:
        """
        Add a sample to the current batch.
        
        Args:
            sample: LLM sample or dictionary with sample data
            
        Returns:
            LLMSample: Added sample
        """
        if isinstance(sample, dict):
            sample = LLMSample(**sample)
        
        self.current_batch.samples.append(sample)
        
        # Save batch if it reaches the configured size
        if len(self.current_batch.samples) >= self.batch_size:
            self.save_current_batch()
        
        return sample
    
    def create_sample(self, 
                      provider_name: str,
                      model_name: str,
                      prompt: str,
                      response: str,
                      **kwargs) -> LLMSample:
        """
        Create and add a new LLM sample.
        
        Args:
            provider_name: Name of the LLM provider
            model_name: Name of the model
            prompt: Input prompt
            response: Model response
            **kwargs: Additional sample metadata
            
        Returns:
            LLMSample: Created sample
        """
        sample = LLMSample(
            provider_name=provider_name,
            model_name=model_name,
            prompt=prompt,
            response=response,
            **kwargs
        )
        
        return self.add_sample(sample)
    
    def save_current_batch(self) -> Optional[Path]:
        """
        Save the current batch to storage and start a new batch.
        
        Returns:
            Optional[Path]: Path to the saved file, or None if batch was empty
        """
        if not self.current_batch.samples:
            return None
        
        file_path = self.current_batch.save(self.storage_path)
        
        # Start a new batch
        self.current_batch = SampleBatch()
        
        return file_path
    
    async def collect_samples_async(self,
                                   provider_name: str,
                                   model_name: str, 
                                   prompts: List[str],
                                   api_func,
                                   **kwargs) -> List[LLMSample]:
        """
        Collect multiple samples asynchronously using the provided API function.
        
        Args:
            provider_name: Name of the LLM provider
            model_name: Name of the model
            prompts: List of prompts to send to the LLM
            api_func: Async function that makes the API request
            **kwargs: Additional arguments to pass to the API function
            
        Returns:
            List[LLMSample]: Collected samples
        """
        samples = []
        
        async def process_prompt(prompt: str) -> LLMSample:
            """Process a single prompt."""
            start_time = time.time()
            error = None
            response_text = None
            metadata = {}
            
            try:
                response = await api_func(prompt, **kwargs)
                
                if isinstance(response, tuple) and len(response) == 2:
                    response_text, metadata = response
                else:
                    response_text = response
                
            except Exception as e:
                error = str(e)
                logger.warning(f"Error collecting sample: {str(e)}")
                response_text = ""
            
            end_time = time.time()
            response_time = end_time - start_time
            
            sample = self.create_sample(
                provider_name=provider_name,
                model_name=model_name,
                prompt=prompt,
                response=response_text,
                response_time=response_time,
                error=error,
                metadata=metadata
            )
            
            return sample
        
        # Process prompts concurrently with a semaphore to limit concurrency
        max_concurrency = kwargs.pop('max_concurrency', 5)
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def bounded_process(prompt: str) -> LLMSample:
            """Process prompt with bounded concurrency."""
            async with semaphore:
                return await process_prompt(prompt)
        
        # Create tasks for all prompts
        tasks = [bounded_process(prompt) for prompt in prompts]
        
        # Wait for all tasks to complete
        samples = await asyncio.gather(*tasks)
        
        return samples
    
    def load_samples(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    provider_filter: Optional[List[str]] = None,
                    model_filter: Optional[List[str]] = None,
                    max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load collected samples from storage.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            provider_filter: Optional list of providers to include
            model_filter: Optional list of models to include
            max_samples: Optional maximum number of samples to load
            
        Returns:
            pd.DataFrame: DataFrame containing the loaded samples
        """
        all_samples = []
        files = sorted(self.storage_path.glob("llm_samples_*.json"))
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    batch_data = json.load(f)
                
                batch = SampleBatch(**batch_data)
                
                # Apply filters
                filtered_samples = []
                for sample in batch.samples:
                    # Convert string timestamp to datetime if needed
                    if isinstance(sample.timestamp, str):
                        sample.timestamp = datetime.fromisoformat(sample.timestamp)
                    
                    # Apply date filters
                    if start_date and sample.timestamp < start_date:
                        continue
                    if end_date and sample.timestamp > end_date:
                        continue
                    
                    # Apply provider filter
                    if provider_filter and sample.provider_name not in provider_filter:
                        continue
                    
                    # Apply model filter
                    if model_filter and sample.model_name not in model_filter:
                        continue
                    
                    filtered_samples.append(sample)
                
                all_samples.extend(filtered_samples)
                
                # Check if we've reached the maximum number of samples
                if max_samples and len(all_samples) >= max_samples:
                    all_samples = all_samples[:max_samples]
                    break
                
            except Exception as e:
                logger.warning(f"Error loading samples from {file_path}: {str(e)}")
        
        # Convert to DataFrame
        if not all_samples:
            return pd.DataFrame()
        
        samples_dict = [s.dict() for s in all_samples]
        return pd.DataFrame(samples_dict)
    
    async def close(self):
        """Close the collector and any open resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        
        # Save any remaining samples
        self.save_current_batch()


# Context manager support
async def collector_context():
    """Async context manager for the data collector."""
    collector = get_collector()
    try:
        yield collector
    finally:
        await collector.close()