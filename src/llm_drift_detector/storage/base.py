"""
Base storage interface for LLM Drift Detector.

This module defines the abstract interface for all storage backends,
ensuring consistent data access across different implementations.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Iterator
from datetime import datetime
import pandas as pd
import numpy as np
import json
import pickle

logger = logging.getLogger(__name__)

class StorageError(Exception):
    """Exception raised for storage-specific errors."""
    pass


class BaseStorage(ABC):
    """
    Abstract base class for storage backends.
    
    This class defines the interface that all storage implementations
    must follow, ensuring consistent data access across different backends.
    """
    
    # Class attribute to identify the storage type
    storage_type: str = "base"
    
    def __init__(self, **kwargs):
        """
        Initialize the storage backend.
        
        Args:
            **kwargs: Backend-specific configuration
        """
        self.initialized = False
    
    @abstractmethod
    def store_samples(self, samples: Union[List[Dict], pd.DataFrame], collection: str = "samples") -> bool:
        """
        Store LLM samples in the storage backend.
        
        Args:
            samples: List of sample dictionaries or DataFrame with samples
            collection: Name of the collection/table to store in
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_samples(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    provider: Optional[str] = None,
                    model: Optional[str] = None,
                    limit: Optional[int] = None,
                    collection: str = "samples") -> pd.DataFrame:
        """
        Load LLM samples from the storage backend.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            provider: Optional provider filter
            model: Optional model filter
            limit: Optional maximum number of samples to return
            collection: Name of the collection/table to load from
            
        Returns:
            pd.DataFrame: DataFrame containing the samples
        """
        pass
    
    @abstractmethod
    def store_reference_distribution(self, distribution: Dict[str, Any], key: str = "reference") -> bool:
        """
        Store a reference distribution.
        
        Args:
            distribution: Dictionary containing reference distribution data
            key: Identifier for the distribution
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_reference_distribution(self, key: str = "reference") -> Optional[Dict[str, Any]]:
        """
        Load a reference distribution.
        
        Args:
            key: Identifier for the distribution
            
        Returns:
            Optional[Dict[str, Any]]: Reference distribution or None if not found
        """
        pass
    
    @abstractmethod
    def store_drift_metrics(self, metrics: Union[List[Dict], pd.DataFrame], collection: str = "metrics") -> bool:
        """
        Store drift detection metrics.
        
        Args:
            metrics: List of metric dictionaries or DataFrame with metrics
            collection: Name of the collection/table to store in
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_drift_metrics(self,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          detector: Optional[str] = None,
                          provider: Optional[str] = None, 
                          model: Optional[str] = None,
                          limit: Optional[int] = None,
                          collection: str = "metrics") -> pd.DataFrame:
        """
        Load drift detection metrics.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            detector: Optional detector type filter
            provider: Optional provider filter
            model: Optional model filter
            limit: Optional maximum number of metrics to return
            collection: Name of the collection/table to load from
            
        Returns:
            pd.DataFrame: DataFrame containing the metrics
        """
        pass
    
    @abstractmethod
    def store_embeddings(self, embeddings: Dict[str, np.ndarray], key: str = "embeddings") -> bool:
        """
        Store embeddings.
        
        Args:
            embeddings: Dictionary mapping IDs to embedding vectors
            key: Identifier for the embeddings set
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_embeddings(self, key: str = "embeddings") -> Optional[Dict[str, np.ndarray]]:
        """
        Load embeddings.
        
        Args:
            key: Identifier for the embeddings set
            
        Returns:
            Optional[Dict[str, np.ndarray]]: Dictionary of embeddings or None if not found
        """
        pass
    
    def serialize_for_storage(self, data: Any) -> bytes:
        """
        Serialize data for storage.
        
        This utility method provides a standard way to serialize data
        across different storage backends.
        
        Args:
            data: Data to serialize
            
        Returns:
            bytes: Serialized data
        """
        try:
            # Try to pickle the data first (handles complex objects including numpy arrays)
            return pickle.dumps(data)
        except:
            # Fall back to JSON for simpler data types
            try:
                # Convert numpy arrays to lists and serialize as JSON
                if isinstance(data, dict):
                    json_data = {}
                    for key, value in data.items():
                        if isinstance(value, np.ndarray):
                            json_data[key] = value.tolist()
                        else:
                            json_data[key] = value
                    return json.dumps(json_data).encode('utf-8')
                else:
                    return json.dumps(data).encode('utf-8')
            except:
                # Last resort: convert to string
                return str(data).encode('utf-8')
    
    def deserialize_from_storage(self, data: bytes) -> Any:
        """
        Deserialize data from storage.
        
        This utility method provides a standard way to deserialize data
        across different storage backends.
        
        Args:
            data: Serialized data
            
        Returns:
            Any: Deserialized data
        """
        try:
            # Try to unpickle the data first
            return pickle.loads(data)
        except:
            # Fall back to JSON
            try:
                json_data = json.loads(data.decode('utf-8'))
                
                # Try to detect and convert lists back to numpy arrays
                if isinstance(json_data, dict):
                    for key, value in json_data.items():
                        if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
                            json_data[key] = np.array(value)
                
                return json_data
            except:
                # Last resort: return as string
                return data.decode('utf-8')
    
    def cleanup(self):
        """
        Clean up resources used by the storage backend.
        
        This method should be implemented by storage backends that need to
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


# Register this class (doesn't do anything for the base class, but included for consistency)
from . import register_storage
register_storage(BaseStorage)