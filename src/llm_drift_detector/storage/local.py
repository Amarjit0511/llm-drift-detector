"""
Local file-based storage for LLM Drift Detector.

This module provides a storage backend that uses the local filesystem
for storing drift detection data, samples, and metrics.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Iterator
from datetime import datetime
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path

from .base import BaseStorage, StorageError

logger = logging.getLogger(__name__)

class LocalStorage(BaseStorage):
    """
    Local file-based storage backend.
    
    This storage backend uses the local filesystem to store data,
    supporting various formats (JSON, CSV, Parquet, Pickle).
    """
    
    storage_type = "local"
    
    def __init__(
        self,
        directory: str = "./data/storage/",
        format: str = "parquet",
        create_dirs: bool = True,
        compression: bool = True,
        **kwargs
    ):
        """
        Initialize the local storage backend.
        
        Args:
            directory: Base directory for storage
            format: File format ('json', 'csv', 'parquet', 'pickle')
            create_dirs: Whether to create directories if they don't exist
            compression: Whether to compress stored data
            **kwargs: Additional configuration options
        """
        super().__init__(**kwargs)
        
        self.base_dir = Path(directory)
        self.format = format.lower()
        self.compression = compression
        
        # Validate format
        valid_formats = ["json", "csv", "parquet", "pickle"]
        if self.format not in valid_formats:
            logger.warning(f"Invalid format '{format}', defaulting to 'parquet'")
            self.format = "parquet"
        
        # Create directories if specified
        if create_dirs:
            self._create_directories()
        
        self.initialized = True
        logger.info(f"Initialized local storage in {self.base_dir} using {self.format} format")
    
    def _create_directories(self):
        """Create required directories if they don't exist."""
        # Create base directory
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Create subdirectories for different types of data
        dirs = ["samples", "metrics", "reference", "embeddings"]
        for subdir in dirs:
            os.makedirs(self.base_dir / subdir, exist_ok=True)
    
    def _get_file_path(self, collection: str, timestamp: Optional[datetime] = None) -> Path:
        """
        Get the file path for a collection.
        
        Args:
            collection: Name of the collection
            timestamp: Optional timestamp to include in filename
            
        Returns:
            Path: File path
        """
        if timestamp:
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{collection}_{timestamp_str}.{self.format}"
        else:
            filename = f"{collection}.{self.format}"
        
        return self.base_dir / collection / filename
    
    def store_samples(self, samples: Union[List[Dict], pd.DataFrame], collection: str = "samples") -> bool:
        """
        Store LLM samples in the local filesystem.
        
        Args:
            samples: List of sample dictionaries or DataFrame with samples
            collection: Name of the collection to store in
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not self.initialized:
            raise StorageError("Storage not initialized")
        
        try:
            # Convert to DataFrame if not already
            if not isinstance(samples, pd.DataFrame):
                df = pd.DataFrame(samples)
            else:
                df = samples
            
            # Create timestamp for the filename
            timestamp = datetime.now()
            file_path = self._get_file_path(collection, timestamp)
            
            # Ensure directory exists
            os.makedirs(file_path.parent, exist_ok=True)
            
            # Save based on format
            if self.format == "json":
                df.to_json(file_path, orient="records", date_format="iso")
            elif self.format == "csv":
                df.to_csv(file_path, index=False)
            elif self.format == "parquet":
                compression = "snappy" if self.compression else None
                df.to_parquet(file_path, compression=compression)
            elif self.format == "pickle":
                df.to_pickle(file_path)
            
            logger.debug(f"Stored {len(df)} samples to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing samples: {str(e)}")
            return False
    
    def load_samples(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    provider: Optional[str] = None,
                    model: Optional[str] = None,
                    limit: Optional[int] = None,
                    collection: str = "samples") -> pd.DataFrame:
        """
        Load LLM samples from the local filesystem.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            provider: Optional provider filter
            model: Optional model filter
            limit: Optional maximum number of samples to return
            collection: Name of the collection to load from
            
        Returns:
            pd.DataFrame: DataFrame containing the samples
        """
        if not self.initialized:
            raise StorageError("Storage not initialized")
        
        collection_dir = self.base_dir / collection
        if not collection_dir.exists():
            logger.warning(f"Collection directory {collection_dir} does not exist")
            return pd.DataFrame()
        
        # Find all files in the collection directory with the right extension
        files = list(collection_dir.glob(f"*.{self.format}"))
        if not files:
            logger.warning(f"No {self.format} files found in {collection_dir}")
            return pd.DataFrame()
        
        # Sort files by name (which includes timestamp)
        files.sort()
        
        # Load and concatenate files
        dfs = []
        for file_path in files:
            try:
                # Load file based on format
                if self.format == "json":
                    df = pd.read_json(file_path, orient="records")
                elif self.format == "csv":
                    df = pd.read_csv(file_path)
                elif self.format == "parquet":
                    df = pd.read_parquet(file_path)
                elif self.format == "pickle":
                    df = pd.read_pickle(file_path)
                
                # Apply filters
                if "timestamp" in df.columns:
                    # Convert to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                    
                    # Apply date filters
                    if start_date:
                        df = df[df["timestamp"] >= start_date]
                    if end_date:
                        df = df[df["timestamp"] <= end_date]
                
                if provider and "provider_name" in df.columns:
                    df = df[df["provider_name"] == provider]
                
                if model and "model_name" in df.columns:
                    df = df[df["model_name"] == model]
                
                # Skip if empty after filtering
                if len(df) == 0:
                    continue
                
                dfs.append(df)
                
                # Check if we've reached the limit
                if limit and sum(len(df) for df in dfs) >= limit:
                    break
                
            except Exception as e:
                logger.warning(f"Error loading file {file_path}: {str(e)}")
        
        # Return empty DataFrame if no data
        if not dfs:
            return pd.DataFrame()
        
        # Concatenate all dataframes
        result = pd.concat(dfs, ignore_index=True)
        
        # Apply final limit if needed
        if limit and len(result) > limit:
            result = result.head(limit)
        
        return result
    
    def store_reference_distribution(self, distribution: Dict[str, Any], key: str = "reference") -> bool:
        """
        Store a reference distribution.
        
        Args:
            distribution: Dictionary containing reference distribution data
            key: Identifier for the distribution
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not self.initialized:
            raise StorageError("Storage not initialized")
        
        try:
            # Create file path
            file_path = self.base_dir / "reference" / f"{key}.pkl"
            
            # Ensure directory exists
            os.makedirs(file_path.parent, exist_ok=True)
            
            # Save using pickle for complex objects
            with open(file_path, 'wb') as f:
                pickle.dump(distribution, f)
            
            logger.debug(f"Stored reference distribution '{key}' to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing reference distribution: {str(e)}")
            return False
    
    def load_reference_distribution(self, key: str = "reference") -> Optional[Dict[str, Any]]:
        """
        Load a reference distribution.
        
        Args:
            key: Identifier for the distribution
            
        Returns:
            Optional[Dict[str, Any]]: Reference distribution or None if not found
        """
        if not self.initialized:
            raise StorageError("Storage not initialized")
        
        file_path = self.base_dir / "reference" / f"{key}.pkl"
        if not file_path.exists():
            logger.warning(f"Reference distribution file {file_path} does not exist")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                distribution = pickle.load(f)
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error loading reference distribution: {str(e)}")
            return None
    
    def store_drift_metrics(self, metrics: Union[List[Dict], pd.DataFrame], collection: str = "metrics") -> bool:
        """
        Store drift detection metrics.
        
        Args:
            metrics: List of metric dictionaries or DataFrame with metrics
            collection: Name of the collection to store in
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        # Implementation is the same as store_samples
        return self.store_samples(metrics, collection=collection)
    
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
            collection: Name of the collection to load from
            
        Returns:
            pd.DataFrame: DataFrame containing the metrics
        """
        # Load the base data
        df = self.load_samples(
            start_date=start_date,
            end_date=end_date,
            provider=provider,
            model=model,
            limit=limit,
            collection=collection
        )
        
        # Apply detector filter if specified
        if detector and "detector_name" in df.columns and not df.empty:
            df = df[df["detector_name"] == detector]
        
        return df
    
    def store_embeddings(self, embeddings: Dict[str, np.ndarray], key: str = "embeddings") -> bool:
        """
        Store embeddings.
        
        Args:
            embeddings: Dictionary mapping IDs to embedding vectors
            key: Identifier for the embeddings set
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not self.initialized:
            raise StorageError("Storage not initialized")
        
        try:
            # Create file path
            file_path = self.base_dir / "embeddings" / f"{key}.npz"
            
            # Ensure directory exists
            os.makedirs(file_path.parent, exist_ok=True)
            
            # Convert to arrays for efficient storage
            ids = list(embeddings.keys())
            vectors = np.array(list(embeddings.values()))
            
            # Save using numpy's compressed format
            np.savez_compressed(file_path, ids=ids, vectors=vectors)
            
            logger.debug(f"Stored {len(embeddings)} embeddings to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            return False
    
    def load_embeddings(self, key: str = "embeddings") -> Optional[Dict[str, np.ndarray]]:
        """
        Load embeddings.
        
        Args:
            key: Identifier for the embeddings set
            
        Returns:
            Optional[Dict[str, np.ndarray]]: Dictionary of embeddings or None if not found
        """
        if not self.initialized:
            raise StorageError("Storage not initialized")
        
        file_path = self.base_dir / "embeddings" / f"{key}.npz"
        if not file_path.exists():
            logger.warning(f"Embeddings file {file_path} does not exist")
            return None
        
        try:
            # Load numpy arrays
            data = np.load(file_path)
            ids = data['ids']
            vectors = data['vectors']
            
            # Reconstruct dictionary
            embeddings = {id: vector for id, vector in zip(ids, vectors)}
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            return None
    
    def cleanup(self):
        """Clean up resources (no-op for local storage)."""
        self.initialized = False


# Register this storage type
from . import register_storage
register_storage(LocalStorage)