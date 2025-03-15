"""
Redis storage for LLM Drift Detector.

This module provides a storage backend that uses Redis for storing
drift detection data, samples, and metrics, offering high-performance
and distributed storage capabilities.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Iterator
from datetime import datetime
import pandas as pd
import numpy as np
import json
import pickle
import hashlib
import time

from .base import BaseStorage, StorageError

logger = logging.getLogger(__name__)

class RedisStorage(BaseStorage):
    """
    Redis storage backend.
    
    This storage backend uses Redis for high-performance, in-memory
    storage with optional persistence. It supports various data types
    and efficient querying capabilities.
    """
    
    storage_type = "redis"
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "llm_drift:",
        expiration_days: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the Redis storage backend.
        
        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Optional Redis password
            prefix: Key prefix for all stored data
            expiration_days: Optional expiration time in days
            **kwargs: Additional Redis configuration
        """
        super().__init__(**kwargs)
        
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.prefix = prefix
        self.expiration_seconds = expiration_days * 86400 if expiration_days else None
        
        # Initialize Redis client
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize the Redis client."""
        try:
            import redis
            
            # Connect to Redis
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False  # Keep binary data as-is
            )
            
            # Test connection
            self._client.ping()
            
            self.initialized = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            
        except ImportError:
            logger.error("Redis package not installed. Install with 'pip install redis'")
            raise StorageError("Redis package not installed")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise StorageError(f"Redis connection error: {str(e)}")
    
    def _get_key(self, collection: str, id: Optional[str] = None) -> str:
        """
        Get a Redis key with the proper prefix.
        
        Args:
            collection: Collection name
            id: Optional ID within the collection
            
        Returns:
            str: Full Redis key
        """
        if id:
            return f"{self.prefix}{collection}:{id}"
        else:
            return f"{self.prefix}{collection}"
    
    def _get_timestamp_key(self, collection: str, timestamp: datetime) -> str:
        """
        Get a time-based key for sorted sets.
        
        Args:
            collection: Collection name
            timestamp: Timestamp
            
        Returns:
            str: Time-based key
        """
        ts_str = timestamp.strftime("%Y%m%d%H%M%S")
        return f"{self.prefix}{collection}:ts:{ts_str}"
    
    def store_samples(self, samples: Union[List[Dict], pd.DataFrame], collection: str = "samples") -> bool:
        """
        Store LLM samples in Redis.
        
        Args:
            samples: List of sample dictionaries or DataFrame with samples
            collection: Name of the collection to store in
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not self.initialized:
            raise StorageError("Redis storage not initialized")
        
        try:
            # Convert to list of dicts if DataFrame
            if isinstance(samples, pd.DataFrame):
                samples_list = samples.to_dict(orient="records")
            else:
                samples_list = samples
            
            # Use pipeline for better performance
            pipeline = self._client.pipeline()
            
            # Collection index key
            collection_key = self._get_key(collection)
            
            # Store each sample
            for sample in samples_list:
                # Generate a unique ID if not present
                if "id" not in sample:
                    sample_id = hashlib.md5(json.dumps(sample, default=str).encode()).hexdigest()
                    sample["id"] = sample_id
                else:
                    sample_id = str(sample["id"])
                
                # Get timestamp for indexing
                if "timestamp" in sample:
                    if isinstance(sample["timestamp"], datetime):
                        timestamp = sample["timestamp"]
                    elif isinstance(sample["timestamp"], str):
                        timestamp = pd.to_datetime(sample["timestamp"])
                    else:
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()
                    sample["timestamp"] = timestamp
                
                # Serialize the sample
                serialized = self.serialize_for_storage(sample)
                
                # Store the sample by ID
                sample_key = self._get_key(collection, sample_id)
                pipeline.set(sample_key, serialized)
                
                # Add to collection index
                pipeline.sadd(collection_key, sample_id)
                
                # Store by timestamp in a sorted set
                score = time.mktime(timestamp.timetuple())
                pipeline.zadd(f"{collection_key}:by_time", {sample_id: score})
                
                # Add provider index if available
                if "provider_name" in sample:
                    provider = sample["provider_name"]
                    pipeline.sadd(f"{collection_key}:provider:{provider}", sample_id)
                
                # Add model index if available
                if "model_name" in sample:
                    model = sample["model_name"]
                    pipeline.sadd(f"{collection_key}:model:{model}", sample_id)
                
                # Set expiration if configured
                if self.expiration_seconds:
                    pipeline.expire(sample_key, self.expiration_seconds)
            
            # Execute pipeline
            pipeline.execute()
            
            logger.debug(f"Stored {len(samples_list)} samples in Redis under {collection_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing samples in Redis: {str(e)}")
            return False
    
    def load_samples(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    provider: Optional[str] = None,
                    model: Optional[str] = None,
                    limit: Optional[int] = None,
                    collection: str = "samples") -> pd.DataFrame:
        """
        Load LLM samples from Redis.
        
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
            raise StorageError("Redis storage not initialized")
        
        try:
            # Collection keys
            collection_key = self._get_key(collection)
            time_key = f"{collection_key}:by_time"
            
            # Check if collection exists
            if not self._client.exists(collection_key):
                logger.warning(f"Collection {collection_key} does not exist in Redis")
                return pd.DataFrame()
            
            # Determine which sample IDs to fetch based on filters
            sample_ids = set()
            
            # Apply date filters if available
            if start_date or end_date:
                min_score = "-inf"
                max_score = "+inf"
                
                if start_date:
                    min_score = time.mktime(start_date.timetuple())
                
                if end_date:
                    max_score = time.mktime(end_date.timetuple())
                
                # Get IDs from sorted set within time range
                if self._client.exists(time_key):
                    time_ids = self._client.zrangebyscore(time_key, min_score, max_score)
                    sample_ids = set(id.decode() for id in time_ids)
                else:
                    # Fall back to loading all and filtering
                    all_ids = self._client.smembers(collection_key)
                    sample_ids = set(id.decode() for id in all_ids)
            else:
                # No date filter, get all IDs
                all_ids = self._client.smembers(collection_key)
                sample_ids = set(id.decode() for id in all_ids)
            
            # Apply provider filter if specified
            if provider and sample_ids:
                provider_key = f"{collection_key}:provider:{provider}"
                if self._client.exists(provider_key):
                    provider_ids = set(id.decode() for id in self._client.smembers(provider_key))
                    sample_ids &= provider_ids
            
            # Apply model filter if specified
            if model and sample_ids:
                model_key = f"{collection_key}:model:{model}"
                if self._client.exists(model_key):
                    model_ids = set(id.decode() for id in self._client.smembers(model_key))
                    sample_ids &= model_ids
            
            # Limit number of samples if specified
            if limit and len(sample_ids) > limit:
                sample_ids = list(sample_ids)[:limit]
            
            # Convert to list
            sample_ids = list(sample_ids)
            
            # No samples match filters
            if not sample_ids:
                return pd.DataFrame()
            
            # Fetch samples using pipeline
            pipeline = self._client.pipeline()
            for sample_id in sample_ids:
                sample_key = self._get_key(collection, sample_id)
                pipeline.get(sample_key)
            
            # Execute pipeline and deserialize results
            raw_samples = pipeline.execute()
            samples = []
            
            for raw_sample in raw_samples:
                if raw_sample:
                    try:
                        sample = self.deserialize_from_storage(raw_sample)
                        samples.append(sample)
                    except Exception as e:
                        logger.warning(f"Error deserializing sample: {str(e)}")
            
            # Convert to DataFrame
            return pd.DataFrame(samples)
            
        except Exception as e:
            logger.error(f"Error loading samples from Redis: {str(e)}")
            return pd.DataFrame()
    
    def store_reference_distribution(self, distribution: Dict[str, Any], key: str = "reference") -> bool:
        """
        Store a reference distribution in Redis.
        
        Args:
            distribution: Dictionary containing reference distribution data
            key: Identifier for the distribution
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not self.initialized:
            raise StorageError("Redis storage not initialized")
        
        try:
            # Serialize the distribution
            serialized = self.serialize_for_storage(distribution)
            
            # Store in Redis
            full_key = self._get_key("reference", key)
            self._client.set(full_key, serialized)
            
            # Set expiration if configured
            if self.expiration_seconds:
                self._client.expire(full_key, self.expiration_seconds)
            
            logger.debug(f"Stored reference distribution '{key}' in Redis")
            return True
            
        except Exception as e:
            logger.error(f"Error storing reference distribution in Redis: {str(e)}")
            return False
    
    def load_reference_distribution(self, key: str = "reference") -> Optional[Dict[str, Any]]:
        """
        Load a reference distribution from Redis.
        
        Args:
            key: Identifier for the distribution
            
        Returns:
            Optional[Dict[str, Any]]: Reference distribution or None if not found
        """
        if not self.initialized:
            raise StorageError("Redis storage not initialized")
        
        try:
            # Get from Redis
            full_key = self._get_key("reference", key)
            serialized = self._client.get(full_key)
            
            if not serialized:
                logger.warning(f"Reference distribution '{key}' not found in Redis")
                return None
            
            # Deserialize
            distribution = self.deserialize_from_storage(serialized)
            return distribution
            
        except Exception as e:
            logger.error(f"Error loading reference distribution from Redis: {str(e)}")
            return None
    
    def store_drift_metrics(self, metrics: Union[List[Dict], pd.DataFrame], collection: str = "metrics") -> bool:
        """
        Store drift detection metrics in Redis.
        
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
        Load drift detection metrics from Redis.
        
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
        Store embeddings in Redis.
        
        Args:
            embeddings: Dictionary mapping IDs to embedding vectors
            key: Identifier for the embeddings set
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not self.initialized:
            raise StorageError("Redis storage not initialized")
        
        try:
            # Use pipeline for better performance
            pipeline = self._client.pipeline()
            
            # Store each embedding
            embeddings_key = self._get_key("embeddings", key)
            count = 0
            
            for id, vector in embeddings.items():
                # Store embedding
                vector_key = f"{embeddings_key}:{id}"
                serialized = self.serialize_for_storage(vector)
                pipeline.set(vector_key, serialized)
                
                # Add to index
                pipeline.sadd(embeddings_key, id)
                
                # Set expiration if configured
                if self.expiration_seconds:
                    pipeline.expire(vector_key, self.expiration_seconds)
                
                count += 1
            
            # Set expiration on the index if configured
            if self.expiration_seconds:
                pipeline.expire(embeddings_key, self.expiration_seconds)
            
            # Execute pipeline
            pipeline.execute()
            
            logger.debug(f"Stored {count} embeddings in Redis under {embeddings_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings in Redis: {str(e)}")
            return False
    
    def load_embeddings(self, key: str = "embeddings") -> Optional[Dict[str, np.ndarray]]:
        """
        Load embeddings from Redis.
        
        Args:
            key: Identifier for the embeddings set
            
        Returns:
            Optional[Dict[str, np.ndarray]]: Dictionary of embeddings or None if not found
        """
        if not self.initialized:
            raise StorageError("Redis storage not initialized")
        
        try:
            # Get embeddings index
            embeddings_key = self._get_key("embeddings", key)
            ids = self._client.smembers(embeddings_key)
            
            if not ids:
                logger.warning(f"Embeddings set '{key}' not found in Redis")
                return None
            
            # Use pipeline to get all embeddings
            pipeline = self._client.pipeline()
            id_map = {}
            
            for id_bytes in ids:
                id = id_bytes.decode()
                vector_key = f"{embeddings_key}:{id}"
                pipeline.get(vector_key)
                id_map[len(id_map)] = id
            
            # Execute pipeline
            results = pipeline.execute()
            
            # Deserialize results
            embeddings = {}
            for i, result in enumerate(results):
                if result:
                    try:
                        vector = self.deserialize_from_storage(result)
                        id = id_map[i]
                        embeddings[id] = vector
                    except Exception as e:
                        logger.warning(f"Error deserializing embedding: {str(e)}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error loading embeddings from Redis: {str(e)}")
            return None
    
    def cleanup(self):
        """Clean up resources used by the Redis storage backend."""
        if self._client:
            try:
                # Close Redis connection
                self._client.close()
                self._client = None
                self.initialized = False
                logger.debug("Closed Redis connection")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {str(e)}")


# Register this storage type
from . import register_storage
register_storage(RedisStorage)