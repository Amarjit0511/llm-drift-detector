"""
SQL database storage for LLM Drift Detector.

This module provides a storage backend using SQL databases for storing
drift detection data, samples, and metrics, offering persistent storage
with querying capabilities.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Iterator
from datetime import datetime
import pandas as pd
import numpy as np
import json
import pickle
import uuid
import os

from .base import BaseStorage, StorageError

logger = logging.getLogger(__name__)

class SQLStorage(BaseStorage):
    """
    SQL database storage backend.
    
    This storage backend uses SQL databases for persistent storage of
    drift detection data, supporting various database engines through
    SQLAlchemy (SQLite, PostgreSQL, MySQL, etc.).
    """
    
    storage_type = "sql"
    
    def __init__(
        self,
        connection_string: str = "sqlite:///data/storage/drift.db",
        table_prefix: str = "llm_drift_",
        echo: bool = False,
        create_tables: bool = True,
        **kwargs
    ):
        """
        Initialize the SQL storage backend.
        
        Args:
            connection_string: SQLAlchemy connection string
            table_prefix: Prefix for all database tables
            echo: Whether to echo SQL statements (for debugging)
            create_tables: Whether to create tables if they don't exist
            **kwargs: Additional SQLAlchemy configuration
        """
        super().__init__(**kwargs)
        
        self.connection_string = connection_string
        self.table_prefix = table_prefix
        self.echo = echo
        
        # Initialize SQLAlchemy components
        self._engine = None
        self._metadata = None
        self._session_factory = None
        self._tables = {}
        
        # Initialize database connection
        self._init_database(create_tables)
    
    def _init_database(self, create_tables: bool = True):
        """
        Initialize the database connection and create tables if needed.
        
        Args:
            create_tables: Whether to create tables if they don't exist
        """
        try:
            from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Integer, DateTime, JSON, LargeBinary, ForeignKey, select
            from sqlalchemy.orm import sessionmaker, declarative_base, Session
            import sqlalchemy as sa
            
            # Create engine
            self._engine = create_engine(
                self.connection_string,
                echo=self.echo
            )
            
            # Create metadata
            self._metadata = MetaData()
            
            # Create session factory
            self._session_factory = sessionmaker(bind=self._engine)
            
            # Create tables if requested
            if create_tables:
                # Define tables
                
                # Samples table
                samples_table = Table(
                    f"{self.table_prefix}samples",
                    self._metadata,
                    Column("id", String(36), primary_key=True),
                    Column("timestamp", DateTime, index=True),
                    Column("provider_name", String(50), index=True),
                    Column("model_name", String(50), index=True),
                    Column("prompt", String),
                    Column("response", String),
                    Column("response_time", Float),
                    Column("token_count", Integer),
                    Column("total_tokens", Integer),
                    Column("input_tokens", Integer),
                    Column("output_tokens", Integer),
                    Column("finish_reason", String(20)),
                    Column("error", String),
                    Column("metadata", JSON)
                )
                self._tables["samples"] = samples_table
                
                # Metrics table
                metrics_table = Table(
                    f"{self.table_prefix}metrics",
                    self._metadata,
                    Column("id", String(36), primary_key=True),
                    Column("timestamp", DateTime, index=True),
                    Column("provider_name", String(50), index=True),
                    Column("model_name", String(50), index=True),
                    Column("detector_name", String(50), index=True),
                    Column("metric_name", String(50), index=True),
                    Column("drift_score", Float),
                    Column("threshold", Float),
                    Column("drift_detected", sa.Boolean),
                    Column("reference_size", Integer),
                    Column("current_size", Integer),
                    Column("details", JSON)
                )
                self._tables["metrics"] = metrics_table
                
                # Reference distributions table
                reference_table = Table(
                    f"{self.table_prefix}reference",
                    self._metadata,
                    Column("key", String(50), primary_key=True),
                    Column("created_at", DateTime),
                    Column("updated_at", DateTime),
                    Column("data", LargeBinary)
                )
                self._tables["reference"] = reference_table
                
                # Embeddings table
                embeddings_table = Table(
                    f"{self.table_prefix}embeddings",
                    self._metadata,
                    Column("key", String(50), primary_key=True),
                    Column("created_at", DateTime),
                    Column("updated_at", DateTime),
                    Column("data", LargeBinary)
                )
                self._tables["embeddings"] = embeddings_table
                
                # Create all tables
                self._metadata.create_all(self._engine)
                logger.info(f"Created SQL tables with prefix {self.table_prefix}")
            
            # Test connection
            with self._session_factory() as session:
                session.execute(select(1))
            
            self.initialized = True
            logger.info(f"Connected to SQL database: {self.connection_string}")
            
        except ImportError:
            logger.error("SQLAlchemy not installed. Install with 'pip install sqlalchemy'")
            raise StorageError("SQLAlchemy not installed")
        except Exception as e:
            logger.error(f"Failed to initialize SQL database: {str(e)}")
            raise StorageError(f"SQL database initialization error: {str(e)}")
    
    def store_samples(self, samples: Union[List[Dict], pd.DataFrame], collection: str = "samples") -> bool:
        """
        Store LLM samples in the SQL database.
        
        Args:
            samples: List of sample dictionaries or DataFrame with samples
            collection: Name of the collection to store in (ignored, always uses samples table)
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not self.initialized:
            raise StorageError("SQL storage not initialized")
        
        if collection != "samples":
            logger.warning(f"SQL storage only supports 'samples' collection, ignoring '{collection}'")
        
        try:
            # Convert to list of dicts if DataFrame
            if isinstance(samples, pd.DataFrame):
                samples_list = samples.to_dict(orient="records")
            else:
                samples_list = samples
            
            # Get samples table
            samples_table = self._tables["samples"]
            
            # Store samples
            with self._session_factory() as session:
                for sample in samples_list:
                    # Generate ID if not present
                    if "id" not in sample or not sample["id"]:
                        sample["id"] = str(uuid.uuid4())
                    
                    # Ensure timestamp is datetime
                    if "timestamp" in sample:
                        if isinstance(sample["timestamp"], str):
                            sample["timestamp"] = pd.to_datetime(sample["timestamp"])
                    else:
                        sample["timestamp"] = datetime.now()
                    
                    # Handle JSON for metadata
                    if "metadata" in sample and sample["metadata"]:
                        # Ensure metadata is JSON serializable
                        for key, value in list(sample["metadata"].items()):
                            if isinstance(value, np.ndarray):
                                sample["metadata"][key] = value.tolist()
                            elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                # Remove non-serializable values
                                sample["metadata"].pop(key)
                    
                    # Insert into database
                    session.execute(samples_table.insert().values(**sample))
                
                # Commit transaction
                session.commit()
            
            logger.debug(f"Stored {len(samples_list)} samples in SQL database")
            return True
            
        except Exception as e:
            logger.error(f"Error storing samples in SQL database: {str(e)}")
            return False
    
    def load_samples(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    provider: Optional[str] = None,
                    model: Optional[str] = None,
                    limit: Optional[int] = None,
                    collection: str = "samples") -> pd.DataFrame:
        """
        Load LLM samples from the SQL database.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            provider: Optional provider filter
            model: Optional model filter
            limit: Optional maximum number of samples to return
            collection: Name of the collection to load from (ignored, always uses samples table)
            
        Returns:
            pd.DataFrame: DataFrame containing the samples
        """
        if not self.initialized:
            raise StorageError("SQL storage not initialized")
        
        if collection != "samples":
            logger.warning(f"SQL storage only supports 'samples' collection, ignoring '{collection}'")
        
        try:
            from sqlalchemy import select
            
            # Get samples table
            samples_table = self._tables["samples"]
            
            # Build query
            query = select(samples_table)
            
            # Apply filters
            if start_date:
                query = query.where(samples_table.c.timestamp >= start_date)
            
            if end_date:
                query = query.where(samples_table.c.timestamp <= end_date)
            
            if provider:
                query = query.where(samples_table.c.provider_name == provider)
            
            if model:
                query = query.where(samples_table.c.model_name == model)
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            # Execute query
            with self._session_factory() as session:
                result = session.execute(query)
                rows = result.all()
            
            # Convert to DataFrame
            if not rows:
                return pd.DataFrame()
            
            # Convert rows to dictionaries
            data = [dict(row) for row in rows]
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error loading samples from SQL database: {str(e)}")
            return pd.DataFrame()
    
    def store_reference_distribution(self, distribution: Dict[str, Any], key: str = "reference") -> bool:
        """
        Store a reference distribution in the SQL database.
        
        Args:
            distribution: Dictionary containing reference distribution data
            key: Identifier for the distribution
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not self.initialized:
            raise StorageError("SQL storage not initialized")
        
        try:
            # Serialize the distribution
            serialized = self.serialize_for_storage(distribution)
            
            # Get reference table
            reference_table = self._tables["reference"]
            
            # Store in database
            with self._session_factory() as session:
                # Check if key already exists
                from sqlalchemy import select
                result = session.execute(
                    select(reference_table).where(reference_table.c.key == key)
                )
                existing = result.first()
                
                now = datetime.now()
                
                if existing:
                    # Update existing record
                    session.execute(
                        reference_table.update()
                        .where(reference_table.c.key == key)
                        .values(
                            updated_at=now,
                            data=serialized
                        )
                    )
                else:
                    # Insert new record
                    session.execute(
                        reference_table.insert().values(
                            key=key,
                            created_at=now,
                            updated_at=now,
                            data=serialized
                        )
                    )
                
                # Commit transaction
                session.commit()
            
            logger.debug(f"Stored reference distribution '{key}' in SQL database")
            return True
            
        except Exception as e:
            logger.error(f"Error storing reference distribution in SQL database: {str(e)}")
            return False
    
    def load_reference_distribution(self, key: str = "reference") -> Optional[Dict[str, Any]]:
        """
        Load a reference distribution from the SQL database.
        
        Args:
            key: Identifier for the distribution
            
        Returns:
            Optional[Dict[str, Any]]: Reference distribution or None if not found
        """
        if not self.initialized:
            raise StorageError("SQL storage not initialized")
        
        try:
            from sqlalchemy import select
            
            # Get reference table
            reference_table = self._tables["reference"]
            
            # Query database
            with self._session_factory() as session:
                result = session.execute(
                    select(reference_table.c.data)
                    .where(reference_table.c.key == key)
                )
                row = result.first()
            
            if not row:
                logger.warning(f"Reference distribution '{key}' not found in SQL database")
                return None
            
            # Deserialize
            distribution = self.deserialize_from_storage(row[0])
            return distribution
            
        except Exception as e:
            logger.error(f"Error loading reference distribution from SQL database: {str(e)}")
            return None
    
    def store_drift_metrics(self, metrics: Union[List[Dict], pd.DataFrame], collection: str = "metrics") -> bool:
        """
        Store drift detection metrics in the SQL database.
        
        Args:
            metrics: List of metric dictionaries or DataFrame with metrics
            collection: Name of the collection to store in (ignored, always uses metrics table)
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not self.initialized:
            raise StorageError("SQL storage not initialized")
        
        if collection != "metrics":
            logger.warning(f"SQL storage only supports 'metrics' collection, ignoring '{collection}'")
        
        try:
            # Convert to list of dicts if DataFrame
            if isinstance(metrics, pd.DataFrame):
                metrics_list = metrics.to_dict(orient="records")
            else:
                metrics_list = metrics
            
            # Get metrics table
            metrics_table = self._tables["metrics"]
            
            # Store metrics
            with self._session_factory() as session:
                for metric in metrics_list:
                    # Generate ID if not present
                    if "id" not in metric or not metric["id"]:
                        metric["id"] = str(uuid.uuid4())
                    
                    # Ensure timestamp is datetime
                    if "timestamp" in metric:
                        if isinstance(metric["timestamp"], str):
                            metric["timestamp"] = pd.to_datetime(metric["timestamp"])
                    else:
                        metric["timestamp"] = datetime.now()
                    
                    # Handle JSON for details
                    if "details" in metric and metric["details"]:
                        # Ensure details is JSON serializable
                        for key, value in list(metric["details"].items()):
                            if isinstance(value, np.ndarray):
                                metric["details"][key] = value.tolist()
                            elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                # Remove non-serializable values
                                metric["details"].pop(key)
                    
                    # Insert into database
                    session.execute(metrics_table.insert().values(**metric))
                
                # Commit transaction
                session.commit()
            
            logger.debug(f"Stored {len(metrics_list)} metrics in SQL database")
            return True
            
        except Exception as e:
            logger.error(f"Error storing metrics in SQL database: {str(e)}")
            return False
    
    def load_drift_metrics(self,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          detector: Optional[str] = None,
                          provider: Optional[str] = None, 
                          model: Optional[str] = None,
                          limit: Optional[int] = None,
                          collection: str = "metrics") -> pd.DataFrame:
        """
        Load drift detection metrics from the SQL database.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            detector: Optional detector type filter
            provider: Optional provider filter
            model: Optional model filter
            limit: Optional maximum number of metrics to return
            collection: Name of the collection to load from (ignored, always uses metrics table)
            
        Returns:
            pd.DataFrame: DataFrame containing the metrics
        """
        if not self.initialized:
            raise StorageError("SQL storage not initialized")
        
        if collection != "metrics":
            logger.warning(f"SQL storage only supports 'metrics' collection, ignoring '{collection}'")
        
        try:
            from sqlalchemy import select
            
            # Get metrics table
            metrics_table = self._tables["metrics"]
            
            # Build query
            query = select(metrics_table)
            
            # Apply filters
            if start_date:
                query = query.where(metrics_table.c.timestamp >= start_date)
            
            if end_date:
                query = query.where(metrics_table.c.timestamp <= end_date)
            
            if provider:
                query = query.where(metrics_table.c.provider_name == provider)
            
            if model:
                query = query.where(metrics_table.c.model_name == model)
            
            if detector:
                query = query.where(metrics_table.c.detector_name == detector)
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            # Execute query
            with self._session_factory() as session:
                result = session.execute(query)
                rows = result.all()
            
            # Convert to DataFrame
            if not rows:
                return pd.DataFrame()
            
            # Convert rows to dictionaries
            data = [dict(row) for row in rows]
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error loading metrics from SQL database: {str(e)}")
            return pd.DataFrame()
    
    def store_embeddings(self, embeddings: Dict[str, np.ndarray], key: str = "embeddings") -> bool:
        """
        Store embeddings in the SQL database.
        
        Args:
            embeddings: Dictionary mapping IDs to embedding vectors
            key: Identifier for the embeddings set
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not self.initialized:
            raise StorageError("SQL storage not initialized")
        
        try:
            # Serialize the embeddings
            serialized = self.serialize_for_storage(embeddings)
            
            # Get embeddings table
            embeddings_table = self._tables["embeddings"]
            
            # Store in database
            with self._session_factory() as session:
                # Check if key already exists
                from sqlalchemy import select
                result = session.execute(
                    select(embeddings_table).where(embeddings_table.c.key == key)
                )
                existing = result.first()
                
                now = datetime.now()
                
                if existing:
                    # Update existing record
                    session.execute(
                        embeddings_table.update()
                        .where(embeddings_table.c.key == key)
                        .values(
                            updated_at=now,
                            data=serialized
                        )
                    )
                else:
                    # Insert new record
                    session.execute(
                        embeddings_table.insert().values(
                            key=key,
                            created_at=now,
                            updated_at=now,
                            data=serialized
                        )
                    )
                
                # Commit transaction
                session.commit()
            
            logger.debug(f"Stored {len(embeddings)} embeddings with key '{key}' in SQL database")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings in SQL database: {str(e)}")
            return False
    
    def load_embeddings(self, key: str = "embeddings") -> Optional[Dict[str, np.ndarray]]:
        """
        Load embeddings from the SQL database.
        
        Args:
            key: Identifier for the embeddings set
            
        Returns:
            Optional[Dict[str, np.ndarray]]: Dictionary of embeddings or None if not found
        """
        if not self.initialized:
            raise StorageError("SQL storage not initialized")
        
        try:
            from sqlalchemy import select
            
            # Get embeddings table
            embeddings_table = self._tables["embeddings"]
            
            # Query database
            with self._session_factory() as session:
                result = session.execute(
                    select(embeddings_table.c.data)
                    .where(embeddings_table.c.key == key)
                )
                row = result.first()
            
            if not row:
                logger.warning(f"Embeddings '{key}' not found in SQL database")
                return None
            
            # Deserialize
            embeddings = self.deserialize_from_storage(row[0])
            return embeddings
            
        except Exception as e:
            logger.error(f"Error loading embeddings from SQL database: {str(e)}")
            return None
    
    def cleanup(self):
        """Clean up resources used by the SQL storage backend."""
        if self._engine:
            try:
                # Close database connection
                self._engine.dispose()
                self._engine = None
                self._session_factory = None
                self._tables = {}
                self.initialized = False
                logger.debug("Closed SQL database connection")
            except Exception as e:
                logger.warning(f"Error closing SQL database connection: {str(e)}")


# Register this storage type
from . import register_storage
register_storage(SQLStorage)