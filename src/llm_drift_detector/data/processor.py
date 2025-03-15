"""
Data processing functionality for LLM Drift Detector.

This module handles processing raw LLM data into features suitable
for drift detection, including embedding generation and feature extraction.
"""

import os
import pickle
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from dataclasses import dataclass, field
import time

from ..config import Config, get_config

logger = logging.getLogger(__name__)

# Import sentence transformers conditionally
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not installed. Embedding functionality will be disabled.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class ProcessedSample:
    """
    Container for a processed LLM sample with extracted features.
    """
    # Original sample ID
    sample_id: str
    
    # Basic metadata
    timestamp: datetime
    provider_name: str
    model_name: str
    
    # Text content
    prompt: str
    response: str
    
    # Extracted features
    response_length: int
    prompt_length: int
    token_count: Optional[int] = None
    response_time: Optional[float] = None
    
    # Embeddings
    prompt_embedding: Optional[np.ndarray] = None
    response_embedding: Optional[np.ndarray] = None
    
    # Additional metrics
    perplexity: Optional[float] = None
    
    # Other metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributionStats:
    """
    Statistical representation of a feature distribution.
    """
    
    def __init__(self, values: np.ndarray = None):
        """
        Initialize with optional values.
        
        Args:
            values: Optional array of values to calculate statistics from
        """
        self.count = 0
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.q25 = None
        self.median = None
        self.q75 = None
        self.values = None
        
        if values is not None and len(values) > 0:
            self.update(values)
    
    def update(self, values: np.ndarray):
        """
        Update statistics with new values.
        
        Args:
            values: Array of values
        """
        if len(values) == 0:
            return
        
        # Store clean values (remove NaN)
        self.values = values[~np.isnan(values)]
        self.count = len(self.values)
        
        if self.count == 0:
            return
        
        # Calculate statistics
        self.mean = np.mean(self.values)
        self.std = np.std(self.values)
        self.min = np.min(self.values)
        self.max = np.max(self.values)
        self.q25 = np.percentile(self.values, 25)
        self.median = np.percentile(self.values, 50)
        self.q75 = np.percentile(self.values, 75)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary with statistics
        """
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "q25": self.q25,
            "median": self.median,
            "q75": self.q75
        }


class ReferenceDistribution:
    """
    Reference distribution for drift detection.
    """
    
    def __init__(self):
        """Initialize an empty reference distribution."""
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.sample_count = 0
        
        # Feature distributions
        self.response_length = DistributionStats()
        self.prompt_length = DistributionStats()
        self.token_count = DistributionStats()
        self.response_time = DistributionStats()
        self.perplexity = DistributionStats()
        
        # Embedding information
        self.embeddings = None
        self.embedding_centroid = None
        
        # PCA for dimensionality reduction
        self.pca = None
    
    def update(self, processed_samples: List[ProcessedSample]):
        """
        Update the reference distribution with new processed samples.
        
        Args:
            processed_samples: List of processed samples
        """
        if not processed_samples:
            return
        
        self.updated_at = datetime.now()
        self.sample_count = len(processed_samples)
        
        # Update feature distributions
        self.response_length.update(np.array([s.response_length for s in processed_samples]))
        self.prompt_length.update(np.array([s.prompt_length for s in processed_samples]))
        
        # Update optional features if available
        token_counts = [s.token_count for s in processed_samples if s.token_count is not None]
        if token_counts:
            self.token_count.update(np.array(token_counts))
        
        response_times = [s.response_time for s in processed_samples if s.response_time is not None]
        if response_times:
            self.response_time.update(np.array(response_times))
        
        perplexities = [s.perplexity for s in processed_samples if s.perplexity is not None]
        if perplexities:
            self.perplexity.update(np.array(perplexities))
        
        # Update embeddings if available
        embeddings = [s.response_embedding for s in processed_samples if s.response_embedding is not None]
        if embeddings:
            self.embeddings = np.vstack(embeddings)
            self.embedding_centroid = np.mean(self.embeddings, axis=0)
            
            # Update PCA if we have enough samples
            if len(embeddings) >= 10:
                try:
                    from sklearn.decomposition import PCA
                    self.pca = PCA(n_components=2)
                    self.pca.fit(self.embeddings)
                except ImportError:
                    logger.warning("scikit-learn not installed. PCA will not be available.")
                except Exception as e:
                    logger.warning(f"Error calculating PCA: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary with reference distribution data
        """
        result = {
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "sample_count": self.sample_count,
            "response_length": self.response_length.to_dict(),
            "prompt_length": self.prompt_length.to_dict(),
            "token_count": self.token_count.to_dict(),
            "response_time": self.response_time.to_dict(),
            "perplexity": self.perplexity.to_dict(),
            "has_embeddings": self.embeddings is not None,
            "embedding_dimensions": self.embeddings.shape[1] if self.embeddings is not None else None,
            "has_pca": self.pca is not None
        }
        return result


class DataProcessor:
    """
    Processes raw LLM data for drift detection.
    
    This class handles feature extraction, embedding generation,
    and reference distribution management.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration object, uses global config if None
        """
        self.config = config or get_config()
        self.reference_path = Path(self.config.get("data.reference.path", "./data/reference_distribution.pkl"))
        
        # Initialize embedding model if enabled
        self.embedding_model = None
        self.embedding_enabled = self.config.get("drift_detection.embedding.enabled", True)
        
        if self.embedding_enabled and SENTENCE_TRANSFORMERS_AVAILABLE:
            model_name = self.config.get("drift_detection.embedding.model", "all-MiniLM-L6-v2")
            device = self.config.get("drift_detection.embedding.device", "cpu")
            
            try:
                logger.info(f"Loading embedding model {model_name} on {device}")
                self.embedding_model = SentenceTransformer(model_name, device=device)
            except Exception as e:
                logger.warning(f"Error loading embedding model: {str(e)}")
                self.embedding_enabled = False
        elif self.embedding_enabled:
            logger.warning("Embedding enabled but sentence-transformers not installed")
            self.embedding_enabled = False
        
        # Load reference distribution if it exists
        self.reference_distribution = self._load_reference_distribution()
    
    def _load_reference_distribution(self) -> ReferenceDistribution:
        """
        Load reference distribution from disk if it exists.
        
        Returns:
            ReferenceDistribution: Loaded or new reference distribution
        """
        if self.reference_path.exists():
            try:
                with open(self.reference_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load reference distribution: {str(e)}")
        
        # Return empty reference if file doesn't exist or loading fails
        return ReferenceDistribution()
    
    def _save_reference_distribution(self):
        """Save the reference distribution to disk."""
        # Create directory if it doesn't exist
        os.makedirs(self.reference_path.parent, exist_ok=True)
        
        try:
            with open(self.reference_path, 'wb') as f:
                pickle.dump(self.reference_distribution, f)
            
            logger.info(f"Saved reference distribution to {self.reference_path}")
        except Exception as e:
            logger.warning(f"Failed to save reference distribution: {str(e)}")
    
    def process_dataframe(self, df: pd.DataFrame) -> List[ProcessedSample]:
        """
        Process a DataFrame of LLM samples.
        
        Args:
            df: DataFrame containing LLM samples
            
        Returns:
            List[ProcessedSample]: List of processed samples
        """
        processed_samples = []
        
        for _, row in df.iterrows():
            try:
                # Extract basic features
                sample = ProcessedSample(
                    sample_id=str(row.get('id', '')),
                    timestamp=row['timestamp'] if isinstance(row['timestamp'], datetime) else 
                               datetime.fromisoformat(row['timestamp']),
                    provider_name=row['provider_name'],
                    model_name=row['model_name'],
                    prompt=row['prompt'],
                    response=row['response'],
                    response_length=len(row['response']),
                    prompt_length=len(row['prompt']),
                    token_count=row.get('token_count'),
                    response_time=row.get('response_time'),
                    metadata=row.get('metadata', {})
                )
                
                # Generate embeddings if model is available
                if self.embedding_model is not None:
                    start_time = time.time()
                    
                    # Generate embeddings
                    sample.response_embedding = self.embedding_model.encode(
                        row['response'], 
                        show_progress_bar=False
                    )
                    
                    sample.prompt_embedding = self.embedding_model.encode(
                        row['prompt'],
                        show_progress_bar=False
                    )
                    
                    logger.debug(f"Generated embeddings in {time.time() - start_time:.2f} seconds")
                
                # TODO: Calculate perplexity if enabled
                
                processed_samples.append(sample)
                
            except Exception as e:
                logger.warning(f"Error processing sample: {str(e)}")
        
        return processed_samples
    
    def process_batch(self, batch: Union[List[Dict[str, Any]], pd.DataFrame]) -> List[ProcessedSample]:
        """
        Process a batch of LLM samples.
        
        Args:
            batch: List of dictionaries or DataFrame with LLM samples
            
        Returns:
            List[ProcessedSample]: List of processed samples
        """
        if isinstance(batch, list):
            df = pd.DataFrame(batch)
        else:
            df = batch
        
        return self.process_dataframe(df)
    
    def update_reference_distribution(self, samples: List[ProcessedSample]) -> bool:
        """
        Update the reference distribution with new samples.
        
        Args:
            samples: List of processed samples
            
        Returns:
            bool: True if reference was updated, False otherwise
        """
        min_samples = self.config.get("data.reference.min_samples", 100)
        
        if len(samples) < min_samples:
            logger.warning(f"Not enough samples to update reference distribution. Need at least {min_samples}, got {len(samples)}")
            return False
        
        self.reference_distribution.update(samples)
        self._save_reference_distribution()
        
        logger.info(f"Updated reference distribution with {len(samples)} samples")
        return True
    
    def get_current_distribution(self, samples: List[ProcessedSample]) -> Dict[str, Any]:
        """
        Calculate distribution statistics for current samples.
        
        Args:
            samples: List of processed samples
            
        Returns:
            Dict[str, Any]: Statistics for the current distribution
        """
        current_dist = ReferenceDistribution()
        current_dist.update(samples)
        return current_dist.to_dict()
    
    def get_reference_distribution(self) -> Dict[str, Any]:
        """
        Get the reference distribution statistics.
        
        Returns:
            Dict[str, Any]: Statistics for the reference distribution
        """
        return self.reference_distribution.to_dict()