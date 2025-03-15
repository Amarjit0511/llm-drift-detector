"""
Embedding-based drift detection.

This module provides drift detection based on the distance between
embedding distributions in the semantic space.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseDriftDetector, DriftResult
from ..data.processor import ProcessedSample
from ..config import Config

logger = logging.getLogger(__name__)

class EmbeddingDriftDetector(BaseDriftDetector):
    """
    Detects drift based on embedding distributions.
    
    This detector compares the distribution of embeddings in the reference set
    with the current set to detect semantic drift in LLM outputs.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the embedding drift detector.
        
        Args:
            config: Configuration object, uses global config if None
        """
        super().__init__(config)
        
        # Get embedding-specific configuration
        self.distance_metric = self.detector_config.get("distance_metric", "cosine")
        
        # Check if we have the required dependencies
        try:
            import sklearn
            self._sklearn_available = True
        except ImportError:
            logger.warning("scikit-learn not installed. Some embedding drift methods will be limited.")
            self._sklearn_available = False
    
    def detect(self, 
              reference_samples: List[ProcessedSample],
              current_samples: List[ProcessedSample]) -> DriftResult:
        """
        Detect drift between reference and current embedding distributions.
        
        Args:
            reference_samples: List of reference processed samples
            current_samples: List of current processed samples
            
        Returns:
            DriftResult: Result of drift detection
        """
        # Validate inputs
        if not self._validate_input(reference_samples, current_samples):
            return self._create_result(
                0.0, reference_samples, current_samples,
                analysis={"error": "Invalid input or detector disabled"}
            )
        
        # Extract embeddings
        ref_embeddings, curr_embeddings = self._extract_embeddings(reference_samples, current_samples)
        
        if ref_embeddings is None or curr_embeddings is None:
            return self._create_result(
                0.0, reference_samples, current_samples,
                analysis={"error": "Could not extract embeddings"}
            )
        
        # Calculate centroid distance
        centroid_distance = self._calculate_centroid_distance(ref_embeddings, curr_embeddings)
        
        # Calculate intra-cluster distances
        ref_intra_distance = self._calculate_intra_cluster_distance(ref_embeddings)
        curr_intra_distance = self._calculate_intra_cluster_distance(curr_embeddings)
        
        # Calculate relative change in variance
        variance_change = abs(curr_intra_distance - ref_intra_distance) / max(ref_intra_distance, 1e-10)
        
        # Calculate average distance from current samples to reference centroid
        ref_centroid = np.mean(ref_embeddings, axis=0)
        distances_to_ref = []
        
        for emb in curr_embeddings:
            if self.distance_metric == "cosine":
                dist = cosine(emb, ref_centroid)
            else:
                dist = euclidean(emb, ref_centroid)
            distances_to_ref.append(dist)
        
        avg_distance_to_ref = np.mean(distances_to_ref)
        
        # Calculate overall drift score (weighted average of metrics)
        drift_score = 0.6 * centroid_distance + 0.2 * variance_change + 0.2 * avg_distance_to_ref
        
        # Prepare analysis results
        analysis = {
            "centroid_distance": centroid_distance,
            "variance_change": variance_change,
            "avg_distance_to_reference": avg_distance_to_ref,
            "ref_intra_distance": ref_intra_distance,
            "curr_intra_distance": curr_intra_distance,
            "reference_centroid": ref_centroid.tolist(),
            "distance_metric": self.distance_metric
        }
        
        # If scikit-learn is available, add additional analysis
        if self._sklearn_available:
            try:
                # Calculate MMD (Maximum Mean Discrepancy) if we have enough samples
                if len(ref_embeddings) > 10 and len(curr_embeddings) > 10:
                    mmd = self._calculate_mmd(ref_embeddings, curr_embeddings)
                    analysis["mmd"] = mmd
                    
                    # Include MMD in drift score
                    drift_score = 0.5 * drift_score + 0.5 * min(mmd, 1.0)
            except Exception as e:
                logger.warning(f"Error calculating MMD: {str(e)}")
        
        return self._create_result(
            drift_score, reference_samples, current_samples,
            metric_name="embedding_distance",
            analysis=analysis
        )
    
    def _extract_embeddings(self, 
                           reference_samples: List[ProcessedSample],
                           current_samples: List[ProcessedSample]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract embeddings from processed samples.
        
        Args:
            reference_samples: List of reference processed samples
            current_samples: List of current processed samples
            
        Returns:
            Tuple of reference and current embeddings arrays, or (None, None) if extraction fails
        """
        # Extract reference embeddings
        ref_embeddings = []
        for sample in reference_samples:
            if sample.response_embedding is not None:
                ref_embeddings.append(sample.response_embedding)
        
        # Extract current embeddings
        curr_embeddings = []
        for sample in current_samples:
            if sample.response_embedding is not None:
                curr_embeddings.append(sample.response_embedding)
        
        # Check if we have enough embeddings
        min_samples = self.detector_config.get("min_samples", 10)
        
        if len(ref_embeddings) < min_samples:
            logger.warning(f"Not enough reference embeddings. Need at least {min_samples}, got {len(ref_embeddings)}")
            return None, None
        
        if len(curr_embeddings) < min_samples:
            logger.warning(f"Not enough current embeddings. Need at least {min_samples}, got {len(curr_embeddings)}")
            return None, None
        
        return np.vstack(ref_embeddings), np.vstack(curr_embeddings)
    
    def _calculate_centroid_distance(self, ref_embeddings: np.ndarray, curr_embeddings: np.ndarray) -> float:
        """
        Calculate distance between centroids of reference and current embeddings.
        
        Args:
            ref_embeddings: Reference embeddings array
            curr_embeddings: Current embeddings array
            
        Returns:
            float: Distance between centroids
        """
        # Calculate centroids
        ref_centroid = np.mean(ref_embeddings, axis=0)
        curr_centroid = np.mean(curr_embeddings, axis=0)
        
        # Calculate distance
        if self.distance_metric == "cosine":
            return cosine(ref_centroid, curr_centroid)
        else:
            return euclidean(ref_centroid, curr_centroid) / np.sqrt(ref_centroid.shape[0])
    
    def _calculate_intra_cluster_distance(self, embeddings: np.ndarray) -> float:
        """
        Calculate average intra-cluster distance for embeddings.
        
        Args:
            embeddings: Embeddings array
            
        Returns:
            float: Average intra-cluster distance
        """
        # Calculate centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Calculate distances to centroid
        distances = []
        for emb in embeddings:
            if self.distance_metric == "cosine":
                distances.append(cosine(emb, centroid))
            else:
                distances.append(euclidean(emb, centroid) / np.sqrt(emb.shape[0]))
        
        return np.mean(distances)
    
    def _calculate_mmd(self, ref_embeddings: np.ndarray, curr_embeddings: np.ndarray) -> float:
        """
        Calculate Maximum Mean Discrepancy between distributions.
        
        Args:
            ref_embeddings: Reference embeddings array
            curr_embeddings: Current embeddings array
            
        Returns:
            float: MMD value
        """
        # Use a subset for efficiency if we have many samples
        max_samples = 1000
        if len(ref_embeddings) > max_samples:
            indices = np.random.choice(len(ref_embeddings), max_samples, replace=False)
            ref_embeddings = ref_embeddings[indices]
        
        if len(curr_embeddings) > max_samples:
            indices = np.random.choice(len(curr_embeddings), max_samples, replace=False)
            curr_embeddings = curr_embeddings[indices]
        
        # Calculate kernel matrices
        XX = cosine_similarity(ref_embeddings, ref_embeddings)
        YY = cosine_similarity(curr_embeddings, curr_embeddings)
        XY = cosine_similarity(ref_embeddings, curr_embeddings)
        
        # Calculate MMD
        m = ref_embeddings.shape[0]
        n = curr_embeddings.shape[0]
        
        mmd = (np.sum(XX) - np.trace(XX)) / (m * (m - 1))
        mmd += (np.sum(YY) - np.trace(YY)) / (n * (n - 1))
        mmd -= 2 * np.sum(XY) / (m * n)
        
        return max(0.0, mmd)  # MMD should be non-negative