"""
Statistical distribution-based drift detection.

This module provides drift detection based on statistical tests
comparing numerical feature distributions.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import pandas as pd

from .base import BaseDriftDetector, DriftResult
from ..data.processor import ProcessedSample
from ..config import Config

logger = logging.getLogger(__name__)

class DistributionDriftDetector(BaseDriftDetector):
    """
    Detects drift based on numerical feature distributions.
    
    This detector applies statistical tests to compare distributions
    of features like response length, token count, and response time.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the distribution drift detector.
        
        Args:
            config: Configuration object, uses global config if None
        """
        super().__init__(config)
        
        # Get features to analyze
        self.features = self.detector_config.get("features", ["response_length", "token_count", "response_time"])
        
        # Get statistical methods to use
        self.methods = self.detector_config.get("methods", [
            {"name": "ks_test", "threshold": 0.05},
            {"name": "js_divergence", "threshold": 0.2}
        ])
    
    def detect(self, 
              reference_samples: List[ProcessedSample],
              current_samples: List[ProcessedSample]) -> DriftResult:
        """
        Detect drift between reference and current feature distributions.
        
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
        
        # Extract features
        ref_features, curr_features = self._extract_features(reference_samples, current_samples)
        
        # Initialize result variables
        feature_results = {}
        overall_drift_score = 0.0
        num_features = 0
        
        # Analyze each feature
        for feature_name in self.features:
            if feature_name not in ref_features or feature_name not in curr_features:
                continue
            
            ref_values = ref_features[feature_name]
            curr_values = curr_features[feature_name]
            
            # Skip features with insufficient data
            if len(ref_values) < 10 or len(curr_values) < 10:
                continue
            
            # Run statistical tests
            test_results = self._run_statistical_tests(ref_values, curr_values)
            
            # Calculate feature drift score
            feature_drift = 0.0
            for method in test_results:
                score = test_results[method]["score"]
                threshold = test_results[method]["threshold"]
                weight = test_results[method].get("weight", 1.0)
                
                normalized_score = min(score / threshold, 1.0) if threshold > 0 else 0.0
                feature_drift += normalized_score * weight
            
            # Normalize by number of methods
            if test_results:
                feature_drift /= sum(test_results[m].get("weight", 1.0) for m in test_results)
                
                # Add to overall drift score
                overall_drift_score += feature_drift
                num_features += 1
            
            # Store feature results
            feature_results[feature_name] = {
                "drift_score": feature_drift,
                "tests": test_results,
                "ref_stats": {
                    "mean": float(np.mean(ref_values)),
                    "std": float(np.std(ref_values)),
                    "min": float(np.min(ref_values)),
                    "max": float(np.max(ref_values)),
                    "median": float(np.median(ref_values))
                },
                "curr_stats": {
                    "mean": float(np.mean(curr_values)),
                    "std": float(np.std(curr_values)),
                    "min": float(np.min(curr_values)),
                    "max": float(np.max(curr_values)),
                    "median": float(np.median(curr_values))
                }
            }
        
        # Calculate overall drift score
        if num_features > 0:
            overall_drift_score /= num_features
        
        # Prepare analysis results
        analysis = {
            "features": feature_results,
            "num_features_analyzed": num_features
        }
        
        return self._create_result(
            overall_drift_score, reference_samples, current_samples,
            metric_name="distribution_drift",
            analysis=analysis
        )
    
    def _extract_features(self, 
                         reference_samples: List[ProcessedSample],
                         current_samples: List[ProcessedSample]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Extract numerical features from processed samples.
        
        Args:
            reference_samples: List of reference processed samples
            current_samples: List of current processed samples
            
        Returns:
            Tuple of dictionaries with feature arrays for reference and current samples
        """
        ref_features = {}
        curr_features = {}
        
        # Extract reference features
        for feature_name in self.features:
            values = []
            for sample in reference_samples:
                value = getattr(sample, feature_name, None)
                if value is not None:
                    values.append(value)
            
            if values:
                ref_features[feature_name] = np.array(values)
        
        # Extract current features
        for feature_name in self.features:
            values = []
            for sample in current_samples:
                value = getattr(sample, feature_name, None)
                if value is not None:
                    values.append(value)
            
            if values:
                curr_features[feature_name] = np.array(values)
        
        return ref_features, curr_features
    
    def _run_statistical_tests(self, ref_values: np.ndarray, curr_values: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Run statistical tests to compare distributions.
        
        Args:
            ref_values: Reference values array
            curr_values: Current values array
            
        Returns:
            Dict[str, Dict[str, Any]]: Results of statistical tests
        """
        results = {}
        
        for method_config in self.methods:
            method_name = method_config["name"]
            threshold = method_config.get("threshold", 0.05)
            weight = method_config.get("weight", 1.0)
            
            try:
                if method_name == "ks_test":
                    # Kolmogorov-Smirnov test
                    statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                    
                    results["ks_test"] = {
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "score": 1.0 - float(p_value),  # Higher score = more drift
                        "threshold": threshold,
                        "weight": weight,
                        "drift_detected": p_value < threshold
                    }
                    
                elif method_name == "js_divergence":
                    # Jensen-Shannon divergence
                    js_div = self._calculate_js_divergence(ref_values, curr_values)
                    
                    results["js_divergence"] = {
                        "divergence": float(js_div),
                        "score": float(js_div),
                        "threshold": threshold,
                        "weight": weight,
                        "drift_detected": js_div > threshold
                    }
                    
                elif method_name == "wasserstein":
                    # Wasserstein distance (Earth Mover's Distance)
                    wd = stats.wasserstein_distance(ref_values, curr_values)
                    
                    # Normalize by the range of values
                    value_range = max(np.max(ref_values) - np.min(ref_values),
                                   np.max(curr_values) - np.min(curr_values))
                    if value_range > 0:
                        normalized_wd = wd / value_range
                    else:
                        normalized_wd = 0.0
                    
                    results["wasserstein"] = {
                        "distance": float(wd),
                        "normalized_distance": float(normalized_wd),
                        "score": float(normalized_wd),
                        "threshold": threshold,
                        "weight": weight,
                        "drift_detected": normalized_wd > threshold
                    }
                    
                elif method_name == "t_test":
                    # T-test for difference in means
                    statistic, p_value = stats.ttest_ind(ref_values, curr_values, equal_var=False)
                    
                    results["t_test"] = {
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "score": 1.0 - float(p_value),
                        "threshold": threshold,
                        "weight": weight,
                        "drift_detected": p_value < threshold
                    }
                
            except Exception as e:
                logger.warning(f"Error running {method_name}: {str(e)}")
        
        return results
    
    def _calculate_js_divergence(self, ref_values: np.ndarray, curr_values: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence between distributions.
        
        Args:
            ref_values: Reference values array
            curr_values: Current values array
            
        Returns:
            float: Jensen-Shannon divergence
        """
        # Create histograms
        min_val = min(np.min(ref_values), np.min(curr_values))
        max_val = max(np.max(ref_values), np.max(curr_values))
        
        # Ensure we have a reasonable range
        if max_val <= min_val:
            return 0.0
        
        # Determine number of bins (Rice rule)
        n = len(ref_values) + len(curr_values)
        num_bins = int(np.ceil(2 * n**(1/3)))
        
        # Calculate histograms
        ref_hist, _ = np.histogram(ref_values, bins=num_bins, range=(min_val, max_val), density=True)
        curr_hist, _ = np.histogram(curr_values, bins=num_bins, range=(min_val, max_val), density=True)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        ref_hist = ref_hist + epsilon
        curr_hist = curr_hist + epsilon
        
        # Normalize
        ref_hist = ref_hist / np.sum(ref_hist)
        curr_hist = curr_hist / np.sum(curr_hist)
        
        # Calculate KL divergences
        m = (ref_hist + curr_hist) / 2
        
        js_div = 0.5 * np.sum(ref_hist * np.log(ref_hist / m)) + 0.5 * np.sum(curr_hist * np.log(curr_hist / m))
        
        return float(js_div)