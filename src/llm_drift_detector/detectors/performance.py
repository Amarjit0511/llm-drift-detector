"""
Performance-based drift detection.

This module provides drift detection based on LLM performance metrics
like response time, token count, and error rates.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats

from .base import BaseDriftDetector, DriftResult
from ..data.processor import ProcessedSample
from ..config import Config

logger = logging.getLogger(__name__)

class PerformanceDriftDetector(BaseDriftDetector):
    """
    Detects drift based on LLM performance metrics.
    
    This detector monitors operational metrics like response time,
    token count, and error rates to detect performance degradation.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the performance drift detector.
        
        Args:
            config: Configuration object, uses global config if None
        """
        super().__init__(config)
        
        # Get performance metrics to monitor
        self.metrics = self.detector_config.get("metrics", [
            {
                "name": "response_time",
                "upper_threshold": 5.0,  # seconds
                "lower_threshold": 0.1   # seconds
            },
            {
                "name": "token_count",
                "upper_threshold": 500,
                "lower_threshold": 10
            },
            {
                "name": "error_rate",
                "upper_threshold": 0.05,  # 5% error rate
                "lower_threshold": 0.0
            }
        ])
    
    def detect(self, 
              reference_samples: List[ProcessedSample],
              current_samples: List[ProcessedSample]) -> DriftResult:
        """
        Detect performance drift between reference and current samples.
        
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
        
        # Analyze each metric
        metric_results = {}
        overall_drift_score = 0.0
        num_metrics = 0
        
        for metric_config in self.metrics:
            metric_name = metric_config["name"]
            upper_threshold = metric_config.get("upper_threshold")
            lower_threshold = metric_config.get("lower_threshold")
            
            # Calculate metric values
            ref_values = self._extract_metric_values(reference_samples, metric_name)
            curr_values = self._extract_metric_values(current_samples, metric_name)
            
            # Skip metrics with insufficient data
            if len(ref_values) < 5 or len(curr_values) < 5:
                continue
            
            # Special handling for error rate
            if metric_name == "error_rate":
                # Calculate error rates
                ref_error_rate = self._calculate_error_rate(reference_samples)
                curr_error_rate = self._calculate_error_rate(current_samples)
                
                # Calculate relative change
                if ref_error_rate > 0:
                    rel_change = (curr_error_rate - ref_error_rate) / ref_error_rate
                else:
                    rel_change = curr_error_rate * 100  # Treat any errors as significant if reference had none
                
                # Scale to a 0-1 range for drift score
                max_rel_change = 2.0  # 200% increase in error rate
                metric_drift = min(max(0, rel_change) / max_rel_change, 1.0)
                
                # Store results
                metric_results[metric_name] = {
                    "reference_value": ref_error_rate,
                    "current_value": curr_error_rate,
                    "relative_change": rel_change,
                    "drift_score": metric_drift,
                    "drift_detected": curr_error_rate > upper_threshold or metric_drift > 0.5
                }
                
                # Add to overall drift score
                overall_drift_score += metric_drift
                num_metrics += 1
                
            else:
                # Calculate summary statistics
                ref_mean = np.mean(ref_values)
                curr_mean = np.mean(curr_values)
                
                # Calculate relative change
                if ref_mean > 0:
                    rel_change = (curr_mean - ref_mean) / ref_mean
                else:
                    rel_change = 0.0
                
                # Detect if current values exceed thresholds
                threshold_exceeded = False
                if upper_threshold is not None and curr_mean > upper_threshold:
                    threshold_exceeded = True
                if lower_threshold is not None and curr_mean < lower_threshold:
                    threshold_exceeded = True
                
                # Calculate statistical significance
                try:
                    _, p_value = stats.ttest_ind(ref_values, curr_values, equal_var=False)
                    statistically_significant = p_value < 0.05
                except:
                    p_value = None
                    statistically_significant = False
                
                # Calculate metric drift score based on relative change
                max_rel_change = 0.5  # 50% change is considered maximum drift
                metric_drift = min(abs(rel_change) / max_rel_change, 1.0)
                
                # Increase drift score if threshold is exceeded
                if threshold_exceeded:
                    metric_drift = max(metric_drift, 0.8)
                
                # Store results
                metric_results[metric_name] = {
                    "reference_mean": float(ref_mean),
                    "reference_std": float(np.std(ref_values)),
                    "current_mean": float(curr_mean),
                    "current_std": float(np.std(curr_values)),
                    "relative_change": float(rel_change),
                    "p_value": float(p_value) if p_value is not None else None,
                    "statistically_significant": statistically_significant,
                    "threshold_exceeded": threshold_exceeded,
                    "drift_score": float(metric_drift),
                    "drift_detected": metric_drift > 0.5 or threshold_exceeded
                }
                
                # Add to overall drift score
                overall_drift_score += metric_drift
                num_metrics += 1
        
        # Calculate overall drift score
        if num_metrics > 0:
            overall_drift_score /= num_metrics
        
        # Prepare analysis results
        analysis = {
            "metrics": metric_results,
            "num_metrics_analyzed": num_metrics
        }
        
        return self._create_result(
            overall_drift_score, reference_samples, current_samples,
            metric_name="performance_drift",
            analysis=analysis
        )
    
    def _extract_metric_values(self, samples: List[ProcessedSample], metric_name: str) -> np.ndarray:
        """
        Extract metric values from processed samples.
        
        Args:
            samples: List of processed samples
            metric_name: Name of the metric to extract
            
        Returns:
            np.ndarray: Array of metric values
        """
        values = []
        
        for sample in samples:
            if metric_name == "response_time" and sample.response_time is not None:
                values.append(sample.response_time)
            elif metric_name == "token_count" and sample.token_count is not None:
                values.append(sample.token_count)
            elif metric_name == "response_length":
                values.append(sample.response_length)
            elif metric_name in sample.metadata:
                values.append(sample.metadata[metric_name])
        
        return np.array(values)
    
    def _calculate_error_rate(self, samples: List[ProcessedSample]) -> float:
        """
        Calculate error rate from processed samples.
        
        Args:
            samples: List of processed samples
            
        Returns:
            float: Error rate (0-1)
        """
        error_count = 0
        
        for sample in samples:
            # Check for explicit error field
            if 'error' in sample.metadata and sample.metadata['error']:
                error_count += 1
                continue
            
            # Check for error in finish reason
            if 'finish_reason' in sample.metadata:
                finish_reason = sample.metadata['finish_reason']
                if finish_reason and finish_reason in ('error', 'timeout', 'content_filter'):
                    error_count += 1
                    continue
        
        return error_count / len(samples) if samples else 0.0
    
    def detect_anomalies(self, 
                       history: List[ProcessedSample],
                       window_size: int = 20) -> List[Dict[str, Any]]:
        """
        Detect performance anomalies in a time series of samples.
        
        This method uses a rolling window approach to detect sudden changes
        in performance metrics that may indicate issues.
        
        Args:
            history: List of processed samples in chronological order
            window_size: Size of the rolling window
            
        Returns:
            List[Dict[str, Any]]: List of detected anomalies
        """
        if len(history) < window_size * 2:
            return []
        
        anomalies = []
        
        for metric_config in self.metrics:
            metric_name = metric_config["name"]
            
            # Extract metric values
            values = []
            timestamps = []
            
            for sample in history:
                if metric_name == "response_time" and sample.response_time is not None:
                    values.append(sample.response_time)
                    timestamps.append(sample.timestamp)
                elif metric_name == "token_count" and sample.token_count is not None:
                    values.append(sample.token_count)
                    timestamps.append(sample.timestamp)
            
            if len(values) < window_size * 2:
                continue
            
            # Convert to numpy array
            values_array = np.array(values)
            
            # Detect anomalies using rolling window
            for i in range(window_size, len(values_array) - window_size + 1):
                prev_window = values_array[i-window_size:i]
                curr_window = values_array[i:i+window_size]
                
                prev_mean = np.mean(prev_window)
                prev_std = np.std(prev_window)
                
                curr_mean = np.mean(curr_window)
                
                # Calculate z-score
                if prev_std > 0:
                    z_score = abs(curr_mean - prev_mean) / prev_std
                else:
                    z_score = 0
                
                # Detect anomaly if z-score exceeds threshold
                if z_score > 3.0:  # 3 standard deviations
                    anomalies.append({
                        "metric": metric_name,
                        "timestamp": timestamps[i],
                        "value": float(curr_mean),
                        "expected": float(prev_mean),
                        "z_score": float(z_score),
                        "direction": "increase" if curr_mean > prev_mean else "decrease"
                    })
        
        return anomalies