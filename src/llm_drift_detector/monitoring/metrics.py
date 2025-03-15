"""
Metrics collection and storage for LLM drift detection.

This module handles tracking and storing drift metrics over time,
enabling historical analysis and visualization.
"""

import logging
import os
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
import threading
import uuid

from ..config import Config, get_config
from ..detectors.base import DriftResult

logger = logging.getLogger(__name__)

@dataclass
class DriftMetric:
    """
    Individual drift metric data point.
    """
    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Source information
    provider_name: str = ""
    model_name: str = ""
    detector_name: str = ""
    metric_name: Optional[str] = None
    
    # Drift values
    drift_score: float = 0.0
    threshold: float = 0.0
    drift_detected: bool = False
    p_value: Optional[float] = None
    
    # Sample information
    reference_size: int = 0
    current_size: int = 0
    
    # Additional data
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the metric
        """
        result = asdict(self)
        result["timestamp"] = result["timestamp"].isoformat()
        return result
    
    @classmethod
    def from_drift_result(cls, 
                        result: DriftResult, 
                        provider_name: str, 
                        model_name: str) -> 'DriftMetric':
        """
        Create a DriftMetric from a DriftResult.
        
        Args:
            result: Drift detection result
            provider_name: Name of the LLM provider
            model_name: Name of the model
            
        Returns:
            DriftMetric: Created metric
        """
        return cls(
            timestamp=result.timestamp,
            provider_name=provider_name,
            model_name=model_name,
            detector_name=result.detector_name,
            metric_name=result.metric_name,
            drift_score=result.drift_score,
            threshold=result.threshold,
            drift_detected=result.drift_detected,
            p_value=result.p_value,
            reference_size=result.reference_size,
            current_size=result.current_size,
            details=result.analysis
        )


class MetricsTracker:
    """
    Tracks and stores drift metrics over time.
    
    This class provides functionality for recording drift metrics,
    retrieving historical data, and managing metric storage.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the metrics tracker.
        
        Args:
            config: Configuration object, uses global config if None
        """
        self.config = config or get_config()
        
        # Get storage configuration
        storage_backend = self.config.get("monitoring.storage_backend", "local")
        self.storage_type = storage_backend
        
        # Set up storage based on type
        if storage_backend == "local":
            self._setup_local_storage()
        elif storage_backend == "prometheus":
            self._setup_prometheus()
        
        # In-memory cache of recent metrics
        self.recent_metrics = []
        self.max_cache_size = 1000
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def _setup_local_storage(self):
        """Set up local file storage for metrics."""
        metrics_file = self.config.get("monitoring.local.metrics_file", "./data/metrics/metrics.csv")
        rotation = self.config.get("monitoring.local.rotation", "daily")
        
        self.metrics_dir = os.path.dirname(metrics_file)
        self.metrics_base_filename = os.path.basename(metrics_file)
        self.rotation = rotation
        
        # Create directory if it doesn't exist
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Current metrics file
        self.current_file = self._get_current_filename()
    
    def _setup_prometheus(self):
        """Set up Prometheus metrics export."""
        try:
            from prometheus_client import Gauge, Counter, REGISTRY, start_http_server
            
            # Set up Prometheus metrics
            self.prom_drift_score = Gauge(
                'llm_drift_score', 
                'LLM Drift Score', 
                ['provider', 'model', 'detector', 'metric']
            )
            
            self.prom_drift_detected = Counter(
                'llm_drift_detected_total', 
                'LLM Drift Detected Count', 
                ['provider', 'model', 'detector', 'metric']
            )
            
            # Start HTTP server if enabled
            if self.config.get("monitoring.prometheus.enabled", False):
                port = self.config.get("monitoring.prometheus.port", 8000)
                start_http_server(port)
                logger.info(f"Started Prometheus HTTP server on port {port}")
            
        except ImportError:
            logger.warning("prometheus_client not installed. Falling back to local storage.")
            self.storage_type = "local"
            self._setup_local_storage()
    
    def _get_current_filename(self) -> str:
        """
        Get the current metrics filename based on rotation policy.
        
        Returns:
            str: Current metrics filename
        """
        base_name, ext = os.path.splitext(self.metrics_base_filename)
        now = datetime.now()
        
        if self.rotation == "hourly":
            date_part = now.strftime("%Y%m%d_%H")
            return os.path.join(self.metrics_dir, f"{base_name}_{date_part}{ext}")
        elif self.rotation == "daily":
            date_part = now.strftime("%Y%m%d")
            return os.path.join(self.metrics_dir, f"{base_name}_{date_part}{ext}")
        elif self.rotation == "weekly":
            # Use ISO week number
            year, week, _ = now.isocalendar()
            return os.path.join(self.metrics_dir, f"{base_name}_{year}_week{week}{ext}")
        elif self.rotation == "monthly":
            date_part = now.strftime("%Y%m")
            return os.path.join(self.metrics_dir, f"{base_name}_{date_part}{ext}")
        else:
            # No rotation
            return os.path.join(self.metrics_dir, self.metrics_base_filename)
    
    def record_metric(self, metric: DriftMetric):
        """
        Record a new drift metric.
        
        Args:
            metric: Drift metric to record
        """
        with self._lock:
            # Add to in-memory cache
            self.recent_metrics.append(metric)
            
            # Trim cache if it gets too large
            if len(self.recent_metrics) > self.max_cache_size:
                self.recent_metrics = self.recent_metrics[-self.max_cache_size:]
            
            # Record to storage
            if self.storage_type == "local":
                self._record_to_file(metric)
            elif self.storage_type == "prometheus":
                self._record_to_prometheus(metric)
    
    def _record_to_file(self, metric: DriftMetric):
        """
        Record metric to local file storage.
        
        Args:
            metric: Drift metric to record
        """
        # Check if we need to rotate file
        current_file = self._get_current_filename()
        if current_file != self.current_file:
            self.current_file = current_file
        
        # Convert to dictionary
        data = metric.to_dict()
        
        # Create DataFrame with a single row
        df = pd.DataFrame([data])
        
        # Append to file or create new file
        file_exists = os.path.isfile(self.current_file)
        
        try:
            if file_exists:
                df.to_csv(self.current_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.current_file, index=False)
        except Exception as e:
            logger.error(f"Error recording metric to file: {str(e)}")
    
    def _record_to_prometheus(self, metric: DriftMetric):
        """
        Record metric to Prometheus.
        
        Args:
            metric: Drift metric to record
        """
        try:
            # Record drift score
            self.prom_drift_score.labels(
                provider=metric.provider_name,
                model=metric.model_name,
                detector=metric.detector_name,
                metric=metric.metric_name or ""
            ).set(metric.drift_score)
            
            # Record drift detection
            if metric.drift_detected:
                self.prom_drift_detected.labels(
                    provider=metric.provider_name,
                    model=metric.model_name,
                    detector=metric.detector_name,
                    metric=metric.metric_name or ""
                ).inc()
                
        except Exception as e:
            logger.error(f"Error recording metric to Prometheus: {str(e)}")
    
    def get_recent_metrics(self, 
                          hours: int = 24,
                          provider_filter: Optional[str] = None,
                          model_filter: Optional[str] = None,
                          detector_filter: Optional[str] = None) -> List[DriftMetric]:
        """
        Get recent metrics from in-memory cache.
        
        Args:
            hours: Number of hours to look back
            provider_filter: Optional provider name filter
            model_filter: Optional model name filter
            detector_filter: Optional detector name filter
            
        Returns:
            List[DriftMetric]: List of recent metrics
        """
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(hours=hours)
            
            filtered_metrics = []
            
            for metric in self.recent_metrics:
                # Apply time filter
                if metric.timestamp < cutoff:
                    continue
                
                # Apply provider filter
                if provider_filter and metric.provider_name != provider_filter:
                    continue
                
                # Apply model filter
                if model_filter and metric.model_name != model_filter:
                    continue
                
                # Apply detector filter
                if detector_filter and metric.detector_name != detector_filter:
                    continue
                
                filtered_metrics.append(metric)
            
            return filtered_metrics
    
    def load_historical_metrics(self,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               provider_filter: Optional[str] = None,
                               model_filter: Optional[str] = None,
                               detector_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical metrics from storage.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            provider_filter: Optional provider name filter
            model_filter: Optional model name filter
            detector_filter: Optional detector name filter
            
        Returns:
            pd.DataFrame: DataFrame with historical metrics
        """
        if self.storage_type != "local":
            logger.warning("Historical metrics loading only supported for local storage")
            return pd.DataFrame()
        
        # Default dates if not specified
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # Find all metrics files in the range
        all_files = []
        for filename in os.listdir(self.metrics_dir):
            if not filename.startswith(os.path.splitext(self.metrics_base_filename)[0]):
                continue
            
            file_path = os.path.join(self.metrics_dir, filename)
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            # Use file modification time as a heuristic for date range
            if file_mtime >= start_date - timedelta(days=1) and file_mtime <= end_date + timedelta(days=1):
                all_files.append(file_path)
        
        if not all_files:
            return pd.DataFrame()
        
        # Load and concat all files
        dfs = []
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error loading metrics file {file_path}: {str(e)}")
        
        if not dfs:
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Convert timestamp strings to datetime
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        
        # Apply filters
        if start_date:
            combined_df = combined_df[combined_df['timestamp'] >= start_date]
        
        if end_date:
            combined_df = combined_df[combined_df['timestamp'] <= end_date]
        
        if provider_filter:
            combined_df = combined_df[combined_df['provider_name'] == provider_filter]
        
        if model_filter:
            combined_df = combined_df[combined_df['model_name'] == model_filter]
        
        if detector_filter:
            combined_df = combined_df[combined_df['detector_name'] == detector_filter]
        
        return combined_df
    
    def get_drift_summary(self,
                         hours: int = 24,
                         provider_filter: Optional[str] = None,
                         model_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of drift metrics.
        
        Args:
            hours: Number of hours to include
            provider_filter: Optional provider name filter
            model_filter: Optional model name filter
            
        Returns:
            Dict[str, Any]: Summary of drift metrics
        """
        # Get recent metrics
        metrics = self.get_recent_metrics(
            hours=hours,
            provider_filter=provider_filter,
            model_filter=model_filter
        )
        
        if not metrics:
            return {
                "providers": [],
                "drift_detected": False,
                "overall_status": "unknown",
                "last_updated": datetime.now().isoformat()
            }
        
        # Group by provider and model
        by_provider = {}
        
        for metric in metrics:
            provider = metric.provider_name
            model = metric.model_name
            
            if provider not in by_provider:
                by_provider[provider] = {}
            
            if model not in by_provider[provider]:
                by_provider[provider][model] = {
                    "detectors": {},
                    "drift_detected": False,
                    "max_drift_score": 0.0
                }
            
            detector = metric.detector_name
            
            if detector not in by_provider[provider][model]["detectors"]:
                by_provider[provider][model]["detectors"][detector] = {
                    "metrics": [],
                    "drift_detected": False,
                    "max_drift_score": 0.0,
                    "latest_timestamp": None
                }
            
            # Add metric
            by_provider[provider][model]["detectors"][detector]["metrics"].append(metric)
            
            # Update summary values
            if metric.drift_detected:
                by_provider[provider][model]["detectors"][detector]["drift_detected"] = True
                by_provider[provider][model]["drift_detected"] = True
            
            by_provider[provider][model]["detectors"][detector]["max_drift_score"] = max(
                by_provider[provider][model]["detectors"][detector]["max_drift_score"],
                metric.drift_score
            )
            
            by_provider[provider][model]["max_drift_score"] = max(
                by_provider[provider][model]["max_drift_score"],
                metric.drift_score
            )
            
            # Update timestamp
            latest = by_provider[provider][model]["detectors"][detector]["latest_timestamp"]
            if latest is None or metric.timestamp > latest:
                by_provider[provider][model]["detectors"][detector]["latest_timestamp"] = metric.timestamp
        
        # Build summary response
        result = {
            "providers": [],
            "drift_detected": False,
            "overall_status": "normal",
            "last_updated": datetime.now().isoformat()
        }
        
        for provider, models in by_provider.items():
            provider_summary = {
                "name": provider,
                "models": [],
                "drift_detected": False
            }
            
            for model, model_data in models.items():
                model_summary = {
                    "name": model,
                    "drift_detected": model_data["drift_detected"],
                    "max_drift_score": model_data["max_drift_score"],
                    "detectors": []
                }
                
                for detector, detector_data in model_data["detectors"].items():
                    detector_summary = {
                        "name": detector,
                        "drift_detected": detector_data["drift_detected"],
                        "max_drift_score": detector_data["max_drift_score"],
                        "latest_timestamp": detector_data["latest_timestamp"].isoformat() if detector_data["latest_timestamp"] else None,
                        "metric_count": len(detector_data["metrics"])
                    }
                    
                    model_summary["detectors"].append(detector_summary)
                
                provider_summary["models"].append(model_summary)
                
                if model_data["drift_detected"]:
                    provider_summary["drift_detected"] = True
            
            result["providers"].append(provider_summary)
            
            if provider_summary["drift_detected"]:
                result["drift_detected"] = True
        
        # Set overall status
        if result["drift_detected"]:
            # Calculate severity based on drift scores
            max_score = 0.0
            for provider in result["providers"]:
                for model in provider["models"]:
                    max_score = max(max_score, model["max_drift_score"])
            
            if max_score > 0.8:
                result["overall_status"] = "critical"
            else:
                result["overall_status"] = "warning"
        
        return result