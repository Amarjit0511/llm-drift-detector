"""
Base drift detector implementation.

This module defines the base interface for all drift detectors
and common functionality for drift detection.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd

from ..config import Config, get_config
from ..data.processor import ProcessedSample

logger = logging.getLogger(__name__)

@dataclass
class DriftResult:
    """
    Result of a drift detection operation.
    """
    # Basic information
    detector_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Drift results
    drift_detected: bool = False
    drift_score: float = 0.0
    threshold: float = 0.0
    p_value: Optional[float] = None
    
    # Analysis details
    metric_name: Optional[str] = None
    analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Sample information
    reference_size: int = 0
    current_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary with result data
        """
        return {
            "detector_name": self.detector_name,
            "timestamp": self.timestamp.isoformat(),
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_score,
            "threshold": self.threshold,
            "p_value": self.p_value,
            "metric_name": self.metric_name,
            "analysis": self.analysis,
            "reference_size": self.reference_size,
            "current_size": self.current_size
        }


class BaseDriftDetector(ABC):
    """
    Base class for all drift detectors.
    
    This abstract class defines the interface that all drift
    detectors must implement and provides common functionality.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the drift detector.
        
        Args:
            config: Configuration object, uses global config if None
        """
        self.config = config or get_config()
        self.name = self.__class__.__name__
        
        # Get configuration specific to this detector type
        detector_type = self._get_detector_type()
        self.detector_config = self.config.get(f"drift_detection.{detector_type}", {})
        
        # Check if detector is enabled
        self.enabled = self.detector_config.get("enabled", True)
        
        # Get threshold
        self.threshold = self.detector_config.get("threshold", 0.1)
        
        # Weight for combining with other detectors
        self.weight = self.detector_config.get("weight", 1.0)
        
        if not self.enabled:
            logger.info(f"{self.name} is disabled in configuration")
    
    def _get_detector_type(self) -> str:
        """
        Get the detector type from the class name.
        
        Returns:
            str: Detector type (e.g., 'embedding' for EmbeddingDriftDetector)
        """
        class_name = self.__class__.__name__
        if class_name.endswith("DriftDetector"):
            return class_name[:-13].lower()
        return "base"
    
    @abstractmethod
    def detect(self, 
              reference_samples: List[ProcessedSample],
              current_samples: List[ProcessedSample]) -> DriftResult:
        """
        Detect drift between reference and current samples.
        
        Args:
            reference_samples: List of reference processed samples
            current_samples: List of current processed samples
            
        Returns:
            DriftResult: Result of drift detection
        """
        pass
    
    def _validate_input(self, 
                       reference_samples: List[ProcessedSample],
                       current_samples: List[ProcessedSample]) -> bool:
        """
        Validate input samples before drift detection.
        
        Args:
            reference_samples: List of reference processed samples
            current_samples: List of current processed samples
            
        Returns:
            bool: True if inputs are valid, False otherwise
        """
        if not self.enabled:
            logger.debug(f"{self.name} is disabled")
            return False
        
        # Check if we have enough samples
        min_samples = self.detector_config.get("min_samples", 10)
        
        if len(reference_samples) < min_samples:
            logger.warning(f"Not enough reference samples for {self.name}. "
                          f"Need at least {min_samples}, got {len(reference_samples)}")
            return False
        
        if len(current_samples) < min_samples:
            logger.warning(f"Not enough current samples for {self.name}. "
                          f"Need at least {min_samples}, got {len(current_samples)}")
            return False
        
        return True
    
    def _create_result(self, 
                      drift_score: float,
                      reference_samples: List[ProcessedSample],
                      current_samples: List[ProcessedSample],
                      metric_name: Optional[str] = None,
                      p_value: Optional[float] = None,
                      analysis: Optional[Dict[str, Any]] = None) -> DriftResult:
        """
        Create a standardized drift result.
        
        Args:
            drift_score: Calculated drift score
            reference_samples: Reference samples used
            current_samples: Current samples used
            metric_name: Optional name of the specific metric
            p_value: Optional p-value from statistical test
            analysis: Optional detailed analysis results
            
        Returns:
            DriftResult: Standardized drift result
        """
        detector_type = self._get_detector_type()
        
        return DriftResult(
            detector_name=detector_type,
            drift_detected=drift_score > self.threshold,
            drift_score=drift_score,
            threshold=self.threshold,
            p_value=p_value,
            metric_name=metric_name,
            analysis=analysis or {},
            reference_size=len(reference_samples),
            current_size=len(current_samples)
        )