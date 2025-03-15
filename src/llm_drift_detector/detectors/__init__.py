"""
Drift detection module for LLM outputs.

This module provides various drift detection algorithms to identify
changes in LLM behavior over time.
"""

from .base import BaseDriftDetector, DriftResult
from .embedding import EmbeddingDriftDetector
from .distribution import DistributionDriftDetector
from .semantic import SemanticDriftDetector
from .performance import PerformanceDriftDetector

__all__ = [
    "BaseDriftDetector",
    "DriftResult",
    "EmbeddingDriftDetector",
    "DistributionDriftDetector",
    "SemanticDriftDetector",
    "PerformanceDriftDetector",
    "get_detector",
    "get_all_detectors"
]

# Factory function for getting detectors
def get_detector(detector_type: str, config=None):
    """
    Get a drift detector instance by type.
    
    Args:
        detector_type: Type of detector ('embedding', 'distribution', 'semantic', or 'performance')
        config: Optional configuration object
        
    Returns:
        BaseDriftDetector: Instance of the requested detector
        
    Raises:
        ValueError: If detector_type is not valid
    """
    from ..config import get_config
    
    if config is None:
        config = get_config()
    
    detector_type = detector_type.lower()
    
    if detector_type == 'embedding':
        return EmbeddingDriftDetector(config)
    elif detector_type == 'distribution':
        return DistributionDriftDetector(config)
    elif detector_type == 'semantic':
        return SemanticDriftDetector(config)
    elif detector_type == 'performance':
        return PerformanceDriftDetector(config)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")

def get_all_detectors(config=None):
    """
    Get instances of all available drift detectors.
    
    Args:
        config: Optional configuration object
        
    Returns:
        dict: Dictionary of detector instances keyed by detector type
    """
    from ..config import get_config
    
    if config is None:
        config = get_config()
    
    detectors = {
        'embedding': EmbeddingDriftDetector(config),
        'distribution': DistributionDriftDetector(config),
        'semantic': SemanticDriftDetector(config),
        'performance': PerformanceDriftDetector(config)
    }
    
    # Filter out disabled detectors
    enabled_detectors = {}
    for name, detector in detectors.items():
        if detector.enabled:
            enabled_detectors[name] = detector
    
    return enabled_detectors