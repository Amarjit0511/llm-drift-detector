"""
Data collection and processing module for LLM Drift Detector.

This module provides functionality for collecting LLM responses,
processing them for analysis, and managing reference distributions.
"""

from .collector import DataCollector, SampleBatch
from .processor import DataProcessor, ProcessedSample

__all__ = [
    "DataCollector",
    "SampleBatch",
    "DataProcessor", 
    "ProcessedSample",
    "get_collector",
    "get_processor"
]

# Singleton instances for common use
_default_collector = None
_default_processor = None

def get_collector() -> DataCollector:
    """
    Get the default data collector instance.
    
    Returns:
        DataCollector: Default data collector
    """
    global _default_collector
    
    if _default_collector is None:
        from ..config import get_config
        config = get_config()
        _default_collector = DataCollector(config)
    
    return _default_collector

def get_processor() -> DataProcessor:
    """
    Get the default data processor instance.
    
    Returns:
        DataProcessor: Default data processor
    """
    global _default_processor
    
    if _default_processor is None:
        from ..config import get_config
        config = get_config()
        _default_processor = DataProcessor(config)
    
    return _default_processor