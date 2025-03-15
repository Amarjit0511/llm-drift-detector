"""
LLM Drift Detector - Monitoring for drift in Large Language Models.

This package provides tools to detect, monitor, and alert on changes in
LLM behavior, helping maintain reliability and quality over time.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import os
import importlib.metadata

# Set up package-level logger
logger = logging.getLogger("llm_drift_detector")
logger.setLevel(logging.INFO)

# Check if a handler is already configured to avoid duplicates
if not logger.handlers:
    # Add console handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Package metadata
try:
    __version__ = importlib.metadata.version("llm_drift_detector")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0.dev0"  # Default during development

# Import main components for easier access
from .config import get_config, load_config, Config
from .data import get_collector, get_processor
from .detectors import get_detector, get_all_detectors
from .monitoring import get_metrics_tracker, get_alert_manager
from .storage import get_storage, get_storage_types

# Main classes for end-user API
from .drift_monitor import DriftMonitor, DriftResult
from .provider_monitor import ProviderMonitor
from .runners import periodic_detection, run_detection, store_samples, collect_and_detect

__all__ = [
    # Version
    "__version__",
    
    # Main components
    "DriftMonitor",
    "ProviderMonitor",
    "DriftResult",
    
    # Runner functions
    "periodic_detection",
    "run_detection",
    "store_samples",
    "collect_and_detect",
    
    # Access functions
    "get_config",
    "load_config",
    "get_collector",
    "get_processor",
    "get_detector",
    "get_all_detectors",
    "get_metrics_tracker",
    "get_alert_manager",
    "get_storage",
    "get_storage_types",
]

# Initialize logging based on config
def _setup_logging():
    """Setup logging based on configuration."""
    config = get_config()
    log_level = config.get("logging.level", "INFO")
    
    # Set package log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Configure log file if specified
    log_file = config.get("logging.file")
    if log_file:
        # Check if we should rotate logs
        if config.get("logging.rotate", True):
            from logging.handlers import RotatingFileHandler
            max_size = config.get("logging.max_size", 10_485_760)  # 10 MB
            backup_count = config.get("logging.backup_count", 5)
            
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=max_size,
                backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(log_file)
        
        # Set formatter
        log_format = config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        
        # Add handler
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")

# Auto-initialize based on environment variable
if os.environ.get("LLM_DRIFT_AUTO_INIT", "1") == "1":
    load_config()
    _setup_logging()
    logger.info(f"LLM Drift Detector {__version__} initialized")