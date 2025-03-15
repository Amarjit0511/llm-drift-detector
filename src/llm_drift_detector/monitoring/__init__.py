"""
Monitoring and alerting functionality for LLM Drift Detector.

This module provides tools for tracking drift metrics over time
and sending alerts when significant drift is detected.
"""

from .metrics import MetricsTracker, DriftMetric
from .alerting import AlertManager, Alert

__all__ = [
    "MetricsTracker",
    "DriftMetric",
    "AlertManager",
    "Alert",
    "get_metrics_tracker",
    "get_alert_manager"
]

# Singleton instances
_metrics_tracker = None
_alert_manager = None

def get_metrics_tracker():
    """
    Get the global metrics tracker instance.
    
    Returns:
        MetricsTracker: Global metrics tracker instance
    """
    global _metrics_tracker
    
    if _metrics_tracker is None:
        from ..config import get_config
        config = get_config()
        _metrics_tracker = MetricsTracker(config)
    
    return _metrics_tracker

def get_alert_manager():
    """
    Get the global alert manager instance.
    
    Returns:
        AlertManager: Global alert manager instance
    """
    global _alert_manager
    
    if _alert_manager is None:
        from ..config import get_config
        config = get_config()
        _alert_manager = AlertManager(config)
    
    return _alert_manager