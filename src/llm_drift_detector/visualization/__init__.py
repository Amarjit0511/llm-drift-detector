"""
Visualization module for LLM Drift Detector.

This module provides tools for visualizing drift detection results,
including interactive dashboards, plots, and data exploration utilities.
"""

from .dashboard import (
    DashboardApp,
    create_dashboard,
    launch_dashboard,
    create_drift_plot,
    create_histogram_plot,
    create_heatmap_plot,
    create_line_plot,
    create_scatter_plot,
    create_embedding_plot
)

__all__ = [
    "DashboardApp",
    "create_dashboard",
    "launch_dashboard",
    "create_drift_plot",
    "create_histogram_plot",
    "create_heatmap_plot",
    "create_line_plot",
    "create_scatter_plot",
    "create_embedding_plot"
]