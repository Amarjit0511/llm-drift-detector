"""
API module for LLM Drift Detector.

This module provides a FastAPI application for interacting with the
drift detection system, allowing users to:
- Query current drift status
- Retrieve historical drift metrics
- Configure detection parameters
- Trigger manual detection runs
"""

from fastapi import FastAPI
from .routes import router as api_router

__all__ = ["create_app"]


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    app = FastAPI(
        title="LLM Drift Detector API",
        description="API for monitoring and managing LLM output drift detection",
        version="0.1.0",
    )
    
    # Include the API router
    app.include_router(api_router, prefix="/api/v1")
    
    return app


# Create a default app instance for simple usage
app = create_app()