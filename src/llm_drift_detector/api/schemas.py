"""
Pydantic schemas for API request and response validation.

This module defines the data models used in API requests and responses,
ensuring proper validation and documentation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field, validator


class ProviderType(str, Enum):
    """Type of LLM provider."""
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    VLLM = "vllm"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class DetectorType(str, Enum):
    """Type of drift detector."""
    EMBEDDING = "embedding"
    DISTRIBUTION = "distribution"
    SEMANTIC = "semantic"
    PERFORMANCE = "performance"


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    type: ProviderType
    model_name: str
    api_key: Optional[str] = Field(None, description="API key (not required for local models)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific parameters")


class DriftDetectionConfig(BaseModel):
    """Configuration for drift detection."""
    detectors: List[DetectorType]
    window_size: int = Field(100, description="Number of samples to consider for current distribution")
    reference_update_frequency: int = Field(7, description="Days between reference distribution updates")
    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "embedding": 0.15,
            "distribution": 0.05,
            "semantic": 0.3,
            "performance": 0.2
        },
        description="Thresholds for each detector type"
    )


class MonitorConfig(BaseModel):
    """Configuration for the drift monitor."""
    providers: List[ProviderConfig]
    detection: DriftDetectionConfig
    alert_email: Optional[str] = None
    alert_webhook: Optional[str] = None


class DriftStatus(BaseModel):
    """Current drift status for a specific detector."""
    detector: DetectorType
    is_drifting: bool
    drift_score: float
    threshold: float
    last_checked: datetime


class ProviderDriftStatus(BaseModel):
    """Drift status for a specific provider."""
    provider_name: str
    provider_type: ProviderType
    model_name: str
    detectors: List[DriftStatus]
    overall_status: str = Field(..., description="Overall drift status: 'normal', 'warning', or 'critical'")


class DriftMetricsResponse(BaseModel):
    """Response with drift metrics for all providers."""
    providers: List[ProviderDriftStatus]
    timestamp: datetime = Field(default_factory=datetime.now)


class TimeSeriesDataPoint(BaseModel):
    """Single data point for time series data."""
    timestamp: datetime
    value: float


class TimeSeriesData(BaseModel):
    """Time series data for a specific metric."""
    metric_name: str
    provider_name: str
    detector: DetectorType
    data: List[TimeSeriesDataPoint]


class HistoricalDataResponse(BaseModel):
    """Response with historical drift data."""
    time_series: List[TimeSeriesData]
    start_time: datetime
    end_time: datetime


class AlertConfig(BaseModel):
    """Configuration for alerts."""
    email: Optional[str] = None
    slack_webhook: Optional[str] = None
    threshold_multiplier: float = Field(1.0, description="Multiplier for base thresholds to trigger alerts")
    cooldown_minutes: int = Field(60, description="Minimum time between alerts")


class ConfigUpdateResponse(BaseModel):
    """Response after updating configuration."""
    success: bool
    message: str
    updated_config: Optional[MonitorConfig] = None


class LLMSample(BaseModel):
    """Sample of LLM input/output for manual drift detection."""
    prompt: str
    response: str
    provider: str
    model: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ManualDetectionRequest(BaseModel):
    """Request for manual drift detection."""
    samples: List[LLMSample]


class ManualDetectionResponse(BaseModel):
    """Response from manual drift detection."""
    drift_detected: bool
    analysis: Dict[str, Any]
    recommendations: List[str]