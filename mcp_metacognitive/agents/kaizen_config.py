"""
Pydantic Configuration Schema for Kaizen
========================================

Provides type-safe configuration with validation and environment variable support.
"""

from pydantic import BaseModel, Field, validator, ValidationError
from typing import Dict, Optional, Any
from pathlib import Path
import os
from datetime import timedelta


class PerformanceThresholds(BaseModel):
    """Performance threshold configuration"""
    response_time: float = Field(
        default=2.0,
        description="Maximum acceptable average response time in seconds",
        ge=0.1,
        le=60.0
    )
    error_rate: float = Field(
        default=0.05,
        description="Maximum acceptable error rate (0-1)",
        ge=0.0,
        le=1.0
    )
    consciousness_level: float = Field(
        default=0.4,
        description="Minimum healthy consciousness level (0-1)", 
        ge=0.0,
        le=1.0
    )


class KaizenConfig(BaseModel):
    """
    Configuration schema for Kaizen Improvement Engine.
    
    Supports environment variable overrides with KAIZEN_ prefix.
    Example: KAIZEN_ANALYSIS_INTERVAL=7200
    """
    
    # Core settings
    analysis_interval: int = Field(
        default=3600,
        description="Interval between analysis cycles in seconds",
        env="KAIZEN_ANALYSIS_INTERVAL",
        ge=60,  # At least 1 minute
        le=86400  # Max 24 hours
    )
    
    min_data_points: int = Field(
        default=10,
        description="Minimum data points required for analysis",
        env="KAIZEN_MIN_DATA_POINTS",
        ge=1,
        le=1000
    )
    
    # Feature flags
    enable_auto_apply: bool = Field(
        default=False,
        description="Automatically apply high-confidence insights",
        env="KAIZEN_ENABLE_AUTO_APPLY"
    )
    
    enable_clustering: bool = Field(
        default=False,
        description="Enable sklearn-based query clustering",
        env="KAIZEN_ENABLE_CLUSTERING"
    )
    
    enable_gap_fill: bool = Field(
        default=True,
        description="Enable automatic gap-fill paper search",
        env="KAIZEN_ENABLE_GAP_FILL"
    )
    
    use_celery_tasks: bool = Field(
        default=False,
        description="Offload heavy analyses to Celery workers",
        env="KAIZEN_USE_CELERY"
    )
    
    # Limits
    max_insights_stored: int = Field(
        default=10000,
        description="Maximum insights to keep in memory",
        env="KAIZEN_MAX_INSIGHTS",
        ge=100,
        le=100000
    )
    
    max_insights_per_cycle: int = Field(
        default=5,
        description="Maximum insights to generate per analysis cycle",
        env="KAIZEN_MAX_INSIGHTS_PER_CYCLE",
        ge=1,
        le=50
    )
    
    max_metric_items: int = Field(
        default=10000,
        description="Maximum items in metric deques",
        ge=1000,
        le=100000
    )
    
    # Thresholds
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence for insight generation",
        env="KAIZEN_CONFIDENCE_THRESHOLD",
        ge=0.0,
        le=1.0
    )
    
    auto_apply_threshold: float = Field(
        default=0.9,
        description="Minimum confidence for auto-applying insights",
        ge=0.7,
        le=1.0
    )
    
    # Storage
    knowledge_base_path: Optional[Path] = Field(
        default=None,
        description="Path to knowledge base JSON file",
        env="KAIZEN_KB_PATH"
    )
    
    metrics_retention_days: int = Field(
        default=7,
        description="Days to retain metrics data",
        env="KAIZEN_RETENTION_DAYS",
        ge=1,
        le=365
    )
    
    enable_kb_rotation: bool = Field(
        default=False,
        description="Enable nightly knowledge base rotation",
        env="KAIZEN_ENABLE_KB_ROTATION"
    )
    
    kb_rotation_keep_days: int = Field(
        default=30,
        description="Days to keep rotated KB files",
        ge=7,
        le=365
    )
    
    # Performance thresholds
    performance_threshold: PerformanceThresholds = Field(
        default_factory=PerformanceThresholds,
        description="Performance metric thresholds"
    )
    
    # Prometheus metrics
    enable_prometheus: bool = Field(
        default=False,
        description="Enable Prometheus metrics export",
        env="KAIZEN_ENABLE_PROMETHEUS"
    )
    
    prometheus_port: int = Field(
        default=9090,
        description="Port for Prometheus metrics endpoint",
        env="KAIZEN_PROMETHEUS_PORT",
        ge=1024,
        le=65535
    )
    
    metrics_update_interval: int = Field(
        default=30,
        description="Interval for updating Prometheus metrics in seconds",
        ge=10,
        le=300
    )
    
    # Circuit breaker settings
    gap_fill_max_retries: int = Field(
        default=3,
        description="Maximum retries for gap-fill searches",
        ge=1,
        le=10
    )
    
    gap_fill_backoff_seconds: int = Field(
        default=300,
        description="Backoff time after gap-fill failures",
        ge=60,
        le=3600
    )
    
    class Config:
        """Pydantic config"""
        env_prefix = "KAIZEN_"
        case_sensitive = False
        
    @validator('knowledge_base_path', pre=True)
    def validate_kb_path(cls, v):
        """Ensure knowledge base path is absolute"""
        if v is None:
            return None
        path = Path(v)
        if not path.is_absolute():
            # Make relative paths absolute from current working directory
            path = Path.cwd() / path
        return path
    
    @validator('analysis_interval')
    def validate_analysis_interval(cls, v):
        """Warn if analysis interval is very short"""
        if v < 300:  # Less than 5 minutes
            import logging
            logging.getLogger(__name__).warning(
                f"Analysis interval {v}s is very short - this may impact performance"
            )
        return v
    
    @classmethod
    def from_env(cls) -> 'KaizenConfig':
        """
        Create config from environment variables.
        
        Example:
            config = KaizenConfig.from_env()
        """
        # Build config dict from environment
        config_dict = {}
        
        # Check each field for env var
        for field_name, field in cls.__fields__.items():
            env_name = f"KAIZEN_{field_name.upper()}"
            if env_name in os.environ:
                value = os.environ[env_name]
                
                # Handle boolean conversion
                if field.type_ == bool:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                # Handle int conversion
                elif field.type_ == int:
                    value = int(value)
                # Handle float conversion
                elif field.type_ == float:
                    value = float(value)
                    
                config_dict[field_name] = value
        
        return cls(**config_dict)
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy dictionary format for backward compatibility.
        """
        return {
            "analysis_interval": self.analysis_interval,
            "min_data_points": self.min_data_points,
            "confidence_threshold": self.confidence_threshold,
            "max_insights_per_cycle": self.max_insights_per_cycle,
            "enable_clustering": self.enable_clustering,
            "enable_auto_apply": self.enable_auto_apply,
            "knowledge_base_path": str(self.knowledge_base_path) if self.knowledge_base_path else None,
            "metrics_retention_days": self.metrics_retention_days,
            "performance_threshold": self.performance_threshold.dict(),
            "max_insights_stored": self.max_insights_stored,
            "enable_gap_fill": self.enable_gap_fill,
            "use_celery_tasks": self.use_celery_tasks,
            "enable_prometheus": self.enable_prometheus,
            "prometheus_port": self.prometheus_port,
            "metrics_update_interval": self.metrics_update_interval,
            "gap_fill_max_retries": self.gap_fill_max_retries,
            "gap_fill_backoff_seconds": self.gap_fill_backoff_seconds,
            "enable_kb_rotation": self.enable_kb_rotation,
            "kb_rotation_keep_days": self.kb_rotation_keep_days
        }
    
    def get_timedelta(self, field: str) -> timedelta:
        """Get a time field as timedelta"""
        seconds = getattr(self, field)
        return timedelta(seconds=seconds)


def load_config(config_path: Optional[Path] = None) -> KaizenConfig:
    """
    Load Kaizen configuration from file or environment.
    
    Args:
        config_path: Optional path to JSON/YAML config file
        
    Returns:
        Validated KaizenConfig instance
    """
    if config_path and config_path.exists():
        import json
        with open(config_path) as f:
            config_data = json.load(f)
        return KaizenConfig(**config_data)
    else:
        # Load from environment
        return KaizenConfig.from_env()


# Example usage and validation
if __name__ == "__main__":
    # Example 1: Default config
    default_config = KaizenConfig()
    print("Default config:")
    print(default_config.json(indent=2))
    
    # Example 2: From environment
    os.environ["KAIZEN_ANALYSIS_INTERVAL"] = "7200"
    os.environ["KAIZEN_ENABLE_AUTO_APPLY"] = "true"
    
    env_config = KaizenConfig.from_env()
    print("\nConfig from environment:")
    print(f"Analysis interval: {env_config.analysis_interval}")
    print(f"Auto-apply enabled: {env_config.enable_auto_apply}")
    
    # Example 3: Validation
    try:
        invalid_config = KaizenConfig(
            analysis_interval=10,  # Too short
            error_rate=2.0  # Invalid percentage
        )
    except ValidationError as e:
        print("\nValidation errors:")
        print(e)
