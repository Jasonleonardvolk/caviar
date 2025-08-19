"""
MCP Metacognitive Configuration Module
Provides configuration for the metacognitive server
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path


class MCPMetacognitiveConfig:
    """Configuration for MCP Metacognitive Server"""
    
    def __init__(self):
        self.host = os.getenv("SERVER_HOST", "localhost")
        self.port = int(os.getenv("SERVER_PORT", "8888"))
        self.transport_type = os.getenv("TRANSPORT_TYPE", "stdio")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Cognitive parameters
        self.cognitive_dimension = int(os.getenv("COGNITIVE_DIMENSION", "10"))
        self.manifold_metric = os.getenv("MANIFOLD_METRIC", "fisher_rao")
        self.consciousness_threshold = float(os.getenv("CONSCIOUSNESS_THRESHOLD", "0.3"))
        self.max_metacognitive_levels = int(os.getenv("MAX_METACOGNITIVE_LEVELS", "3"))
        
        # TORI integration
        self.tori_integration = os.getenv("TORI_INTEGRATION", "false").lower() == "true"
        
        # Phase 2 Components
        self.daniel_auto_start = os.getenv("DANIEL_AUTO_START", "false").lower() == "true"
        self.kaizen_auto_start = os.getenv("KAIZEN_AUTO_START", "false").lower() == "true"
        self.daniel_model_backend = os.getenv("DANIEL_MODEL_BACKEND", "mock")
        self.kaizen_analysis_interval = int(os.getenv("KAIZEN_ANALYSIS_INTERVAL", "3600"))
        self.enable_celery = os.getenv("ENABLE_CELERY", "false").lower() == "true"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "host": self.host,
            "port": self.port,
            "transport_type": self.transport_type,
            "debug": self.debug,
            "cognitive_dimension": self.cognitive_dimension,
            "manifold_metric": self.manifold_metric,
            "consciousness_threshold": self.consciousness_threshold,
            "max_metacognitive_levels": self.max_metacognitive_levels,
            "tori_integration": self.tori_integration,
            "daniel_auto_start": self.daniel_auto_start,
            "kaizen_auto_start": self.kaizen_auto_start,
            "daniel_model_backend": self.daniel_model_backend,
            "kaizen_analysis_interval": self.kaizen_analysis_interval,
            "enable_celery": self.enable_celery
        }


# Global config instance
config = MCPMetacognitiveConfig()


def get_config() -> MCPMetacognitiveConfig:
    """Get the global configuration instance"""
    return config


def update_config(**kwargs) -> None:
    """Update configuration parameters"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)


# Export for easy imports
__all__ = ['config', 'get_config', 'update_config', 'MCPMetacognitiveConfig']
