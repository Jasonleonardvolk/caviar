"""
Configuration loader utility for TORI Ingest Bus.

This module provides functions for loading and validating configuration settings.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("ingest-bus.config")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from the specified path or the default location.
    
    Args:
        config_path: Optional path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Default configuration
    default_config = {
        "scholar_sphere": {
            "enabled": True,
            "encoder_version": "v2.5.0",
            "chunk_size": 512,
            "chunk_overlap": 128,
            "max_concepts_per_chunk": 12,
            "phase_vector_dim": 1024,
            "citation_enabled": True,
            "citation_preview": True
        },
        "extraction": {
            "default_format": "default",
            "auto_verify": False,
            "scholar_sphere_integration": True
        },
        "verification": {
            "auto_fix": False,
            "max_search_depth": 5
        },
        "integration": {
            "extraction_timeout_ms": 30000,
            "verification_timeout_ms": 15000,
            "max_recent_conversations": 5
        }
    }
    
    # Try to load configuration
    config = default_config.copy()
    
    try:
        # If config_path is provided, use it
        if config_path:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        
        # Otherwise, try to find the configuration file
        # First in the current directory
        if os.path.exists("conversation_config.json"):
            with open("conversation_config.json", "r") as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                logger.info("Loaded configuration from ./conversation_config.json")
                return config
        
        # Then in the parent directory
        parent_config = Path(__file__).parent.parent.parent / "conversation_config.json"
        if os.path.exists(parent_config):
            with open(parent_config, "r") as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                logger.info(f"Loaded configuration from {parent_config}")
                return config
                
        logger.warning("No configuration file found, using default settings")
        return config
        
    except Exception as e:
        logger.warning(f"Error loading configuration: {str(e)}")
        logger.warning("Using default configuration settings")
        return config

def get_config_value(key: str, default_value: Any = None, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get a specific configuration value.
    
    Args:
        key: Configuration key in dot notation (e.g., 'scholar_sphere.enabled')
        default_value: Default value to return if key not found
        config: Optional configuration dictionary
        
    Returns:
        Any: Configuration value or default value
    """
    if config is None:
        config = load_config()
    
    # Parse dot notation
    keys = key.split(".")
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default_value
    
    return value
