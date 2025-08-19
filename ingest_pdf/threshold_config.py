# threshold_config.py — Central Confidence Thresholds and Fallbacks
"""
Centralized configuration for concept ingestion thresholds and fallback logic.

This module provides tunable parameters for concept filtering and ensures
consistent threshold application across the entire ingestion pipeline.

Key Features:
- Configurable confidence thresholds (lowered from 0.8 to 0.5 per triage)
- Fallback minimum concept counts to prevent empty results
- Per-media-type threshold adjustment capability
- Dynamic threshold adaptation based on content length
"""

import os
import json
from typing import Dict, Any
from pathlib import Path

# Default confidence threshold (lowered from previous 0.8 to address Issue #2)
MIN_CONFIDENCE = 0.5

# Minimum number of concepts to always retain (prevents empty results)
FALLBACK_MIN_COUNT = 3

# Maximum concepts to process (configurable, removes hard cap of 5)
MAX_CONCEPTS_DEFAULT = 20  # Raised from 5 to address Issue #3

# Per-media-type threshold adjustments
MEDIA_TYPE_ADJUSTMENTS = {
    "pdf": 0.0,           # No adjustment for clean PDF text
    "audio": -0.1,        # Lower threshold for transcribed speech
    "video": -0.1,        # Lower threshold for video transcripts
    "conversation": -0.05, # Slightly lower for conversation exports
    "ocr": -0.15          # Much lower for OCR'd text (more noise)
}

# Adaptive threshold settings
ADAPTIVE_THRESHOLDS = {
    "enable": True,
    "min_content_length": 500,    # Minimum chars to trigger adaptation
    "length_multiplier": 0.0001,  # Threshold adjustment per char
    "max_adjustment": 0.2         # Maximum threshold adjustment
}

def load_config_from_file(config_path: str = None) -> Dict[str, Any]:
    """
    Load threshold configuration from JSON file.
    
    Args:
        config_path: Path to config file (defaults to config/ingestion_settings.json)
        
    Returns:
        Dictionary with configuration settings
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "ingestion_settings.json"
    
    if not os.path.exists(config_path):
        return {}
    
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {}

def get_threshold_for_media_type(media_type: str, base_threshold: float = None) -> float:
    """
    Get adjusted confidence threshold for specific media type.
    
    Args:
        media_type: Type of media (pdf, audio, video, etc.)
        base_threshold: Base threshold to adjust (defaults to MIN_CONFIDENCE)
        
    Returns:
        Adjusted confidence threshold
    """
    if base_threshold is None:
        base_threshold = MIN_CONFIDENCE
    
    adjustment = MEDIA_TYPE_ADJUSTMENTS.get(media_type.lower(), 0.0)
    adjusted = base_threshold + adjustment
    
    # Ensure threshold stays within valid range
    return max(0.0, min(1.0, adjusted))

def get_adaptive_threshold(content_length: int, media_type: str = "pdf") -> float:
    """
    Calculate adaptive threshold based on content length and media type.
    
    Args:
        content_length: Length of content in characters
        media_type: Type of media being processed
        
    Returns:
        Adaptive confidence threshold
    """
    base_threshold = get_threshold_for_media_type(media_type)
    
    if not ADAPTIVE_THRESHOLDS["enable"] or content_length < ADAPTIVE_THRESHOLDS["min_content_length"]:
        return base_threshold
    
    # Calculate length-based adjustment
    length_adjustment = min(
        content_length * ADAPTIVE_THRESHOLDS["length_multiplier"],
        ADAPTIVE_THRESHOLDS["max_adjustment"]
    )
    
    # For longer content, we can afford to be slightly more selective
    adjusted = base_threshold + length_adjustment
    return max(0.0, min(1.0, adjusted))

def get_fallback_count(total_candidates: int, media_type: str = "pdf") -> int:
    """
    Get adaptive fallback minimum concept count.
    
    Args:
        total_candidates: Total number of candidate concepts extracted
        media_type: Type of media being processed
        
    Returns:
        Minimum number of concepts to retain
    """
    base_count = FALLBACK_MIN_COUNT
    
    # For very short content, reduce minimum count
    if total_candidates < 5:
        return min(base_count, total_candidates)
    
    # For longer content, allow more concepts but cap it
    if total_candidates > 20:
        return min(8, total_candidates // 3)
    
    return base_count

# Load configuration from file if available
_config = load_config_from_file()

# Override defaults with config file values
if "min_confidence" in _config:
    MIN_CONFIDENCE = _config["min_confidence"]
if "fallback_min_count" in _config:
    FALLBACK_MIN_COUNT = _config["fallback_min_count"]
if "max_concepts_default" in _config:
    MAX_CONCEPTS_DEFAULT = _config["max_concepts_default"]
if "media_type_adjustments" in _config:
    MEDIA_TYPE_ADJUSTMENTS.update(_config["media_type_adjustments"])
if "adaptive_thresholds" in _config:
    ADAPTIVE_THRESHOLDS.update(_config["adaptive_thresholds"])
