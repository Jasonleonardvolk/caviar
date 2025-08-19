"""
pipeline/utils.py

Utility functions for safe operations and data handling.
Provides bulletproof math operations and data sanitization.
UPDATED WITH SIMPLIFIED SAFE MATH FROM PATCH #3
"""

import logging
import math
from typing import Dict, Any, Optional, Union

# Setup logger
logger = logging.getLogger("pdf_ingestion.utils")

# === SIMPLIFIED SAFE MATH (Patch #3) ===
def safe_num(val, default=0.0, cast=float):
    """Return `cast(val)` or `default` on any exception / None."""
    try:
        if val is None:
            return default
        return cast(val)
    except Exception:
        return default

def safe_divide(a, b, default=0.0):
    """Bulletproof division using safe_num"""
    return default if b in (0, None) else safe_num(a)/safe_num(b)

def safe_multiply(a, b, default=0.0):
    """Bulletproof multiplication"""
    return safe_num(a)*safe_num(b)

def safe_percentage(part, whole, d=0.0):
    """Bulletproof percentage calculation"""
    return safe_divide(part, whole, d)*100

def safe_round(v, dec=3):
    """Bulletproof rounding"""
    return round(safe_num(v), dec)

# === LEGACY COMPATIBILITY FUNCTIONS ===
def safe_get(obj: Optional[Dict], key: str, default: Any = 0) -> Any:
    """
    Absolutely safe dictionary access.
    
    Args:
        obj: Dictionary to access (can be None)
        key: Key to retrieve
        default: Default value if key not found or value is None
        
    Returns:
        Value from dictionary or default
    """
    if obj is None:
        return default
    value = obj.get(key, default)
    return value if value is not None else default


def sanitize_dict(data_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sanitize any dictionary to remove all None values.
    Replaces None with appropriate defaults based on key names.
    
    Args:
        data_dict: Dictionary to sanitize (can be None)
        
    Returns:
        Sanitized dictionary with no None values
    """
    if not data_dict:
        return {}
    
    clean_dict = {}
    for key, value in data_dict.items():
        if value is None:
            # Numeric keys default to 0
            if key in ['total', 'selected', 'pruned', 'count', 'frequency']:
                clean_dict[key] = 0
            # Float keys default to 0.0
            elif key in ['score', 'final_entropy', 'avg_similarity', 'efficiency']:
                clean_dict[key] = 0.0
            # Everything else defaults to 0
            else:
                clean_dict[key] = 0
        else:
            clean_dict[key] = value
    return clean_dict


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with consistent configuration.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
