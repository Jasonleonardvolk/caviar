"""
Logging utility for TORI Ingest Bus.

This module provides configured logging for the ingest bus service.
"""

import os
import sys
import logging
import logging.handlers
from typing import Optional

# Configure root logger
def setup_logger(
    logger_name: str = "ingest-bus",
    log_level: int = logging.INFO,
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        logger_name: Name of the logger
        log_level: Logging level (default: INFO)
        log_format: Format string for log messages
        log_file: Optional file path to write logs to
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Get a logger with the specified name
def get_logger(name: str = "ingest-bus") -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name of the logger
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

# Shortcut for getting component-specific loggers
def get_component_logger(component_name: str) -> logging.Logger:
    """
    Get a logger for a specific component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        logging.Logger: Logger instance for the component
    """
    return logging.getLogger(f"ingest-bus.{component_name}")
