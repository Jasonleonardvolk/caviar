"""
logging_config.py

Centralized logging configuration for the PDF ingestion system.
Should be configured once at application startup.
"""

import logging
import sys
import os
from typing import Optional, Dict, Any
from pathlib import Path
import json


class EmojiFormatter(logging.Formatter):
    """Custom formatter that adds emoji prefixes based on log level."""
    
    EMOJI_MAP = {
        logging.DEBUG: "🔍",
        logging.INFO: "ℹ️",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🚨"
    }
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, 
                 enable_emoji: bool = True):
        super().__init__(fmt, datefmt)
        self.enable_emoji = enable_emoji
        
    def format(self, record: logging.LogRecord) -> str:
        # Add emoji prefix if enabled
        if self.enable_emoji and record.levelno in self.EMOJI_MAP:
            record.emoji = self.EMOJI_MAP[record.levelno] + " "
        else:
            record.emoji = ""
        
        return super().format(record)


class ToriLoggerConfig:
    """Configuration manager for Tori PDF ingestion logging."""
    
    # Default configuration
    DEFAULT_FORMAT = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(emoji)s%(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    DEFAULT_LEVEL = "INFO"
    
    # Logger name hierarchy
    ROOT_LOGGER_NAME = "tori"
    INGEST_LOGGER_NAME = "tori.ingest_pdf"
    
    @classmethod
    def configure(cls, 
                  root_level: Optional[str] = None,
                  module_levels: Optional[Dict[str, str]] = None,
                  enable_emoji: Optional[bool] = None,
                  log_file: Optional[str] = None,
                  format_string: Optional[str] = None,
                  date_format: Optional[str] = None) -> None:
        """
        Configure logging for the entire application.
        
        Args:
            root_level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            module_levels: Dict of module-specific log levels {"tori.ingest_pdf.pipeline": "DEBUG"}
            enable_emoji: Whether to enable emoji in logs
            log_file: Optional file path for file logging
            format_string: Custom format string
            date_format: Custom date format
        """
        # Get configuration from environment or parameters
        root_level = root_level or os.environ.get('LOG_LEVEL', cls.DEFAULT_LEVEL).upper()
        enable_emoji = enable_emoji if enable_emoji is not None else (
            os.environ.get('ENABLE_EMOJI_LOGS', 'false').lower() == 'true'
        )
        format_string = format_string or os.environ.get('LOG_FORMAT', cls.DEFAULT_FORMAT)
        date_format = date_format or os.environ.get('LOG_DATE_FORMAT', cls.DEFAULT_DATE_FORMAT)
        
        # Load module-specific levels from environment or file
        if module_levels is None:
            module_levels = cls._load_module_levels()
        
        # Create formatter
        formatter = EmojiFormatter(
            fmt=format_string,
            datefmt=date_format,
            enable_emoji=enable_emoji
        )
        
        # Configure root tori logger
        root_logger = logging.getLogger(cls.ROOT_LOGGER_NAME)
        root_logger.setLevel(getattr(logging, root_level, logging.INFO))
        root_logger.handlers.clear()  # Remove any existing handlers
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file or os.environ.get('LOG_FILE'):
            file_path = log_file or os.environ.get('LOG_FILE')
            try:
                file_handler = logging.FileHandler(file_path, encoding='utf-8')
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                root_logger.warning(f"Failed to create file logger: {e}")
        
        # Configure module-specific levels
        for module_name, level in module_levels.items():
            module_logger = logging.getLogger(module_name)
            try:
                module_logger.setLevel(getattr(logging, level.upper()))
            except AttributeError:
                root_logger.warning(f"Invalid log level '{level}' for module '{module_name}'")
        
        # Prevent propagation to avoid duplicate logs from Python's root logger
        root_logger.propagate = False
        
        # Log configuration summary
        root_logger.info(f"Logging configured: level={root_level}, emoji={enable_emoji}")
        if module_levels:
            root_logger.debug(f"Module-specific levels: {module_levels}")
    
    @classmethod
    def _load_module_levels(cls) -> Dict[str, str]:
        """Load module-specific log levels from environment or config file."""
        module_levels = {}
        
        # Check environment variable
        env_levels = os.environ.get('LOG_MODULE_LEVELS')
        if env_levels:
            # Format: "module1:LEVEL1,module2:LEVEL2"
            for item in env_levels.split(','):
                if ':' in item:
                    module, level = item.split(':', 1)
                    module_levels[module.strip()] = level.strip()
        
        # Check config file
        config_file = os.environ.get('LOG_CONFIG_FILE')
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    module_levels.update(config.get('module_levels', {}))
            except Exception:
                pass  # Ignore config file errors
        
        return module_levels
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance with proper hierarchy.
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Configured logger instance
        """
        # Ensure the logger is under our hierarchy
        if not name.startswith(cls.ROOT_LOGGER_NAME):
            # For modules using this, prepend our root
            if name.startswith('__main__') or name.startswith('ingest_pdf'):
                name = f"{cls.INGEST_LOGGER_NAME}.{name.split('.')[-1]}"
        
        return logging.getLogger(name)
    
    @classmethod
    def configure_from_dict(cls, config: Dict[str, Any]) -> None:
        """Configure logging from a dictionary."""
        cls.configure(
            root_level=config.get('root_level'),
            module_levels=config.get('module_levels'),
            enable_emoji=config.get('enable_emoji'),
            log_file=config.get('log_file'),
            format_string=config.get('format'),
            date_format=config.get('date_format')
        )


# Convenience function for backward compatibility
def setup_logging(**kwargs):
    """Setup logging with the provided configuration."""
    ToriLoggerConfig.configure(**kwargs)


# Example usage function
def get_logger(name: str) -> logging.Logger:
    """
    Get a properly configured logger.
    
    Usage:
        from .logging_config import get_logger
        logger = get_logger(__name__)
    """
    return ToriLoggerConfig.get_logger(name)


# Configuration presets
PRESETS = {
    "development": {
        "root_level": "DEBUG",
        "enable_emoji": True,
        "module_levels": {
            "tori.ingest_pdf.pipeline": "DEBUG",
            "tori.ingest_pdf.io": "INFO",
            "tori.ingest_pdf.storage": "INFO"
        }
    },
    "production": {
        "root_level": "INFO",
        "enable_emoji": False,
        "module_levels": {
            "tori.ingest_pdf.pipeline": "INFO",
            "tori.ingest_pdf.io": "WARNING",
            "tori.ingest_pdf.storage": "INFO"
        }
    },
    "testing": {
        "root_level": "WARNING",
        "enable_emoji": False,
        "module_levels": {}
    }
}


def configure_preset(preset: str, **overrides):
    """Configure logging using a preset with optional overrides."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    
    config = PRESETS[preset].copy()
    config.update(overrides)
    ToriLoggerConfig.configure_from_dict(config)


if __name__ == "__main__":
    # Example configuration
    configure_preset("development")
    
    # Test logging
    logger = get_logger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
