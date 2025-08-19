"""
Logging bootstrap module - initializes logging from YAML configuration.
Ensures log directories exist and configures all loggers for the TORI system.
"""
import os
import sys
import logging
import logging.config
import yaml
from pathlib import Path
from typing import Optional

def setup_logging(
    config_path: Optional[str] = None,
    default_level: int = logging.INFO,
    env_key: str = 'TORI_LOG_CONFIG'
) -> None:
    """
    Setup logging configuration from YAML file.
    
    Args:
        config_path: Path to logging config YAML file
        default_level: Default logging level if config not found
        env_key: Environment variable to check for config path
    """
    # Determine config path
    if config_path is None:
        config_path = os.getenv(env_key, None)
    
    if config_path is None:
        # Default to logging_config.yaml in same directory
        config_path = Path(__file__).parent / 'logging_config.yaml'
    
    config_path = Path(config_path)
    
    # Create log directories
    log_dirs = [
        'logs',
        'logs/inference',
        'logs/mesh',
        'logs/hybrid',
    ]
    
    for log_dir in log_dirs:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and apply config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
            # Ensure directories exist for all file handlers
            for handler in config.get('handlers', {}).values():
                if 'filename' in handler:
                    log_file = Path(handler['filename'])
                    log_file.parent.mkdir(parents=True, exist_ok=True)
            
            logging.config.dictConfig(config)
            logging.info(f"Logging configured from {config_path}")
    else:
        # Fallback to basic config
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
        logging.warning(f"Logging config not found at {config_path}, using defaults")

def get_audit_logger(name: str) -> logging.Logger:
    """
    Get a logger configured for audit logging.
    
    Args:
        name: Logger name (e.g., 'adapter_loader', 'mesh_exporter')
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

# Auto-initialize on import
if __name__ != '__main__':
    setup_logging()
