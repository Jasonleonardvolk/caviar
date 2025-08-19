#!/usr/bin/env python3
"""
Logging configuration with rotation for the soliton memory system
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict, Optional
import yaml

def setup_logging(config_path: Optional[str] = None):
    """
    Set up logging with rotation based on configuration
    """
    # Load config
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logging_config = config.get('logging', {})
    else:
        logging_config = get_default_logging_config()
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, logging_config.get('level', 'INFO')))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Main rotating file handler
    main_log_file = logging_config.get('file', 'logs/soliton_memory.log')
    rotation_config = logging_config.get('rotation', {})
    
    if rotation_config.get('enabled', True):
        max_bytes = rotation_config.get('max_size_mb', 100) * 1024 * 1024
        backup_count = rotation_config.get('backup_count', 10)
        
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # Optionally compress rotated logs
        if rotation_config.get('compression', True):
            file_handler.rotator = CompressingRotator()
    else:
        # Simple file handler without rotation
        file_handler = logging.FileHandler(main_log_file, encoding='utf-8')
    
    file_handler.setLevel(getattr(logging, logging_config.get('level', 'INFO')))
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Set up component-specific loggers
    component_logs = logging_config.get('component_logs', {})
    
    for component, log_file in component_logs.items():
        setup_component_logger(component, log_file, rotation_config)
    
    # Set specific log levels for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    logging.info("Logging system initialized")
    
    return root_logger

def setup_component_logger(component_name: str, log_file: str, rotation_config: Dict):
    """Set up a logger for a specific component"""
    logger = logging.getLogger(f"soliton.{component_name}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Don't propagate to root logger
    
    # Create component log handler
    if rotation_config.get('enabled', True):
        max_bytes = rotation_config.get('max_size_mb', 100) * 1024 * 1024
        backup_count = rotation_config.get('backup_count', 10)
        
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
    else:
        handler = logging.FileHandler(log_file, encoding='utf-8')
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class CompressingRotator:
    """Custom rotator that compresses log files after rotation"""
    
    def __call__(self, source: str, dest: str):
        """Rotate and compress log file"""
        import gzip
        import shutil
        
        # Rotate the file
        with open(source, 'rb') as f_in:
            with gzip.open(f"{dest}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove the original file
        os.remove(source)

def get_default_logging_config() -> Dict:
    """Get default logging configuration"""
    return {
        'level': 'INFO',
        'file': 'logs/soliton_memory.log',
        'rotation': {
            'enabled': True,
            'max_size_mb': 100,
            'backup_count': 10,
            'compression': True
        },
        'component_logs': {
            'topology_changes': 'logs/topology.log',
            'memory_operations': 'logs/memory_ops.log',
            'crystallization': 'logs/crystallization.log',
            'comfort_metrics': 'logs/comfort.log'
        }
    }

# Specialized loggers for different components
def get_topology_logger():
    """Get logger for topology changes"""
    return logging.getLogger("soliton.topology_changes")

def get_memory_logger():
    """Get logger for memory operations"""
    return logging.getLogger("soliton.memory_operations")

def get_crystallization_logger():
    """Get logger for crystallization process"""
    return logging.getLogger("soliton.crystallization")

def get_comfort_logger():
    """Get logger for comfort metrics"""
    return logging.getLogger("soliton.comfort_metrics")

# Convenience logging functions
def log_topology_change(from_topology: str, to_topology: str, reason: str = ""):
    """Log a topology change event"""
    logger = get_topology_logger()
    logger.info(f"Topology change: {from_topology} -> {to_topology} | Reason: {reason}")

def log_memory_operation(operation: str, memory_id: str, details: Dict = None):
    """Log a memory operation"""
    logger = get_memory_logger()
    details_str = f" | Details: {details}" if details else ""
    logger.info(f"Memory {operation}: {memory_id}{details_str}")

def log_crystallization_event(event: str, metrics: Dict):
    """Log a crystallization event"""
    logger = get_crystallization_logger()
    logger.info(f"Crystallization {event}: {metrics}")

def log_comfort_update(memory_id: str, comfort_metrics: Dict):
    """Log comfort metrics update"""
    logger = get_comfort_logger()
    logger.debug(f"Comfort update for {memory_id}: stress={comfort_metrics.get('stress', 0):.2f}, "
                f"energy={comfort_metrics.get('energy', 0):.2f}, "
                f"flux={comfort_metrics.get('flux', 0):.2f}")

# Performance logging
class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger("soliton.performance")
        self.timings = {}
    
    def start_timing(self, operation: str):
        """Start timing an operation"""
        import time
        self.timings[operation] = time.time()
    
    def end_timing(self, operation: str, metadata: Dict = None):
        """End timing and log the duration"""
        import time
        if operation in self.timings:
            duration = time.time() - self.timings[operation]
            del self.timings[operation]
            
            log_msg = f"Operation '{operation}' completed in {duration:.3f}s"
            if metadata:
                log_msg += f" | {metadata}"
            
            self.logger.info(log_msg)

# Global performance logger instance
perf_logger = PerformanceLogger()

# Example usage in code:
if __name__ == "__main__":
    # Set up logging
    setup_logging("conf/soliton_memory_config.yaml")
    
    # Test different loggers
    log_topology_change("kagome", "hexagonal", "High system load")
    log_memory_operation("store", "mem_12345", {"concepts": ["physics", "quantum"]})
    log_crystallization_event("started", {"total_memories": 1000})
    
    # Test performance logging
    perf_logger.start_timing("memory_fusion")
    import time
    time.sleep(0.1)  # Simulate work
    perf_logger.end_timing("memory_fusion", {"fused": 10})
