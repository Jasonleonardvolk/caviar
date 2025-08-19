"""
Log Rotation Configuration
Implements automatic log rotation to prevent disk space issues
"""

import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime

def setup_log_rotation(
    log_file: str = "logs/session.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 14,  # Keep 2 weeks of logs
    when: str = "midnight",
    interval: int = 1
):
    """
    Set up log rotation for the application
    
    Args:
        log_file: Path to the log file
        max_bytes: Maximum size of each log file (for RotatingFileHandler)
        backup_count: Number of backup files to keep
        when: When to rotate (for TimedRotatingFileHandler) - 'midnight', 'H', 'D', 'W0-W6'
        interval: Interval for time-based rotation
    """
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Remove existing file handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        if isinstance(handler, (logging.FileHandler, logging.handlers.RotatingFileHandler, 
                               logging.handlers.TimedRotatingFileHandler)):
            root_logger.removeHandler(handler)
    
    # Create time-based rotating handler
    time_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when=when,
        interval=interval,
        backupCount=backup_count,
        encoding='utf-8'
    )
    
    # Set formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    time_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(time_handler)
    
    # Also add a size-based handler for safety
    size_handler = logging.handlers.RotatingFileHandler(
        filename=log_file.replace('.log', '_size.log'),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    size_handler.setFormatter(formatter)
    root_logger.addHandler(size_handler)
    
    # Log rotation info
    logging.info(f"Log rotation configured: {log_file}")
    logging.info(f"Time-based: rotate {when}, keep {backup_count} backups")
    logging.info(f"Size-based: rotate at {max_bytes/1024/1024:.1f}MB, keep {backup_count} backups")

def configure_dynamic_logging(initial_lines: int = 100):
    """
    Configure dynamic logging that reduces verbosity after initial lines
    
    Args:
        initial_lines: Number of lines to log at INFO level before switching to DEBUG
    """
    class DynamicLoggingFilter(logging.Filter):
        def __init__(self, initial_lines=100):
            super().__init__()
            self.line_count = 0
            self.initial_lines = initial_lines
            self.reduced_logging = False
            
        def filter(self, record):
            self.line_count += 1
            
            # After initial lines, only allow WARNING and above
            if self.line_count > self.initial_lines and not self.reduced_logging:
                self.reduced_logging = True
                logging.info(f"Reducing log verbosity after {self.initial_lines} lines")
                
            if self.reduced_logging and record.levelno < logging.WARNING:
                return False
                
            return True
    
    # Add filter to all handlers
    root_logger = logging.getLogger()
    dynamic_filter = DynamicLoggingFilter(initial_lines)
    
    for handler in root_logger.handlers:
        handler.addFilter(dynamic_filter)
    
    logging.info(f"Dynamic logging configured: INFO for first {initial_lines} lines, then WARNING+")

def cleanup_old_logs(log_dir: str = "logs", days_to_keep: int = 30):
    """
    Clean up log files older than specified days
    
    Args:
        log_dir: Directory containing log files
        days_to_keep: Number of days to keep logs
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return
        
    current_time = datetime.now()
    removed_count = 0
    
    for log_file in log_path.glob("*.log*"):
        # Get file modification time
        mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
        age_days = (current_time - mtime).days
        
        if age_days > days_to_keep:
            try:
                log_file.unlink()
                removed_count += 1
            except Exception as e:
                logging.warning(f"Failed to remove old log {log_file}: {e}")
    
    if removed_count > 0:
        logging.info(f"Cleaned up {removed_count} old log files")

# Auto-configure on import
if __name__ != "__main__":
    # Only auto-configure if imported as a module
    setup_log_rotation()
    configure_dynamic_logging()
