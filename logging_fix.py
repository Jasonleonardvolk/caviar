"""
UTF-8 Safe Logging Configuration for Windows Consciousness System
================================================================

Fixes UnicodeEncodeError issues when logging emoji to Windows PowerShell (cp1252)
"""

import logging
import sys
import io
from typing import Optional

def setup_utf8_logging(logger_name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup UTF-8 safe logging for consciousness system components.
    
    This eliminates UnicodeEncodeError when logging emoji to Windows PowerShell.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create UTF-8 safe console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Force UTF-8 encoding on the stream
    try:
        # For Python 3.7+
        console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # Fallback for older Python versions
        try:
            console_handler.stream = io.TextIOWrapper(
                sys.stdout.buffer, 
                encoding='utf-8', 
                errors='replace'
            )
        except AttributeError:
            # Ultimate fallback - just use regular stdout
            console_handler.stream = sys.stdout
    except Exception:
        # Any other issues, just use stdout
        console_handler.stream = sys.stdout
    
    # Windows-safe formatter (no problematic characters)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")
    
    return logger

def setup_windows_safe_stdout():
    """
    Configure stdout for Windows UTF-8 compatibility.
    Call this at the start of your main module.
    """
    try:
        # Try to reconfigure stdout to UTF-8
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # For older Python versions, try environment variable
        import os
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    except Exception:
        # If all else fails, just continue
        pass

def safe_log_message(message: str) -> str:
    """
    Convert log message to Windows-safe format by replacing problematic Unicode.
    """
    # Replace common emoji with text equivalents for Windows compatibility
    emoji_replacements = {
        '🧠': '[BRAIN]',
        '🧬': '[DNA]', 
        '🚀': '[ROCKET]',
        '✅': '[CHECK]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '🔍': '[SEARCH]',
        '📊': '[CHART]',
        '📚': '[BOOKS]',
        '🎆': '[FIREWORKS]',
        '🌟': '[STAR]',
        '⚡': '[LIGHTNING]',
        '🔧': '[WRENCH]',
        '🎯': '[TARGET]',
        '💾': '[DISK]',
        '🔄': '[REFRESH]',
        '🛑': '[STOP]',
        '🎭': '[THEATER]',
        '🏥': '[HOSPITAL]',
        '👽': '[ALIEN]',
        '👻': '[GHOST]',
        '🗣️': '[SPEAKING]',
        '🕸️': '[WEB]',
        '🌊': '[WAVE]',
        '🦕': '[DINOSAUR]',
        '📈': '[TRENDING_UP]',
        '📄': '[DOCUMENT]',
        '🧪': '[TEST_TUBE]',
        '📝': '[MEMO]',
        '📁': '[FOLDER]',
        '📜': '[SCROLL]',
        '✨': '[SPARKLES]',
        '🎉': '[PARTY]'
    }
    
    safe_message = message
    for emoji, replacement in emoji_replacements.items():
        safe_message = safe_message.replace(emoji, replacement)
    
    return safe_message

class WindowsSafeLogger:
    """
    Windows-safe logger wrapper that automatically converts problematic Unicode.
    """
    
    def __init__(self, logger_name: str, log_file: Optional[str] = None):
        self.logger = setup_utf8_logging(logger_name, log_file)
    
    def info(self, message: str):
        self.logger.info(safe_log_message(message))
    
    def debug(self, message: str):
        self.logger.debug(safe_log_message(message))
    
    def warning(self, message: str):
        self.logger.warning(safe_log_message(message))
    
    def error(self, message: str):
        self.logger.error(safe_log_message(message))
    
    def critical(self, message: str):
        self.logger.critical(safe_log_message(message))

if __name__ == "__main__":
    # Test the logging system
    setup_windows_safe_stdout()
    
    logger = WindowsSafeLogger("test.logger")
    
    # Test with emoji (should work on Windows now)
    logger.info("🧠 Testing consciousness logging system")
    logger.info("🚀 System initialization complete")
    logger.error("❌ This is a test error with emoji")
    logger.warning("⚠️ This is a test warning")
    
    print("Logging test complete - no UnicodeEncodeError should occur!")
