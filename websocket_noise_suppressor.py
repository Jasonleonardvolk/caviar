#!/usr/bin/env python3
"""
WebSocket Noise Suppression Utility
Provides fine-grained control over WebSocket logging
"""

import logging
from typing import Optional, List

class WebSocketNoiseFilter(logging.Filter):
    """
    Custom filter to suppress specific WebSocket errors while keeping others
    """
    
    def __init__(self, suppress_messages: Optional[List[str]] = None):
        super().__init__()
        self.suppress_messages = suppress_messages or [
            "opening handshake failed",
            "connection closed while reading HTTP request",
            "did not receive a valid HTTP request",
            "InvalidMessage",
            "EOFError"
        ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter out records containing suppressed messages
        Returns True to keep the record, False to suppress it
        """
        message = record.getMessage()
        
        # Check if this is a message we want to suppress
        for suppress_text in self.suppress_messages:
            if suppress_text in message:
                # Downgrade to DEBUG level instead of suppressing entirely
                record.levelno = logging.DEBUG
                record.levelname = "DEBUG"
                return True  # Still log it, but at DEBUG level
        
        return True  # Keep all other messages

def suppress_websocket_handshake_noise(
    logger_name: str = "websockets.server",
    level: int = logging.ERROR,
    use_filter: bool = False,
    custom_messages: Optional[List[str]] = None
):
    """
    Suppress WebSocket handshake noise with various options
    
    Args:
        logger_name: Name of the logger to configure (default: "websockets.server")
        level: Logging level to set (default: ERROR to suppress INFO/WARNING)
        use_filter: Use custom filter for fine-grained control
        custom_messages: Custom list of messages to suppress (if use_filter=True)
    
    Examples:
        # Simple suppression (recommended)
        suppress_websocket_handshake_noise()
        
        # With custom filter
        suppress_websocket_handshake_noise(use_filter=True)
        
        # Suppress specific messages only
        suppress_websocket_handshake_noise(
            use_filter=True,
            custom_messages=["opening handshake failed"]
        )
    """
    logger = logging.getLogger(logger_name)
    
    if use_filter:
        # Use custom filter for fine-grained control
        noise_filter = WebSocketNoiseFilter(custom_messages)
        logger.addFilter(noise_filter)
        print(f"Added custom filter to {logger_name} logger")
    else:
        # Simple level-based suppression
        logger.setLevel(level)
        print(f"Set {logger_name} logger level to {logging.getLevelName(level)}")

def configure_all_websocket_loggers(level: int = logging.ERROR):
    """
    Configure all WebSocket-related loggers at once
    """
    loggers = [
        "websockets",
        "websockets.server",
        "websockets.client",
        "websockets.protocol",
        "websockets.http"
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
    
    print(f"Configured all WebSocket loggers to {logging.getLevelName(level)}")

# Example usage in bridge scripts:
if __name__ == "__main__":
    # Example 1: Simple suppression (what we're using now)
    suppress_websocket_handshake_noise()
    
    # Example 2: With custom filter
    # suppress_websocket_handshake_noise(use_filter=True)
    
    # Example 3: Suppress all WebSocket logging
    # configure_all_websocket_loggers(logging.CRITICAL)
    
    # Example 4: Custom messages only
    # suppress_websocket_handshake_noise(
    #     use_filter=True,
    #     custom_messages=["specific error to suppress"]
    # )
