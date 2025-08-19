"""
Error handling and documentation utilities for the ELFIN framework.

This module provides utilities for handling errors in the ELFIN framework,
including error code documentation and formatting.
"""

from .error_handler import ErrorHandler, VerificationError

__all__ = [
    "ErrorHandler",
    "VerificationError"
]
