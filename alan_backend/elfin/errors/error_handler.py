"""
Error handling utilities for the ELFIN framework.

This module provides utilities for handling errors in the ELFIN framework,
including standardized error formatting and documentation lookup.
"""

import os
import pathlib
import webbrowser
from typing import Dict, Any, Optional, List, Tuple, Union

import logging
logger = logging.getLogger(__name__)


class VerificationError(Exception):
    """
    Standardized error format for verification failures.
    
    This class creates a structured error object with a code, message,
    and reference to the interaction that caused the error.
    
    Attributes:
        code: Error code (e.g., "E-LYAP-001")
        title: Human-readable error title
        detail: Detailed error message
        system_id: ID of the system being verified
        interaction_ref: Reference to the interaction that caused the error
        extra_fields: Additional fields to include in the error
        doc_url: URL to documentation about the error
    """
    
    ERROR_CODES = {
        "LYAP_001": "Function not positive definite",
        "LYAP_002": "Function not decreasing",
        "LYAP_003": "Convergence rate not satisfied",
        "VERIF_001": "Verification failed due to solver error",
        "VERIF_002": "Verification timeout",
        "PARAM_001": "Invalid parameter values",
    }
    
    def __init__(
        self,
        code: str,
        detail: str,
        system_id: str,
        interaction_ref: Optional[str] = None,
        **extra_fields
    ):
        """
        Initialize a verification error.
        
        Args:
            code: Error code (e.g., "LYAP_001")
            detail: Detailed error message
            system_id: ID of the system being verified
            interaction_ref: Reference to the interaction that caused the error
            **extra_fields: Additional fields to include in the error
        """
        self.code = code
        self.title = self.ERROR_CODES.get(code, "Unknown error")
        self.detail = detail
        self.system_id = system_id
        self.interaction_ref = interaction_ref
        self.extra_fields = extra_fields
        self.doc_url = f"https://elfin.dev/errors/E-{code}"
        
        # Construct error message
        message = f"E-{code}: {detail}"
        if system_id:
            message += f" (system: {system_id})"
        message += f" - See {self.doc_url} for more information."
        
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to a dictionary.
        
        Returns:
            Dictionary representation of the error
        """
        error_dict = {
            "code": f"E-{self.code}",
            "title": self.title,
            "detail": self.detail,
            "system_id": self.system_id,
            "doc": self.doc_url,
        }
        
        if self.interaction_ref:
            error_dict["interaction_ref"] = self.interaction_ref
            
        # Add any extra fields
        error_dict.update(self.extra_fields)
        
        return error_dict


class ErrorHandler:
    """
    Utilities for handling errors in the ELFIN framework.
    
    This class provides methods for looking up error documentation,
    formatting errors, and opening documentation in a web browser.
    """
    
    def __init__(self, docs_dir: Optional[pathlib.Path] = None):
        """
        Initialize error handler.
        
        Args:
            docs_dir: Path to documentation directory
        """
        # Default to docs/errors relative to repo root
        if docs_dir is None:
            # Try to find docs directory
            module_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
            self.docs_dir = module_dir / "docs" / "errors"
        else:
            self.docs_dir = docs_dir
    
    def get_error_doc_path(self, error_code: str) -> pathlib.Path:
        """
        Get path to error documentation.
        
        Args:
            error_code: Error code (e.g., "E-LYAP-001" or "LYAP_001")
            
        Returns:
            Path to error documentation
        """
        # Normalize error code
        if error_code.startswith("E-"):
            error_code = error_code[2:]
            
        return self.docs_dir / f"E-{error_code}.md"
    
    def error_doc_exists(self, error_code: str) -> bool:
        """
        Check if error documentation exists.
        
        Args:
            error_code: Error code (e.g., "E-LYAP-001" or "LYAP_001")
            
        Returns:
            True if documentation exists, False otherwise
        """
        doc_path = self.get_error_doc_path(error_code)
        return doc_path.exists()
    
    def get_error_doc(self, error_code: str) -> Optional[str]:
        """
        Get error documentation.
        
        Args:
            error_code: Error code (e.g., "E-LYAP-001" or "LYAP_001")
            
        Returns:
            Error documentation or None if not found
        """
        doc_path = self.get_error_doc_path(error_code)
        if not doc_path.exists():
            return None
            
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read error documentation: {e}")
            return None
    
    def open_error_doc(self, error_code: str) -> bool:
        """
        Open error documentation in the default web browser.
        
        Args:
            error_code: Error code (e.g., "E-LYAP-001" or "LYAP_001")
            
        Returns:
            True if documentation was opened, False otherwise
        """
        doc_path = self.get_error_doc_path(error_code)
        if not doc_path.exists():
            logger.warning(f"Documentation not found for error code: {error_code}")
            return False
            
        try:
            # Try to open in browser
            url = f"file://{doc_path.resolve()}"
            webbrowser.open(url)
            return True
        except Exception as e:
            logger.warning(f"Failed to open documentation in browser: {e}")
            return False


# For convenient imports
def create_verification_error(
    code: str,
    detail: str,
    system_id: str,
    interaction_ref: Optional[str] = None,
    **extra_fields
) -> VerificationError:
    """
    Create a verification error.
    
    Args:
        code: Error code (e.g., "LYAP_001")
        detail: Detailed error message
        system_id: ID of the system being verified
        interaction_ref: Reference to the interaction that caused the error
        **extra_fields: Additional fields to include in the error
        
    Returns:
        VerificationError instance
    """
    return VerificationError(code, detail, system_id, interaction_ref, **extra_fields)
