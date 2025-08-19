"""
error_handling.py

Enhanced error handling system with specific exception types,
error categorization, and recovery strategies.
"""

import logging
import traceback
import os
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import time
import json
from pathlib import Path

# Get logger
logger = logging.getLogger("tori.ingest_pdf.error_handling")

# Error severity levels
class ErrorSeverity(Enum):
    LOW = "low"          # Minor issues, processing continues
    MEDIUM = "medium"    # Some functionality impaired
    HIGH = "high"        # Major functionality impaired
    CRITICAL = "critical" # Processing cannot continue

# Error categories
class ErrorCategory(Enum):
    IO_ERROR = "io_error"
    PARSE_ERROR = "parse_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    EXTRACTION_ERROR = "extraction_error"
    STORAGE_ERROR = "storage_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"

# Recovery strategies
class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    DEGRADE = "degrade"  # Continue with reduced functionality

# Custom exception hierarchy
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 recoverable: bool = False,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.recoverable = recoverable
        self.details = details or {}
        self.timestamp = time.time()

class IOError(PipelineError):
    """File I/O related errors."""
    def __init__(self, message: str, path: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCategory.IO_ERROR, **kwargs)
        self.path = path
        if path:
            self.details['path'] = path

class PDFParseError(PipelineError):
    """PDF parsing errors."""
    def __init__(self, message: str, page: Optional[int] = None, **kwargs):
        super().__init__(message, ErrorCategory.PARSE_ERROR, **kwargs)
        self.page = page
        if page is not None:
            self.details['page'] = page

class ExtractionError(PipelineError):
    """Concept extraction errors."""
    def __init__(self, message: str, chunk_index: Optional[int] = None, **kwargs):
        super().__init__(message, ErrorCategory.EXTRACTION_ERROR, **kwargs)
        self.chunk_index = chunk_index
        if chunk_index is not None:
            self.details['chunk_index'] = chunk_index

class MemoryError(PipelineError):
    """Memory-related errors."""
    def __init__(self, message: str, size_mb: Optional[float] = None, **kwargs):
        super().__init__(message, ErrorCategory.MEMORY_ERROR, 
                        severity=ErrorSeverity.HIGH, **kwargs)
        if size_mb:
            self.details['size_mb'] = size_mb

class TimeoutError(PipelineError):
    """Processing timeout errors."""
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        super().__init__(message, ErrorCategory.TIMEOUT_ERROR, **kwargs)
        if timeout_seconds:
            self.details['timeout_seconds'] = timeout_seconds

class ValidationError(PipelineError):
    """Input validation errors."""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION_ERROR, **kwargs)
        if field:
            self.details['field'] = field

class ConfigurationError(PipelineError):
    """Configuration errors."""
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCategory.CONFIGURATION_ERROR,
                        severity=ErrorSeverity.HIGH, **kwargs)
        if config_key:
            self.details['config_key'] = config_key

# Error context manager
@dataclass
class ErrorContext:
    """Context for error handling and recovery."""
    operation: str
    file_path: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    errors: List[PipelineError] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    def add_error(self, error: PipelineError):
        """Add an error to the context."""
        self.errors.append(error)
        
    def should_retry(self, error: PipelineError) -> bool:
        """Determine if operation should be retried."""
        if not error.recoverable:
            return False
        if self.retry_count >= self.max_retries:
            return False
        if error.severity == ErrorSeverity.CRITICAL:
            return False
        return True
        
    def get_recovery_strategy(self, error: PipelineError) -> RecoveryStrategy:
        """Determine recovery strategy for an error."""
        if error.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ABORT
            
        if error.recoverable and self.retry_count < self.max_retries:
            return RecoveryStrategy.RETRY
            
        # Category-specific strategies
        strategy_map = {
            ErrorCategory.IO_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.PARSE_ERROR: RecoveryStrategy.FALLBACK,
            ErrorCategory.MEMORY_ERROR: RecoveryStrategy.DEGRADE,
            ErrorCategory.TIMEOUT_ERROR: RecoveryStrategy.SKIP,
            ErrorCategory.VALIDATION_ERROR: RecoveryStrategy.ABORT,
            ErrorCategory.EXTRACTION_ERROR: RecoveryStrategy.SKIP,
            ErrorCategory.STORAGE_ERROR: RecoveryStrategy.DEGRADE,
            ErrorCategory.CONFIGURATION_ERROR: RecoveryStrategy.ABORT,
        }
        
        return strategy_map.get(error.category, RecoveryStrategy.SKIP)

# Error handler with retry logic
class ErrorHandler:
    """Centralized error handling with retry and recovery."""
    
    def __init__(self, enable_retries: bool = True, 
                 log_errors_to_file: bool = False,
                 error_log_dir: Optional[str] = None):
        self.enable_retries = enable_retries
        self.log_to_file = log_errors_to_file
        self.error_log_dir = Path(error_log_dir or "./error_logs")
        if self.log_to_file:
            self.error_log_dir.mkdir(exist_ok=True)
            
    def handle_error(self, error: Exception, context: ErrorContext) -> RecoveryStrategy:
        """Handle an error and determine recovery strategy."""
        # Convert to PipelineError if needed
        if not isinstance(error, PipelineError):
            pipeline_error = self._convert_to_pipeline_error(error)
        else:
            pipeline_error = error
            
        # Add to context
        context.add_error(pipeline_error)
        
        # Log the error
        self._log_error(pipeline_error, context)
        
        # Save to file if enabled
        if self.log_to_file:
            self._save_error_to_file(pipeline_error, context)
            
        # Determine recovery strategy
        strategy = context.get_recovery_strategy(pipeline_error)
        
        # Log recovery decision
        logger.info(f"Recovery strategy for {pipeline_error.category.value}: {strategy.value}")
        
        return strategy
        
    def _convert_to_pipeline_error(self, error: Exception) -> PipelineError:
        """Convert standard exception to PipelineError."""
        error_type = type(error).__name__
        
        # Map common exceptions
        if isinstance(error, FileNotFoundError):
            return IOError(str(error), recoverable=True, severity=ErrorSeverity.HIGH)
        elif isinstance(error, PermissionError):
            return IOError(str(error), recoverable=False, severity=ErrorSeverity.HIGH)
        elif isinstance(error, json.JSONDecodeError):
            return PDFParseError(str(error), recoverable=False)
        elif isinstance(error, MemoryError):
            return MemoryError(str(error), recoverable=False)
        elif isinstance(error, TimeoutError):
            return TimeoutError(str(error), recoverable=True)
        elif isinstance(error, ValueError):
            return ValidationError(str(error))
        elif isinstance(error, KeyError):
            return ConfigurationError(str(error))
        else:
            # Generic error
            return PipelineError(
                f"{error_type}: {str(error)}",
                category=ErrorCategory.UNKNOWN_ERROR,
                severity=ErrorSeverity.MEDIUM,
                recoverable=False,
                details={"exception_type": error_type}
            )
            
    def _log_error(self, error: PipelineError, context: ErrorContext):
        """Log error with appropriate severity."""
        log_method = {
            ErrorSeverity.LOW: logger.warning,
            ErrorSeverity.MEDIUM: logger.error,
            ErrorSeverity.HIGH: logger.error,
            ErrorSeverity.CRITICAL: logger.critical
        }.get(error.severity, logger.error)
        
        log_message = (
            f"{error.category.value} in {context.operation}: {str(error)} "
            f"[Severity: {error.severity.value}, Recoverable: {error.recoverable}]"
        )
        
        if error.details:
            log_message += f" Details: {json.dumps(error.details)}"
            
        log_method(log_message, exc_info=(error.severity == ErrorSeverity.CRITICAL))
        
    def _save_error_to_file(self, error: PipelineError, context: ErrorContext):
        """Save error details to file for analysis."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.error_log_dir / f"error_{timestamp}_{context.operation}.json"
        
        error_data = {
            "timestamp": error.timestamp,
            "operation": context.operation,
            "file_path": context.file_path,
            "category": error.category.value,
            "severity": error.severity.value,
            "recoverable": error.recoverable,
            "message": str(error),
            "details": error.details,
            "retry_count": context.retry_count,
            "traceback": traceback.format_exc()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(error_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save error log: {e}")

# Retry decorator
def with_retry(max_retries: int = 3, delay: float = 1.0, 
               backoff: float = 2.0, recoverable_errors: Optional[List[type]] = None):
    """Decorator for automatic retry logic."""
    if recoverable_errors is None:
        recoverable_errors = [IOError, TimeoutError]
        
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None
            delay_time = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    # Check if error is recoverable
                    if not any(isinstance(e, err_type) for err_type in recoverable_errors):
                        raise
                        
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}"
                            f" Retrying in {delay_time}s..."
                        )
                        time.sleep(delay_time)
                        delay_time *= backoff
                    else:
                        logger.error(f"All retry attempts failed for {func.__name__}")
                        raise
                        
            if last_error:
                raise last_error
                
        return wrapper
    return decorator

# Enhanced error response builder
class ErrorResponseBuilder:
    """Build consistent error responses with detailed context."""
    
    @staticmethod
    def build_error_response(
        file_path: str,
        error: PipelineError,
        context: ErrorContext,
        start_time: float,
        admin_mode: bool = False
    ) -> Dict[str, Any]:
        """Build detailed error response maintaining schema consistency."""
        
        processing_time = time.time() - start_time
        
        response = {
            # Standard fields
            "filename": Path(file_path).name if file_path else "unknown",
            "concept_count": 0,
            "concept_names": [],
            "concepts": [],
            "status": "error",
            "admin_mode": admin_mode,
            "processing_time_seconds": round(processing_time, 2),
            
            # Error details
            "error": {
                "message": str(error),
                "category": error.category.value,
                "severity": error.severity.value,
                "recoverable": error.recoverable,
                "operation": context.operation,
                "retry_count": context.retry_count,
                "details": error.details
            },
            
            # Processing metadata
            "metadata": {
                "error_timestamp": error.timestamp,
                "processing_stage": context.operation,
                "partial_results": len(context.errors) > 1  # Multiple errors indicate partial processing
            }
        }
        
        # Add traceback in admin mode
        if admin_mode:
            response["error"]["traceback"] = traceback.format_exc()
            
        # Add all errors if multiple
        if len(context.errors) > 1:
            response["error"]["all_errors"] = [
                {
                    "message": str(e),
                    "category": e.category.value,
                    "severity": e.severity.value
                }
                for e in context.errors
            ]
            
        return response
        
    @staticmethod
    def build_partial_success_response(
        file_path: str,
        results: Dict[str, Any],
        errors: List[PipelineError],
        start_time: float,
        admin_mode: bool = False
    ) -> Dict[str, Any]:
        """Build response for partial success (some errors but processing continued)."""
        
        # Start with normal results
        response = results.copy()
        
        # Add error information
        response["status"] = "partial_success"
        response["warnings"] = [
            {
                "message": str(e),
                "category": e.category.value,
                "severity": e.severity.value,
                "details": e.details
            }
            for e in errors
        ]
        
        # Add warning count to metadata
        if "metadata" not in response:
            response["metadata"] = {}
        response["metadata"]["warning_count"] = len(errors)
        
        return response

# Global error handler instance
error_handler = ErrorHandler(
    enable_retries=True,
    log_errors_to_file=os.environ.get('LOG_ERRORS_TO_FILE', 'false').lower() == 'true',
    error_log_dir=os.environ.get('ERROR_LOG_DIR', './error_logs')
)
