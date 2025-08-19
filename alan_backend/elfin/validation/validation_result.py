"""
Validation Result Objects

This module defines the data structures for representing validation results.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np


class ValidationStatus(Enum):
    """Enumeration of possible validation statuses."""
    
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"
    INCONCLUSIVE = "inconclusive"


class ValidationResult:
    """
    Class representing the result of a validation operation.
    
    Attributes:
        status (ValidationStatus): Overall status of the validation.
        message (str): Human-readable description of the validation result.
        details (Dict[str, Any]): Additional details about the validation.
        violations (List[Dict[str, Any]]): List of specific validation violations.
        counter_examples (List[Dict[str, Any]]): Specific counter-examples found.
    """
    
    def __init__(
        self,
        status: ValidationStatus,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        violations: Optional[List[Dict[str, Any]]] = None,
        counter_examples: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize a validation result.
        
        Args:
            status: Overall status of the validation.
            message: Human-readable description of the validation result.
            details: Additional details about the validation.
            violations: List of specific validation violations.
            counter_examples: Specific counter-examples found.
        """
        self.status = status
        self.message = message
        self.details = details or {}
        self.violations = violations or []
        self.counter_examples = counter_examples or []
    
    @property
    def passed(self) -> bool:
        """Check if the validation passed."""
        return self.status == ValidationStatus.PASSED
    
    @property
    def failed(self) -> bool:
        """Check if the validation failed."""
        return self.status == ValidationStatus.FAILED
    
    @property
    def has_warnings(self) -> bool:
        """Check if the validation has warnings."""
        return self.status == ValidationStatus.WARNING
    
    @property
    def has_errors(self) -> bool:
        """Check if the validation has errors."""
        return self.status == ValidationStatus.ERROR
    
    @property
    def is_inconclusive(self) -> bool:
        """Check if the validation is inconclusive."""
        return self.status == ValidationStatus.INCONCLUSIVE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation result to a dictionary."""
        return {
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "violations": self.violations,
            "counter_examples": self.counter_examples
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create a validation result from a dictionary."""
        return cls(
            status=ValidationStatus(data["status"]),
            message=data["message"],
            details=data.get("details", {}),
            violations=data.get("violations", []),
            counter_examples=data.get("counter_examples", [])
        )
    
    def add_violation(self, violation: Dict[str, Any]) -> None:
        """Add a violation to the validation result."""
        self.violations.append(violation)
    
    def add_counter_example(self, counter_example: Dict[str, Any]) -> None:
        """Add a counter-example to the validation result."""
        self.counter_examples.append(counter_example)
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """
        Merge another validation result into this one.
        
        The resulting status is the worse of the two statuses.
        
        Args:
            other: Another validation result to merge.
            
        Returns:
            A new merged validation result.
        """
        # Determine the worse status
        status_order = {
            ValidationStatus.PASSED: 0,
            ValidationStatus.WARNING: 1,
            ValidationStatus.INCONCLUSIVE: 2,
            ValidationStatus.ERROR: 3,
            ValidationStatus.FAILED: 4
        }
        
        if status_order[self.status] >= status_order[other.status]:
            worse_status = self.status
        else:
            worse_status = other.status
        
        # Merge messages
        message = f"{self.message}; {other.message}"
        
        # Merge details
        merged_details = {**self.details, **other.details}
        
        # Merge violations and counter_examples
        merged_violations = self.violations + other.violations
        merged_counter_examples = self.counter_examples + other.counter_examples
        
        return ValidationResult(
            status=worse_status,
            message=message,
            details=merged_details,
            violations=merged_violations,
            counter_examples=merged_counter_examples
        )
    
    def __str__(self) -> str:
        """String representation of the validation result."""
        result = f"Validation {self.status.value}: {self.message}"
        
        if self.violations:
            result += f"\nViolations ({len(self.violations)}):"
            for i, v in enumerate(self.violations, 1):
                result += f"\n  {i}. {v.get('message', 'Unknown violation')}"
        
        if self.counter_examples:
            result += f"\nCounter-examples ({len(self.counter_examples)}):"
            for i, ce in enumerate(self.counter_examples, 1):
                result += f"\n  {i}. {ce.get('description', 'Unknown counter-example')}"
        
        return result
    
    def __repr__(self) -> str:
        """Detailed string representation of the validation result."""
        return (f"ValidationResult(status={self.status}, "
                f"message='{self.message}', "
                f"violations={len(self.violations)}, "
                f"counter_examples={len(self.counter_examples)})")


# Convenience functions for creating validation results

def passed(message: str, details: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """Create a validation result with PASSED status."""
    return ValidationResult(ValidationStatus.PASSED, message, details)


def failed(message: str, details: Optional[Dict[str, Any]] = None,
           violations: Optional[List[Dict[str, Any]]] = None,
           counter_examples: Optional[List[Dict[str, Any]]] = None) -> ValidationResult:
    """Create a validation result with FAILED status."""
    return ValidationResult(ValidationStatus.FAILED, message, details, violations, counter_examples)


def warning(message: str, details: Optional[Dict[str, Any]] = None,
            violations: Optional[List[Dict[str, Any]]] = None) -> ValidationResult:
    """Create a validation result with WARNING status."""
    return ValidationResult(ValidationStatus.WARNING, message, details, violations)


def error(message: str, details: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """Create a validation result with ERROR status."""
    return ValidationResult(ValidationStatus.ERROR, message, details)


def inconclusive(message: str, details: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """Create a validation result with INCONCLUSIVE status."""
    return ValidationResult(ValidationStatus.INCONCLUSIVE, message, details)
