# ingest_pdf/pipeline_validator.py — Runtime Validator for Extracted Concepts
"""
Pipeline validator for concept extraction quality assurance.

This module provides runtime validation of extracted concepts to ensure
schema compliance, data quality, and proper metadata preservation.
It implements the validation layer mentioned in Issue #4 of the triage document.

Key Features:
- Schema validation for concept dictionaries
- Confidence score validation and bounds checking
- Required field presence validation
- Metadata completeness verification
- Real-time logging of validation issues
"""

import logging
from typing import List, Dict, Any, Set, Optional, Tuple
from datetime import datetime

# Configure logging
logger = logging.getLogger("concept_validator")

# Required fields for concept validation (addresses Issue #4 - metadata preservation)
REQUIRED_FIELDS = {"name", "confidence", "method", "source"}

# Optional fields that enhance concept quality
RECOMMENDED_FIELDS = {"context", "embedding", "eigenfunction_id"}

# Valid confidence bounds
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0

# Valid extraction methods
VALID_METHODS = {
    "embedding_cluster",
    "tfidf_extraction",
    "keyword_extraction", 
    "llm_summarization",
    "spectral_analysis",
    "phrase_extraction",
    "topic_modeling"
}

class ConceptValidationError(Exception):
    """Exception raised when concept validation fails critically."""
    pass

class ConceptValidator:
    """
    Validator for concept extraction pipeline.
    
    Provides comprehensive validation of extracted concepts including
    schema compliance, data quality checks, and metadata verification.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize concept validator.
        
        Args:
            strict_mode: If True, raises exceptions on validation failures
        """
        self.strict_mode = strict_mode
        self.validation_stats = {
            "total_validated": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }
    
    def validate_concept(self, concept: Dict[str, Any], segment_id: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate a single concept dictionary.
        
        Args:
            concept: Concept dictionary to validate
            segment_id: Optional segment identifier for context
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if concept is a dictionary
        if not isinstance(concept, dict):
            issues.append(f"Concept is not a dictionary: {type(concept)}")
            return False, issues
        
        # Validate required fields
        missing_fields = REQUIRED_FIELDS - concept.keys()
        if missing_fields:
            issues.append(f"Missing required fields: {missing_fields}")
        
        # Validate concept name
        name = concept.get("name")
        if not name or not isinstance(name, str):
            issues.append("Invalid or missing concept name")
        elif len(name.strip()) == 0:
            issues.append("Concept name is empty")
        
        # Validate confidence score
        confidence = concept.get("confidence")
        if confidence is None:
            issues.append("Missing confidence score")
        elif not isinstance(confidence, (int, float)):
            issues.append(f"Confidence must be numeric, got {type(confidence)}")
        elif not (CONFIDENCE_MIN <= confidence <= CONFIDENCE_MAX):
            issues.append(f"Confidence {confidence} outside valid range [{CONFIDENCE_MIN}, {CONFIDENCE_MAX}]")
        
        # Validate extraction method
        method = concept.get("method")
        if not method:
            issues.append("Missing extraction method")
        elif not isinstance(method, str):
            issues.append(f"Method must be string, got {type(method)}")
        elif method not in VALID_METHODS:
            issues.append(f"Unknown extraction method: {method}")
        
        # Validate source information
        source = concept.get("source")
        if not source:
            issues.append("Missing source information")
        elif not isinstance(source, dict):
            issues.append(f"Source must be dictionary, got {type(source)}")
        else:
            # Check for common source fields
            if not any(key in source for key in ["page", "segment", "timestamp", "line"]):
                issues.append("Source lacks location information (page, segment, timestamp, or line)")
        
        # Validate optional but recommended fields
        warnings = []
        for field in RECOMMENDED_FIELDS:
            if field not in concept:
                warnings.append(f"Missing recommended field: {field}")
        
        # Validate context if present
        context = concept.get("context")
        if context is not None:
            if not isinstance(context, str):
                issues.append(f"Context must be string, got {type(context)}")
            elif len(context.strip()) == 0:
                warnings.append("Context field is empty")
        
        # Validate embedding if present
        embedding = concept.get("embedding")
        if embedding is not None:
            if not isinstance(embedding, (list, tuple)):
                issues.append(f"Embedding must be list/tuple, got {type(embedding)}")
            elif len(embedding) == 0:
                issues.append("Embedding is empty")
            elif not all(isinstance(x, (int, float)) for x in embedding):
                issues.append("Embedding contains non-numeric values")
        
        # Log warnings
        if warnings:
            self.validation_stats["warnings"] += len(warnings)
            for warning in warnings:
                logger.warning(f"[Validator] {segment_id or 'Unknown'}: {warning} in concept '{name}'")
        
        # Determine overall validity
        is_valid = len(issues) == 0
        
        # Log validation result
        if issues:
            for issue in issues:
                logger.warning(f"[Validator] {segment_id or 'Unknown'}: {issue} in concept '{name}'")
        
        return is_valid, issues
    
    def validate_concepts(self, concepts: List[Dict[str, Any]], segment_id: Optional[str] = None) -> int:
        """
        Validate a list of concepts and return count of valid concepts.
        
        Args:
            concepts: List of concept dictionaries
            segment_id: Optional segment identifier for context
            
        Returns:
            Number of valid concepts
        """
        if not concepts:
            logger.warning(f"[Validator] {segment_id or 'Unknown'}: No concepts to validate")
            return 0
        
        valid_count = 0
        total_issues = []
        
        for i, concept in enumerate(concepts):
            self.validation_stats["total_validated"] += 1
            
            is_valid, issues = self.validate_concept(concept, segment_id)
            
            if is_valid:
                valid_count += 1
                self.validation_stats["passed"] += 1
            else:
                self.validation_stats["failed"] += 1
                total_issues.extend(issues)
                
                # In strict mode, raise exception on validation failure
                if self.strict_mode:
                    raise ConceptValidationError(
                        f"Concept validation failed for concept {i} in {segment_id}: {issues}"
                    )
        
        # Log summary
        logger.info(f"[Validator] {segment_id or 'Unknown'}: {valid_count}/{len(concepts)} concepts passed validation")
        
        if total_issues and not self.strict_mode:
            logger.warning(f"[Validator] {segment_id or 'Unknown'}: {len(total_issues)} total validation issues found")
        
        return valid_count
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        stats = dict(self.validation_stats)
        if stats["total_validated"] > 0:
            stats["pass_rate"] = stats["passed"] / stats["total_validated"]
            stats["fail_rate"] = stats["failed"] / stats["total_validated"]
        else:
            stats["pass_rate"] = 0.0
            stats["fail_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_stats = {
            "total_validated": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0
        }

# Global validator instance
default_validator = ConceptValidator(strict_mode=False)

def validate_concept(concept: Dict[str, Any], segment_id: Optional[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate a single concept (convenience function).
    
    Args:
        concept: Concept dictionary to validate
        segment_id: Optional segment identifier
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    return default_validator.validate_concept(concept, segment_id)

def validate_concepts(concepts: List[Dict[str, Any]], segment_id: Optional[str] = None) -> int:
    """
    Validate a list of concepts (convenience function).
    
    Args:
        concepts: List of concept dictionaries
        segment_id: Optional segment identifier
        
    Returns:
        Number of valid concepts
    """
    return default_validator.validate_concepts(concepts, segment_id)

def get_validation_stats() -> Dict[str, Any]:
    """Get validation statistics."""
    return default_validator.get_validation_stats()

def configure_validation(strict_mode: bool = False, reset_stats: bool = False):
    """
    Configure global validation settings.
    
    Args:
        strict_mode: Whether to raise exceptions on validation failures
        reset_stats: Whether to reset validation statistics
    """
    global default_validator
    default_validator.strict_mode = strict_mode
    
    if reset_stats:
        default_validator.reset_stats()
    
    logger.info(f"[Validator] Configuration updated: strict_mode={strict_mode}")
