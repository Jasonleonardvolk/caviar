"""
Dynamic configuration for the TORI pipeline.

Uses Pydantic BaseSettings so every value can be overridden via
environment variables or a .env file (loadable with python-dotenv).
"""

from pathlib import Path
from typing import Optional, Dict, Set, Tuple, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Additional fields that might come from environment
    skip_preflight_check: bool = Field(default=False)
    port: int = Field(default=3001)

    # ---------- feature flags ----------
    enable_context_extraction: bool = True
    enable_frequency_tracking: bool = True
    enable_smart_filtering: bool = True
    enable_entropy_pruning: bool = True
    enable_ocr_fallback: bool = True
    enable_parallel_processing: bool = True
    enable_enhanced_memory_storage: bool = True

    # ---------- OCR ----------
    ocr_max_pages: Optional[int] = Field(
        None, description="None = unlimited; otherwise max OCR pages"
    )

    # ---------- parallelism ----------
    max_parallel_workers: Optional[int] = Field(
        None, env="MAX_PARALLEL_WORKERS"
    )

    # ---------- entropy pruning ----------
    entropy_threshold: float = 0.0001
    similarity_threshold: float = 0.85
    max_diverse_concepts: Optional[int] = None
    concepts_per_category: Optional[int] = None

    # ---------- file-size limits ----------
    small_file_mb:  int = 1
    small_chunks:   int = 300
    small_concepts: int = 250

    medium_file_mb:  int = 5
    medium_chunks:   int = 500
    medium_concepts: int = 700

    large_file_mb:  int = 25
    large_chunks:   int = 1200
    large_concepts: int = 1500

    xlarge_chunks:   int = 2000
    xlarge_concepts: int = 3000

    # ---------- section weights ----------
    section_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "title": 2.0,
            "abstract": 1.5,
            "introduction": 1.2,
            "conclusion": 1.2,
            "methodology": 1.1,
            "results": 1.1,
            "discussion": 1.0,
            "body": 1.0,
            "references": 0.7,
        },
        env="SECTION_WEIGHTS_JSON",
    )

    # ---------- generic terms ----------
    generic_terms: Set[str] = Field(
        default_factory=lambda: {
            "document", "paper", "analysis", "method", "approach",
            "study", "research", "results", "data", "figure", "table",
        }
    )

    # ---------- academic sections ----------
    academic_sections: Set[str] = Field(
        default_factory=lambda: {
            "abstract", "introduction", "methodology", "methods",
            "results", "discussion", "conclusion", "references",
            "bibliography", "appendix", "acknowledgments"
        }
    )

    # ---------- paths & versions ----------
    base_dir: Path = Path(__file__).resolve().parent.parent
    extractor_version: str = "tori_enhanced_v2.2"

    # ---------- validators / helpers ----------
    @field_validator("section_weights", mode="before")
    @classmethod
    def _parse_section_weights(cls, v):
        """
        Allow env override as JSON or 'title=2.0,abstract=1.4,...'
        """
        if isinstance(v, str):
            import json, re
            if v.strip().startswith("{"):
                return json.loads(v)
            return {
                k: float(val) for k, val in
                (pair.split("=") for pair in re.split(r"[;,]", v) if pair)
            }
        return v
    
    @field_validator("skip_preflight_check", mode="before")
    @classmethod
    def _parse_bool(cls, v):
        """Parse string boolean values"""
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes', 'on')
        return v
    
    @field_validator("port", mode="before")
    @classmethod
    def _parse_port(cls, v):
        """Parse string port values"""
        if isinstance(v, str):
            return int(v)
        return v

    model_config = SettingsConfigDict(
        env_prefix="",  # read env vars as-is
        env_file=".env",  # optional – keeps docker-compose tidy
        case_sensitive=False,
        extra="allow",  # Allow extra fields to prevent validation errors
        env_file_encoding='utf-8'
    )

# singleton – import anywhere in the codebase
settings = Settings()

# ============================================
# BACKWARD COMPATIBILITY EXPORTS
# ============================================
# Export constants for modules expecting the old config format

# Feature flags
ENABLE_CONTEXT_EXTRACTION = settings.enable_context_extraction
ENABLE_FREQUENCY_TRACKING = settings.enable_frequency_tracking
ENABLE_SMART_FILTERING = settings.enable_smart_filtering
ENABLE_ENTROPY_PRUNING = settings.enable_entropy_pruning
ENABLE_OCR_FALLBACK = settings.enable_ocr_fallback
ENABLE_PARALLEL_PROCESSING = settings.enable_parallel_processing
ENABLE_ENHANCED_MEMORY_STORAGE = settings.enable_enhanced_memory_storage

# OCR settings
OCR_MAX_PAGES = settings.ocr_max_pages

# Parallelism
MAX_PARALLEL_WORKERS = settings.max_parallel_workers

# Entropy configuration as expected by the pipeline
ENTROPY_CONFIG = {
    "entropy_threshold": settings.entropy_threshold,
    "similarity_threshold": settings.similarity_threshold,
    "max_diverse_concepts": settings.max_diverse_concepts,
    "concepts_per_category": settings.concepts_per_category,
    "enable_categories": True  # Default value expected by pipeline
}

# File size limits in the format expected by the pipeline
# Format: {size_category: (max_bytes, chunk_limit, concept_limit)}
FILE_SIZE_LIMITS = {
    "small": (
        settings.small_file_mb * 1024 * 1024,    # Convert MB to bytes
        settings.small_chunks,
        settings.small_concepts
    ),
    "medium": (
        settings.medium_file_mb * 1024 * 1024,   # Convert MB to bytes
        settings.medium_chunks,
        settings.medium_concepts
    ),
    "large": (
        settings.large_file_mb * 1024 * 1024,    # Convert MB to bytes
        settings.large_chunks,
        settings.large_concepts
    ),
    "xlarge": (
        float('inf'),                            # No upper limit for xlarge
        settings.xlarge_chunks,
        settings.xlarge_concepts
    )
}

# Section weights
SECTION_WEIGHTS = settings.section_weights

# Generic terms
GENERIC_TERMS = settings.generic_terms

# Academic sections
ACADEMIC_SECTIONS = settings.academic_sections

# Version info
EXTRACTOR_VERSION = settings.extractor_version

# Base directory
BASE_DIR = settings.base_dir

# Concept database path
CONCEPT_DB_PATH = settings.base_dir / "data" / "concept_db.json"

# Universal seed path for PDF processing
UNIVERSAL_SEED_PATH = settings.base_dir / "data" / "universal_seed.json"

# Minimum score threshold for concept filtering
MIN_CONCEPT_SCORE = 0.15

# High quality threshold for filtering premium concepts
HIGH_QUALITY_THRESHOLD = 0.85

# Minimum length for valid concepts (filters out very short terms)
MIN_CONCEPT_LENGTH = 6

# Maximum number of words in a concept (longer concepts are truncated)
MAX_CONCEPT_WORDS = 12

# Maximum number of database boost passes during ingest
MAX_DATABASE_BOOSTS = 5

# ---------- legacy constants expected by ingest pipeline ----------
CHUNK_SIZE = 800           # Default chunk size in characters
MIN_DATABASE_BOOSTS = 5    # Minimum database boosts
MAX_CHUNKS_DEFAULT = 100   # Default maximum chunks
MAX_CONCEPTS_DEFAULT = 50  # Default maximum concepts

# Additional exports that might be expected
CONFIG = {
    "feature_flags": {
        "enable_context_extraction": settings.enable_context_extraction,
        "enable_frequency_tracking": settings.enable_frequency_tracking,
        "enable_smart_filtering": settings.enable_smart_filtering,
        "enable_entropy_pruning": settings.enable_entropy_pruning,
        "enable_ocr_fallback": settings.enable_ocr_fallback,
        "enable_parallel_processing": settings.enable_parallel_processing,
        "enable_enhanced_memory_storage": settings.enable_enhanced_memory_storage,
    },
    "ocr": {
        "max_pages": settings.ocr_max_pages,
    },
    "parallelism": {
        "max_workers": settings.max_parallel_workers,
    },
    "entropy": ENTROPY_CONFIG,
    "file_limits": FILE_SIZE_LIMITS,
    "section_weights": settings.section_weights,
    "generic_terms": list(settings.generic_terms),
    "academic_sections": list(settings.academic_sections),
    "version": settings.extractor_version,
}

# All available exports for modules that use star imports
__all__ = [
    'settings',  # The Pydantic settings object
    'Settings',  # The settings class
    # Feature flags
    'ENABLE_CONTEXT_EXTRACTION',
    'ENABLE_FREQUENCY_TRACKING', 
    'ENABLE_SMART_FILTERING',
    'ENABLE_ENTROPY_PRUNING',
    'ENABLE_OCR_FALLBACK',
    'ENABLE_PARALLEL_PROCESSING',
    'ENABLE_ENHANCED_MEMORY_STORAGE',
    # Configuration values
    'OCR_MAX_PAGES',
    'MAX_PARALLEL_WORKERS',
    'ENTROPY_CONFIG',
    'FILE_SIZE_LIMITS',
    'SECTION_WEIGHTS',
    'GENERIC_TERMS',
    'ACADEMIC_SECTIONS',
    'EXTRACTOR_VERSION',
    'BASE_DIR',
    'CONCEPT_DB_PATH',
    'CONFIG',
    'UNIVERSAL_SEED_PATH',
    'MIN_CONCEPT_SCORE',
    'HIGH_QUALITY_THRESHOLD',
    'MIN_CONCEPT_LENGTH',
    'MAX_CONCEPT_WORDS',
    'MAX_DATABASE_BOOSTS',
    # Legacy constants
    'CHUNK_SIZE',
    'MIN_DATABASE_BOOSTS',
    'MAX_CHUNKS_DEFAULT',
    'MAX_CONCEPTS_DEFAULT',
]
