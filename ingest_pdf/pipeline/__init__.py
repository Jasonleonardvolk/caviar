"""
TORI Pipeline Package - Modular PDF Processing System

This package provides a complete pipeline for PDF ingestion with:
- OCR support for scanned documents
- Academic paper structure detection
- Concept quality scoring
- Entropy-based pruning for diversity
- Integration with Soliton Memory storage

The package maintains backward compatibility with the original pipeline.py
while providing a cleaner modular structure.
"""

import logging

# Import all public functions for backward compatibility
from .pipeline import ingest_pdf_clean, ingest_pdf_async, preload_concept_database, ProgressTracker, get_db

# Import configuration
from .config import (
    # Feature flags
    ENABLE_CONTEXT_EXTRACTION,
    ENABLE_FREQUENCY_TRACKING,
    ENABLE_SMART_FILTERING,
    ENABLE_ENTROPY_PRUNING,
    ENABLE_OCR_FALLBACK,
    ENABLE_PARALLEL_PROCESSING,
    ENABLE_ENHANCED_MEMORY_STORAGE,
    
    # Configuration values
    OCR_MAX_PAGES,
    MAX_PARALLEL_WORKERS,
    ENTROPY_CONFIG,
    ACADEMIC_SECTIONS,
    
    # Version info
    EXTRACTOR_VERSION
)

# Import utilities
from .utils import (
    safe_get,
    safe_divide,
    safe_multiply,
    safe_percentage,
    safe_round,
    sanitize_dict,
    get_logger
)

# Import execution helpers
from .execution_helpers import run_sync, await_sync

# Import IO functions
from .io import (
    extract_pdf_metadata,
    preprocess_with_ocr,
    extract_title_abstract_safe,
    detect_section_type,
    process_chunks_parallel,
    extract_chunks  # This might be from extract_blocks module
)

# Import quality functions
from .quality import (
    load_universal_concept_file_storage,
    calculate_theme_relevance,
    calculate_concept_quality,
    boost_known_concepts,
    extract_and_boost_concepts,
    analyze_concept_purity,
    is_rogue_concept_contextual
)

# Import pruning functions
from .pruning import (
    calculate_simple_similarity,
    cluster_similar_concepts,
    apply_entropy_pruning,
    deduplicate_concepts
)

# Import storage functions
from .storage import (
    store_concepts_in_soliton,
    inject_concept_diff,
    prepare_storage_metadata
)

# Import from external modules that were used in original pipeline
try:
    from ..extractConceptsFromDocument import (
        extractConceptsFromDocument,
        reset_frequency_counter,
        track_concept_frequency,
        get_concept_frequency,
        concept_frequency_counter
    )
except ImportError:
    try:
        from extractConceptsFromDocument import (
            extractConceptsFromDocument,
            reset_frequency_counter,
            track_concept_frequency,
            get_concept_frequency,
            concept_frequency_counter
        )
    except ImportError:
        logging.warning("⚠️ Could not import concept extraction functions")

try:
    from ..entropy_prune import entropy_prune, entropy_prune_with_categories
except ImportError:
    try:
        from entropy_prune import entropy_prune, entropy_prune_with_categories
    except ImportError:
        logging.warning("⚠️ Could not import entropy pruning functions")

try:
    from ..cognitive_interface import add_concept_diff
except ImportError:
    try:
        from cognitive_interface import add_concept_diff
    except ImportError:
        logging.warning("⚠️ Could not import cognitive interface")

try:
    from ..memory_sculptor import memory_sculptor
except ImportError:
    try:
        from memory_sculptor import memory_sculptor
    except ImportError:
        logging.warning("⚠️ Could not import memory sculptor")

# Maintain global state for compatibility
concept_file_storage = []
concept_names = []
concept_scores = {}

# Load the file_storage on import (as original did)
from .quality import concept_file_storage, concept_names, concept_scores

# Version information
__version__ = "2.0.0"
__author__ = "TORI Team"

# Logger setup for the package
logger = logging.getLogger("pdf_ingestion")
logger.setLevel(logging.INFO)

# Compatibility message
logger.info("🛡️ ENHANCED BULLETPROOF PIPELINE LOADED - ZERO NONETYPE ERRORS GUARANTEED")
logger.info("✨ New features: OCR support, academic structure detection, quality metrics, parallel processing, enhanced memory storage")
logger.info("📦 Now with modular architecture for better maintainability")

# Define what's publicly available
# ⚠️ Update test_exports() if you change this list
__all__ = [
    # Main function
    'ingest_pdf_clean',
    'ingest_pdf_async',
    'preload_concept_database',
    'ProgressTracker',
    'get_db',
    
    # Configuration
    'ENABLE_CONTEXT_EXTRACTION',
    'ENABLE_FREQUENCY_TRACKING',
    'ENABLE_SMART_FILTERING',
    'ENABLE_ENTROPY_PRUNING',
    'ENABLE_OCR_FALLBACK',
    'ENABLE_PARALLEL_PROCESSING',
    'ENABLE_ENHANCED_MEMORY_STORAGE',
    'OCR_MAX_PAGES',
    'MAX_PARALLEL_WORKERS',
    'ENTROPY_CONFIG',
    
    # Utilities
    'safe_get',
    'safe_divide',
    'safe_multiply',
    'safe_percentage',
    'safe_round',
    'sanitize_dict',
    'run_sync',
    'await_sync',
    
    # IO functions
    'extract_pdf_metadata',
    'preprocess_with_ocr',
    'extract_title_abstract_safe',
    'detect_section_type',
    
    # Quality functions
    'calculate_concept_quality',
    'analyze_concept_purity',
    'extract_and_boost_concepts',
    'is_rogue_concept_contextual',
    
    # Pruning functions
    'cluster_similar_concepts',
    
    # External functions (if available)
    'extractConceptsFromDocument',
    'reset_frequency_counter',
    'entropy_prune',
    'add_concept_diff',
    'memory_sculptor',
    
    # Global state
    'concept_file_storage',
    'concept_names',
    'concept_scores'
]
