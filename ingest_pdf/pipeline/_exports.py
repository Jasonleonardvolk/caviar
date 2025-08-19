"""
Single source of truth for TORI pipeline exports.
This prevents export lists from drifting out of sync.
"""

# Core pipeline functions
PIPELINE_EXPORTS = [
    'ingest_pdf_clean',
    'ingest_pdf_async', 
    'handle',
    'handle_pdf',
    'get_db',
    'preload_concept_database',
    'ProgressTracker',
]

# Configuration exports
CONFIG_EXPORTS = [
    'settings',
    'Settings',
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
    'FILE_SIZE_LIMITS',
    'SECTION_WEIGHTS',
    'GENERIC_TERMS',
    'ACADEMIC_SECTIONS',
    'EXTRACTOR_VERSION',
    'BASE_DIR',
    'CONFIG',
]

# Utility exports
UTILITY_EXPORTS = [
    'safe_get',
    'safe_divide',
    'safe_multiply',
    'safe_percentage',
    'safe_round',
    'sanitize_dict',
    'run_sync',
    'await_sync',
]

# Combined public API
PUBLIC_API = PIPELINE_EXPORTS + ['settings'] + UTILITY_EXPORTS
