"""
type_definitions.py

Type definitions for pipeline data structures.
Provides TypedDict and other type hints for better IDE support and type safety.
"""

from typing import TypedDict, Dict, List, Any, Optional, Union, Literal, Callable
from datetime import datetime


class PdfMetadata(TypedDict):
    """PDF file metadata structure."""
    filename: str
    file_path: str
    extraction_timestamp: str
    extractor_version: str
    file_size_bytes: int
    sha256: str
    file_size_mb: float
    page_count: int
    estimated_uncompressed_mb: float
    pdf_metadata: Optional[Dict[str, str]]
    ocr_used: Optional[bool]


class ConceptMetadata(TypedDict, total=False):
    """Metadata for an extracted concept."""
    frequency: int
    section: str
    category: str
    in_title: bool
    in_abstract: bool
    chunk_index: int


class ConceptDict(TypedDict):
    """Structure of an extracted concept."""
    name: str
    score: float
    method: str
    metadata: ConceptMetadata
    quality_score: Optional[float]
    source: Optional[Dict[str, Any]]


class PurityAnalysis(TypedDict):
    """Purity analysis statistics."""
    raw_concepts: int
    pure_concepts: int
    final_concepts: int
    purity_efficiency_percent: float
    diversity_efficiency_percent: float
    top_concepts: List[Dict[str, Any]]


class EntropyAnalysis(TypedDict):
    """Entropy pruning analysis."""
    enabled: bool
    admin_mode: Optional[bool]
    total_before_entropy: Optional[int]
    selected_diverse: Optional[int]
    pruned_similar: Optional[int]
    diversity_efficiency_percent: Optional[float]
    final_entropy: Optional[float]
    avg_similarity: Optional[float]
    by_category: Optional[Dict[str, int]]
    config: Optional[Dict[str, Any]]
    performance: Optional[Dict[str, Any]]
    reason: Optional[str]


class IngestResponse(TypedDict):
    """Complete response from PDF ingestion."""
    filename: str
    concept_count: int
    concept_names: List[str]
    concepts: List[ConceptDict]
    status: Literal["success", "error", "no_concepts", "empty", "partial_success"]
    purity_based: bool
    entropy_pruned: bool
    admin_mode: bool
    equal_access: bool
    performance_limited: bool
    chunks_processed: int
    chunks_available: int
    semantic_extracted: int
    file_storage_boosted: int
    average_concept_score: float
    high_confidence_concepts: int
    high_quality_concepts: int
    total_extraction_time: float
    domain_distribution: Dict[str, int]
    section_distribution: Dict[str, int]
    title_found: bool
    abstract_found: bool
    ocr_used: bool
    parallel_processing: bool
    enhanced_memory_storage: bool
    processing_time_seconds: float
    sha256: str
    purity_analysis: PurityAnalysis
    entropy_analysis: EntropyAnalysis
    error_message: Optional[str]
    error: Optional[Dict[str, Any]]
    warnings: Optional[List[Dict[str, Any]]]
    metadata: Optional[Dict[str, Any]]


class ErrorResponse(TypedDict):
    """Error response structure."""
    filename: str
    concept_count: int
    concept_names: List[str]
    concepts: List[Any]
    status: Literal["error"]
    admin_mode: bool
    processing_time_seconds: float
    error_message: str
    error: Dict[str, Any]
    metadata: Optional[Dict[str, Any]]


class ChunkDict(TypedDict):
    """Structure of a text chunk."""
    text: str
    index: int
    section: str
    metadata: Optional[Dict[str, Any]]


class ExtractionParams(TypedDict):
    """Parameters for concept extraction."""
    threshold: float
    title: str
    abstract: str


class ProgressState(TypedDict):
    """Progress tracking state."""
    current: int
    total: int
    percentage: float
    is_complete: bool
    last_reported_pct: Optional[float]
    description: str


class ExecutorStatsDict(TypedDict):
    """Executor statistics."""
    tasks_submitted: int
    tasks_completed: int
    tasks_failed: int
    average_duration: float
    active_tasks: int
    max_concurrent: int
    success_rate: float


class SafetyCheckResult(TypedDict):
    """Result of PDF safety check."""
    is_safe: bool
    message: str
    metadata: PdfMetadata


# Type aliases for common patterns
ConceptList = List[ConceptDict]
ChunkList = List[ChunkDict]

# Signature: (stage name, percentage, human-readable message) -> None
ProgressCallback = Optional[Callable[[str, int, str], None]]

# Export all types
__all__ = [
    'PdfMetadata',
    'ConceptMetadata',
    'ConceptDict',
    'PurityAnalysis',
    'EntropyAnalysis',
    'IngestResponse',
    'ErrorResponse',
    'ChunkDict',
    'ExtractionParams',
    'ProgressState',
    'ExecutorStatsDict',
    'SafetyCheckResult',
    'ConceptList',
    'ChunkList',
    'ProgressCallback'
]
