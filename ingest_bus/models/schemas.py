"""
Data schemas for the TORI Ingest Bus system.

This module defines the Pydantic models used for data validation and serialization
throughout the ingest bus service.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator

class IngestStatus(str, Enum):
    """Status values for ingest jobs."""
    QUEUED = "queued"
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    VECTORIZING = "vectorizing"
    CONCEPT_MAPPING = "concept_mapping"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentType(str, Enum):
    """Types of documents that can be ingested."""
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    CONVERSATION = "conversation"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"

class FailureCode(str, Enum):
    """Standardized failure codes for ingest operations."""
    INVALID_FORMAT = "invalid_format"
    PARSING_ERROR = "parsing_error"
    EXTRACTION_ERROR = "extraction_error"
    CHUNKING_ERROR = "chunking_error"
    VECTORIZATION_ERROR = "vectorization_error"
    CONCEPT_MAPPING_ERROR = "concept_mapping_error"
    STORAGE_ERROR = "storage_error"
    TIMEOUT = "timeout"
    AUTHENTICATION_ERROR = "authentication_error"
    UNKNOWN = "unknown"

class Chunk(BaseModel):
    """A chunk of text from a document."""
    id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Text content of the chunk")
    index: int = Field(..., description="Position in the sequence of chunks")
    sha256: str = Field(..., description="SHA256 hash of the chunk content")
    start_offset: int = Field(..., description="Start character offset in original document")
    end_offset: int = Field(..., description="End character offset in original document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the chunk")

class ConceptVectorLink(BaseModel):
    """A link between a chunk and a concept in ScholarSphere."""
    concept_id: str = Field(..., description="Concept ID in ScholarSphere")
    chunk_id: str = Field(..., description="Chunk ID this concept is linked to")
    strength: float = Field(..., description="Strength of association (0-1)")
    phase_vector: List[float] = Field(default_factory=list, description="Phase vector for this concept link")
    encoder_version: str = Field(..., description="Version of the encoder used to generate the vector")

class IngestRequest(BaseModel):
    """Request to ingest a document."""
    document_type: DocumentType = Field(..., description="Type of document being ingested")
    source_url: Optional[str] = Field(None, description="Source URL of the document if applicable")
    title: Optional[str] = Field(None, description="Title of the document")
    description: Optional[str] = Field(None, description="Description of the document")
    tags: List[str] = Field(default_factory=list, description="Tags for the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the document")
    priority: int = Field(default=1, description="Priority of the ingest job (1-5, higher is higher priority)")
    callback_url: Optional[str] = Field(None, description="Webhook to call when ingest is complete")

class IngestJob(BaseModel):
    """An ingest job in the system."""
    id: str = Field(..., description="Unique identifier for the ingest job")
    request: IngestRequest = Field(..., description="Original ingest request")
    status: IngestStatus = Field(default=IngestStatus.QUEUED, description="Current status of the job")
    created_at: datetime = Field(default_factory=datetime.now, description="When the job was created")
    updated_at: datetime = Field(default_factory=datetime.now, description="When the job was last updated")
    completed_at: Optional[datetime] = Field(None, description="When the job was completed")
    percent_complete: float = Field(default=0.0, description="Percentage of completion (0-100)")
    chunks_processed: int = Field(default=0, description="Number of chunks processed")
    chunks_total: int = Field(default=0, description="Total number of chunks")
    concepts_mapped: int = Field(default=0, description="Number of concepts mapped")
    failure_code: Optional[FailureCode] = Field(None, description="Failure code if the job failed")
    failure_message: Optional[str] = Field(None, description="Detailed failure message if the job failed")
    chunk_ids: List[str] = Field(default_factory=list, description="IDs of the chunks created from this document")
    concept_ids: List[str] = Field(default_factory=list, description="IDs of the concepts mapped from this document")

class IngestJobUpdate(BaseModel):
    """An update to an ingest job."""
    status: Optional[IngestStatus] = Field(None, description="Updated status of the job")
    percent_complete: Optional[float] = Field(None, description="Updated percentage of completion (0-100)")
    chunks_processed: Optional[int] = Field(None, description="Updated number of chunks processed")
    chunks_total: Optional[int] = Field(None, description="Updated total number of chunks")
    concepts_mapped: Optional[int] = Field(None, description="Updated number of concepts mapped")
    failure_code: Optional[FailureCode] = Field(None, description="Failure code if the job failed")
    failure_message: Optional[str] = Field(None, description="Detailed failure message if the job failed")
    chunk_ids: Optional[List[str]] = Field(None, description="IDs of the chunks created from this document")
    concept_ids: Optional[List[str]] = Field(None, description="IDs of the concepts mapped from this document")

class MetricsResponse(BaseModel):
    """Response containing ingest metrics."""
    timestamp: datetime = Field(default_factory=datetime.now, description="When the metrics were collected")
    ingest_files_queued_total: int = Field(..., description="Total number of files queued for ingestion")
    ingest_files_processed_total: int = Field(..., description="Total number of files processed")
    ingest_failures_total: int = Field(..., description="Total number of ingest failures")
    chunk_avg_len_chars: float = Field(..., description="Average length of chunks in characters")
    concept_recall_accuracy: float = Field(..., description="Accuracy of concept recall (0-1)")
    active_jobs: int = Field(..., description="Number of active ingest jobs")
    queue_depth: int = Field(..., description="Number of jobs waiting in the queue")
    processing_time_avg_ms: float = Field(..., description="Average processing time in milliseconds")
    failure_by_code: Dict[str, int] = Field(default_factory=dict, description="Count of failures by code")
