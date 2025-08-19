"""
Job and related models for ingest-bus service.

This module defines the data structures for ingest jobs and their status.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid


class JobStatus(str, Enum):
    """Status of an ingest job"""
    QUEUED = "queued"        # Job is queued, waiting to be processed
    PROCESSING = "processing"  # Job is currently being processed
    COMPLETED = "completed"   # Job completed successfully
    FAILED = "failed"         # Job failed during processing
    CANCELED = "canceled"     # Job was canceled


class ProcessingStage(str, Enum):
    """Stage of processing for an ingest job"""
    QUEUED = "queued"               # Initial state, job is queued
    DOWNLOADING = "downloading"     # Downloading file from source
    MIME_CHECK = "mime_check"       # Checking MIME type
    TEXT_EXTRACTION = "text_extraction"  # Extracting text from file
    IMAGE_EXTRACTION = "image_extraction"  # Extracting images
    CHUNKING = "chunking"           # Breaking into chunks
    EMBEDDING = "embedding"         # Creating embeddings
    GRAPH_INSERTION = "graph_insertion"  # Inserting into knowledge graph
    COMPLETED = "completed"         # All processing completed
    FAILED = "failed"               # Processing failed


class ChunkInfo:
    """Information about a chunk created from a document"""
    
    def __init__(self, 
                 id: str,
                 text: str,
                 start_offset: int,
                 end_offset: int,
                 metadata: Dict[str, Any] = None):
        """
        Create a new chunk info
        
        Args:
            id: Unique identifier for the chunk
            text: Text content of the chunk
            start_offset: Starting character offset in the original document
            end_offset: Ending character offset in the original document
            metadata: Additional metadata about the chunk
        """
        self.id = id
        self.text = text
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "text": self.text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkInfo':
        """Create from dictionary representation"""
        return cls(
            id=data["id"],
            text=data["text"],
            start_offset=data["start_offset"],
            end_offset=data["end_offset"],
            metadata=data.get("metadata", {})
        )


class IngestJob:
    """
    Ingest job model
    
    Represents a document processing job in the ingest system
    """
    
    def __init__(self,
                 job_id: str = None,
                 file_url: str = None,
                 file_name: str = None,
                 file_size: int = None,
                 file_sha256: str = None,
                 track: str = None,
                 status: JobStatus = JobStatus.QUEUED,
                 created_at: datetime = None,
                 updated_at: datetime = None,
                 completed_at: Optional[datetime] = None,
                 processing_stage: ProcessingStage = ProcessingStage.QUEUED,
                 progress: float = 0.0,
                 error: Optional[str] = None,
                 metadata: Dict[str, Any] = None,
                 chunk_count: int = 0,
                 chunks: List[ChunkInfo] = None,
                 concept_ids: List[str] = None):
        """
        Create a new ingest job
        
        Args:
            job_id: Unique identifier for the job
            file_url: URL to the file to process
            file_name: Name of the file
            file_size: Size of the file in bytes
            file_sha256: SHA-256 hash of the file
            track: Track to assign the document to (e.g., "programming", "math_physics")
            status: Current status of the job
            created_at: When the job was created
            updated_at: When the job was last updated
            completed_at: When the job was completed (if completed)
            processing_stage: Current processing stage
            progress: Processing progress (0-100%)
            error: Error message (if failed)
            metadata: Additional metadata
            chunk_count: Number of chunks created
            chunks: Information about chunks created
            concept_ids: Concept IDs created in knowledge graph
        """
        # Set default values
        now = datetime.utcnow()
        
        self.job_id = job_id or str(uuid.uuid4())
        self.file_url = file_url
        self.file_name = file_name
        self.file_size = file_size
        self.file_sha256 = file_sha256
        self.track = track
        self.status = status
        self.created_at = created_at or now
        self.updated_at = updated_at or now
        self.completed_at = completed_at
        self.processing_stage = processing_stage
        self.progress = progress
        self.error = error
        self.metadata = metadata or {}
        self.chunk_count = chunk_count
        self.chunks = chunks or []
        self.concept_ids = concept_ids or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "job_id": self.job_id,
            "file_url": self.file_url,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "file_sha256": self.file_sha256,
            "track": self.track,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_stage": self.processing_stage.value,
            "progress": self.progress,
            "error": self.error,
            "metadata": self.metadata,
            "chunk_count": self.chunk_count,
            "chunks": [c.to_dict() for c in self.chunks] if self.chunks else [],
            "concept_ids": self.concept_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IngestJob':
        """Create from dictionary representation"""
        job = cls(
            job_id=data.get("job_id"),
            file_url=data.get("file_url"),
            file_name=data.get("file_name"),
            file_size=data.get("file_size"),
            file_sha256=data.get("file_sha256"),
            track=data.get("track"),
            status=JobStatus(data.get("status", JobStatus.QUEUED.value)),
            created_at=datetime.fromisoformat(data.get("created_at")) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data.get("updated_at")) if data.get("updated_at") else None,
            completed_at=datetime.fromisoformat(data.get("completed_at")) if data.get("completed_at") else None,
            processing_stage=ProcessingStage(data.get("processing_stage", ProcessingStage.QUEUED.value)),
            progress=data.get("progress", 0.0),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
            chunk_count=data.get("chunk_count", 0),
            concept_ids=data.get("concept_ids", [])
        )
        
        if "chunks" in data and data["chunks"]:
            job.chunks = [ChunkInfo.from_dict(c) for c in data["chunks"]]
        
        return job
    
    def update_status(self, status: JobStatus, error: Optional[str] = None) -> None:
        """
        Update the job status
        
        Args:
            status: New status
            error: Error message (if failed)
        """
        self.status = status
        self.updated_at = datetime.utcnow()
        
        if error:
            self.error = error
        
        if status == JobStatus.COMPLETED or status == JobStatus.FAILED:
            self.completed_at = datetime.utcnow()
    
    def update_progress(self, stage: ProcessingStage, progress: float) -> None:
        """
        Update processing progress
        
        Args:
            stage: Current processing stage
            progress: Progress percentage (0-100)
        """
        self.processing_stage = stage
        self.progress = progress
        self.updated_at = datetime.utcnow()
        
        # Update status based on stage
        if stage == ProcessingStage.FAILED:
            self.status = JobStatus.FAILED
            self.completed_at = datetime.utcnow()
        elif stage == ProcessingStage.COMPLETED:
            self.status = JobStatus.COMPLETED
            self.completed_at = datetime.utcnow()
        elif stage != ProcessingStage.QUEUED and self.status == JobStatus.QUEUED:
            self.status = JobStatus.PROCESSING
    
    def add_chunk(self, chunk: ChunkInfo) -> None:
        """
        Add a chunk to the job
        
        Args:
            chunk: Chunk information
        """
        self.chunks.append(chunk)
        self.chunk_count = len(self.chunks)
        self.updated_at = datetime.utcnow()
    
    def add_concept_id(self, concept_id: str) -> None:
        """
        Add a concept ID to the job
        
        Args:
            concept_id: Concept ID in knowledge graph
        """
        if concept_id not in self.concept_ids:
            self.concept_ids.append(concept_id)
            self.updated_at = datetime.utcnow()
