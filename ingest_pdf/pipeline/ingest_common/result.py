"""
ingest_common/result.py

Common result dataclass for all ingestion handlers.
Provides a unified interface regardless of input media type.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

@dataclass
class IngestResult:
    """
    Unified result structure for all media ingestion types.
    
    This provides a consistent interface whether ingesting
    PDF, audio, video, or images.
    """
    # === Core Fields ===
    filename: str
    file_path: str
    media_type: str  # "pdf", "audio", "video", "image", "text"
    mime_type: Optional[str] = None
    
    # === Extraction Results ===
    concepts: List[Dict[str, Any]] = field(default_factory=list)
    concept_count: int = 0
    concept_names: List[str] = field(default_factory=list)
    
    # === Processing Metadata ===
    status: str = "success"  # "success", "partial", "error", "no_concepts"
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # === Performance Metrics ===
    processing_time_seconds: float = 0.0
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # === Media-Specific Metadata ===
    # For PDFs
    page_count: Optional[int] = None
    ocr_used: bool = False
    
    # For Audio/Video
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    
    # For Video
    frame_count: Optional[int] = None
    fps: Optional[float] = None
    resolution: Optional[tuple] = None  # (width, height)
    
    # For Images
    dimensions: Optional[tuple] = None  # (width, height)
    color_mode: Optional[str] = None  # "RGB", "RGBA", "L", etc.
    
    # === Quality Metrics ===
    average_concept_score: float = 0.0
    high_confidence_concepts: int = 0
    high_quality_concepts: int = 0
    
    # === Source Information ===
    file_size_bytes: int = 0
    file_size_mb: float = 0.0
    sha256: str = "unknown"
    
    # === Processing Details ===
    chunks_processed: int = 0
    chunks_available: int = 0
    parallel_processing: bool = False
    
    # === Analysis Results ===
    purity_analysis: Dict[str, Any] = field(default_factory=dict)
    entropy_analysis: Dict[str, Any] = field(default_factory=dict)
    section_distribution: Dict[str, int] = field(default_factory=dict)
    
    # === Holographic Metadata ===
    psi_state: Optional[Dict[str, Any]] = None  # Wavefunction data
    spectral_features: Optional[Dict[str, Any]] = None  # FFT analysis
    motion_vectors: Optional[List[float]] = None  # For video
    
    # === Administrative ===
    admin_mode: bool = False
    doc_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            # Core
            "filename": self.filename,
            "file_path": self.file_path,
            "media_type": self.media_type,
            "mime_type": self.mime_type,
            
            # Results
            "concepts": self.concepts,
            "concept_count": self.concept_count,
            "concept_names": self.concept_names,
            
            # Status
            "status": self.status,
            "error_message": self.error_message,
            "warnings": self.warnings,
            
            # Performance
            "processing_time_seconds": self.processing_time_seconds,
            "extraction_timestamp": self.extraction_timestamp,
            
            # Media metadata
            "metadata": self._get_media_metadata(),
            
            # Quality
            "quality_metrics": {
                "average_concept_score": self.average_concept_score,
                "high_confidence_concepts": self.high_confidence_concepts,
                "high_quality_concepts": self.high_quality_concepts,
            },
            
            # File info
            "file_info": {
                "size_bytes": self.file_size_bytes,
                "size_mb": self.file_size_mb,
                "sha256": self.sha256,
            },
            
            # Processing
            "processing_info": {
                "chunks_processed": self.chunks_processed,
                "chunks_available": self.chunks_available,
                "parallel_processing": self.parallel_processing,
            },
            
            # Analysis
            "purity_analysis": self.purity_analysis,
            "entropy_analysis": self.entropy_analysis,
            "section_distribution": self.section_distribution,
            
            # Holographic
            "holographic": {
                "psi_state": self.psi_state,
                "spectral_features": self.spectral_features,
                "motion_vectors": self.motion_vectors,
            } if any([self.psi_state, self.spectral_features, self.motion_vectors]) else None,
            
            # Admin
            "admin_mode": self.admin_mode,
            "doc_id": self.doc_id,
        }
    
    def _get_media_metadata(self) -> Dict[str, Any]:
        """Get media-specific metadata based on type"""
        metadata = {}
        
        if self.media_type == "pdf":
            metadata.update({
                "page_count": self.page_count,
                "ocr_used": self.ocr_used,
            })
        elif self.media_type in ["audio", "video"]:
            metadata.update({
                "duration_seconds": self.duration_seconds,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
            })
            if self.media_type == "video":
                metadata.update({
                    "frame_count": self.frame_count,
                    "fps": self.fps,
                    "resolution": self.resolution,
                })
        elif self.media_type == "image":
            metadata.update({
                "dimensions": self.dimensions,
                "color_mode": self.color_mode,
            })
        
        return metadata
    
    @classmethod
    def from_legacy_dict(cls, data: Dict[str, Any], media_type: str = "pdf") -> "IngestResult":
        """
        Create IngestResult from legacy pipeline dictionary.
        Maintains backward compatibility.
        """
        return cls(
            filename=data.get("filename", ""),
            file_path=data.get("file_path", ""),
            media_type=media_type,
            
            concepts=data.get("concepts", []),
            concept_count=data.get("concept_count", 0),
            concept_names=data.get("concept_names", []),
            
            status=data.get("status", "success"),
            processing_time_seconds=data.get("processing_time_seconds", 0.0),
            
            page_count=data.get("page_count"),
            ocr_used=data.get("ocr_used", False),
            
            average_concept_score=data.get("average_concept_score", 0.0),
            high_confidence_concepts=data.get("high_confidence_concepts", 0),
            high_quality_concepts=data.get("high_quality_concepts", 0),
            
            file_size_bytes=data.get("file_size_bytes", 0),
            file_size_mb=data.get("file_size_mb", 0.0),
            sha256=data.get("sha256", "unknown"),
            
            chunks_processed=data.get("chunks_processed", 0),
            chunks_available=data.get("chunks_available", 0),
            parallel_processing=data.get("parallel_processing", False),
            
            purity_analysis=data.get("purity_analysis", {}),
            entropy_analysis=data.get("entropy_analysis", {}),
            section_distribution=data.get("section_distribution", {}),
            
            admin_mode=data.get("admin_mode", False),
            doc_id=data.get("doc_id"),
        )
