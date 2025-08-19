from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import uuid

@dataclass
class ConceptTuple:
    name: str
    embedding: np.ndarray
    context: str
    passage_embedding: np.ndarray
    cluster_members: List[int]
    resonance_score: float
    narrative_centrality: float
    predictability_score: float = 0.5  # New field for Lyapunov-based predictability
    
    # Enhanced metadata fields (addresses Issue #4 - metadata preservation)
    confidence: float = 0.0  # Extraction confidence score
    method: str = "unknown"  # Extraction method used
    source_reference: Dict[str, Any] = field(default_factory=dict)  # Source location info
    
    # Existing enhanced fields for transparency and provenance tracking
    eigenfunction_id: str = ""  # Unique spectral identifier
    source_provenance: Dict[str, Any] = field(default_factory=dict)  # Document source metadata
    spectral_lineage: List[Tuple[float, float]] = field(default_factory=list)  # Eigenvalue, magnitude pairs
    cluster_coherence: float = 0.0  # Internal similarity measure
    
    # Additional metadata for enhanced tracking
    extraction_timestamp: Optional[str] = None  # When concept was extracted
    processing_metadata: Dict[str, Any] = field(default_factory=dict)  # Processing pipeline info
    quality_metrics: Dict[str, float] = field(default_factory=dict)  # Quality assessment scores
    
    def __post_init__(self):
        """Generate a unique eigenfunction ID if not provided."""
        if not self.eigenfunction_id:
            # Create a deterministic ID based on embedding fingerprint if possible
            if hasattr(self, 'embedding') and self.embedding is not None:
                fingerprint = hash(self.embedding.tobytes()) % 10000000
                self.eigenfunction_id = f"eigen-{fingerprint}"
            else:
                # Fallback to random UUID
                self.eigenfunction_id = f"eigen-{str(uuid.uuid4())[:8]}"
        
        # Set extraction timestamp if not provided
        if not self.extraction_timestamp:
            from datetime import datetime
            self.extraction_timestamp = datetime.now().isoformat()
        
        # Ensure method is set
        if not self.method or self.method == "unknown":
            # Try to infer from processing metadata
            if "extraction_method" in self.processing_metadata:
                self.method = self.processing_metadata["extraction_method"]
            else:
                self.method = "embedding_cluster"  # Default method
        
        # Initialize quality metrics if empty
        if not self.quality_metrics:
            self.quality_metrics = {
                "resonance": self.resonance_score,
                "centrality": self.narrative_centrality,
                "predictability": self.predictability_score,
                "confidence": self.confidence,
                "coherence": self.cluster_coherence
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a fully traceable dictionary representation for visualization and debugging."""
        return {
            # Core concept information
            "name": self.name,
            "context": self.context,
            
            # Quality and scoring metrics
            "confidence": self.confidence,
            "resonance_score": self.resonance_score,
            "narrative_centrality": self.narrative_centrality,
            "predictability_score": self.predictability_score,
            "cluster_coherence": self.cluster_coherence,
            
            # Extraction metadata (addresses Issue #4)
            "method": self.method,
            "source_reference": self.source_reference,
            "extraction_timestamp": self.extraction_timestamp,
            
            # Provenance and lineage
            "eigenfunction_id": self.eigenfunction_id,
            "source_provenance": self.source_provenance,
            "spectral_lineage": [(float(real), float(mag)) for real, mag in self.spectral_lineage],
            
            # Processing metadata
            "processing_metadata": self.processing_metadata,
            "quality_metrics": self.quality_metrics,
            
            # Cluster information
            "cluster_size": len(self.cluster_members),
            "cluster_members": self.cluster_members
        }
    
    def to_minimal_dict(self) -> Dict[str, Any]:
        """Convert to minimal dictionary for storage efficiency."""
        return {
            "name": self.name,
            "confidence": self.confidence,
            "method": self.method,
            "source": self.source_reference,
            "context": self.context[:200] + "..." if len(self.context) > 200 else self.context,
            "eigenfunction_id": self.eigenfunction_id,
            "extraction_timestamp": self.extraction_timestamp
        }
    
    def update_confidence(self, new_confidence: float, reason: str = ""):
        """Update confidence score with tracking."""
        old_confidence = self.confidence
        self.confidence = max(0.0, min(1.0, new_confidence))  # Clamp to [0,1]
        
        # Track confidence changes
        if "confidence_history" not in self.processing_metadata:
            self.processing_metadata["confidence_history"] = []
        
        self.processing_metadata["confidence_history"].append({
            "old_value": old_confidence,
            "new_value": self.confidence,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update quality metrics
        self.quality_metrics["confidence"] = self.confidence
    
    def add_processing_step(self, step_name: str, details: Dict[str, Any]):
        """Add processing step metadata."""
        if "processing_steps" not in self.processing_metadata:
            self.processing_metadata["processing_steps"] = []
        
        self.processing_metadata["processing_steps"].append({
            "step": step_name,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    def validate_schema(self) -> Tuple[bool, List[str]]:
        """Validate concept schema and return issues."""
        issues = []
        
        # Check required fields
        if not self.name or not isinstance(self.name, str):
            issues.append("Invalid or missing name")
        
        if not isinstance(self.confidence, (int, float)) or not (0.0 <= self.confidence <= 1.0):
            issues.append("Invalid confidence score (must be float 0.0-1.0)")
        
        if not self.method or not isinstance(self.method, str):
            issues.append("Invalid or missing extraction method")
        
        if not isinstance(self.source_reference, dict):
            issues.append("Invalid source reference (must be dict)")
        
        # Check embeddings
        if self.embedding is None or not hasattr(self.embedding, 'shape'):
            issues.append("Invalid embedding (must be numpy array)")
        elif self.embedding.size == 0:
            issues.append("Empty embedding array")
        
        # Check context
        if not self.context or not isinstance(self.context, str):
            issues.append("Invalid or missing context")
        
        return len(issues) == 0, issues
    
    def get_quality_score(self) -> float:
        """Calculate overall quality score."""
        weights = {
            "confidence": 0.3,
            "resonance": 0.2,
            "centrality": 0.2,
            "coherence": 0.2,
            "predictability": 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric == "confidence":
                score += weight * self.confidence
            elif metric == "resonance":
                score += weight * min(1.0, self.resonance_score)
            elif metric == "centrality":
                score += weight * min(1.0, self.narrative_centrality / 10.0)  # Normalize centrality
            elif metric == "coherence":
                score += weight * self.cluster_coherence
            elif metric == "predictability":
                score += weight * self.predictability_score
        
        return min(1.0, max(0.0, score))

@dataclass
class ConceptExtractionResult:
    """Container for concept extraction results with metadata."""
    
    concepts: List[ConceptTuple]
    document_metadata: Dict[str, Any]
    extraction_summary: Dict[str, Any]
    processing_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize extraction result."""
        if not self.extraction_summary:
            self.extraction_summary = {
                "total_concepts": len(self.concepts),
                "extraction_timestamp": datetime.now().isoformat(),
                "average_confidence": self.get_average_confidence(),
                "quality_distribution": self.get_quality_distribution()
            }
    
    def get_average_confidence(self) -> float:
        """Calculate average confidence across all concepts."""
        if not self.concepts:
            return 0.0
        return sum(c.confidence for c in self.concepts) / len(self.concepts)
    
    def get_quality_distribution(self) -> Dict[str, int]:
        """Get distribution of quality tiers."""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for concept in self.concepts:
            if concept.confidence >= 0.8:
                distribution["high"] += 1
            elif concept.confidence >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    def filter_by_confidence(self, min_confidence: float) -> 'ConceptExtractionResult':
        """Return new result with concepts filtered by confidence."""
        filtered_concepts = [c for c in self.concepts if c.confidence >= min_confidence]
        
        return ConceptExtractionResult(
            concepts=filtered_concepts,
            document_metadata=self.document_metadata,
            extraction_summary={
                **self.extraction_summary,
                "filtered_by_confidence": min_confidence,
                "concepts_after_filter": len(filtered_concepts)
            },
            processing_log=self.processing_log + [{
                "operation": "confidence_filter",
                "threshold": min_confidence,
                "concepts_before": len(self.concepts),
                "concepts_after": len(filtered_concepts),
                "timestamp": datetime.now().isoformat()
            }]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "concepts": [c.to_dict() for c in self.concepts],
            "document_metadata": self.document_metadata,
            "extraction_summary": self.extraction_summary,
            "processing_log": self.processing_log
        }
    
    def to_minimal_dict(self) -> Dict[str, Any]:
        """Convert to minimal dictionary for lightweight storage."""
        return {
            "concepts": [c.to_minimal_dict() for c in self.concepts],
            "summary": self.extraction_summary,
            "document_id": self.document_metadata.get("filename", "unknown")
        }

# For compatibility with existing code
def create_concept_tuple_from_dict(data: Dict[str, Any]) -> ConceptTuple:
    """Create ConceptTuple from dictionary (for backward compatibility)."""
    # Handle numpy arrays
    embedding = data.get("embedding", [])
    if isinstance(embedding, list):
        embedding = np.array(embedding, dtype=np.float32)
    
    passage_embedding = data.get("passage_embedding", [])
    if isinstance(passage_embedding, list):
        passage_embedding = np.array(passage_embedding, dtype=np.float32)
    
    return ConceptTuple(
        name=data.get("name", "Unnamed"),
        embedding=embedding,
        context=data.get("context", ""),
        passage_embedding=passage_embedding,
        cluster_members=data.get("cluster_members", []),
        resonance_score=data.get("resonance_score", 0.0),
        narrative_centrality=data.get("narrative_centrality", 0.0),
        predictability_score=data.get("predictability_score", 0.5),
        confidence=data.get("confidence", 0.0),
        method=data.get("method", "unknown"),
        source_reference=data.get("source_reference", data.get("source", {})),
        eigenfunction_id=data.get("eigenfunction_id", ""),
        source_provenance=data.get("source_provenance", {}),
        spectral_lineage=data.get("spectral_lineage", []),
        cluster_coherence=data.get("cluster_coherence", 0.0),
        extraction_timestamp=data.get("extraction_timestamp"),
        processing_metadata=data.get("processing_metadata", {}),
        quality_metrics=data.get("quality_metrics", {})
    )
