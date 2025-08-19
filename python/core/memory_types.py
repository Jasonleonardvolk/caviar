"""
Memory Types Module
Defines core data structures for the memory vault system
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import uuid


class MemoryType(Enum):
    """Types of memories that can be stored in the vault"""
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    EXTRACTED = "extracted"
    

@dataclass
class Relationship:
    """Represents a relationship between memory concepts"""
    type: str  # e.g., "subject_of", "object_of", "related_to", "similar_to"
    target: str  # The target concept ID or label
    source: Optional[str] = None  # The source concept ID or label
    verb: Optional[str] = None  # The connecting verb (for SVO relationships)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryConcept:
    """
    Core memory concept structure for vault storage
    Designed to be JSON-serializable and extensible
    """
    id: str
    label: str
    method: str = "extracted"  # e.g., "yake", "spacy", "semantic", "manual"
    score: Optional[float] = None
    relationships: List[Dict[str, Any]] = field(default_factory=list)  # List of Relationship dicts
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_type: str = MemoryType.SEMANTIC.value
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __post_init__(self):
        """Initialize timestamps if not provided"""
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        # Ensure ID is string
        if not isinstance(self.id, str):
            self.id = str(self.id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Ensure relationships are dictionaries
        if self.relationships and isinstance(self.relationships[0], Relationship):
            data['relationships'] = [r.to_dict() for r in self.relationships]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryConcept':
        """Create from dictionary (for deserialization)"""
        return cls(**data)
    
    def add_relationship(self, relationship: Relationship):
        """Add a relationship to this concept"""
        if isinstance(relationship, Relationship):
            self.relationships.append(relationship.to_dict())
        else:
            self.relationships.append(relationship)
        self.updated_at = datetime.utcnow().isoformat()
    
    def merge_with(self, other: 'MemoryConcept'):
        """Merge another concept into this one"""
        # Take the higher score
        if other.score and (not self.score or other.score > self.score):
            self.score = other.score
        
        # Merge relationships (deduplicate)
        existing_rels = {(r['type'], r['target']) for r in self.relationships}
        for rel in other.relationships:
            key = (rel['type'], rel['target'])
            if key not in existing_rels:
                self.relationships.append(rel)
                existing_rels.add(key)
        
        # Merge metadata
        for key, value in other.metadata.items():
            if key not in self.metadata:
                self.metadata[key] = value
            elif key == 'sources' and isinstance(value, list):
                # Special handling for source lists
                self.metadata[key] = list(set(self.metadata[key] + value))
        
        self.updated_at = datetime.utcnow().isoformat()


@dataclass
class MemoryEntry:
    """Complete memory entry for vault storage"""
    concept: MemoryConcept
    context: Optional[str] = None
    source_document: Optional[str] = None
    extraction_method: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    bps_tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'concept': self.concept.to_dict(),
            'context': self.context,
            'source_document': self.source_document,
            'extraction_method': self.extraction_method,
            'tags': self.tags,
            'bps_tags': self.bps_tags
        }


# Utility function for generating unique IDs
def generate_concept_id(prefix: str = "concept") -> str:
    """Generate a unique ID for a concept"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
