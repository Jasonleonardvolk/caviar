"""concept_metadata.py - Implements standardized metadata for concept traceability.

This module defines the ConceptMetadata class which acts as the semantic spine
for ALAN's concept management, providing:

1. Standardized provenance tracking
2. Age and stability metrics
3. Activation history 
4. Phase coherence measurements
5. Temporal integration with TimeContext

Together with TimeContext, ConceptMetadata forms the foundation for
ALAN's memory sculpting capabilities, enabling concepts to function as
temporally-anchored ψ-resonance agents with full traceability.
"""

import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
import json
import logging

try:
    # Try absolute import first
    from time_context import TimeContext, default_time_context
except ImportError:
    # Fallback to relative import
    from .time_context import TimeContext, default_time_context

# Configure logger
logger = logging.getLogger("concept_metadata")

@dataclass
class ConceptMetadata:
    """
    Standardized metadata for concept tracking and provenance.
    
    ConceptMetadata provides a complete record of a concept's origin,
    history, activation patterns, and stability metrics. This enables
    advanced memory management, concept resonance tracking, and
    provenance-aware operations.
    
    Attributes:
        ψ_id: Eigenfunction ID (unique spectral identifier)
        source_hash: Content hash for source validation
        ingestion_ts: Timestamp of concept ingestion
        provenance: Filename or semantic ID of source
        stability: Time-integrated resonance score
        merge_lineage: IDs of concepts merged to form this one
        phase_coherence: Phase alignment with related concepts
        last_activation: When concept was last accessed
        activation_count: How often concept has been accessed
        activation_history: Timestamped record of activations
        tags: Set of semantic tags applied to this concept
        spatial_position: Optional position in concept space
    """
    
    ψ_id: str  # Eigenfunction ID
    source_hash: str = ""  # Content hash for source validation
    ingestion_ts: float = field(default_factory=time.time)  # Timestamp of ingestion
    provenance: str = ""  # Filename or semantic ID
    stability: float = 1.0  # Time-integrated resonance
    merge_lineage: List[str] = field(default_factory=list)  # IDs of merged concepts
    phase_coherence: float = 0.0  # Phase alignment with related concepts
    last_activation: float = 0.0  # When concept was last accessed/activated
    activation_count: int = 0  # How often concept has been accessed
    activation_history: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, strength) pairs
    tags: Set[str] = field(default_factory=set)  # Semantic tags
    spatial_position: Optional[Tuple[float, float, float]] = None  # 3D position in concept space
    
    # Additional contextual information
    domain: str = ""  # Knowledge domain classification
    complexity: float = 0.5  # Concept complexity score
    adjacent_concepts: Dict[str, float] = field(default_factory=dict)  # ID-to-similarity mapping
    
    def __post_init__(self) -> None:
        """Initialize timestamp fields if not already set."""
        if not self.last_activation:
            self.last_activation = self.ingestion_ts
    
    def update_stability(self, resonance: float, time_factor: float) -> None:
        """
        Update stability score based on current resonance and time factor.
        
        Stability is a time-integrated measure of concept relevance that
        combines historical stability with current resonance.
        
        Args:
            resonance: Current resonance measurement (0.0-1.0)
            time_factor: Weight for new measurement vs. history (0.0-1.0)
        """
        old_stability = self.stability
        self.stability = self.stability * (1 - time_factor) + resonance * time_factor
        
        # Log significant changes
        if abs(self.stability - old_stability) > 0.2:
            direction = "↑" if self.stability > old_stability else "↓"
            logger.info(f"Concept stability change: {self.ψ_id} | {old_stability:.2f} → {self.stability:.2f} {direction}")
    
    def record_activation(self, strength: float = 1.0, timestamp: Optional[float] = None) -> None:
        """
        Record concept activation with optional strength parameter.
        
        Args:
            strength: Activation strength, typically 0.0-1.0
            timestamp: Optional explicit timestamp (default: current time)
        """
        ts = timestamp or time.time()
        self.last_activation = ts
        self.activation_count += 1
        
        # Keep activation history with recent bias (max 100 entries)
        self.activation_history.append((ts, strength))
        if len(self.activation_history) > 100:
            # Remove oldest entries when over limit
            self.activation_history = sorted(self.activation_history, key=lambda x: x[0])[-100:]
    
    def age(self, time_ctx: Optional[TimeContext] = None) -> float:
        """
        Calculate concept age in seconds using TimeContext.
        
        Args:
            time_ctx: TimeContext instance (uses default if None)
            
        Returns:
            float: Age in seconds, adjusted by TimeContext clock rate
        """
        ctx = time_ctx or default_time_context
        return ctx.get_time_since(self.ingestion_ts)
    
    def age_factor(self, time_ctx: Optional[TimeContext] = None, scale_factor: float = 1.0) -> float:
        """
        Calculate normalized age factor from 0.0 (new) to 1.0 (old).
        
        Args:
            time_ctx: TimeContext instance (uses default if None)
            scale_factor: Controls aging rate (lower = slower aging)
            
        Returns:
            float: Age factor between 0.0 (new) and 1.0 (old)
        """
        ctx = time_ctx or default_time_context
        return ctx.get_age_factor(self.ingestion_ts, scale_factor)
    
    def time_since_activation(self, time_ctx: Optional[TimeContext] = None) -> float:
        """
        Calculate time elapsed since last activation.
        
        Args:
            time_ctx: TimeContext instance (uses default if None)
            
        Returns:
            float: Seconds since last activation, adjusted by clock rate
        """
        ctx = time_ctx or default_time_context
        return ctx.get_time_since(self.last_activation)
    
    def activation_frequency(self, time_window_hours: float = 24.0) -> float:
        """
        Calculate activation frequency within a specified time window.
        
        Args:
            time_window_hours: Time window to consider, in hours
            
        Returns:
            float: Activations per hour within the specified window
        """
        if not self.activation_history:
            return 0.0
            
        now = time.time()
        window_start = now - (time_window_hours * 3600)
        
        # Count activations within window
        recent_activations = sum(1 for ts, _ in self.activation_history if ts >= window_start)
        
        # Calculate hours since first activation or window start
        earliest_ts = min(ts for ts, _ in self.activation_history)
        hours_span = (now - max(earliest_ts, window_start)) / 3600
        
        # Avoid division by zero
        if hours_span < 0.001:
            return 0.0
            
        return recent_activations / hours_span
    
    def add_to_lineage(self, concept_id: str) -> None:
        """
        Add a concept ID to this concept's merge lineage.
        
        Args:
            concept_id: ID of concept to add to lineage
        """
        if concept_id not in self.merge_lineage:
            self.merge_lineage.append(concept_id)
    
    def add_tag(self, tag: str) -> None:
        """
        Add a semantic tag to this concept.
        
        Args:
            tag: Semantic tag to add
        """
        tag = tag.strip().lower()
        if tag:
            self.tags.add(tag)
    
    def update_source_hash(self, content: str) -> str:
        """
        Update source hash based on content.
        
        Args:
            content: Source content to hash
            
        Returns:
            str: Generated hash
        """
        if not content:
            return self.source_hash
            
        self.source_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        return self.source_hash
    
    def set_adjacent_concept(self, concept_id: str, similarity: float) -> None:
        """
        Record adjacent concept with similarity score.
        
        Args:
            concept_id: ID of adjacent concept
            similarity: Similarity score (0.0-1.0)
        """
        self.adjacent_concepts[concept_id] = similarity
    
    def get_activation_recency_score(self) -> float:
        """
        Calculate normalized recency score based on last activation.
        
        Returns:
            float: Recency score (1.0 = very recent, 0.0 = long ago)
        """
        if not self.last_activation:
            return 0.0
            
        # Exponential decay based on time since activation
        elapsed_days = (time.time() - self.last_activation) / 86400  # seconds to days
        recency = math.exp(-elapsed_days / 7)  # 7-day half-life
        
        return max(0.0, min(1.0, recency))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary format for serialization.
        
        Returns:
            Dict: Dictionary representation of metadata
        """
        data = asdict(self)
        # Convert sets to lists for JSON serialization
        data['tags'] = list(data['tags'])
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptMetadata':
        """
        Create ConceptMetadata instance from dictionary.
        
        Args:
            data: Dictionary with metadata fields
            
        Returns:
            ConceptMetadata: New instance with provided data
        """
        # Convert lists back to sets where needed
        if 'tags' in data and isinstance(data['tags'], list):
            data['tags'] = set(data['tags'])
            
        return cls(**data)

import math  # Import needed for recency calculation
