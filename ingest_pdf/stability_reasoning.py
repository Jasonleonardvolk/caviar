"""stability_reasoning.py - Implements stability-aware concept reasoning.

This module provides mechanisms for stability-aware reasoning with concepts,
ensuring that ALAN's inferences are grounded in stable concepts and avoid
blind reasoning. It:

1. Checks concept stability before using in logical chains
2. Provides alternative concept routing when stability is low
3. Assigns confidence scores to inferences based on concept stability
4. Detects and flags phase decoherence in reasoning paths

This system supports ALAN's "No Blind Inference" commitment by ensuring that
all reasoning emerges from phase-stable activations and meets spectral
confidence thresholds.
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Set, Union, Callable
import math
import numpy as np
from dataclasses import dataclass, field

try:
    # Try absolute import first
    from concept_metadata import ConceptMetadata
except ImportError:
    # Fallback to relative import
    from .concept_metadata import ConceptMetadata
try:
    # Try absolute import first
    from concept_logger import ConceptLogger, default_concept_logger
except ImportError:
    # Fallback to relative import
    from .concept_logger import ConceptLogger, default_concept_logger
try:
    # Try absolute import first
    from time_context import TimeContext, default_time_context
except ImportError:
    # Fallback to relative import
    from .time_context import TimeContext, default_time_context
try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple

# Configure logger
logger = logging.getLogger("stability_reasoning")

@dataclass
class ReasoningPath:
    """
    Represents a chain of concepts used in reasoning.
    
    A ReasoningPath tracks concepts used in an inference chain, including
    stability metrics, alternative paths, and confidence scoring.
    
    Attributes:
        concepts: List of concepts in reasoning path
        path_confidence: Overall confidence in reasoning path (0.0-1.0)
        alternatives: List of alternative concepts for low-stability nodes
        stability_warnings: List of stability warnings for concepts
        chain_breaks: Points where reasoning coherence is broken
    """
    
    concepts: List[Tuple[str, float]] = field(default_factory=list)  # (concept_id, confidence)
    path_confidence: float = 1.0
    alternatives: Dict[int, List[Tuple[str, float]]] = field(default_factory=dict)  # position -> [(concept_id, confidence)]
    stability_warnings: List[Tuple[int, str, float]] = field(default_factory=list)  # (position, concept_id, stability)
    chain_breaks: List[int] = field(default_factory=list)  # positions of coherence breaks
    
    def add_concept(self, concept_id: str, confidence: float) -> None:
        """
        Add a concept to the reasoning path.
        
        Args:
            concept_id: ID of the concept to add
            confidence: Confidence score for this concept (0.0-1.0)
        """
        self.concepts.append((concept_id, confidence))
        
        # Update overall path confidence (multiplicative)
        self.path_confidence *= confidence
        
    def add_alternative(self, position: int, concept_id: str, confidence: float) -> None:
        """
        Add an alternative concept for a specific position.
        
        Args:
            position: Position in the path (0-based index)
            concept_id: ID of the alternative concept
            confidence: Confidence score for this alternative
        """
        if position not in self.alternatives:
            self.alternatives[position] = []
            
        self.alternatives[position].append((concept_id, confidence))
        
    def add_stability_warning(self, position: int, concept_id: str, stability: float) -> None:
        """
        Add a stability warning for a concept.
        
        Args:
            position: Position in the path (0-based index)
            concept_id: ID of the unstable concept
            stability: Current stability value
        """
        self.stability_warnings.append((position, concept_id, stability))
        
    def mark_chain_break(self, position: int) -> None:
        """
        Mark a position as a coherence break in the reasoning chain.
        
        Args:
            position: Position where coherence is broken
        """
        if position not in self.chain_breaks:
            self.chain_breaks.append(position)
            
    def get_strongest_path(self) -> List[str]:
        """
        Get the strongest complete reasoning path.
        
        This may use alternatives for low-confidence nodes if they
        increase overall path confidence.
        
        Returns:
            List of concept IDs forming the strongest path
        """
        # Start with the original path
        strongest_path = [concept_id for concept_id, _ in self.concepts]
        
        # Check if alternatives improve confidence
        for position, alts in self.alternatives.items():
            if position >= len(self.concepts):
                continue
                
            original_confidence = self.concepts[position][1]
            
            # Find best alternative if any
            best_alt = None
            best_confidence = original_confidence
            
            for alt_id, alt_confidence in alts:
                if alt_confidence > best_confidence:
                    best_alt = alt_id
                    best_confidence = alt_confidence
                    
            # Replace with best alternative if found
            if best_alt is not None:
                strongest_path[position] = best_alt
                
        return strongest_path
        
    def get_confidence_profile(self) -> Dict[str, Any]:
        """
        Get a detailed confidence profile for this reasoning path.
        
        Returns:
            Dict with confidence metrics and warning information
        """
        # Calculate normalized confidence (accounting for path length)
        path_length = len(self.concepts)
        normalized_confidence = self.path_confidence ** (1.0 / max(1, path_length)) if path_length > 0 else 0.0
        
        # Find weakest link (lowest confidence concept)
        weakest_confidence = 1.0
        weakest_position = -1
        
        for i, (concept_id, confidence) in enumerate(self.concepts):
            if confidence < weakest_confidence:
                weakest_confidence = confidence
                weakest_position = i
                
        return {
            "path_confidence": self.path_confidence,
            "normalized_confidence": normalized_confidence,
            "path_length": path_length,
            "weakest_link": weakest_position,
            "weakest_confidence": weakest_confidence,
            "stability_warnings": len(self.stability_warnings),
            "chain_breaks": len(self.chain_breaks),
            "has_alternatives": len(self.alternatives) > 0
        }


class StabilityReasoning:
    """
    Stability-aware concept reasoning system.
    
    StabilityReasoning ensures that ALAN's inferences are grounded in stable
    concepts and maintains confidence metrics throughout reasoning chains.
    
    Attributes:
        concept_store: Dictionary mapping concept IDs to metadata
        time_context: TimeContext for temporal references
        logger: ConceptLogger for event logging
        stability_threshold: Minimum stability threshold for confident reasoning
        coherence_threshold: Minimum phase coherence for reasoning paths
    """
    
    def __init__(
        self,
        concept_store: Optional[Dict[str, Union[ConceptTuple, ConceptMetadata]]] = None,
        time_context: Optional[TimeContext] = None,
        logger: Optional[ConceptLogger] = None,
        stability_threshold: float = 0.4,
        coherence_threshold: float = 0.6
    ):
        """
        Initialize the StabilityReasoning system.
        
        Args:
            concept_store: Map of concept IDs to metadata/concepts
            time_context: TimeContext for temporal references
            logger: ConceptLogger for event logging
            stability_threshold: Minimum stability for confident reasoning
            coherence_threshold: Minimum phase coherence for reasoning
        """
        self.concept_store = concept_store or {}
        self.time_context = time_context or default_time_context
        self.logger = logger or default_concept_logger
        self.stability_threshold = stability_threshold
        self.coherence_threshold = coherence_threshold
        
        # Track recent desync events by concept
        self.desync_events: Dict[str, List[Tuple[float, float]]] = {}  # concept_id -> [(timestamp, coherence)]
        
        # Cache for concept similarity to find alternatives
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
    def get_concept_stability(self, concept_id: str) -> float:
        """
        Get current stability score for a concept.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Stability score (0.0-1.0) or 0.0 if concept not found
        """
        # Check if we have ConceptMetadata
        if concept_id in self.concept_store:
            concept = self.concept_store[concept_id]
            
            # ConceptMetadata
            if hasattr(concept, 'stability'):
                return concept.stability
                
            # ConceptTuple with source_provenance containing stability
            elif hasattr(concept, 'source_provenance') and 'stability' in concept.source_provenance:
                return float(concept.source_provenance['stability'])
                
            # Check resonance score as fallback
            elif hasattr(concept, 'resonance_score'):
                return concept.resonance_score
                
        # Concept not found or no stability information
        return 0.0
        
    def get_concept_coherence(self, concept_id: str) -> float:
        """
        Get current phase coherence for a concept.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Phase coherence (0.0-1.0) or 0.0 if not available
        """
        # Check if we have ConceptMetadata
        if concept_id in self.concept_store:
            concept = self.concept_store[concept_id]
            
            # ConceptMetadata
            if hasattr(concept, 'phase_coherence'):
                return concept.phase_coherence
                
            # ConceptTuple with coherence in source_provenance
            elif hasattr(concept, 'source_provenance') and 'phase_coherence' in concept.source_provenance:
                return float(concept.source_provenance['phase_coherence'])
                
            # Use cluster_coherence as fallback
            elif hasattr(concept, 'cluster_coherence'):
                return concept.cluster_coherence
                
        # Concept not found or no coherence information
        return 0.0
        
    def has_recent_desync(self, concept_id: str, window_hours: float = 24.0) -> bool:
        """
        Check if a concept has experienced recent phase desynchronization.
        
        Args:
            concept_id: ID of the concept
            window_hours: Time window to consider
            
        Returns:
            True if recent desync events detected
        """
        if concept_id not in self.desync_events:
            return False
            
        # Calculate time window
        now = time.time()
        window_start = now - (window_hours * 3600)
        
        # Check for desync events in window
        return any(ts >= window_start for ts, _ in self.desync_events[concept_id])
        
    def record_desync_event(self, concept_id: str, coherence: float) -> None:
        """
        Record a phase desynchronization event for a concept.
        
        Args:
            concept_id: ID of the desynchronized concept
            coherence: Coherence measure at time of desync
        """
        if concept_id not in self.desync_events:
            self.desync_events[concept_id] = []
            
        self.desync_events[concept_id].append((time.time(), coherence))
        
        # Keep only recent events (last week)
        week_ago = time.time() - (7 * 24 * 3600)
        self.desync_events[concept_id] = [
            (ts, coh) for ts, coh in self.desync_events[concept_id]
            if ts >= week_ago
        ]
        
    def get_concept_confidence(self, concept_id: str) -> float:
        """
        Calculate confidence score for using a concept in reasoning.
        
        The confidence score combines stability, coherence, and desync history.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Get basic metrics
        stability = self.get_concept_stability(concept_id)
        coherence = self.get_concept_coherence(concept_id)
        
        # Check for desync events
        desync_penalty = 0.0
        if self.has_recent_desync(concept_id):
            # Calculate recency-weighted penalty
            desync_events = self.desync_events.get(concept_id, [])
            if desync_events:
                now = time.time()
                max_penalty = 0.3  # Maximum penalty for very recent desync
                
                # Find most recent event and its age in hours
                most_recent = max(desync_events, key=lambda x: x[0])
                hours_ago = (now - most_recent[0]) / 3600
                
                # Exponential decay of penalty based on recency
                # Fresh events have higher penalty
                desync_penalty = max_penalty * math.exp(-hours_ago / 24)
                
        # Combine factors for final confidence score
        # Weight stability more heavily than coherence
        base_confidence = 0.7 * stability + 0.3 * coherence
        
        # Apply desync penalty
        confidence = max(0.0, base_confidence - desync_penalty)
        
        return confidence
        
    def find_alternative_concepts(
        self,
        concept_id: str,
        max_alternatives: int = 3,
        min_similarity: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find alternative concepts for a low-stability concept.
        
        Args:
            concept_id: ID of the concept to find alternatives for
            max_alternatives: Maximum number of alternatives to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (concept_id, confidence) pairs for alternatives
        """
        alternatives = []
        
        # Concept must exist in store
        if concept_id not in self.concept_store:
            return alternatives
            
        target_concept = self.concept_store[concept_id]
        
        # Find similar concepts with better stability
        for cid, concept in self.concept_store.items():
            # Skip self
            if cid == concept_id:
                continue
                
            # Calculate similarity if not cached
            cache_key = (min(concept_id, cid), max(concept_id, cid))
            if cache_key not in self.similarity_cache:
                similarity = self._calculate_concept_similarity(target_concept, concept)
                self.similarity_cache[cache_key] = similarity
            else:
                similarity = self.similarity_cache[cache_key]
                
            # Skip if not similar enough
            if similarity < min_similarity:
                continue
                
            # Get confidence for this alternative
            confidence = self.get_concept_confidence(cid)
            
            # Only consider if better than original
            original_confidence = self.get_concept_confidence(concept_id)
            if confidence > original_confidence:
                alternatives.append((cid, confidence))
                
        # Sort by confidence and limit
        alternatives.sort(key=lambda x: x[1], reverse=True)
        return alternatives[:max_alternatives]
        
    def _calculate_concept_similarity(
        self,
        concept1: Union[ConceptTuple, ConceptMetadata],
        concept2: Union[ConceptTuple, ConceptMetadata]
    ) -> float:
        """
        Calculate similarity between two concepts.
        
        Args:
            concept1, concept2: Concepts to compare
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Check if we have ConceptTuple with embeddings
        if (hasattr(concept1, 'embedding') and 
            hasattr(concept2, 'embedding') and 
            concept1.embedding is not None and 
            concept2.embedding is not None):
            
            # Cosine similarity between embeddings
            c1 = concept1.embedding / np.linalg.norm(concept1.embedding)
            c2 = concept2.embedding / np.linalg.norm(concept2.embedding)
            return float(np.dot(c1, c2))
            
        # Check if we have adjacent concepts in metadata
        elif (hasattr(concept1, 'adjacent_concepts') and
              hasattr(concept2, 'adjacent_concepts')):
            
            c1_id = concept1.ψ_id if hasattr(concept1, 'ψ_id') else getattr(concept1, 'eigenfunction_id', '')
            c2_id = concept2.ψ_id if hasattr(concept2, 'ψ_id') else getattr(concept2, 'eigenfunction_id', '')
            
            # Check if in each other's adjacency list
            if c1_id in concept2.adjacent_concepts:
                return concept2.adjacent_concepts[c1_id]
            elif c2_id in concept1.adjacent_concepts:
                return concept1.adjacent_concepts[c2_id]
                
        # Fallback to tag/domain similarity for metadata
        if hasattr(concept1, 'tags') and hasattr(concept2, 'tags'):
            # Jaccard similarity of tags
            c1_tags = concept1.tags
            c2_tags = concept2.tags
            
            if c1_tags and c2_tags:
                intersection = len(c1_tags.intersection(c2_tags))
                union = len(c1_tags.union(c2_tags))
                return intersection / union if union > 0 else 0.0
                
        # Fallback to domain similarity
        if (hasattr(concept1, 'domain') and hasattr(concept2, 'domain') and
            concept1.domain and concept2.domain):
            
            return 1.0 if concept1.domain == concept2.domain else 0.2
            
        # Check source provenance domains as last resort
        if (hasattr(concept1, 'source_provenance') and 
            hasattr(concept2, 'source_provenance')):
            
            domain1 = concept1.source_provenance.get('domain', '')
            domain2 = concept2.source_provenance.get('domain', '')
            
            if domain1 and domain2:
                return 1.0 if domain1 == domain2 else 0.2
        
        # No similarity data available
        return 0.0
        
    def verify_concept_stability(
        self,
        concept_id: str,
        find_alternatives: bool = True
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Verify stability of a concept for reasoning use.
        
        Args:
            concept_id: ID of the concept to verify
            find_alternatives: Whether to find alternatives for low stability
            
        Returns:
            Tuple of (confidence, alternatives)
        """
        # Get confidence score
        confidence = self.get_concept_confidence(concept_id)
        
        # Find alternatives if confidence is low
        alternatives = []
        if confidence < self.stability_threshold and find_alternatives:
            alternatives = self.find_alternative_concepts(concept_id)
            
            # Log stability warning
            concept_name = None
            if concept_id in self.concept_store:
                concept = self.concept_store[concept_id]
                if hasattr(concept, 'name'):
                    concept_name = concept.name
                    
            self.logger.log_error(
                operation="stability_verification",
                error=f"Low stability concept used in reasoning: {confidence:.2f}",
                concept_id=concept_id,
                severity="WARNING"
            )
            
        return confidence, alternatives
        
    def create_reasoning_path(
        self,
        concept_chain: List[str],
        verify_all: bool = True
    ) -> ReasoningPath:
        """
        Create and verify a reasoning path from a chain of concepts.
        
        Args:
            concept_chain: List of concept IDs in reasoning chain
            verify_all: Whether to verify all concepts in the chain
            
        Returns:
            ReasoningPath with confidence and alternatives
        """
        path = ReasoningPath()
        
        for i, concept_id in enumerate(concept_chain):
            # Verify concept stability
            confidence, alternatives = self.verify_concept_stability(concept_id)
            
            # Add to reasoning path
            path.add_concept(concept_id, confidence)
            
            # Add stability warning if needed
            stability = self.get_concept_stability(concept_id)
            if stability < self.stability_threshold:
                path.add_stability_warning(i, concept_id, stability)
                
            # Add alternatives
            for alt_id, alt_confidence in alternatives:
                path.add_alternative(i, alt_id, alt_confidence)
                
            # Check coherence with previous concept
            if i > 0:
                prev_id = concept_chain[i-1]
                coherence = self._check_concept_coherence(prev_id, concept_id)
                
                if coherence < self.coherence_threshold:
                    path.mark_chain_break(i)
                    
                    # Log coherence break
                    self.logger.log_phase_alert(
                        concept_ids=[prev_id, concept_id],
                        coherence=coherence,
                        event="reasoning_chain_break",
                        concept_names=self._get_concept_names([prev_id, concept_id])
                    )
        
        return path
        
    def _check_concept_coherence(self, concept_id1: str, concept_id2: str) -> float:
        """
        Check phase coherence between two concepts.
        
        Args:
            concept_id1, concept_id2: IDs of concepts to check
            
        Returns:
            Phase coherence (0.0-1.0) between concepts
        """
        # Check cached similarity first
        cache_key = (min(concept_id1, concept_id2), max(concept_id1, concept_id2))
        if cache_key in self.similarity_cache:
            # Use similarity as proxy for coherence if no better measure available
            return self.similarity_cache[cache_key]
            
        # Get concepts
        if concept_id1 in self.concept_store and concept_id2 in self.concept_store:
            concept1 = self.concept_store[concept_id1]
            concept2 = self.concept_store[concept_id2]
            
            # Calculate similarity and use as proxy for coherence
            coherence = self._calculate_concept_similarity(concept1, concept2)
            self.similarity_cache[cache_key] = coherence
            
            return coherence
            
        return 0.0
        
    def _get_concept_names(self, concept_ids: List[str]) -> List[str]:
        """Get names for a list of concept IDs."""
        names = []
        for cid in concept_ids:
            if cid in self.concept_store:
                concept = self.concept_store[cid]
                if hasattr(concept, 'name'):
                    names.append(concept.name)
                else:
                    names.append(cid)
            else:
                names.append(cid)
        return names
        
    def evaluate_inference(
        self,
        premises: List[str],
        conclusion: str
    ) -> Dict[str, Any]:
        """
        Evaluate confidence in an inference based on stability.
        
        Args:
            premises: List of premise concept IDs
            conclusion: Conclusion concept ID
            
        Returns:
            Dict with confidence metrics and warnings
        """
        # Create reasoning path from premises
        premise_path = self.create_reasoning_path(premises)
        
        # Verify conclusion
        conclusion_confidence, alternatives = self.verify_concept_stability(conclusion)
        
        # Create full path including conclusion
        full_chain = premises + [conclusion]
        full_path = self.create_reasoning_path(full_chain)
        
        # Calculate inference confidence
        # Weighted by premise path confidence and conclusion confidence
        if premise_path.path_confidence > 0:
            inference_confidence = 0.7 * premise_path.path_confidence + 0.3 * conclusion_confidence
        else:
            inference_confidence = conclusion_confidence
            
        # Evaluate coherence of conclusion with premises
        coherence_with_premises = 0.0
        if premises:
            # Average coherence with all premises
            coherences = [
                self._check_concept_coherence(premise_id, conclusion)
                for premise_id in premises
            ]
            coherence_with_premises = sum(coherences) / len(coherences)
            
        # Prepare result
        result = {
            "inference_confidence": inference_confidence,
            "premise_confidence": premise_path.path_confidence,
            "conclusion_confidence": conclusion_confidence,
            "coherence_with_premises": coherence_with_premises,
            "chain_breaks": len(full_path.chain_breaks),
            "stability_warnings": len(full_path.stability_warnings),
            "conclusion_alternatives": alternatives,
            "reasoning_profile": full_path.get_confidence_profile()
        }
        
        # Categorize inference quality
        if inference_confidence >= 0.8:
            result["quality"] = "high"
        elif inference_confidence >= 0.5:
            result["quality"] = "medium"
        else:
            result["quality"] = "low"
            
        # Log significant inferences
        self._log_inference_evaluation(premises, conclusion, result)
            
        return result
        
    def _log_inference_evaluation(
        self,
        premises: List[str],
        conclusion: str,
        result: Dict[str, Any]
    ) -> None:
        """Log inference evaluation results."""
        # Only log interesting cases (very high or low confidence)
        if result["inference_confidence"] >= 0.9 or result["inference_confidence"] <= 0.3:
            # Get names for log readability
            premise_names = self._get_concept_names(premises)
            conclusion_name = self._get_concept_names([conclusion])[0]
            
            confidence = result["inference_confidence"]
            quality = result["quality"]
            
            # Log as phase event
            all_concepts = premises + [conclusion]
            self.logger.log_phase_alert(
                concept_ids=all_concepts,
                coherence=confidence,  # Use confidence as coherence proxy
                event=f"inference_{quality}_confidence",
                concept_names=self._get_concept_names(all_concepts)
            )

# For convenience, create a singleton instance
default_stability_reasoning = StabilityReasoning()
