"""eigenfunction_labeler.py - Implements semantic labeling for eigenfunctions.

This module provides mechanisms to label eigenfunctions with human-interpretable names
based on the concepts that share those eigenfunctions. It transforms spectral IDs into
meaningful semantic labels, making the system's internal representations more transparent.
"""

import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
import re
import logging
try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple

# Configure logger
logger = logging.getLogger("eigenfunction_labeler")

class EigenfunctionLabeler:
    """Labels eigenfunctions based on concept clusters that share them."""
    
    def __init__(self, concept_store=None):
        self.concept_store = concept_store  # Storage containing concepts
        self.eigenfunction_labels: Dict[str, str] = {}  # Maps eigenfunction_id to label
        self.eigenfunction_concepts: Dict[str, List[str]] = {}  # Maps eigenfunction_id to concept names
        self.confidence_scores: Dict[str, float] = {}  # Confidence in label assignments
        
    def set_concepts(self, concepts: List[ConceptTuple]) -> None:
        """Set the concepts to use for labeling."""
        self.concepts = concepts
        
    def extract_common_terms(self, names: List[str], min_freq: int = 2) -> List[str]:
        """Extract common terms from a list of concept names."""
        # Tokenize all names
        all_terms = []
        for name in names:
            # Extract words, filtering out very short ones
            terms = [term.lower() for term in re.findall(r'\b[a-zA-Z]{3,}\b', name)]
            all_terms.extend(terms)
            
        # Count term frequency
        term_counts = Counter(all_terms)
        
        # Filter by minimum frequency
        common_terms = [term for term, count in term_counts.items() 
                       if count >= min_freq]
        
        # Sort by frequency (most common first)
        common_terms.sort(key=lambda term: term_counts[term], reverse=True)
        
        return common_terms[:5]  # Return top 5 common terms
    
    def find_central_concept(self, concepts: List[ConceptTuple]) -> Optional[ConceptTuple]:
        """Find the most central concept in a cluster based on resonance and centrality."""
        if not concepts:
            return None
            
        # Score concepts by resonance and centrality
        scored_concepts = []
        for concept in concepts:
            # Combined score favoring both resonance and centrality
            score = (0.6 * concept.resonance_score + 
                     0.4 * concept.narrative_centrality)
            scored_concepts.append((concept, score))
            
        # Return highest scoring concept
        return max(scored_concepts, key=lambda x: x[1])[0]
        
    def label_eigenfunctions(self) -> Dict[str, str]:
        """
        Generate human-interpretable labels for eigenfunctions based on concept clusters.
        
        This method:
        1. Groups concepts by eigenfunction_id
        2. Extracts common terms from concepts sharing an eigenfunction
        3. Identifies the central concept in each eigenfunction
        4. Generates descriptive labels combining common terms and central concept
        
        Returns:
            Dictionary mapping eigenfunction IDs to human-readable labels
        """
        if not hasattr(self, 'concepts') or not self.concepts:
            logger.warning("No concepts available for labeling eigenfunctions")
            return {}
            
        # Reset current labels
        self.eigenfunction_labels = {}
        self.eigenfunction_concepts = {}
        self.confidence_scores = {}
        
        # Group concepts by eigenfunction_id
        eigen_concepts = defaultdict(list)
        for concept in self.concepts:
            if not concept.eigenfunction_id:
                continue
            eigen_concepts[concept.eigenfunction_id].append(concept)
            
            # Also track concept names by eigenfunction
            if concept.eigenfunction_id not in self.eigenfunction_concepts:
                self.eigenfunction_concepts[concept.eigenfunction_id] = []
            self.eigenfunction_concepts[concept.eigenfunction_id].append(concept.name)
            
        # Process each eigenfunction with sufficient concepts
        logger.info(f"Labeling {len(eigen_concepts)} eigenfunctions")
        for eigen_id, concepts in eigen_concepts.items():
            # Skip eigenfunctions with too few concepts
            if len(concepts) < 2:
                continue
                
            # Extract common terms from concept names
            concept_names = [c.name for c in concepts]
            common_terms = self.extract_common_terms(concept_names)
            
            # Find central concept in cluster
            central = self.find_central_concept(concepts)
            
            # Skip if no central concept found
            if not central:
                continue
                
            # Confidence based on cluster size and term frequency
            confidence = min(1.0, 0.5 + 0.1 * len(concepts))
                
            # Generate label
            if common_terms:
                # Combine common terms with central concept name
                label = f"{' '.join(common_terms[:3]).title()} [{central.name}]"
            else:
                # Fall back to central concept name
                label = f"Eigen [{central.name}]"
                
            self.eigenfunction_labels[eigen_id] = label
            self.confidence_scores[eigen_id] = confidence
            
            logger.info(f"Labeled eigenfunction {eigen_id} as '{label}' (confidence: {confidence:.2f})")
            
        return self.eigenfunction_labels
        
    def get_label(self, eigenfunction_id: str) -> Optional[str]:
        """Get the label for a specific eigenfunction."""
        return self.eigenfunction_labels.get(eigenfunction_id)
        
    def get_concepts_for_eigenfunction(self, eigenfunction_id: str) -> List[str]:
        """Get the concepts that share a specific eigenfunction."""
        return self.eigenfunction_concepts.get(eigenfunction_id, [])
        
    def get_confidence(self, eigenfunction_id: str) -> float:
        """Get the confidence score for a specific eigenfunction label."""
        return self.confidence_scores.get(eigenfunction_id, 0.0)
        
    def export_labels(self) -> Dict[str, Dict[str, Any]]:
        """Export all eigenfunction labels with metadata."""
        result = {}
        for eigen_id, label in self.eigenfunction_labels.items():
            result[eigen_id] = {
                "label": label,
                "confidence": self.confidence_scores.get(eigen_id, 0.0),
                "concepts": self.eigenfunction_concepts.get(eigen_id, []),
                "concept_count": len(self.eigenfunction_concepts.get(eigen_id, []))
            }
        return result
        
    def analyze_eigenspace(self) -> Dict[str, Any]:
        """Analyze the overall eigenspace structure."""
        if not self.eigenfunction_labels:
            return {}
            
        # Count concepts per eigenfunction
        concept_counts = [len(concepts) for concepts in self.eigenfunction_concepts.values()]
        
        # Calculate statistics
        avg_concepts = sum(concept_counts) / len(concept_counts) if concept_counts else 0
        max_concepts = max(concept_counts) if concept_counts else 0
        
        # Identify "hub" eigenfunctions (those with many concepts)
        threshold = max(3, avg_concepts * 1.5)
        hubs = [eigen_id for eigen_id, concepts in self.eigenfunction_concepts.items()
               if len(concepts) >= threshold]
        
        return {
            "eigenfunction_count": len(self.eigenfunction_labels),
            "labeled_count": len(self.eigenfunction_labels),
            "avg_concepts_per_eigenfunction": avg_concepts,
            "max_concepts_per_eigenfunction": max_concepts,
            "hub_eigenfunctions": hubs,
            "hub_count": len(hubs)
        }

def label_concept_eigenfunctions(concepts: List[ConceptTuple]) -> Tuple[Dict[str, str], List[ConceptTuple]]:
    """
    Convenience function to label eigenfunctions for a list of concepts.
    
    Args:
        concepts: List of ConceptTuple objects
        
    Returns:
        Tuple of (eigenfunction_labels, updated_concepts)
    """
    # Create labeler and label eigenfunctions
    labeler = EigenfunctionLabeler()
    labeler.set_concepts(concepts)
    labels = labeler.label_eigenfunctions()
    
    # Update concepts with eigenfunction labels
    updated_concepts = []
    for concept in concepts:
        if concept.eigenfunction_id in labels:
            # Add label to source provenance
            if "eigenfunction_label" not in concept.source_provenance:
                concept.source_provenance["eigenfunction_label"] = labels[concept.eigenfunction_id]
                concept.source_provenance["eigenfunction_confidence"] = labeler.get_confidence(concept.eigenfunction_id)
        updated_concepts.append(concept)
        
    return labels, updated_concepts
