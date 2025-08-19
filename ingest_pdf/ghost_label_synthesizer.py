"""ghost_label_synthesizer.py - Implements conceptual name generation for ALAN.

This module provides mechanisms for generating meaningful names for emergent concepts
based on their spectral fingerprints, cluster properties, and relational context.
It implements techniques for:

1. Analyzing concept embeddings to extract semantic essence
2. Generating expressive concept labels from spectral properties
3. Creating names that capture conceptual relationships
4. Incorporating context from surrounding phase-locked clusters

These capabilities allow ALAN to assign meaningful, interpretable labels to concepts
that emerge from phase-locked clusters and resonant interactions.

References:
- Linguistic composition theory
- Semantic fingerprinting
- Spectral naming patterns
- Phase-coherent ontology engineering
"""

import os
import numpy as np
import logging
import time
import random
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import math
from collections import defaultdict
import uuid

try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple
try:
    # Try absolute import first
    from koopman_phase_graph import get_koopman_phase_graph, ConceptNode
except ImportError:
    # Fallback to relative import
    from .koopman_phase_graph import get_koopman_phase_graph, ConceptNode
try:
    # Try absolute import first
    from memory_sculptor import get_memory_sculptor
except ImportError:
    # Fallback to relative import
    from .memory_sculptor import get_memory_sculptor
try:
    # Try absolute import first
    from ontology_refactor_engine import get_ontology_refactor_engine
except ImportError:
    # Fallback to relative import
    from .ontology_refactor_engine import get_ontology_refactor_engine

# Configure logger
logger = logging.getLogger("alan_ghost_label_synthesizer")

# Linguistic elements for name synthesis
ABSTRACT_PREFIXES = [
    "Meta", "Trans", "Proto", "Quasi", "Neo", "Hyper", "Iso", "Para", "Arch", "Sub",
    "Eigen", "Spectral", "Resonant", "Harmonic", "Dampened", "Phase", "Recursive", 
    "Latent", "Emergent", "Manifold", "Coherent", "Entropic", "Attractor"
]

CONCEPTUAL_ROOTS = [
    "Structure", "Dynamics", "Pattern", "System", "Process", "Theorem", "Principle", 
    "Formation", "Interface", "Spectrum", "Manifold", "Field", "Operator", "Function",
    "Attractor", "Resonance", "Stability", "Convergence", "Nullification", "Amplification"
]

RELATIONAL_TERMS = [
    "Mapping", "Transformation", "Correspondence", "Projection", "Embedding", "Inference",
    "Transduction", "Integration", "Bifurcation", "Transition", "Convergence", "Alignment",
    "Synchronization", "Coupling", "Binding", "Reflection", "Propagation", "Emergence"
]

SUFFIXES = [
    "Theorem", "Principle", "Law", "Gate", "Node", "Path", "Graph", "Cycle", "Field",
    "Vector", "System", "Network", "Domain", "Wave", "Flux", "Space", "Manifold",
    "Mapping", "Operator", "Tensor", "Matrix", "Function", "Flow", "State", "Mode"
]

CONNECTORS = [
    "of", "through", "via", "under", "with", "across", "between", "beyond", 
    "within", "above", "below", "around", "from", "into", "toward", "against"
]

PHASE_TERMS = [
    "Oscillation", "Wave", "Resonance", "Synchrony", "Coherence", "Alignment", 
    "Stability", "Entrainment", "Coupling", "Periodicity", "Harmony", "Rhythm",
    "Cycle", "Interference", "Phase-Lock", "Eigenflow", "Attractor", "Orbit"
]

QUALITY_TERMS = [
    "Stable", "Unstable", "Convergent", "Divergent", "Chaotic", "Ordered", 
    "Complex", "Sparse", "Dense", "Dampened", "Amplified", "Constrained",
    "Bounded", "Invariant", "Harmonic", "Dissonant", "Recursive", "Emergent"
]


class GhostLabelSynthesizer:
    """
    Main class for synthesizing meaningful labels for emergent concepts in ALAN.
    
    This class provides mechanisms for analyzing concept properties and generating
    expressive, interpretable names that capture the essence of concepts.
    """
    
    def __init__(
        self,
        embedding_influence: float = 0.4,  # How much the embedding affects the name
        relations_influence: float = 0.3,  # How much relations affect the name
        phase_influence: float = 0.3,      # How much phase properties affect the name 
        compound_name_ratio: float = 0.6,  # Ratio of compound to simple names
        complex_name_ratio: float = 0.4,   # Ratio of complex to moderate names
        log_dir: str = "logs/ghost_labels"
    ):
        """
        Initialize the ghost label synthesizer.
        
        Args:
            embedding_influence: Weight for embedding-based terms
            relations_influence: Weight for relation-based terms
            phase_influence: Weight for phase-based terms
            compound_name_ratio: Ratio of compound to simple names
            complex_name_ratio: Ratio of complex to moderate names
            log_dir: Directory for logging
        """
        self.embedding_influence = embedding_influence
        self.relations_influence = relations_influence
        self.phase_influence = phase_influence
        self.compound_name_ratio = compound_name_ratio
        self.complex_name_ratio = complex_name_ratio
        self.log_dir = log_dir
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Get required components
        self.koopman_graph = get_koopman_phase_graph()
        self.memory_sculptor = get_memory_sculptor()
        
        # Track name synthesis operations
        self.generated_names: Dict[str, Dict[str, Any]] = {}
        
        # Domain-specific term pools
        self.domain_terms = defaultdict(list)
        self._initialize_domain_terms()
        
        logger.info("Ghost label synthesizer initialized")
        
    def _initialize_domain_terms(self):
        """Initialize domain-specific term pools."""
        # Mathematics
        self.domain_terms["mathematics"] = [
            "Theorem", "Axiom", "Topology", "Algebra", "Analysis", "Geometry",
            "Calculus", "Matrix", "Vector", "Tensor", "Manifold", "Symmetry",
            "Group", "Ring", "Field", "Integral", "Differential", "Invariant",
            "Transform", "Metric", "Norm", "Eigenvector", "Eigenvalue", "Spectrum" 
        ]
        
        # Physics
        self.domain_terms["physics"] = [
            "Energy", "Momentum", "Force", "Field", "Potential", "Charge",
            "Wave", "Particle", "Quantum", "Relativity", "Entropy", "Oscillation",
            "Resonance", "Dampening", "Dissipation", "Symmetry", "Conservation",
            "Radiation", "Emission", "Absorption", "Polarization", "Phase"
        ]
        
        # Computer Science
        self.domain_terms["computer_science"] = [
            "Algorithm", "Complexity", "Recursion", "Graph", "Tree", "Network",
            "Interface", "Protocol", "Cache", "Memory", "Process", "Thread",
            "Compiler", "Interpreter", "Parser", "Hash", "Encryption", "Query",
            "Sorting", "Search", "Optimization", "Parallel", "Distributed", "Cluster"
        ]
        
        # Neuroscience
        self.domain_terms["neuroscience"] = [
            "Neuron", "Synapse", "Receptor", "Axon", "Dendrite", "Cortex",
            "Plasticity", "Potential", "Membrane", "Signal", "Pathway", "Circuit",
            "Network", "Oscillation", "Firing", "Inhibition", "Excitation", "Stimulus",
            "Response", "Processing", "Attention", "Memory", "Learning", "Encoding"
        ]
        
        # Dynamical Systems
        self.domain_terms["dynamical_systems"] = [
            "Attractor", "Bifurcation", "Chaos", "Stability", "Trajectory", "Orbit",
            "Basin", "Manifold", "Fixed-Point", "Limit-Cycle", "Torus", "Resonance",
            "Phase-Space", "Coherence", "Synchronization", "Dissipation", "Ergodic",
            "Recurrence", "Transient", "Emergence", "Complexity", "Self-Organization"
        ]
        
    def _extract_embedding_semantics(
        self,
        embedding: np.ndarray,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract semantic elements from an embedding vector.
        
        Args:
            embedding: The concept embedding vector
            domain: Optional domain for specialized terms
            
        Returns:
            Dictionary with semantic elements
        """
        # For now, we'll use a simple approach of calculating some statistical
        # properties of the embedding and using them to guide term selection
        
        # Calculate basic statistics
        mean = np.mean(embedding)
        std = np.std(embedding)
        max_val = np.max(embedding)
        min_val = np.min(embedding)
        percentile_75 = np.percentile(embedding, 75)
        percentile_25 = np.percentile(embedding, 25)
        
        # Calculate number of significant dimensions (approximation)
        sorted_values = np.sort(np.abs(embedding))[::-1]  # Sort descending
        cumulative = np.cumsum(sorted_values)
        significant_dims = np.sum(cumulative < 0.9 * cumulative[-1])
        
        # Calculate entropy-like measure
        normalized = np.abs(embedding) / np.sum(np.abs(embedding))
        entropy = -np.sum(normalized * np.log(normalized + 1e-10))
        
        # Determine semantic qualities
        is_sparse = significant_dims < len(embedding) * 0.1
        is_balanced = abs(mean) < 0.1 and std > 0.1
        is_skewed = abs(percentile_75 + percentile_25) > 0.2
        is_peaked = max_val > 0.5
        is_flat = std < 0.1
        is_diverse = entropy > 3.0
        
        # Select terms based on embedding properties
        terms = []
        
        if is_sparse:
            terms.extend(["Sparse", "Discrete", "Focused"])
        else:
            terms.extend(["Distributed", "Continuous", "Expansive"])
            
        if is_balanced:
            terms.extend(["Balanced", "Symmetric", "Harmonic"])
        else:
            terms.extend(["Skewed", "Asymmetric", "Biased"])
            
        if is_peaked:
            terms.extend(["Peaked", "Singular", "Concentrated"])
        else:
            terms.extend(["Diffuse", "Distributed", "Gradual"])
            
        if is_flat:
            terms.extend(["Uniform", "Stable", "Regular"])
        else:
            terms.extend(["Varied", "Dynamic", "Irregular"])
            
        if is_diverse:
            terms.extend(["Complex", "Diverse", "Heterogeneous"])
        else:
            terms.extend(["Simple", "Homogeneous", "Streamlined"])
            
        # Add domain-specific terms if available
        if domain and domain in self.domain_terms:
            # Pick a few terms based on embedding properties
            domain_specific = []
            percentile = int((np.mean(embedding) + 1) * 50)  # 0-100 range
            idx = min(len(self.domain_terms[domain]) - 1, 
                      max(0, int(percentile / 100 * len(self.domain_terms[domain]))))
            
            domain_specific.append(self.domain_terms[domain][idx])
            
            # Add a few more based on other stats
            idx2 = min(len(self.domain_terms[domain]) - 1,
                       max(0, int(std * 10) % len(self.domain_terms[domain])))
            domain_specific.append(self.domain_terms[domain][idx2])
            
            terms.extend(domain_specific)
            
        return {
            "statistics": {
                "mean": float(mean),
                "std": float(std),
                "max": float(max_val),
                "min": float(min_val),
                "significant_dims": int(significant_dims),
                "entropy": float(entropy)
            },
            "qualities": {
                "sparse": is_sparse,
                "balanced": is_balanced,
                "skewed": is_skewed,
                "peaked": is_peaked,
                "flat": is_flat,
                "diverse": is_diverse
            },
            "terms": terms
        }
        
    def _analyze_phase_properties(
        self,
        concept_id: str
    ) -> Dict[str, Any]:
        """
        Analyze phase-related properties of a concept.
        
        Args:
            concept_id: ID of the concept to analyze
            
        Returns:
            Dictionary with phase-related characteristics
        """
        # Get concept state from memory sculptor
        phase_terms = []
        phase_stats = {}
        
        # Check if we have state information
        if hasattr(self.memory_sculptor, "concept_states"):
            state = self.memory_sculptor.concept_states.get(concept_id)
            
            if state:
                # Use state properties to select phase-related terms
                
                # Check stability
                if state.stability_score > 0.7:
                    phase_terms.extend(["Stable", "Resilient", "Persistent", "Anchored"])
                elif state.stability_score < 0.4:
                    phase_terms.extend(["Unstable", "Volatile", "Transient", "Drifting"])
                else:
                    phase_terms.extend(["Metastable", "Balanced", "Conditional", "Semi-stable"])
                    
                # Check for phase desynchronizations
                if state.phase_desyncs > 10:
                    phase_terms.extend(["Desynchronizing", "Unstable", "Decoherent", "Dissonant"])
                elif state.phase_desyncs < 3:
                    phase_terms.extend(["Synchronous", "Coherent", "Aligned", "Harmonic"])
                    
                # Check recurrence/resonance patterns
                if state.recurrence_count > 5:
                    phase_terms.extend(["Recurrent", "Cyclical", "Returning", "Periodic"])
                
                if state.resonance_count > 10:
                    phase_terms.extend(["Resonant", "Amplifying", "Coupling", "Sympathetic"])
                elif state.resonance_count < 3:
                    phase_terms.extend(["Isolated", "Independent", "Decoupled", "Autonomous"])
                    
                # Get cluster membership info
                if len(state.cluster_membership) > 2:
                    phase_terms.extend(["Network", "Clustered", "Connected", "Hub"])
                elif len(state.cluster_membership) == 0:
                    phase_terms.extend(["Singular", "Disconnected", "Independent", "Unique"])
                    
                # Record stats
                phase_stats = {
                    "stability": state.stability_score,
                    "desync_count": state.phase_desyncs,
                    "resonance_count": state.resonance_count,
                    "recurrence_count": state.recurrence_count,
                    "cluster_count": len(state.cluster_membership)
                }
        
        # If no state information, use generic phase terms
        if not phase_terms:
            phase_terms = random.sample(PHASE_TERMS, 3)
            
        return {
            "terms": phase_terms,
            "stats": phase_stats
        }
        
    def _analyze_relational_context(
        self,
        concept_id: str,
        max_neighbors: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze the relational context of a concept.
        
        Args:
            concept_id: ID of the concept to analyze
            max_neighbors: Maximum number of neighbors to analyze
            
        Returns:
            Dictionary with relational characteristics
        """
        # Get the concept
        concept = self.koopman_graph.get_concept_by_id(concept_id)
        
        if concept is None:
            return {"terms": random.sample(RELATIONAL_TERMS, 3), "stats": {}}
            
        # Get neighbor concepts
        neighbor_concepts = []
        for neighbor_id, weight in concept.edges[:max_neighbors]:
            neighbor = self.koopman_graph.get_concept_by_id(neighbor_id)
            if neighbor:
                neighbor_concepts.append((neighbor, weight))
                
        # If no neighbors, return generic terms
        if not neighbor_concepts:
            return {"terms": random.sample(RELATIONAL_TERMS, 3), "stats": {}}
            
        # Analyze relationship patterns
        relationship_terms = []
        
        # Calculate average similarity
        similarities = []
        for neighbor, _ in neighbor_concepts:
            similarity = np.dot(concept.embedding, neighbor.embedding) / (
                np.linalg.norm(concept.embedding) * np.linalg.norm(neighbor.embedding)
            )
            similarities.append(similarity)
            
        avg_similarity = sum(similarities) / len(similarities)
        
        # Analyze similarity patterns
        if avg_similarity > 0.8:
            relationship_terms.extend(["Aligned", "Coherent", "Consistent", "Harmonious"])
        elif avg_similarity < 0.5:
            relationship_terms.extend(["Diverse", "Contrasting", "Bridging", "Connecting"])
        else:
            relationship_terms.extend(["Mediating", "Balancing", "Integrating", "Synthesizing"])
            
        # Extract terms from neighbor names
        from collections import Counter
        
        word_counter = Counter()
        for neighbor, _ in neighbor_concepts:
            for word in neighbor.name.split():
                # Filter out very short words
                if len(word) > 3:
                    word_counter[word] += 1
                    
        # Get most common words from neighbors
        common_words = [word for word, count in word_counter.most_common(3) if count > 1]
        
        # If we have common words across neighbors, they may indicate a theme
        if common_words:
            relationship_terms.extend(common_words)
            
        # Check edge weights
        avg_weight = sum(weight for _, weight in neighbor_concepts) / len(neighbor_concepts)
        
        if avg_weight > 0.8:
            relationship_terms.extend(["Strong", "Coupled", "Binding", "Fused"])
        elif avg_weight < 0.5:
            relationship_terms.extend(["Weak", "Loose", "Tentative", "Potential"])
            
        return {
            "terms": relationship_terms,
            "stats": {
                "neighbor_count": len(neighbor_concepts),
                "avg_similarity": avg_similarity,
                "avg_weight": avg_weight,
                "common_words": common_words
            }
        }
        
    def generate_name(
        self,
        concept_id: str,
        domain: Optional[str] = None,
        complexity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a meaningful name for a concept.
        
        Args:
            concept_id: ID of the concept
            domain: Optional domain classification
            complexity: Optional complexity level ('simple', 'moderate', 'complex')
            
        Returns:
            Dictionary with name generation results
        """
        # Get the concept
        concept = self.koopman_graph.get_concept_by_id(concept_id)
        
        if concept is None:
            return {
                "status": "error",
                "message": f"Concept {concept_id} not found"
            }
            
        # Analyze concept from different perspectives
        try:
            # Analyze embedding
            embedding_semantics = self._extract_embedding_semantics(
                concept.embedding, domain
            )
            
            # Analyze phase properties
            phase_properties = self._analyze_phase_properties(concept_id)
            
            # Analyze relational context
            relational_context = self._analyze_relational_context(concept_id)
            
            # Determine name complexity if not specified
            if complexity is None:
                rand_val = random.random()
                if rand_val < self.complex_name_ratio:
                    complexity = "complex"
                elif rand_val < self.complex_name_ratio + (1 - self.complex_name_ratio) / 2:
                    complexity = "moderate"
                else:
                    complexity = "simple"
                    
            # Generate name based on complexity
            name = self._synthesize_name(
                embedding_semantics, 
                phase_properties, 
                relational_context,
                complexity,
                domain
            )
            
            # Record the name generation in history
            generation_record = {
                "name": name,
                "concept_id": concept_id,
                "original_name": concept.name,
                "complexity": complexity,
                "domain": domain,
                "timestamp": time.time(),
                "embedding_semantics": embedding_semantics.get("terms", []),
                "phase_terms": phase_properties.get("terms", []),
                "relational_terms": relational_context.get("terms", [])
            }
            
            self.generated_names[concept_id] = generation_record
            
            logger.info(f"Generated name '{name}' for concept {concept_id}")
            
            return {
                "status": "success",
                "name": name,
                "complexity": complexity,
                "original_name": concept.name,
                "concept_id": concept_id,
                "generation_record": generation_record
            }
                
        except Exception as e:
            logger.error(f"Error generating name for concept {concept_id}: {e}")
            return {
                "status": "error",
                "message": f"Error generating name: {str(e)}"
            }
            
    def _synthesize_name(
        self,
        embedding_semantics: Dict[str, Any],
        phase_properties: Dict[str, Any],
        relational_context: Dict[str, Any],
        complexity: str,
        domain: Optional[str] = None
    ) -> str:
        """
        Synthesize a name from the analyzed concept properties.
        
        Args:
            embedding_semantics: Semantic elements from embedding
            phase_properties: Phase-related properties
            relational_context: Relational context
            complexity: Desired complexity level
            domain: Optional domain classification
            
        Returns:
            Synthesized name
        """
        # Get all available terms
        embedding_terms = embedding_semantics.get("terms", [])
        phase_terms = phase_properties.get("terms", [])
        relational_terms = relational_context.get("terms", [])
        
        # Weight the terms according to influences
        all_terms = []
        all_terms.extend([(term, self.embedding_influence) for term in embedding_terms])
        all_terms.extend([(term, self.phase_influence) for term in phase_terms])
        all_terms.extend([(term, self.relations_influence) for term in relational_terms])
        
        # Sort by weights (descending)
        all_terms.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates, keeping the highest weight
        unique_terms = []
        seen = set()
        for term, weight in all_terms:
            if term.lower() not in seen:
                unique_terms.append((term, weight))
                seen.add(term.lower())
                
        # Generate name based on complexity
        if complexity == "simple":
            return self._generate_simple_name(unique_terms, domain)
        elif complexity == "moderate":
            return self._generate_moderate_name(unique_terms, domain)
        else:  # complex
            return self._generate_complex_name(unique_terms, domain)
            
    def _generate_simple_name(
        self,
        weighted_terms: List[Tuple[str, float]],
        domain: Optional[str] = None
    ) -> str:
        """Generate a simple name using 1-2 terms."""
        # Extract terms
        terms = [term for term, _ in weighted_terms]
        
        # For simple names, prefer domain terms if available
        if domain and domain in self.domain_terms:
            domain_terms = self.domain_terms[domain]
            if domain_terms:
                terms = [*terms, *domain_terms[:3]]
                
        # Simple formats:
        # 1. Quality + Root (e.g., "Coherent Resonance")
        # 2. Domain + Root (e.g., "Matrix Principle")
        
        # Select preferred terms
        quality_options = [t for t in terms if any(t == q for q in QUALITY_TERMS)]
        if not quality_options:
            quality_options = QUALITY_TERMS
            
        root_options = [t for t in terms if any(t == r for r in CONCEPTUAL_ROOTS)]
        if not root_options:
            root_options = CONCEPTUAL_ROOTS
            
        # Generate name
        if random.random() < 0.5:
            # Quality + Root
            quality = random.choice(quality_options)
            root = random.choice(root_options)
            return f"{quality} {root}"
        else:
            # Domain + Root
            if domain and domain in self.domain_terms:
                domain_term = random.choice(self.domain_terms[domain][:5])
            else:
                domain_term = random.choice(ABSTRACT_PREFIXES)
                
            root = random.choice(root_options)
            return f"{domain_term} {root}"
            
    def _generate_moderate_name(
        self,
        weighted_terms: List[Tuple[str, float]],
        domain: Optional[str] = None
    ) -> str:
        """Generate a moderate complexity name using multiple terms."""
        # Extract terms
        terms = [term for term, _ in weighted_terms]
        
        # Moderate formats:
        # 1. Prefix + Root + "of" + Domain (e.g., "Meta-Stability of Resonance")
        # 2. Root + "through" + Relation (e.g., "Synchronization through Phase-Locking")
        # 3. Quality + Domain + Process (e.g., "Stable Harmonic Oscillation")
        
        # Get options for each slot
        prefix_options = [t for t in terms if any(t == p for p in ABSTRACT_PREFIXES)]
        if not prefix_options:
            prefix_options = ABSTRACT_PREFIXES
            
        root_options = [t for t in terms if any(t == r for r in CONCEPTUAL_ROOTS)]
        if not root_options:
            root_options = CONCEPTUAL_ROOTS
            
        relation_options = [t for t in terms if any(t == r for r in RELATIONAL_TERMS)]
        if not relation_options:
            relation_options = RELATIONAL_TERMS
            
        quality_options = [t for t in terms if any(t == q for q in QUALITY_TERMS)]
        if not quality_options:
            quality_options = QUALITY_TERMS
            
        phase_options = [t for t in terms if any(t == p for p in PHASE_TERMS)]
        if not phase_options:
            phase_options = PHASE_TERMS
            
        # Domain-specific term
        if domain and domain in self.domain_terms:
            domain_term = random.choice(self.domain_terms[domain][:5])
        else:
            domain_term = random.choice(ABSTRACT_PREFIXES)
            
        # Choose format randomly
        format_choice = random.randint(1, 3)
        
        if format_choice == 1:
            # Prefix + Root + "of" + Domain
            prefix = random.choice(prefix_options)
            root = random.choice(root_options)
            return f"{prefix} {root} of {domain_term}"
        elif format_choice == 2:
            # Root + "through" + Relation
            root = random.choice(root_options)
            relation = random.choice(relation_options)
            connector = random.choice(CONNECTORS)
            return f"{root} {connector} {relation}"
        else:
            # Quality + Domain + Process
            quality = random.choice(quality_options)
            phase = random.choice(phase_options)
            return f"{quality} {domain_term} {phase}"
            
    def _generate_complex_name(
        self,
        weighted_terms: List[Tuple[str, float]],
        domain: Optional[str] = None
    ) -> str:
        """Generate a complex name using sophisticated linguistic patterns."""
        # Extract terms
        terms = [term for term, _ in weighted_terms]
        
        # Complex formats:
        # 1. "The" + Prefix + Root + "of" + Quality + Domain (e.g., "The Meta-Principle of Coherent Dynamics")
        # 2. Prefix + Root + "of" + Domain + "through" + Relation (e.g., "Arch-Resonance of Topology through Synchronization")
        # 3. Quality + Domain + Process + "under" + Condition (e.g., "Stable Harmonic Oscillation under Coupling")
        # 4. Prefix + Domain + Process + "of" + Root + Relation (e.g., "Trans-Spectral Mapping of Bifurcation Dynamics")
        
        # Get options for each slot
        prefix_options = [t for t in terms if any(t == p for p in ABSTRACT_PREFIXES)]
        if not prefix_options:
            prefix_options = ABSTRACT_PREFIXES
            
        root_options = [t for t in terms if any(t == r for r in CONCEPTUAL_ROOTS)]
        if not root_options:
            root_options = CONCEPTUAL_ROOTS
            
        relation_options = [t for t in terms if any(t == r for r in RELATIONAL_TERMS)]
        if not relation_options:
            relation_options = RELATIONAL_TERMS
            
        quality_options = [t for t in terms if any(t == q for q in QUALITY_TERMS)]
        if not quality_options:
            quality_options = QUALITY_TERMS
            
        phase_options = [t for t in terms if any(t == p for p in PHASE_TERMS)]
        if not phase_options:
            phase_options = PHASE_TERMS
            
        suffix_options = SUFFIXES
            
        # Domain-specific term
        if domain and domain in self.domain_terms:
            domain_terms = self.domain_terms[domain]
            domain_term = random.choice(domain_terms[:5])
            domain_term2 = random.choice(domain_terms[5:] if len(domain_terms) > 5 else domain_terms)
        else:
            domain_term = random.choice(ABSTRACT_PREFIXES)
            domain_term2 = random.choice(root_options)
            
        # Choose format randomly
        format_choice = random.randint(1, 4)
        
        if format_choice == 1:
            # "The" + Prefix + Root + "of" + Quality + Domain
            prefix = random.choice(prefix_options)
            root = random.choice(root_options)
            quality = random.choice(quality_options)
            return f"The {prefix} {root} of {quality} {domain_term}"
        elif format_choice == 2:
            # Prefix + Root + "of" + Domain + "through" + Relation
            prefix = random.choice(prefix_options)
            root = random.choice(root_options)
            relation = random.choice(relation_options)
            return f"{prefix} {root} of {domain_term} through {relation}"
        elif format_choice == 3:
            # Quality + Domain + Process + "under" + Condition
            quality = random.choice(quality_options)
            process = random.choice(phase_options)
            condition = random.choice(relation_options)
            return f"{quality} {domain_term} {process} under {condition}"
        else:
            # Prefix + Domain + Process + "of" + Root + Relation
            prefix = random.choice(prefix_options)
            process = random.choice(phase_options)
            root = random.choice(root_options)
            relation = random.choice(relation_options)
            return f"{prefix} {domain_term} {process} of {root} {relation}"
            
    def generate_names_for_cluster(
        self,
        concept_ids: List[str],
        domain: Optional[str] = None,
        complexity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate coordinated names for a cluster of related concepts.
        
        Args:
            concept_ids: List of concept IDs in the cluster
            domain: Optional domain classification
            complexity: Optional complexity level
            
        Returns:
            Dictionary with name generation results
        """
        if not concept_ids:
            return {
                "status": "error",
                "message": "No concept IDs provided"
            }
            
        names = {}
        
        # Generate name for first concept
        first_result = self.generate_name(
            concept_ids[0], domain, complexity
        )
        
        if first_result.get("status") != "success":
            return first_result
            
        names[concept_ids[0]] = first_result.get("name")
        
        # Use similar patterns for other concepts in cluster
        first_name_parts = first_result.get("name").split()
        
        # Generate names for other concepts with shared elements
        for concept_id in concept_ids[1:]:
            # Generate standalone name
            result = self.generate_name(concept_id, domain, complexity)
            
            if result.get("status") == "success":
                # Use elements from first name to create coherence across the cluster
                standalone_name = result.get("name")
                name_parts = standalone_name.split()
                
                # Mix elements from both names
                if random.random() < 0.7 and len(first_name_parts) >= 3:
                    # Take elements from first name
                    shared_element = random.choice(first_name_parts)
                    
                    # Replace a random part of the name with shared element
                    if name_parts:
                        replace_idx = random.randint(0, len(name_parts) - 1)
                        name_parts[replace_idx] = shared_element
                        
                # Rejoin parts
                names[concept_id] = " ".join(name_parts)
            else:
                # Fallback
                names[concept_id] = f"Cluster-{concept_id[:6]}"
                
        return {
            "status": "success",
            "names": names,
            "concept_count": len(names)
        }
    
    def get_generation_history(
        self,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get history of name generation operations.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            Dictionary with generation history
        """
        # Sort by timestamp (newest first)
        sorted_entries = sorted(
            self.generated_names.values(),
            key=lambda x: x.get("timestamp", 0),
            reverse=True
        )
        
        # Take the most recent entries
        recent_entries = sorted_entries[:limit]
        
        return {
            "status": "success",
            "count": len(recent_entries),
            "total_generated": len(self.generated_names),
            "entries": recent_entries
        }
        
    def reset_generation_history(self) -> Dict[str, Any]:
        """
        Reset name generation history.
        
        Returns:
            Dictionary with reset results
        """
        old_count = len(self.generated_names)
        self.generated_names = {}
        
        return {
            "status": "success",
            "cleared_entries": old_count
        }


# Singleton instance
_ghost_label_synthesizer_instance = None

def get_ghost_label_synthesizer() -> GhostLabelSynthesizer:
    """
    Get the singleton instance of the ghost label synthesizer.
    
    Returns:
        GhostLabelSynthesizer instance
    """
    global _ghost_label_synthesizer_instance
    
    if _ghost_label_synthesizer_instance is None:
        _ghost_label_synthesizer_instance = GhostLabelSynthesizer()
        
    return _ghost_label_synthesizer_instance
