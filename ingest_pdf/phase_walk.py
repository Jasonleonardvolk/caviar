"""phase_walk.py - Implements phase-coherent traversal through concept space.

This module replaces token-based language generation with phase-coherent walks
through concept space, fulfilling ALAN's "No Token Imitation" commitment by 
generating output based on spectral resonance rather than learned token transitions.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
import math
import logging

# Configure logger
logger = logging.getLogger("phase_walk")

class ConceptNode:
    """Represents a concept with phase, activation, and relationships."""
    
    def __init__(
        self, 
        name: str, 
        embedding: np.ndarray, 
        phase: float = 0.0,
        activation: float = 0.0,
        eigenfunction_id: str = ""
    ):
        self.name = name
        self.embedding = embedding
        self.phase = phase  # θ in [0, 2π)
        self.activation = activation  # level of conceptual activation
        self.eigenfunction_id = eigenfunction_id
        self.neighbors: Dict[str, float] = {}  # name -> coupling strength
    
    def update_phase(self, dt: float, all_nodes: Dict[str, 'ConceptNode'], 
                    noise: float = 0.01) -> float:
        """Update phase based on Kuramoto oscillator dynamics."""
        # Intrinsic frequency (derived from embedding norm)
        omega = 1.0 + 0.5 * np.linalg.norm(self.embedding) / 10.0
        
        # Calculate coupling effects
        coupling_sum = 0.0
        for neighbor_name, coupling_strength in self.neighbors.items():
            if neighbor_name in all_nodes:
                neighbor = all_nodes[neighbor_name]
                phase_diff = neighbor.phase - self.phase
                coupling_sum += coupling_strength * math.sin(phase_diff)
        
        # Add noise for robustness
        random_factor = np.random.normal(0, noise)
        
        # Phase update equation
        delta_phase = omega + coupling_sum + random_factor
        new_phase = (self.phase + dt * delta_phase) % (2 * math.pi)
        
        return new_phase

class PhaseCoherentWalk:
    """Generates traversal paths through concept space based on phase coherence."""
    
    def __init__(self):
        self.concepts: Dict[str, ConceptNode] = {}
        self.coupling_matrix: Dict[Tuple[str, str], float] = {}
        self.time: float = 0.0
        
    def add_concept(
        self, 
        name: str, 
        embedding: np.ndarray,
        phase: Optional[float] = None,
        eigenfunction_id: str = "",
        activation: float = 0.0
    ) -> None:
        """Add a concept to the phase space."""
        if phase is None:
            # Calculate initial phase from embedding fingerprint for determinism
            hash_val = hash(embedding.tobytes()) % 1000
            phase = (hash_val / 1000) * 2 * math.pi
            
        self.concepts[name] = ConceptNode(
            name=name,
            embedding=embedding,
            phase=phase,
            activation=activation,
            eigenfunction_id=eigenfunction_id
        )
    
    def add_concepts_from_tuples(self, tuples: List[Any]) -> None:
        """Add multiple concepts from ConceptTuple objects."""
        for t in tuples:
            self.add_concept(
                name=t.name,
                embedding=t.embedding,
                eigenfunction_id=getattr(t, 'eigenfunction_id', ""),
                activation=0.0
            )
        
        # Build connections based on embedding similarity
        self._build_connections()
    
    def _build_connections(self, threshold: float = 0.5) -> None:
        """Build connections between concepts based on embedding similarity."""
        names = list(self.concepts.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                # Compute cosine similarity
                emb1 = self.concepts[name1].embedding
                emb2 = self.concepts[name2].embedding
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
                
                # Only connect concepts with similarity above threshold
                if similarity > threshold:
                    coupling = similarity  # Coupling strength based on similarity
                    self.concepts[name1].neighbors[name2] = coupling
                    self.concepts[name2].neighbors[name1] = coupling
                    self.coupling_matrix[(name1, name2)] = coupling
                    self.coupling_matrix[(name2, name1)] = coupling
    
    def run_dynamics(self, steps: int, dt: float = 0.1, noise: float = 0.01) -> None:
        """Run phase dynamics for specified number of steps."""
        for _ in range(steps):
            # Calculate new phases for all concepts
            new_phases = {}
            for name, node in self.concepts.items():
                new_phases[name] = node.update_phase(dt, self.concepts, noise)
            
            # Update all phases
            for name, phase in new_phases.items():
                self.concepts[name].phase = phase
                
            self.time += dt
    
    def activate_concept(self, name: str, level: float = 1.0) -> bool:
        """Activate a concept with specified level."""
        if name in self.concepts:
            self.concepts[name].activation = level
            logger.info(f"Activated concept '{name}' to level {level}")
            return True
        return False
    
    def phase_coherence(self, name1: str, name2: str) -> float:
        """Calculate phase coherence between two concepts."""
        if name1 not in self.concepts or name2 not in self.concepts:
            return 0.0
            
        phase1 = self.concepts[name1].phase
        phase2 = self.concepts[name2].phase
        
        # Phase difference in [0, π]
        phase_diff = abs((phase1 - phase2) % (2 * math.pi))
        if phase_diff > math.pi:
            phase_diff = 2 * math.pi - phase_diff
            
        # Convert to coherence value in [0, 1]
        coherence = 1.0 - phase_diff / math.pi
        
        return coherence
    
    def mean_coherence(self, concept_names: List[str]) -> float:
        """Calculate mean phase coherence across multiple concepts."""
        if len(concept_names) < 2:
            return 1.0  # Perfect coherence for 0 or 1 concept
            
        coherence_values = []
        for i, name1 in enumerate(concept_names):
            for name2 in concept_names[i+1:]:
                coherence_values.append(self.phase_coherence(name1, name2))
                
        return sum(coherence_values) / len(coherence_values) if coherence_values else 0.0
    
    def find_resonant_concepts(
        self, 
        source_name: str, 
        threshold: float = 0.7, 
        max_count: int = 5
    ) -> List[Tuple[str, float]]:
        """Find concepts that resonate with the source concept."""
        if source_name not in self.concepts:
            return []
            
        results = []
        source = self.concepts[source_name]
        
        for name, node in self.concepts.items():
            if name == source_name:
                continue
                
            coherence = self.phase_coherence(source_name, name)
            if coherence >= threshold:
                results.append((name, coherence))
                
        # Sort by coherence strength
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_count]
    
    def walk_phase_coherent_path(
        self, 
        start_name: str, 
        steps: int = 5, 
        threshold: float = 0.7,
        exclude_visited: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Walk a phase-coherent path through concept space.
        
        This is the core method for generating phase-locked walks through concept space,
        replacing token-based language generation with traversals governed by phase coherence.
        
        Args:
            start_name: Starting concept name
            steps: Maximum path length
            threshold: Minimum coherence threshold
            exclude_visited: Whether to avoid revisiting concepts
        
        Returns:
            List of (concept_name, coherence) tuples representing the path
        """
        if start_name not in self.concepts:
            return []
            
        path = [(start_name, 1.0)]  # Start with perfect coherence
        visited = {start_name}
        current = start_name
        
        for _ in range(steps):
            # Find resonant neighbors
            neighbors = self.find_resonant_concepts(current, threshold)
            
            # Filter out visited concepts if requested
            if exclude_visited:
                neighbors = [(name, coh) for name, coh in neighbors if name not in visited]
                
            if not neighbors:
                break  # No valid next steps
                
            # Select next concept (highest coherence)
            next_name, coherence = neighbors[0]
            path.append((next_name, coherence))
            visited.add(next_name)
            current = next_name
            
        return path
    
    def generate_narrative_path(
        self, 
        seed_concepts: List[str], 
        path_length: int = 10,
        coherence_threshold: float = 0.7,
        dynamics_steps: int = 20,
        dt: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Generate a coherent narrative path starting from seed concepts.
        
        This method:
        1. Activates seed concepts
        2. Runs phase dynamics to stabilize the system
        3. Walks a phase-coherent path
        
        Args:
            seed_concepts: Initial concepts to activate
            path_length: Length of path to generate
            coherence_threshold: Minimum coherence for path connections
            dynamics_steps: Steps of phase dynamics to run before path generation
            dt: Time step for dynamics
            
        Returns:
            List of concept, coherence pairs forming a narrative path
        """
        # Activate seed concepts
        for name in seed_concepts:
            if name in self.concepts:
                self.activate_concept(name)
            else:
                logger.warning(f"Seed concept '{name}' not found.")
                
        # Run dynamics to synchronize phases
        self.run_dynamics(dynamics_steps, dt)
        
        # Find most active concept to start path
        start_concept = None
        if seed_concepts:
            valid_seeds = [name for name in seed_concepts if name in self.concepts]
            if valid_seeds:
                start_concept = valid_seeds[0]
        
        if not start_concept:
            # Fall back to most densely connected concept
            connection_counts = {name: len(node.neighbors) for name, node in self.concepts.items()}
            if connection_counts:
                start_concept = max(connection_counts.items(), key=lambda x: x[1])[0]
            else:
                logger.error("No valid start concept found.")
                return []
        
        # Generate the coherent path
        path = self.walk_phase_coherent_path(
            start_concept, 
            steps=path_length,
            threshold=coherence_threshold
        )
        
        # Log the path for debugging
        logger.info(f"Generated narrative path: {[name for name, _ in path]}")
        
        return path
