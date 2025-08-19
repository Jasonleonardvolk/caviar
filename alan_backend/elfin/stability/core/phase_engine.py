"""
Phase Engine for ELFIN.

This module implements the phase-coupled oscillator synchronization engine
based on the ψ-coupling design. It handles the oscillator update loop that 
synchronizes concept phases based on their relationships.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Set
import logging

logger = logging.getLogger(__name__)


class PhaseEngine:
    """
    Implements the ψ-coupling phase synchronization for concepts.
    
    This engine updates phase values of concepts based on their connections,
    adjusting phases toward alignment (or intended phase offset) for connected
    concepts.
    """
    
    def __init__(self, coupling_strength: float = 0.1, natural_frequencies: Optional[Dict[str, float]] = None):
        """
        Initialize the phase engine.
        
        Args:
            coupling_strength: Global coupling strength parameter (K)
            natural_frequencies: Dictionary mapping concept IDs to their natural frequencies
        """
        self.coupling_strength = coupling_strength
        self.natural_frequencies = natural_frequencies or {}
        self.phases: Dict[str, float] = {}  # Current phase values for each concept
        self.graph = nx.DiGraph()  # Concept graph with edge weights for coupling
        self.spectral_feedback = 1.0  # Feedback factor from spectral analysis
    
    def add_concept(self, concept_id: str, initial_phase: float = 0.0, natural_frequency: float = 0.0):
        """
        Add a concept to the phase engine.
        
        Args:
            concept_id: Unique identifier for the concept
            initial_phase: Initial phase value (in radians)
            natural_frequency: Natural oscillation frequency
        """
        self.phases[concept_id] = initial_phase
        self.natural_frequencies[concept_id] = natural_frequency
        self.graph.add_node(concept_id)
    
    def add_edge(self, source_id: str, target_id: str, weight: float = 1.0, 
                 phase_offset: float = 0.0):
        """
        Add a directed edge between concepts.
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            weight: Edge weight for coupling strength
            phase_offset: Desired phase difference between concepts
        """
        # Make sure nodes exist
        if source_id not in self.graph:
            self.add_concept(source_id)
        if target_id not in self.graph:
            self.add_concept(target_id)
            
        # Add edge with weight and desired phase offset
        self.graph.add_edge(source_id, target_id, weight=weight, phase_offset=phase_offset)
    
    def set_phase(self, concept_id: str, phase: float):
        """
        Set the phase value for a specific concept.
        
        Args:
            concept_id: Concept ID
            phase: New phase value (in radians)
        """
        if concept_id not in self.phases:
            self.add_concept(concept_id)
        
        # Normalize phase to [0, 2π)
        self.phases[concept_id] = phase % (2 * np.pi)
    
    def set_spectral_feedback(self, feedback_factor: float):
        """
        Set the feedback factor from spectral analysis.
        
        Args:
            feedback_factor: Value to modulate coupling strength
        """
        self.spectral_feedback = max(0.0, min(2.0, feedback_factor))
    
    def step(self, dt: float) -> Dict[str, float]:
        """
        Perform one step of phase updates for all concepts.
        
        Args:
            dt: Time step size in seconds. Natural frequencies should be in radians/sec.
            
        Returns:
            Dictionary of updated phase values
        """
        # Initialize phase updates with natural frequencies
        phase_updates = {node_id: self.natural_frequencies.get(node_id, 0.0) 
                        for node_id in self.graph.nodes}
        
        # Edge error accumulator for sync ratio calculation
        total_error = 0.0
        total_weight = 0.0
        
        # Single pass over all edges - O(E) instead of O(E*N)
        for source, target, edge_data in self.graph.edges(data=True):
            weight = edge_data.get('weight', 1.0)
            phase_offset = edge_data.get('phase_offset', 0.0)
            
            # Calculate phase difference with desired offset
            # No modulo - using sine's periodicity instead
            source_phase = self.phases.get(source, 0.0)
            target_phase = self.phases.get(target, 0.0)
            phase_diff = source_phase - target_phase - phase_offset
            
            # Apply coupling effect
            effective_coupling = self.coupling_strength * weight * self.spectral_feedback
            coupling_effect = effective_coupling * np.sin(phase_diff)
            
            # Apply to target node
            phase_updates[target] += coupling_effect
            
            # Calculate error for sync ratio (optional optimization)
            error = abs(np.sin(phase_diff/2))  # Proportional to phase difference
            total_error += error * weight
            total_weight += weight
        
        # Apply all updates simultaneously
        for node_id, d_phase in phase_updates.items():
            new_phase = (self.phases.get(node_id, 0.0) + d_phase * dt) % (2 * np.pi)
            self.phases[node_id] = new_phase
        
        # Store sync ratio data for potential reuse
        self._last_total_error = total_error
        self._last_total_weight = total_weight
        
        return self.phases
    
    def calculate_sync_ratio(self) -> float:
        """
        Calculate the synchronization ratio of the concept graph.
        
        Returns:
            Synchronization ratio between 0 (no sync) and 1 (perfect sync)
        """
        if len(self.phases) <= 1:
            return 1.0  # Single node or empty graph is perfectly "synchronized"
        
        # Extract edges in the graph
        edges = list(self.graph.edges(data=True))
        if not edges:
            return 1.0  # No edges means no synchronization constraints
        
        # Use cached calculations if available (faster)
        if hasattr(self, '_last_total_error') and hasattr(self, '_last_total_weight'):
            if self._last_total_weight > 0:
                avg_error = self._last_total_error / self._last_total_weight
                return 1.0 - min(avg_error, 1.0)  # Cap at 0 sync ratio
        
        # Otherwise calculate phase error for each edge
        total_error = 0.0
        for source, target, edge_data in edges:
            weight = edge_data.get('weight', 1.0)
            phase_offset = edge_data.get('phase_offset', 0.0)
            
            source_phase = self.phases.get(source, 0.0)
            target_phase = self.phases.get(target, 0.0)
            
            # Calculate phase difference with desired offset
            # Use sin^2(phase_diff/2) which is proportional to 1-cos(phase_diff)
            # and provides smoother error metric
            phase_diff = source_phase - target_phase - phase_offset
            error = np.sin(phase_diff/2)**2
            
            total_error += error * weight
        
        # Normalize by sum of weights
        total_weight = sum(edge_data.get('weight', 1.0) for _, _, edge_data in edges)
        if total_weight > 0:
            avg_error = total_error / total_weight
        else:
            avg_error = 0.0
        
        # Convert error to sync ratio (0 error = 1.0 sync, max error = 0.0 sync)
        return 1.0 - min(avg_error, 1.0)  # Cap at 0 sync ratio
    
    def get_phase_diff_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Calculate the phase difference matrix for all concept pairs.
        
        Returns:
            Tuple of (concept_ids, phase_diff_matrix)
        """
        concept_ids = list(self.phases.keys())
        n = len(concept_ids)
        
        # Create matrix of phase differences
        phase_diff_matrix = np.zeros((n, n))
        
        for i, id1 in enumerate(concept_ids):
            for j, id2 in enumerate(concept_ids):
                if i != j:
                    phase1 = self.phases.get(id1, 0.0)
                    phase2 = self.phases.get(id2, 0.0)
                    
                    # Calculate minimum phase difference in [0, π]
                    diff = abs((phase1 - phase2) % (2 * np.pi))
                    if diff > np.pi:
                        diff = 2 * np.pi - diff
                    
                    phase_diff_matrix[i, j] = diff
        
        return concept_ids, phase_diff_matrix
    
    def export_state(self) -> Dict:
        """
        Export the current state of the phase engine.
        
        Returns:
            Dictionary containing phases, graph structure, and parameters
        """
        return {
            'phases': self.phases.copy(),
            'coupling_strength': self.coupling_strength,
            'natural_frequencies': self.natural_frequencies.copy(),
            'spectral_feedback': self.spectral_feedback,
            'sync_ratio': self.calculate_sync_ratio(),
            'graph': {
                'nodes': list(self.graph.nodes()),
                'edges': [(u, v, d) for u, v, d in self.graph.edges(data=True)]
            }
        }
    
    def update_matrix(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate the coupling matrix and phase offset matrix for testing.
        
        This method extracts the graph structure as NumPy arrays for testing
        and verification against ground-truth implementations.
        
        Returns:
            Tuple of (coupling_matrix, offset_matrix, node_ids)
            
            coupling_matrix: Matrix A where A[i,j] is coupling strength from j to i
            offset_matrix: Matrix Δ where Δ[i,j] is phase offset from j to i
            node_ids: List of node IDs corresponding to matrix indices
        """
        # Get ordered list of node IDs
        node_ids = list(self.graph.nodes())
        n = len(node_ids)
        
        # Initialize matrices
        coupling_matrix = np.zeros((n, n))
        offset_matrix = np.zeros((n, n))
        
        # Build adjacency matrix with weights and offsets
        for i, target in enumerate(node_ids):
            for j, source in enumerate(node_ids):
                if self.graph.has_edge(source, target):
                    edge_data = self.graph.get_edge_data(source, target)
                    weight = edge_data.get('weight', 1.0)
                    phase_offset = edge_data.get('phase_offset', 0.0)
                    
                    # Apply global coupling strength and spectral feedback
                    effective_coupling = self.coupling_strength * weight * self.spectral_feedback
                    
                    coupling_matrix[i, j] = effective_coupling
                    offset_matrix[i, j] = phase_offset
        
        return coupling_matrix, offset_matrix, node_ids
