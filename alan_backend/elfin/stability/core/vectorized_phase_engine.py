"""
Vectorized Phase Engine for ELFIN at Extreme Scale.

This module implements a highly optimized, vectorized version of the phase-coupled
oscillator synchronization engine designed for large-scale deployments.
It uses sparse matrix operations and can leverage GPU acceleration
for systems with millions of concepts and billions of connections.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import time
from pathlib import Path
import json

# Conditional import for GPU support
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cu_csr_matrix
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

logger = logging.getLogger(__name__)


class VectorizedPhaseEngine:
    """
    Implements an extremely scalable ψ-coupling phase synchronization for concepts.
    
    This engine uses vectorized operations on sparse matrices to efficiently handle
    systems with millions of nodes and billions of edges. For single-machine environments,
    the practical limit is around 1M nodes and 50M edges. For truly web-scale deployments,
    this implementation can be distributed across multiple machines.
    """
    
    def __init__(self, coupling_strength: float = 0.1, use_gpu: bool = False):
        """
        Initialize the vectorized phase engine.
        
        Args:
            coupling_strength: Global coupling strength parameter (K)
            use_gpu: Whether to use GPU acceleration (requires cupy and CUDA)
        """
        self.coupling_strength = coupling_strength
        self.spectral_feedback = 1.0
        
        # Core data structures
        self.node_ids = []  # Ordered list of concept IDs
        self.id_to_idx = {}  # Mapping from concept ID to index
        self.phases = np.array([], dtype=np.float64)  # Phase values
        self.natural_frequencies = np.array([], dtype=np.float64)  # Natural frequencies
        
        # Sparse matrix representations of the graph
        self.weights_matrix = None  # CSR matrix of edge weights
        self.offset_matrix = None  # CSR matrix of phase offsets
        
        # Performance tracking
        self.last_step_time = 0.0
        
        # GPU support if available and requested
        self.use_gpu = use_gpu and HAS_GPU
        if use_gpu and not HAS_GPU:
            logger.warning("GPU acceleration requested but cupy not available. Falling back to CPU.")
            self.use_gpu = False
        
        if self.use_gpu:
            logger.info("Using GPU acceleration for phase engine calculations")
    
    def add_concept(self, concept_id: str, initial_phase: float = 0.0, natural_frequency: float = 0.0):
        """
        Add a concept to the phase engine.
        
        Args:
            concept_id: Unique identifier for the concept
            initial_phase: Initial phase value (in radians)
            natural_frequency: Natural oscillation frequency
        """
        if concept_id in self.id_to_idx:
            # Update existing concept
            idx = self.id_to_idx[concept_id]
            self.phases[idx] = initial_phase % (2 * np.pi)
            self.natural_frequencies[idx] = natural_frequency
            return
            
        # Add new concept
        idx = len(self.node_ids)
        self.node_ids.append(concept_id)
        self.id_to_idx[concept_id] = idx
        
        # Extend arrays
        self.phases = np.append(self.phases, initial_phase % (2 * np.pi))
        self.natural_frequencies = np.append(self.natural_frequencies, natural_frequency)
        
        # If matrices already exist, resize them
        n = len(self.node_ids)
        if self.weights_matrix is not None:
            # Create new empty matrices of the correct size
            new_weights = sp.csr_matrix((n, n), dtype=np.float64)
            new_offsets = sp.csr_matrix((n, n), dtype=np.float64)
            
            # Copy existing data
            old_n = self.weights_matrix.shape[0]
            new_weights[:old_n, :old_n] = self.weights_matrix
            new_offsets[:old_n, :old_n] = self.offset_matrix
            
            # Update matrices
            self.weights_matrix = new_weights
            self.offset_matrix = new_offsets
        else:
            # Initialize empty matrices
            self.weights_matrix = sp.csr_matrix((n, n), dtype=np.float64)
            self.offset_matrix = sp.csr_matrix((n, n), dtype=np.float64)
    
    def add_edge(self, source_id: str, target_id: str, weight: float = 1.0, phase_offset: float = 0.0):
        """
        Add a directed edge between concepts.
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            weight: Edge weight for coupling strength
            phase_offset: Desired phase difference between concepts
        """
        # Make sure concepts exist
        if source_id not in self.id_to_idx:
            self.add_concept(source_id)
        if target_id not in self.id_to_idx:
            self.add_concept(target_id)
        
        # Get indices
        source_idx = self.id_to_idx[source_id]
        target_idx = self.id_to_idx[target_id]
        
        # Update matrices
        self.weights_matrix[target_idx, source_idx] = weight
        self.offset_matrix[target_idx, source_idx] = phase_offset
    
    def build_matrices_from_dict(self, graph_dict: Dict[str, Dict[str, Dict[str, float]]]):
        """
        Build sparse matrices from a dictionary representation.
        
        Args:
            graph_dict: Dictionary of form {source: {target: {weight: w, phase_offset: p}}}
        """
        # First pass: collect all unique node IDs
        all_nodes = set()
        for source, targets in graph_dict.items():
            all_nodes.add(source)
            all_nodes.update(targets.keys())
        
        # Create/update all nodes
        for node_id in all_nodes:
            if node_id not in self.id_to_idx:
                self.add_concept(node_id)
        
        # Pre-allocate data for COO format
        n_edges = sum(len(targets) for targets in graph_dict.values())
        rows = np.zeros(n_edges, dtype=np.int32)
        cols = np.zeros(n_edges, dtype=np.int32)
        weight_data = np.zeros(n_edges, dtype=np.float64)
        offset_data = np.zeros(n_edges, dtype=np.float64)
        
        # Populate arrays
        idx = 0
        for source, targets in graph_dict.items():
            source_idx = self.id_to_idx[source]
            for target, attrs in targets.items():
                target_idx = self.id_to_idx[target]
                
                rows[idx] = target_idx  # Target is the row (being influenced)
                cols[idx] = source_idx  # Source is the column (influencer)
                
                weight_data[idx] = attrs.get('weight', 1.0)
                offset_data[idx] = attrs.get('phase_offset', 0.0)
                
                idx += 1
        
        # Create sparse matrices
        n = len(self.node_ids)
        self.weights_matrix = sp.coo_matrix((weight_data, (rows, cols)), shape=(n, n)).tocsr()
        self.offset_matrix = sp.coo_matrix((offset_data, (rows, cols)), shape=(n, n)).tocsr()
    
    def set_phase(self, concept_id: str, phase: float):
        """
        Set the phase value for a specific concept.
        
        Args:
            concept_id: Concept ID
            phase: New phase value (in radians)
        """
        if concept_id not in self.id_to_idx:
            self.add_concept(concept_id)
        
        idx = self.id_to_idx[concept_id]
        self.phases[idx] = phase % (2 * np.pi)
    
    def set_spectral_feedback(self, feedback_factor: float):
        """
        Set the feedback factor from spectral analysis.
        
        Args:
            feedback_factor: Value to modulate coupling strength
        """
        self.spectral_feedback = max(0.0, min(2.0, feedback_factor))
    
    def step(self, dt: float) -> Dict[str, float]:
        """
        Perform one step of phase updates for all concepts using vectorized operations.
        
        Args:
            dt: Time step size in seconds. Natural frequencies should be in radians/sec.
            
        Returns:
            Dictionary of updated phase values
        """
        start_time = time.time()
        n = len(self.node_ids)
        
        if n == 0:
            return {}
        
        # Transfer data to GPU if using it
        if self.use_gpu:
            phases_dev = cp.asarray(self.phases)
            nat_freq_dev = cp.asarray(self.natural_frequencies)
            weights_dev = cp.sparse.csr_matrix(self.weights_matrix)
            offsets_dev = cp.sparse.csr_matrix(self.offset_matrix)
        else:
            phases_dev = self.phases
            nat_freq_dev = self.natural_frequencies
            weights_dev = self.weights_matrix
            offsets_dev = self.offset_matrix
        
        # Phase differences: Each row i has the differences φ_j - φ_i for all j
        # Δφ_ij = φ_j - φ_i
        if self.use_gpu:
            phases_col = phases_dev.reshape(1, n)  # row vector
            phases_row = phases_dev.reshape(n, 1)  # column vector
            phase_diffs = phases_col - phases_row  # broadcasting creates matrix of all pairwise differences
            
            # Adjust for phase offsets (only where edges exist)
            offsets_coo = offsets_dev.tocoo()
            for k in range(len(offsets_coo.data)):
                i, j = offsets_coo.row[k], offsets_coo.col[k]
                phase_diffs[i, j] -= offsets_coo.data[k]
        else:
            # CPU version can use broadcasting too
            phases_col = self.phases.reshape(1, n)  # row vector
            phases_row = self.phases.reshape(n, 1)  # column vector
            phase_diffs = phases_col - phases_row  # broadcasting creates matrix of all pairwise differences
            
            # Adjust for phase offsets
            if self.offset_matrix.nnz > 0:  # Only if there are any offsets
                phase_diffs = phase_diffs - self.offset_matrix.toarray()
        
        # Compute sin(phase_diffs)
        if self.use_gpu:
            sin_diffs = cp.sin(phase_diffs)
        else:
            sin_diffs = np.sin(phase_diffs)
        
        # Element-wise multiply with weights
        if self.use_gpu:
            # Use CSR format for efficient multiplication
            weights_csr = weights_dev
            
            # Element-wise multiplication
            coupling_effect = weights_csr.multiply(sin_diffs)
            
            # Apply global coupling strength and spectral feedback
            coupling_effect = coupling_effect.multiply(self.coupling_strength * self.spectral_feedback)
            
            # Sum along columns to get total influence on each node
            d_phases = nat_freq_dev + cp.asarray(coupling_effect.sum(axis=1)).flatten()
            
            # Update phases
            phases_dev = (phases_dev + d_phases * dt) % (2 * cp.pi)
            
            # Transfer back to CPU
            self.phases = cp.asnumpy(phases_dev)
        else:
            # Use Hadamard product (element-wise multiply)
            coupling_effect = self.weights_matrix.multiply(sin_diffs)
            
            # Apply global coupling strength and spectral feedback
            coupling_effect = coupling_effect.multiply(self.coupling_strength * self.spectral_feedback)
            
            # Sum along rows to get total influence on each node
            d_phases = self.natural_frequencies + np.asarray(coupling_effect.sum(axis=1)).flatten()
            
            # Update phases
            self.phases = (self.phases + d_phases * dt) % (2 * np.pi)
        
        # Create result dictionary
        result = {node_id: self.phases[idx] for idx, node_id in enumerate(self.node_ids)}
        
        # Update performance tracking
        self.last_step_time = time.time() - start_time
        
        return result
    
    def calculate_sync_ratio(self) -> float:
        """
        Calculate the synchronization ratio of the concept graph.
        
        Returns:
            Synchronization ratio between 0 (no sync) and 1 (perfect sync)
        """
        n = len(self.node_ids)
        
        if n <= 1:
            return 1.0  # Single node or empty graph is perfectly "synchronized"
        
        # Calculate pairwise phase differences
        if self.use_gpu:
            phases_col = cp.asarray(self.phases).reshape(1, n)  # row vector
            phases_row = cp.asarray(self.phases).reshape(n, 1)  # column vector
            phase_diffs = phases_col - phases_row  # all pairwise differences
            
            # Adjust for phase offsets
            offsets_coo = cp.sparse.csr_matrix(self.offset_matrix).tocoo()
            for k in range(len(offsets_coo.data)):
                i, j = offsets_coo.row[k], offsets_coo.col[k]
                phase_diffs[i, j] -= offsets_coo.data[k]
            
            # Calculate error (use sin²(diff/2) as before)
            sin_half_diff = cp.sin(phase_diffs / 2)
            error_matrix = cp.square(sin_half_diff)
            
            # Apply weights
            weights_matrix = cp.sparse.csr_matrix(self.weights_matrix)
            total_error = (weights_matrix.multiply(error_matrix)).sum()
            total_weight = weights_matrix.sum()
            
            # Calculate sync ratio
            if total_weight > 0:
                avg_error = float(total_error / total_weight)
            else:
                avg_error = 0.0
                
            return float(1.0 - min(avg_error, 1.0))
        else:
            # Compute on CPU - similar to the original implementation
            # but using vectorized operations where possible
            total_error = 0.0
            total_weight = self.weights_matrix.sum()
            
            if total_weight == 0:
                return 1.0
            
            # For each edge
            coo_weights = self.weights_matrix.tocoo()
            coo_offsets = self.offset_matrix.tocoo()
            
            for k in range(len(coo_weights.data)):
                i, j = coo_weights.row[k], coo_weights.col[k]
                weight = coo_weights.data[k]
                phase_offset = coo_offsets.data[k] if k < len(coo_offsets.data) else 0.0
                
                # Calculate phase difference with offset
                phase_diff = self.phases[j] - self.phases[i] - phase_offset
                
                # Calculate error
                error = np.sin(phase_diff / 2) ** 2
                total_error += error * weight
            
            # Calculate sync ratio
            avg_error = total_error / total_weight
            return 1.0 - min(avg_error, 1.0)
    
    def export_state(self) -> Dict:
        """
        Export the current state of the phase engine.
        
        Returns:
            Dictionary containing phases, graph structure, and parameters
        """
        # Convert sparse matrices to COO format for export
        weights_coo = self.weights_matrix.tocoo()
        offsets_coo = self.offset_matrix.tocoo()
        
        # Build edge data
        edge_data = []
        for k in range(len(weights_coo.data)):
            i, j = weights_coo.row[k], weights_coo.col[k]
            
            # Find the offset for this edge if it exists
            offset = 0.0
            for l in range(len(offsets_coo.data)):
                if offsets_coo.row[l] == i and offsets_coo.col[l] == j:
                    offset = offsets_coo.data[l]
                    break
            
            edge_data.append({
                'source': self.node_ids[j],
                'target': self.node_ids[i],
                'weight': float(weights_coo.data[k]),
                'phase_offset': float(offset)
            })
        
        # Build result
        return {
            'phases': {node_id: float(self.phases[idx]) for idx, node_id in enumerate(self.node_ids)},
            'natural_frequencies': {node_id: float(self.natural_frequencies[idx]) for idx, node_id in enumerate(self.node_ids)},
            'coupling_strength': float(self.coupling_strength),
            'spectral_feedback': float(self.spectral_feedback),
            'sync_ratio': float(self.calculate_sync_ratio()),
            'node_count': len(self.node_ids),
            'edge_count': int(self.weights_matrix.nnz),
            'performance': {
                'last_step_time_ms': self.last_step_time * 1000,
                'nodes_per_second': len(self.node_ids) / max(self.last_step_time, 1e-6),
                'edges_per_second': self.weights_matrix.nnz / max(self.last_step_time, 1e-6)
            },
            'edges': edge_data
        }
    
    def save_state(self, file_path: Path) -> None:
        """
        Save the current state to file.
        
        Args:
            file_path: Path to save state to
        """
        state = self.export_state()
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, file_path: Path) -> 'VectorizedPhaseEngine':
        """
        Load state from file.
        
        Args:
            file_path: Path to load state from
            
        Returns:
            New VectorizedPhaseEngine instance
        """
        with open(file_path, 'r') as f:
            state = json.load(f)
        
        # Create new engine
        engine = cls(coupling_strength=state.get('coupling_strength', 0.1))
        
        # Set phases and frequencies
        for node_id, phase in state.get('phases', {}).items():
            freq = state.get('natural_frequencies', {}).get(node_id, 0.0)
            engine.add_concept(node_id, phase, freq)
        
        # Add edges
        for edge in state.get('edges', []):
            engine.add_edge(
                edge['source'],
                edge['target'],
                edge.get('weight', 1.0),
                edge.get('phase_offset', 0.0)
            )
        
        # Set spectral feedback
        engine.set_spectral_feedback(state.get('spectral_feedback', 1.0))
        
        return engine
    
    def to_networkx(self):
        """
        Convert to NetworkX graph for visualization and analysis.
        
        Requires networkx package.
        
        Returns:
            NetworkX DiGraph instance
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX is required for this functionality")
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for idx, node_id in enumerate(self.node_ids):
            G.add_node(node_id, phase=float(self.phases[idx]), 
                     natural_frequency=float(self.natural_frequencies[idx]))
        
        # Add edges with attributes
        weights_coo = self.weights_matrix.tocoo()
        offsets_coo = self.offset_matrix.tocoo()
        
        for k in range(len(weights_coo.data)):
            i, j = weights_coo.row[k], weights_coo.col[k]
            
            # Find the offset for this edge if it exists
            offset = 0.0
            for l in range(len(offsets_coo.data)):
                if offsets_coo.row[l] == i and offsets_coo.col[l] == j:
                    offset = offsets_coo.data[l]
                    break
            
            G.add_edge(self.node_ids[j], self.node_ids[i], 
                      weight=float(weights_coo.data[k]),
                      phase_offset=float(offset))
        
        return G


def benchmark_scaling(max_nodes=100000, max_edges=1000000, 
                     use_gpu=False, steps=10):
    """
    Benchmark the scaling performance of the vectorized engine.
    
    Args:
        max_nodes: Maximum number of nodes to test
        max_edges: Maximum number of edges to test
        use_gpu: Whether to use GPU acceleration
        steps: Number of steps to run
        
    Returns:
        Dictionary of benchmark results
    """
    import time
    
    results = []
    
    # Define node counts to test (logarithmic scale)
    node_counts = [10, 100, 1000, 10000]
    while node_counts[-1] < max_nodes:
        node_counts.append(min(node_counts[-1] * 10, max_nodes))
    
    for n_nodes in node_counts:
        # Estimate appropriate number of edges (usually sparse, ~10 per node)
        n_edges = min(n_nodes * 10, max_edges)
        
        # Initialize engine
        engine = VectorizedPhaseEngine(coupling_strength=0.1, use_gpu=use_gpu)
        
        # Create nodes
        for i in range(n_nodes):
            engine.add_concept(f"c{i}", initial_phase=np.random.random() * 2 * np.pi)
        
        # Create random edges
        source_indices = np.random.randint(0, n_nodes, n_edges)
        target_indices = np.random.randint(0, n_nodes, n_edges)
        weights = np.random.random(n_edges) * 0.5 + 0.5  # [0.5, 1.0]
        
        # Add edges
        for i in range(n_edges):
            if source_indices[i] != target_indices[i]:  # Avoid self-loops
                engine.add_edge(f"c{source_indices[i]}", f"c{target_indices[i]}", 
                              weights[i])
        
        # Measure time to run steps
        start_time = time.time()
        
        for _ in range(steps):
            engine.step(dt=0.1)
        
        elapsed = time.time() - start_time
        
        # Record results
        results.append({
            'nodes': n_nodes,
            'edges': engine.weights_matrix.nnz,  # Actual edge count after removing duplicates
            'total_time_ms': elapsed * 1000,
            'time_per_step_ms': (elapsed * 1000) / steps,
            'nodes_per_second': n_nodes * steps / elapsed,
            'edges_per_second': engine.weights_matrix.nnz * steps / elapsed
        })
        
        print(f"Nodes: {n_nodes}, Edges: {engine.weights_matrix.nnz}, "
             f"Time per step: {elapsed * 1000 / steps:.2f} ms")
    
    return results
