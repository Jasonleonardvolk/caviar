"""reasoning_coherence.py - Implements coherence mechanisms for ALAN's reasoning processes.

This module provides tools for ensuring logical consistency and coherence in reasoning
through spectral graph analysis methods. It enables ALAN to:
- Map reasoning as a spectral graph problem
- Detect logical inconsistencies through spectral properties
- Maintain phase synchronization across inference paths
- Validate proof completeness and consistency

References:
- Network spectral theory for coherent reasoning
- Phase-locked inference validation
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable
import logging
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import math
from collections import defaultdict

try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple

# Configure logger
logger = logging.getLogger("alan_reasoning_coherence")

@dataclass
class InferenceNode:
    """Represents a node in the inference graph."""
    id: str  # Unique identifier
    type: str  # Type of node (premise, inference, conclusion)
    content: str  # Textual content of the node
    confidence: float = 1.0  # Confidence in this node (0-1)
    source_concept_ids: List[str] = field(default_factory=list)  # Concepts supporting this node
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "confidence": float(self.confidence),
            "source_concept_ids": self.source_concept_ids,
            "metadata": self.metadata
        }

@dataclass
class InferenceEdge:
    """Represents an edge in the inference graph."""
    source_id: str  # Source node ID
    target_id: str  # Target node ID
    relation_type: str  # Type of logical relation (supports, contradicts, etc.)
    weight: float = 1.0  # Edge weight (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": float(self.weight),
            "metadata": self.metadata
        }

@dataclass
class InferenceGraph:
    """Represents a reasoning structure as a graph."""
    nodes: Dict[str, InferenceNode] = field(default_factory=dict)  # Map of node ID to node
    edges: List[InferenceEdge] = field(default_factory=list)  # List of edges
    metadata: Dict[str, Any] = field(default_factory=dict)  # Graph metadata
    
    def add_node(self, node: InferenceNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        
    def add_edge(self, edge: InferenceEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata
        }
        
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph."""
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, **node.to_dict())
            
        # Add edges
        for edge in self.edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                relation_type=edge.relation_type,
                weight=edge.weight,
                **edge.metadata
            )
            
        return G


class InferenceGraphAnalyzer:
    """
    Analyzes the spectral properties of inference graphs to detect inconsistencies.
    
    Using graph spectral theory, this class computes eigenvalues and eigenvectors
    of the graph Laplacian to assess the coherence of reasoning structures.
    """
    
    def __init__(
        self, 
        coherence_threshold: float = 0.7,
        n_eigenvalues: int = 5
    ):
        """
        Initialize the inference graph analyzer.
        
        Args:
            coherence_threshold: Threshold for minimum acceptable coherence
            n_eigenvalues: Number of eigenvalues to compute
        """
        self.coherence_threshold = coherence_threshold
        self.n_eigenvalues = n_eigenvalues
        
    def compute_laplacian(
        self, 
        graph: Union[InferenceGraph, nx.Graph],
        normalized: bool = True,
        weighted: bool = True
    ) -> np.ndarray:
        """
        Compute the graph Laplacian for an inference graph.
        
        Args:
            graph: InferenceGraph or NetworkX graph
            normalized: Whether to normalize the Laplacian
            weighted: Whether to use edge weights
            
        Returns:
            Laplacian matrix as numpy array
        """
        # Convert to NetworkX if needed
        if isinstance(graph, InferenceGraph):
            G = graph.to_networkx()
        else:
            G = graph
            
        # Create adjacency matrix (weighted or unweighted)
        if weighted:
            # With weights
            A = nx.adjacency_matrix(G, weight='weight')
        else:
            # Unweighted
            A = nx.adjacency_matrix(G)
            
        # Compute degree matrix
        if weighted:
            # Weighted degrees
            degrees = np.array([sum(w.get('weight', 1.0) for _, _, w in G.edges(node, data=True)) 
                              for node in G.nodes()])
        else:
            # Unweighted degrees
            degrees = np.array([G.degree(node) for node in G.nodes()])
            
        # Create diagonal degree matrix
        D = sp.diags(degrees)
        
        # Compute Laplacian
        L = D - A
        
        # Normalize if requested
        if normalized and not np.all(degrees == 0):
            # Handle zero degrees (isolated nodes)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                D_sqrt_inv = sp.diags(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
                L = D_sqrt_inv @ L @ D_sqrt_inv
                
        return L
        
    def compute_spectral_properties(
        self, 
        graph: InferenceGraph
    ) -> Dict[str, Any]:
        """
        Compute spectral properties of an inference graph.
        
        Args:
            graph: InferenceGraph to analyze
            
        Returns:
            Dictionary with spectral analysis results
        """
        if len(graph.nodes) < 2:
            return {
                "status": "insufficient_nodes",
                "message": f"Graph needs at least 2 nodes, has {len(graph.nodes)}"
            }
            
        # Convert to NetworkX
        G = graph.to_networkx()
        
        # Check if the graph is empty
        if G.number_of_edges() == 0:
            return {
                "status": "no_edges",
                "message": "Graph has no edges"
            }
            
        # Compute Laplacian
        L = self.compute_laplacian(G)
        
        # Compute eigenvalues and eigenvectors
        n_vals = min(self.n_eigenvalues, L.shape[0])
        
        try:
            if sp.issparse(L):
                # For sparse matrices, use ARPACK
                eigenvalues, eigenvectors = spla.eigsh(
                    L, 
                    k=n_vals,
                    which='SM',  # Smallest magnitudes first
                    return_eigenvectors=True
                )
            else:
                # For dense matrices, use standard eigendecomposition
                L_dense = L.toarray() if sp.issparse(L) else L
                eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
                eigenvalues = eigenvalues[:n_vals]
                eigenvectors = eigenvectors[:, :n_vals]
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Eigendecomposition failed: {str(e)}"
            }
            
        # Sort eigenvalues in ascending order
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate spectral gap (difference between first non-zero and second eigenvalues)
        non_zero_idx = np.where(np.abs(eigenvalues) > 1e-10)[0]
        
        if len(non_zero_idx) >= 2:
            spectral_gap = eigenvalues[non_zero_idx[1]] - eigenvalues[non_zero_idx[0]]
        else:
            spectral_gap = 0.0
            
        # Compute coherence score from spectral gap
        if spectral_gap > 0:
            coherence_score = min(1.0, spectral_gap / self.coherence_threshold)
        else:
            coherence_score = 0.0
            
        # Check if graph is disconnected (multiple components)
        is_connected = nx.is_connected(G.to_undirected())
        
        # Compute algebraic connectivity (second smallest eigenvalue of Laplacian)
        algebraic_connectivity = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        
        return {
            "status": "success",
            "eigenvalues": eigenvalues.tolist(),
            "spectral_gap": spectral_gap,
            "coherence_score": coherence_score,
            "is_coherent": coherence_score >= self.coherence_threshold,
            "is_connected": is_connected,
            "algebraic_connectivity": algebraic_connectivity,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges()
        }
        
    def detect_inconsistencies(
        self, 
        graph: InferenceGraph
    ) -> Dict[str, Any]:
        """
        Detect logical inconsistencies in an inference graph.
        
        Args:
            graph: InferenceGraph to analyze
            
        Returns:
            Dictionary with inconsistency analysis results
        """
        # Convert to NetworkX
        G = graph.to_networkx()
        
        # Compute spectral properties
        spectral = self.compute_spectral_properties(graph)
        
        if spectral["status"] != "success":
            return spectral
            
        # Identify potential inconsistencies
        inconsistencies = []
        
        # 1. Check for contradictory edges
        contradictory_relations = []
        for node in G.nodes():
            # Track contradictory relations to this node
            in_relations = {}  # source -> relation_type
            
            for pred in G.predecessors(node):
                edge_data = G.get_edge_data(pred, node)
                relation = edge_data.get('relation_type', 'supports')
                
                if relation in in_relations.values():
                    # Find which source has the same relation
                    for src, rel in in_relations.items():
                        if rel == relation and src != pred:
                            contradictory_relations.append({
                                "node_id": node,
                                "source1": src,
                                "source2": pred,
                                "relation": relation
                            })
                
                # Add this relation
                in_relations[pred] = relation
                
        if contradictory_relations:
            inconsistencies.append({
                "type": "contradictory_relations",
                "description": "Multiple sources have contradictory relations to the same node",
                "instances": contradictory_relations
            })
            
        # 2. Check for cycles (excluding self-loops)
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                inconsistencies.append({
                    "type": "cycles",
                    "description": "Circular reasoning detected (inference cycles)",
                    "instances": [{"nodes": cycle} for cycle in cycles]
                })
        except nx.NetworkXNoCycle:
            pass  # No cycles found
            
        # 3. Check for disconnected reasoning chains
        if not spectral["is_connected"]:
            # Find connected components
            components = list(nx.connected_components(G.to_undirected()))
            
            if len(components) > 1:
                inconsistencies.append({
                    "type": "disconnected_components",
                    "description": "Reasoning contains disconnected chains",
                    "instances": [{"component_size": len(comp), "nodes": list(comp)} 
                                for comp in components]
                })
                
        # 4. Check for nodes with no support (leaves)
        leaves = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) > 0]
        if leaves:
            inconsistencies.append({
                "type": "unsupported_premises",
                "description": "Nodes used in reasoning without supporting evidence",
                "instances": [{"node_id": node} for node in leaves]
            })
            
        # 5. Check for spectral inconsistency
        if not spectral["is_coherent"]:
            inconsistencies.append({
                "type": "spectral_incoherence",
                "description": "Reasoning structure lacks overall coherence",
                "details": {
                    "coherence_score": spectral["coherence_score"],
                    "threshold": self.coherence_threshold,
                    "spectral_gap": spectral["spectral_gap"]
                }
            })
            
        # Compute overall consistency score
        consistency_score = 1.0
        
        # Reduce score for each type of inconsistency
        if inconsistencies:
            # Base reduction on number and types of inconsistencies
            reduction = min(0.8, 0.2 * len(inconsistencies))
            consistency_score -= reduction
            
        # Further reduce based on spectral coherence
        consistency_score = min(consistency_score, spectral["coherence_score"])
        
        return {
            "status": "success",
            "consistency_score": max(0.0, consistency_score),
            "is_consistent": consistency_score > 0.7 and not inconsistencies,
            "inconsistencies": inconsistencies,
            "spectral_properties": spectral
        }
    
    def identify_critical_nodes(
        self, 
        graph: InferenceGraph
    ) -> Dict[str, Any]:
        """
        Identify critical nodes in the reasoning structure.
        
        Critical nodes are those whose removal would significantly impact
        the coherence or connectivity of the reasoning.
        
        Args:
            graph: InferenceGraph to analyze
            
        Returns:
            Dictionary with critical node analysis
        """
        # Convert to NetworkX
        G = graph.to_networkx()
        
        if G.number_of_nodes() < 3:
            return {
                "status": "insufficient_nodes",
                "message": "Need at least 3 nodes for meaningful critical node analysis"
            }
            
        # Calculate node centrality metrics
        try:
            # Betweenness centrality - nodes that bridge different parts of reasoning
            betweenness = nx.betweenness_centrality(G)
            
            # Eigenvector centrality - nodes connected to other important nodes
            eigenvector = nx.eigenvector_centrality_numpy(G)
            
            # PageRank - alternative importance metric
            pagerank = nx.pagerank(G)
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Centrality calculation failed: {str(e)}"
            }
            
        # Identify articulation points (nodes whose removal disconnects the graph)
        articulation_points = list(nx.articulation_points(G.to_undirected()))
        
        # Combine metrics to identify critical nodes
        critical_nodes = []
        
        for node in G.nodes():
            # Calculate combined importance score
            importance = (
                0.4 * betweenness.get(node, 0) +
                0.3 * eigenvector.get(node, 0) +
                0.3 * pagerank.get(node, 0)
            )
            
            # Check if it's an articulation point
            is_articulation = node in articulation_points
            
            # Store node data
            node_data = G.nodes[node]
            
            critical_nodes.append({
                "node_id": node,
                "content": node_data.get("content", ""),
                "type": node_data.get("type", ""),
                "importance_score": importance,
                "is_articulation_point": is_articulation,
                "betweenness": betweenness.get(node, 0),
                "eigenvector_centrality": eigenvector.get(node, 0),
                "pagerank": pagerank.get(node, 0)
            })
            
        # Sort by importance
        critical_nodes.sort(key=lambda x: x["importance_score"], reverse=True)
        
        # Calculate how many nodes are needed to disconnect the graph
        node_connectivity = nx.node_connectivity(G)
        
        return {
            "status": "success",
            "critical_nodes": critical_nodes[:min(5, len(critical_nodes))],  # Top 5
            "articulation_points": len(articulation_points),
            "node_connectivity": node_connectivity,
            "total_nodes": G.number_of_nodes()
        }


class LogicalPhaseLocker:
    """
    Ensures coherence of reasoning by maintaining phase synchronization.
    
    This class treats logical inference as a phase synchronization problem,
    ensuring that all inference paths maintain coherence through phase alignment.
    """
    
    def __init__(
        self, 
        coupling_strength: float = 0.1,
        target_phase_coherence: float = 0.9,
        max_iterations: int = 100
    ):
        """
        Initialize the logical phase locker.
        
        Args:
            coupling_strength: Strength of coupling between connected nodes
            target_phase_coherence: Target phase coherence level (0-1)
            max_iterations: Maximum iterations for phase synchronization
        """
        self.coupling_strength = coupling_strength
        self.target_phase_coherence = target_phase_coherence
        self.max_iterations = max_iterations
        
    def initialize_phases(
        self, 
        graph: InferenceGraph,
        fixed_nodes: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Initialize phases for all nodes in the graph.
        
        Args:
            graph: InferenceGraph to initialize
            fixed_nodes: Optional list of node IDs with fixed phases
            
        Returns:
            Dictionary mapping node ID to phase angle (0-2π)
        """
        phases = {}
        fixed_nodes = fixed_nodes or []
        fixed_nodes_set = set(fixed_nodes)
        
        # Assign random phases to non-fixed nodes
        for node_id in graph.nodes:
            if node_id in fixed_nodes_set:
                # Fixed nodes get phase 0
                phases[node_id] = 0.0
            else:
                # Random phase between 0 and 2π
                phases[node_id] = np.random.uniform(0, 2 * np.pi)
                
        return phases
        
    def calculate_phase_coherence(self, phases: Dict[str, float]) -> float:
        """
        Calculate the overall phase coherence of the system.
        
        Args:
            phases: Dictionary mapping node ID to phase angle
            
        Returns:
            Phase coherence value (0-1)
        """
        if not phases:
            return 0.0
            
        # Convert phases to complex numbers on unit circle
        z = np.array([np.exp(1j * phase) for phase in phases.values()])
        
        # Calculate order parameter R
        R = np.abs(np.mean(z))
        
        return R
        
    def synchronize_phases(
        self, 
        graph: InferenceGraph,
        fixed_nodes: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Synchronize phases of nodes in the inference graph.
        
        Args:
            graph: InferenceGraph to synchronize
            fixed_nodes: Optional list of node IDs with fixed phases
            verbose: Whether to print detailed progress
            
        Returns:
            Dictionary with synchronization results
        """
        # Initialize phases
        phases = self.initialize_phases(graph, fixed_nodes)
        
        # Convert to NetworkX for easier manipulation
        G = graph.to_networkx()
        
        # Get edge weights dictionary
        edge_weights = {}
        for edge in graph.edges:
            edge_weights[(edge.source_id, edge.target_id)] = edge.weight
            
        # Create adjacency map for faster access
        adjacency = defaultdict(list)
        for edge in graph.edges:
            source, target = edge.source_id, edge.target_id
            weight = edge.weight
            
            # For each relationship type, adjust the sign of interaction
            if edge.relation_type == "supports":
                sign = 1  # In-phase coupling
            elif edge.relation_type == "contradicts":
                sign = -1  # Anti-phase coupling
            else:
                sign = 0  # No coupling
                
            adjacency[source].append((target, weight, sign))
            adjacency[target].append((source, weight, sign))  # Bidirectional influence
            
        # Track iteration history
        history = []
        
        # Fixed nodes set
        fixed_nodes = fixed_nodes or []
        fixed_nodes_set = set(fixed_nodes)
        
        # Perform phase synchronization iterations (Kuramoto model with adjacency)
        for i in range(self.max_iterations):
            # Calculate current coherence
            coherence = self.calculate_phase_coherence(phases)
            
            # Store current state
            history.append({
                "iteration": i,
                "coherence": coherence,
                "phases": dict(phases)  # Copy current phases
            })
            
            # Check if we've reached target coherence
            if coherence >= self.target_phase_coherence:
                if verbose:
                    print(f"Reached target coherence {coherence:.4f} at iteration {i}")
                break
                
            # Update phases for next iteration
            new_phases = dict(phases)
            
            # Update each node's phase based on its neighbors
            for node_id in G.nodes():
                # Skip fixed nodes
                if node_id in fixed_nodes_set:
                    continue
                    
                # Current phase
                phase_i = phases[node_id]
                
                # Calculate phase update
                phase_diff_sum = 0.0
                
                # Consider influence from all neighbors
                for neighbor, weight, sign in adjacency[node_id]:
                    phase_j = phases[neighbor]
                    
                    if sign > 0:
                        # In-phase coupling (wants to synchronize)
                        phase_diff_sum += weight * np.sin(phase_j - phase_i)
                    elif sign < 0:
                        # Anti-phase coupling (wants to be opposite)
                        phase_diff_sum += weight * np.sin(phase_j - phase_i - np.pi)
                        
                # Update phase
                new_phases[node_id] = phase_i + self.coupling_strength * phase_diff_sum
                
                # Ensure phase is in [0, 2π]
                new_phases[node_id] = new_phases[node_id] % (2 * np.pi)
                
            # Update phases for next iteration
            phases = new_phases
            
            if verbose and (i+1) % 10 == 0:
                print(f"Iteration {i+1}: coherence = {coherence:.4f}")
                
        # Calculate final coherence
        final_coherence = self.calculate_phase_coherence(phases)
        
        # Calculate phase differences for each edge
        edge_coherence = {}
        for edge in graph.edges:
            source, target = edge.source_id, edge.target_id
            relation = edge.relation_type
            
            # Get phases
            phase_i = phases[source]
            phase_j = phases[target]
            
            # Calculate relative phase (considering relationship type)
            if relation == "supports":
                # For "supports", we want phases to align
                target_diff = 0.0
            elif relation == "contradicts":
                # For "contradicts", we want phases to be opposite
                target_diff = np.pi
            else:
                target_diff = 0.0
                
            # Actual phase difference
            actual_diff = np.abs((phase_j - phase_i) % (2 * np.pi))
            
            # Normalized coherence for this edge (0 = perfectly in/out of phase as appropriate, 1 = maximally incoherent)
            edge_coh = 1.0 - np.abs(np.cos(actual_diff - target_diff))
            
            edge_coherence[(source, target)] = edge_coh
            
        return {
            "status": "success",
            "initial_coherence": history[0]["coherence"] if history else 0.0,
            "final_coherence": final_coherence,
            "iterations": len(history),
            "phases": phases,
            "edge_coherence": edge_coherence,
            "reached_target": final_coherence >= self.target_phase_coherence,
            "history": history if verbose else history[::10]  # Save space if not verbose
        }
    
    def detect_synchronization_clusters(
        self, 
        phases: Dict[str, float],
        threshold: float = 0.2
    ) -> List[List[str]]:
        """
        Detect clusters of synchronized nodes.
        
        Args:
            phases: Dictionary mapping node ID to phase angle
            threshold: Maximum phase difference to consider nodes synchronized
            
        Returns:
            List of clusters (each a list of node IDs)
        """
        if not phases:
            return []
            
        # Convert phases dictionary to a list of (node_id, phase) pairs
        phase_items = list(phases.items())
        
        # Initialize clusters
        clusters = []
        used_nodes = set()
        
        # For each node not yet in a cluster
        for i, (node_i, phase_i) in enumerate(phase_items):
            if node_i in used_nodes:
                continue
                
            # Start a new cluster
            cluster = [node_i]
            used_nodes.add(node_i)
            
            # Find all nodes synchronized with this one
            for j, (node_j, phase_j) in enumerate(phase_items):
                if i == j or node_j in used_nodes:
                    continue
                    
                # Check if phases are close (within threshold)
                phase_diff = np.abs((phase_j - phase_i) % (2 * np.pi))
                phase_diff = min(phase_diff, 2 * np.pi - phase_diff)
                
                if phase_diff < threshold:
                    cluster.append(node_j)
                    used_nodes.add(node_j)
                    
            clusters.append(cluster)
            
        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)
        
        return clusters


class ProofConsistencyVerifier:
    """
    Verifies the consistency and completeness of proofs or logical arguments.
    
    This class provides mechanisms to validate formal reasoning structures,
    ensuring that all conclusions follow logically from premises without gaps.
    """
    
    def __init__(
        self, 
        consistency_threshold: float = 0.7,
        completeness_threshold: float = 0.8
    ):
        """
        Initialize the proof consistency verifier.
        
        Args:
            consistency_threshold: Threshold for minimum consistency
            completeness_threshold: Threshold for minimum completeness
        """
        self.consistency_threshold = consistency_threshold
        self.completeness_threshold = completeness_threshold
        
        # Create helper objects
        self.graph_analyzer = InferenceGraphAnalyzer(coherence_threshold=consistency_threshold)
        self.phase_locker = LogicalPhaseLocker()
        
    def verify_proof(
        self, 
        graph: InferenceGraph
    ) -> Dict[str, Any]:
        """
        Verify the consistency and completeness of a proof.
        
        Args:
            graph: InferenceGraph representing the proof structure
            
        Returns:
            Dictionary with verification results
        """
        # Check for structural consistency
        consistency_result = self.graph_analyzer.detect_inconsistencies(graph)
        
        if consistency_result["status"] != "success":
            return consistency_result
            
        # Check for logical coherence through phase synchronization
        # Fix the phases of premise nodes
        premises = [node_id for node_id, node in graph.nodes.items() if node.type == "premise"]
        
        phase_result = self.phase_locker.synchronize_phases(
            graph=graph,
            fixed_nodes=premises,
            verbose=False
        )
        
        # Check for completeness
        completeness_score, gaps = self._check_completeness(graph)
        
        # Calculate overall validity
        validity_score = (
            0.5 * consistency_result["consistency_score"] +
            0.3 * phase_result["final_coherence"] +
            0.2 * completeness_score
        )
        
        return {
            "status": "success",
            "validity_score": validity_score,
            "is_valid": validity_score >= self.consistency_threshold,
            "consistency": {
                "score": consistency_result["consistency_score"],
                "is_consistent": consistency_result["is_consistent"],
                "inconsistencies": consistency_result["inconsistencies"]
            },
            "coherence": {
                "score": phase_result["final_coherence"],
                "is_coherent": phase_result["reached_target"],
                "synchronization_iterations": phase_result["iterations"]
            },
            "completeness": {
                "score": completeness_score,
                "is_complete": completeness_score >= self.completeness_threshold,
                "gaps": gaps
            }
        }
        
    def _check_completeness(
        self, 
        graph: InferenceGraph
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Check for gaps or missing steps in the proof.
        
        Args:
            graph: InferenceGraph to check
            
        Returns:
            Tuple of (completeness_score, list_of_gaps)
        """
        # Convert to NetworkX
        G = graph.to_networkx()
        
        # Identify premise and conclusion nodes
        premises = [node_id for node_id, node in graph.nodes.items() if node.type == "premise"]
        conclusions = [node_id for node_id, node in graph.nodes.items() if node.type == "conclusion"]
        
        # No premises or conclusions means incomplete proof
        if not premises or not conclusions:
            return 0.0, [{
                "type": "missing_elements",
                "description": "Proof is missing premises or conclusions",
                "details": {"premises": len(premises), "conclusions": len(conclusions)}
            }]
            
        # Initialize gaps list
        gaps = []
        
        # 1. Check if every conclusion is reachable from premises
        for conclusion in conclusions:
            # Check if there's a path from any premise to this conclusion
            reachable = False
            for premise in premises:
                try:
                    path = nx.shortest_path(G, source=premise, target=conclusion)
                    reachable = True
                    break
                except nx.NetworkXNoPath:
                    continue
                    
            if not reachable:
                gaps.append({
                    "type": "unreachable_conclusion",
                    "description": f"Conclusion '{graph.nodes[conclusion].content}' is not supported by any premise",
                    "node_id": conclusion
                })
                
        # 2. Check for "jumps" in reasoning (too few intermediate steps)
        for conclusion in conclusions:
            for premise in premises:
                try:
                    path = nx.shortest_path(G, source=premise, target=conclusion)
                    
                    # If path is direct (premise -> conclusion) but nodes are complex, may need more steps
                    if len(path) == 2:
                        # Get content length as proxy for complexity
                        premise_complexity = len(graph.nodes[premise].content.split())
                        conclusion_complexity = len(graph.nodes[conclusion].content.split()) 
                        
                        # If both premise and conclusion are complex, might be a leap
                        if premise_complexity > 10 and conclusion_complexity > 10:
                            gaps.append({
                                "type": "reasoning_leap",
                                "description": "Complex conclusion drawn directly from premise without intermediate steps",
                                "premise": premise,
                                "conclusion": conclusion,
                                "path_length": len(path)
                            })
                except nx.NetworkXNoPath:
                    continue
                    
        # 3. Check for "orphaned" intermediate nodes (not leading to a conclusion)
        for node_id in G.nodes():
            # Skip premises and conclusions
            if node_id in premises or node_id in conclusions:
                continue
                
            # Check if this intermediate node leads to any conclusion
            leads_to_conclusion = False
            for conclusion in conclusions:
                try:
                    path = nx.shortest_path(G, source=node_id, target=conclusion)
                    leads_to_conclusion = True
                    break
                except nx.NetworkXNoPath:
                    continue
                    
            if not leads_to_conclusion:
                gaps.append({
                    "type": "orphaned_inference",
                    "description": "Intermediate inference does not support any conclusion",
                    "node_id": node_id
                })
                
        # Calculate completeness score based on gaps
        base_score = 1.0
        
        # Deductions for different types of gaps
        for gap in gaps:
            if gap["type"] == "unreachable_conclusion":
                # Major issue: deduct up to 0.5
                base_score -= 0.5 / max(1, len(conclusions))
            elif gap["type"] == "reasoning_leap":
                # Moderate issue: deduct up to 0.3
                base_score -= 0.3 / max(1, len(premises) * len(conclusions))
            elif gap["type"] == "orphaned_inference":
                # Minor issue: deduct up to 0.2
                base_score -= 0.2 / max(1, G.number_of_nodes() - len(premises) - len(conclusions))
                
        # Ensure score is between 0 and 1
        completeness_score = max(0.0, min(1.0, base_score))
        
        return completeness_score, gaps
        
    def suggest_improvements(
        self, 
        verification_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Suggest improvements to fix issues in the proof.
        
        Args:
            verification_result: Result from verify_proof
            
        Returns:
            List of improvement suggestions
        """
        if verification_result["status"] != "success":
            return []
            
        suggestions = []
        
        # 1. Suggestions based on consistency issues
        if "inconsistencies" in verification_result["consistency"]:
            for inconsistency in verification_result["consistency"]["inconsistencies"]:
                if inconsistency["type"] == "contradictory_relations":
                    suggestions.append({
                        "issue_type": "contradiction",
                        "description": "Resolve contradicting logical relations",
                        "severity": "high",
                        "suggestion": "Examine the contradictory edges and resolve their relationship types"
                    })
                elif inconsistency["type"] == "cycles":
                    suggestions.append({
                        "issue_type": "circular_reasoning",
                        "description": "Remove circular dependencies in reasoning",
                        "severity": "high",
                        "suggestion": "Break circular references by adding external premises or restructuring the argument"
                    })
                elif inconsistency["type"] == "disconnected_components":
                    suggestions.append({
                        "issue_type": "fragmented_reasoning",
                        "description": "Connect isolated reasoning chains",
                        "severity": "medium",
                        "suggestion": "Add logical connections between separate argument components"
                    })
                    
        # 2. Suggestions based on completeness gaps
        if "gaps" in verification_result["completeness"]:
            for gap in verification_result["completeness"]["gaps"]:
                if gap["type"] == "unreachable_conclusion":
                    suggestions.append({
                        "issue_type": "unsupported_conclusion",
                        "description": f"Provide support for conclusion: {gap.get('node_id', 'unknown')}",
                        "severity": "high",
                        "suggestion": "Add intermediate inferences connecting premises to this conclusion"
                    })
                elif gap["type"] == "reasoning_leap":
                    suggestions.append({
                        "issue_type": "reasoning_leap",
                        "description": "Add intermediate steps to explain complex inference",
                        "severity": "medium",
                        "suggestion": "Break down complex reasoning into smaller, explicit steps"
                    })
                    
        # 3. Suggestions based on phase coherence
        if verification_result["coherence"]["score"] < 0.7:
            suggestions.append({
                "issue_type": "phase_incoherence",
                "description": "Improve logical flow and coherence",
                "severity": "medium",
                "suggestion": "Review the overall logical structure to ensure smooth transitions between ideas"
            })
            
        return suggestions


class ReasoningCoherenceManager:
    """
    Main class integrating all reasoning coherence components.
    
    This provides a unified interface for ensuring logical consistency
    and coherence in ALAN's reasoning processes.
    """
    
    def __init__(self, log_dir: str = "logs/reasoning"):
        """
        Initialize the reasoning coherence manager.
        
        Args:
            log_dir: Directory for logging reasoning data
        """
        self.log_dir = log_dir
        
        # Initialize components
        self.graph_analyzer = InferenceGraphAnalyzer()
        self.phase_locker = LogicalPhaseLocker()
        self.proof_verifier = ProofConsistencyVerifier()
        
        # Track inference graphs
        self.graphs = {}
        
        logger.info("Reasoning coherence manager initialized")
        
    def create_inference_graph(
        self, 
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> InferenceGraph:
        """
        Create a new inference graph.
        
        Args:
            name: Name of the graph
            metadata: Optional metadata for the graph
            
        Returns:
            New InferenceGraph instance
        """
        graph = InferenceGraph(metadata=metadata or {})
        self.graphs[name] = graph
        
        return graph
        
    def add_node_to_graph(
        self, 
        graph_name: str,
        node: InferenceNode
    ) -> None:
        """
        Add a node to an existing inference graph.
        
        Args:
            graph_name: Name of the graph
            node: Node to add
        """
        if graph_name not in self.graphs:
            logger.warning(f"Graph '{graph_name}' not found")
            return
            
        self.graphs[graph_name].add_node(node)
        
    def add_edge_to_graph(
        self, 
        graph_name: str,
        edge: InferenceEdge
    ) -> None:
        """
        Add an edge to an existing inference graph.
        
        Args:
            graph_name: Name of the graph
            edge: Edge to add
        """
        if graph_name not in self.graphs:
            logger.warning(f"Graph '{graph_name}' not found")
            return
            
        self.graphs[graph_name].add_edge(edge)
        
    def create_node_from_concept(
        self, 
        concept: ConceptTuple,
        node_type: str = "premise",
        node_id: Optional[str] = None
    ) -> InferenceNode:
        """
        Create an inference node from a concept.
        
        Args:
            concept: Concept to convert to a node
            node_type: Type of node to create
            node_id: Optional ID for the node (if None, use eigenfunction_id)
            
        Returns:
            New InferenceNode
        """
        node_id = node_id or concept.eigenfunction_id
        
        return InferenceNode(
            id=node_id,
            type=node_type,
            content=concept.name,
            confidence=getattr(concept, 'resonance_score', 1.0),
            source_concept_ids=[concept.eigenfunction_id],
            metadata={
                "concept_name": concept.name,
                "creation_time": datetime.now().isoformat()
            }
        )
        
    def analyze_graph(
        self, 
        graph_name: str
    ) -> Dict[str, Any]:
        """
        Analyze the coherence of an inference graph.
        
        Args:
            graph_name: Name of the graph to analyze
            
        Returns:
            Analysis results
        """
        if graph_name not in self.graphs:
            return {
                "status": "error",
                "message": f"Graph '{graph_name}' not found"
            }
            
        # Get the graph
        graph = self.graphs[graph_name]
        
        # Analyze inconsistencies
        inconsistencies = self.graph_analyzer.detect_inconsistencies(graph)
        
        # Identify critical nodes
        critical_nodes = self.graph_analyzer.identify_critical_nodes(graph)
        
        # Synchronize phases
        phases = self.phase_locker.synchronize_phases(graph)
        
        # Combine results
        return {
            "status": "success",
            "graph_name": graph_name,
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "inconsistencies": inconsistencies,
            "critical_nodes": critical_nodes,
            "phase_synchronization": phases
        }
        
    def verify_reasoning(
        self, 
        graph_name: str
    ) -> Dict[str, Any]:
        """
        Verify the consistency and completeness of reasoning.
        
        Args:
            graph_name: Name of the graph to verify
            
        Returns:
            Verification results
        """
        if graph_name not in self.graphs:
            return {
                "status": "error",
                "message": f"Graph '{graph_name}' not found"
            }
            
        # Get the graph
        graph = self.graphs[graph_name]
        
        # Verify proof
        verification = self.proof_verifier.verify_proof(graph)
        
        # Generate improvement suggestions
        if verification["status"] == "success":
            suggestions = self.proof_verifier.suggest_improvements(verification)
            verification["improvement_suggestions"] = suggestions
            
        return verification

# Singleton instance for easy access
_reasoning_coherence_manager = None

def get_reasoning_coherence_manager() -> ReasoningCoherenceManager:
    """
    Get or create the singleton reasoning coherence manager.
    
    Returns:
        ReasoningCoherenceManager instance
    """
    global _reasoning_coherence_manager
    if _reasoning_coherence_manager is None:
        _reasoning_coherence_manager = ReasoningCoherenceManager()
    return _reasoning_coherence_manager
