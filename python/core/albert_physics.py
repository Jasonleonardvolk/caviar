"""
ALBERT - TORI's ψ-Physics Engine
Computes cognitive geometry properties per tenant mesh
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import threading

logger = logging.getLogger(__name__)

@dataclass
class AlbertState:
    """State of ALBERT physics for a specific mesh"""
    mesh_id: str
    free_energy: float = 0.0
    ricci_curvature: Dict[str, float] = None
    concept_positions: Dict[str, np.ndarray] = None
    temperature: float = 1.0
    entropy: float = 0.0
    last_update: float = 0.0
    
    def __post_init__(self):
        if self.ricci_curvature is None:
            self.ricci_curvature = {}
        if self.concept_positions is None:
            self.concept_positions = {}

class ALBERT:
    """
    Adaptive Learning-Based Entropy Regulation for TORI
    Per-tenant cognitive physics simulation
    """
    
    def __init__(self):
        self.states: Dict[str, AlbertState] = {}
        self.lock = threading.Lock()
        
        # Physics constants
        self.boltzmann_constant = 1.0  # Simplified units
        self.coupling_strength = 0.1
        self.damping_factor = 0.95
        
        logger.info("🧠 ALBERT physics engine initialized")
    
    def get_state(self, mesh_id: str) -> AlbertState:
        """Get or create state for a mesh"""
        if mesh_id not in self.states:
            with self.lock:
                if mesh_id not in self.states:
                    self.states[mesh_id] = AlbertState(mesh_id=mesh_id)
        return self.states[mesh_id]
    
    def compute_free_energy(self, mesh_id: str, concept_mesh) -> float:
        """
        Compute Helmholtz free energy F = U - TS for the concept mesh
        
        Args:
            mesh_id: Identifier for the mesh
            concept_mesh: The ConceptMesh instance
            
        Returns:
            Free energy value
        """
        state = self.get_state(mesh_id)
        
        # Get graph representation
        graph = concept_mesh.graph
        if len(graph.nodes) == 0:
            return 0.0
        
        # Compute internal energy (U)
        # Based on concept relationships and strengths
        internal_energy = 0.0
        
        for u, v, data in graph.edges(data=True):
            strength = data.get('strength', 1.0)
            # Energy contribution from each edge
            internal_energy += -strength * self.coupling_strength
        
        # Add self-energy from concepts
        for node_id in graph.nodes:
            concept = concept_mesh.concepts.get(node_id)
            if concept:
                internal_energy += concept.importance * 0.5
        
        # Compute entropy (S)
        # Based on concept distribution and connectivity
        degree_sequence = [graph.degree(n) for n in graph.nodes]
        if sum(degree_sequence) > 0:
            # Shannon entropy of degree distribution
            degree_probs = np.array(degree_sequence) / sum(degree_sequence)
            degree_probs = degree_probs[degree_probs > 0]  # Remove zeros
            entropy = -np.sum(degree_probs * np.log(degree_probs))
        else:
            entropy = 0.0
        
        # Update state
        state.entropy = entropy
        
        # Compute free energy
        free_energy = internal_energy - state.temperature * entropy
        state.free_energy = free_energy
        
        logger.debug(f"ALBERT[{mesh_id}] F={free_energy:.3f}, U={internal_energy:.3f}, "
                    f"S={entropy:.3f}, T={state.temperature:.3f}")
        
        return free_energy
    
    def compute_ricci_curvature(self, mesh_id: str, concept_mesh) -> Dict[Tuple[str, str], float]:
        """
        Compute Ollivier-Ricci curvature for edges in the concept graph
        
        This measures how "curved" the cognitive space is around each edge,
        indicating conceptual bottlenecks or expansions.
        
        Args:
            mesh_id: Identifier for the mesh
            concept_mesh: The ConceptMesh instance
            
        Returns:
            Dictionary mapping edges to curvature values
        """
        state = self.get_state(mesh_id)
        graph = concept_mesh.graph
        
        if len(graph.nodes) < 2:
            return {}
        
        curvatures = {}
        
        # For each edge, compute Ollivier-Ricci curvature
        for u, v in graph.edges:
            # Get neighbors
            neighbors_u = set(graph.neighbors(u))
            neighbors_v = set(graph.neighbors(v))
            
            # Compute Wasserstein distance approximation
            # Using Jaccard similarity as a proxy
            intersection = len(neighbors_u & neighbors_v)
            union = len(neighbors_u | neighbors_v)
            
            if union > 0:
                jaccard = intersection / union
                # Ricci curvature approximation
                curvature = 2 * jaccard - 1
            else:
                curvature = 0.0
            
            curvatures[(u, v)] = curvature
            
            # Store in state
            edge_key = f"{u}-{v}"
            state.ricci_curvature[edge_key] = curvature
        
        return curvatures
    
    def compute_concept_positions(self, mesh_id: str, concept_mesh, dimensions: int = 3) -> Dict[str, np.ndarray]:
        """
        Compute spatial positions for concepts using force-directed layout
        This creates a "cognitive manifold" embedding
        
        Args:
            mesh_id: Identifier for the mesh
            concept_mesh: The ConceptMesh instance
            dimensions: Number of dimensions for embedding
            
        Returns:
            Dictionary mapping concept IDs to position vectors
        """
        state = self.get_state(mesh_id)
        graph = concept_mesh.graph
        
        if len(graph.nodes) == 0:
            return {}
        
        # Use spring layout for force-directed positioning
        if len(graph.nodes) == 1:
            # Single node at origin
            positions = {list(graph.nodes)[0]: np.zeros(dimensions)}
        else:
            # Compute layout
            pos_2d = nx.spring_layout(graph, k=1.0, iterations=50)
            
            # Extend to requested dimensions
            positions = {}
            for node, pos in pos_2d.items():
                if dimensions == 2:
                    positions[node] = np.array(pos)
                elif dimensions == 3:
                    # Add z-coordinate based on node importance
                    concept = concept_mesh.concepts.get(node)
                    z = concept.importance if concept else 0.5
                    positions[node] = np.array([pos[0], pos[1], z])
                else:
                    # Use MDS for higher dimensions
                    positions = self._compute_mds_positions(graph, dimensions)
                    break
        
        # Update state
        state.concept_positions = positions
        
        return positions
    
    def _compute_mds_positions(self, graph: nx.Graph, dimensions: int) -> Dict[str, np.ndarray]:
        """Use multidimensional scaling for positioning"""
        nodes = list(graph.nodes)
        n = len(nodes)
        
        # Compute shortest path distances
        distances = np.zeros((n, n))
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i != j:
                    try:
                        dist = nx.shortest_path_length(graph, u, v)
                    except nx.NetworkXNoPath:
                        dist = n  # Max distance
                    distances[i, j] = dist
        
        # Apply MDS
        mds = MDS(n_components=dimensions, dissimilarity='precomputed')
        positions_array = mds.fit_transform(distances)
        
        # Convert to dictionary
        return {node: positions_array[i] for i, node in enumerate(nodes)}
    
    def simulate_dynamics(self, mesh_id: str, concept_mesh, time_step: float = 0.1) -> Dict[str, Any]:
        """
        Simulate one step of cognitive dynamics
        
        This updates positions based on forces and computes new physics properties
        
        Args:
            mesh_id: Identifier for the mesh
            concept_mesh: The ConceptMesh instance  
            time_step: Simulation time step
            
        Returns:
            Dictionary of updated physics properties
        """
        state = self.get_state(mesh_id)
        
        # Ensure positions are initialized
        if not state.concept_positions:
            self.compute_concept_positions(mesh_id, concept_mesh)
        
        # Update positions based on forces
        new_positions = {}
        graph = concept_mesh.graph
        
        for node in graph.nodes:
            if node not in state.concept_positions:
                continue
                
            pos = state.concept_positions[node].copy()
            force = np.zeros_like(pos)
            
            # Spring forces from edges
            for neighbor in graph.neighbors(node):
                if neighbor in state.concept_positions:
                    neighbor_pos = state.concept_positions[neighbor]
                    delta = neighbor_pos - pos
                    distance = np.linalg.norm(delta)
                    
                    if distance > 0:
                        # Spring force
                        edge_data = graph[node][neighbor]
                        strength = edge_data.get('strength', 1.0)
                        spring_force = strength * delta / distance
                        force += spring_force * self.coupling_strength
            
            # Repulsion from all nodes
            for other in graph.nodes:
                if other != node and other in state.concept_positions:
                    other_pos = state.concept_positions[other]
                    delta = pos - other_pos
                    distance = np.linalg.norm(delta)
                    
                    if distance > 0:
                        # Coulomb repulsion
                        repulsion = delta / (distance ** 3)
                        force += repulsion * 0.1
            
            # Update position with damping
            new_pos = pos + force * time_step * self.damping_factor
            new_positions[node] = new_pos
        
        # Update state
        state.concept_positions = new_positions
        
        # Recompute physics properties
        free_energy = self.compute_free_energy(mesh_id, concept_mesh)
        curvatures = self.compute_ricci_curvature(mesh_id, concept_mesh)
        
        # Compute average curvature
        avg_curvature = np.mean(list(curvatures.values())) if curvatures else 0.0
        
        return {
            "free_energy": free_energy,
            "entropy": state.entropy,
            "temperature": state.temperature,
            "average_curvature": avg_curvature,
            "num_concepts": len(graph.nodes),
            "num_relations": len(graph.edges)
        }
    
    def get_cognitive_temperature(self, mesh_id: str) -> float:
        """Get the cognitive temperature of a mesh"""
        state = self.get_state(mesh_id)
        return state.temperature
    
    def set_cognitive_temperature(self, mesh_id: str, temperature: float):
        """Set the cognitive temperature (affects entropy weight)"""
        state = self.get_state(mesh_id)
        state.temperature = max(0.1, temperature)  # Minimum temperature
        logger.info(f"ALBERT[{mesh_id}] temperature set to {state.temperature}")
    
    def get_mesh_health(self, mesh_id: str) -> Dict[str, Any]:
        """
        Compute overall health metrics for a concept mesh
        
        Returns indicators of mesh quality and coherence
        """
        state = self.get_state(mesh_id)
        
        health = {
            "mesh_id": mesh_id,
            "free_energy": state.free_energy,
            "entropy": state.entropy,
            "temperature": state.temperature,
            "health_score": 0.0
        }
        
        # Compute health score (0-100)
        # Lower free energy is better (more stable)
        # Moderate entropy is good (not too rigid, not too chaotic)
        
        energy_score = np.exp(-abs(state.free_energy) / 10) * 50
        entropy_score = 50 * np.exp(-abs(state.entropy - 2.0))  # Target entropy ~2.0
        
        health["health_score"] = min(100, energy_score + entropy_score)
        health["energy_contribution"] = energy_score
        health["entropy_contribution"] = entropy_score
        
        # Add curvature statistics
        if state.ricci_curvature:
            curvatures = list(state.ricci_curvature.values())
            health["avg_curvature"] = np.mean(curvatures)
            health["curvature_std"] = np.std(curvatures)
            
            # Negative curvature indicates bottlenecks
            bottlenecks = sum(1 for c in curvatures if c < -0.5)
            health["bottleneck_edges"] = bottlenecks
        
        return health
    
    def reset_state(self, mesh_id: str):
        """Reset physics state for a mesh"""
        with self.lock:
            if mesh_id in self.states:
                del self.states[mesh_id]
        logger.info(f"ALBERT[{mesh_id}] state reset")
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all mesh states"""
        return {
            mesh_id: {
                "free_energy": state.free_energy,
                "entropy": state.entropy,
                "temperature": state.temperature,
                "num_positions": len(state.concept_positions),
                "num_curvatures": len(state.ricci_curvature)
            }
            for mesh_id, state in self.states.items()
        }


# Singleton instance
_albert_instance = None

def get_albert() -> ALBERT:
    """Get singleton ALBERT instance"""
    global _albert_instance
    if _albert_instance is None:
        _albert_instance = ALBERT()
    return _albert_instance
