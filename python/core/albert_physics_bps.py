"""
ALBERT - TORI's ψ-Physics Engine (BPS-Enhanced Version)
=========================================================
Computes cognitive geometry properties per tenant mesh
Now fully integrated with BPS supersymmetric configuration
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import threading
import time
from datetime import datetime

# BPS Configuration Integration
try:
    from .bps_config import (
        # Feature flags
        ENABLE_BPS_CHARGE_TRACKING, ENABLE_BPS_DIAGNOSTICS,
        ENABLE_BPS_ADAPTIVE_SCALING, ENABLE_PHASE_SPONGE,
        
        # Energy parameters
        ENERGY_PER_Q, BPS_ENERGY_QUANTUM, ENERGY_EXTRACTION_EFFICIENCY,
        
        # Charge parameters
        ALLOWED_Q_VALUES, MAX_ALLOWED_CHARGE_MAGNITUDE,
        
        # Phase sponge parameters
        PHASE_SPONGE_DAMPING_FACTOR, PHASE_SPONGE_BOUNDARY_WIDTH,
        
        # Tolerances
        BPS_SATURATION_TOLERANCE, ENERGY_CONSISTENCY_TOLERANCE,
        
        # Performance
        PERFORMANCE_PROFILING_ENABLED, SLOW_OPERATION_THRESHOLD
    )
    BPS_INTEGRATED = True
    logger = logging.getLogger("ALBERT-BPS")
    logger.info("🚀 ALBERT using BPS supersymmetric configuration!")
except ImportError:
    BPS_INTEGRATED = False
    logger = logging.getLogger("ALBERT")
    logger.warning("BPS config not available - using standalone mode")
    
    # Fallback values
    ENABLE_BPS_CHARGE_TRACKING = False
    ENABLE_BPS_DIAGNOSTICS = False
    ENABLE_BPS_ADAPTIVE_SCALING = False
    ENABLE_PHASE_SPONGE = False
    ENERGY_PER_Q = 1.0
    BPS_ENERGY_QUANTUM = 1.0
    PERFORMANCE_PROFILING_ENABLED = False

@dataclass
class AlbertState:
    """Enhanced state of ALBERT physics with BPS integration"""
    mesh_id: str
    free_energy: float = 0.0
    ricci_curvature: Dict[str, float] = field(default_factory=dict)
    concept_positions: Dict[str, np.ndarray] = field(default_factory=dict)
    temperature: float = 1.0
    entropy: float = 0.0
    last_update: float = 0.0
    
    # BPS-specific fields
    topological_charge: float = 0.0
    bps_energy: float = 0.0
    phase_coherence: float = 1.0
    soliton_density: float = 0.0
    boundary_nodes: List[str] = field(default_factory=list)
    phase_sponge_active: bool = False

class ALBERT:
    """
    Adaptive Learning-Based Entropy Regulation for TORI
    Per-tenant cognitive physics simulation with BPS supersymmetry
    """
    
    def __init__(self):
        self.states: Dict[str, AlbertState] = {}
        self.lock = threading.Lock()
        
        # Physics constants
        self.boltzmann_constant = 1.0  # Simplified units
        self.coupling_strength = 0.1
        self.damping_factor = 0.95
        
        # BPS integration parameters
        self.bps_integrated = BPS_INTEGRATED
        self.enable_charge_tracking = ENABLE_BPS_CHARGE_TRACKING if BPS_INTEGRATED else False
        self.enable_phase_sponge = ENABLE_PHASE_SPONGE if BPS_INTEGRATED else False
        
        logger.info(f"🧠 ALBERT physics engine initialized (BPS: {self.bps_integrated})")
    
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
        Enhanced with BPS energy contributions
        
        Args:
            mesh_id: Identifier for the mesh
            concept_mesh: The ConceptMesh instance
            
        Returns:
            Free energy value
        """
        start_time = time.time() if PERFORMANCE_PROFILING_ENABLED else 0
        state = self.get_state(mesh_id)
        
        # Get graph representation
        graph = concept_mesh.graph
        if len(graph.nodes) == 0:
            return 0.0
        
        # Compute internal energy (U)
        internal_energy = 0.0
        
        for u, v, data in graph.edges(data=True):
            strength = data.get('strength', 1.0)
            # Energy contribution from each edge
            internal_energy += -strength * self.coupling_strength
            
            # BPS contribution if enabled
            if self.enable_charge_tracking:
                edge_charge = data.get('topological_charge', 0)
                if edge_charge != 0:
                    internal_energy += abs(edge_charge) * ENERGY_PER_Q
        
        # Add self-energy from concepts
        for node_id in graph.nodes:
            concept = concept_mesh.concepts.get(node_id)
            if concept:
                internal_energy += concept.importance * 0.5
                
                # BPS soliton energy if present
                if self.enable_charge_tracking and hasattr(concept, 'charge'):
                    internal_energy += abs(concept.charge) * BPS_ENERGY_QUANTUM
        
        # Compute entropy (S)
        degree_sequence = [graph.degree(n) for n in graph.nodes]
        if sum(degree_sequence) > 0:
            # Shannon entropy of degree distribution
            degree_probs = np.array(degree_sequence) / sum(degree_sequence)
            degree_probs = degree_probs[degree_probs > 0]
            entropy = -np.sum(degree_probs * np.log(degree_probs))
        else:
            entropy = 0.0
        
        # Update state
        state.entropy = entropy
        state.bps_energy = internal_energy if self.enable_charge_tracking else 0.0
        
        # Compute free energy with BPS corrections
        free_energy = internal_energy - state.temperature * entropy
        
        # Apply BPS saturation bound if enabled
        if self.enable_charge_tracking and state.topological_charge != 0:
            min_energy = abs(state.topological_charge) * ENERGY_PER_Q
            if free_energy < min_energy - BPS_SATURATION_TOLERANCE:
                logger.warning(f"BPS bound violation: F={free_energy} < |Q|={min_energy}")
                free_energy = min_energy
        
        state.free_energy = free_energy
        
        if PERFORMANCE_PROFILING_ENABLED:
            elapsed = time.time() - start_time
            if elapsed > SLOW_OPERATION_THRESHOLD:
                logger.warning(f"Slow free energy computation: {elapsed:.3f}s")
        
        logger.debug(f"ALBERT[{mesh_id}] F={free_energy:.3f}, U={internal_energy:.3f}, "
                    f"S={entropy:.3f}, T={state.temperature:.3f}, BPS={state.bps_energy:.3f}")
        
        return free_energy
    
    def compute_topological_charge(self, mesh_id: str, concept_mesh) -> float:
        """
        Compute total topological charge of the concept mesh
        
        Args:
            mesh_id: Identifier for the mesh
            concept_mesh: The ConceptMesh instance
            
        Returns:
            Total topological charge
        """
        if not self.enable_charge_tracking:
            return 0.0
        
        state = self.get_state(mesh_id)
        graph = concept_mesh.graph
        
        total_charge = 0.0
        
        # Sum charges from nodes
        for node_id in graph.nodes:
            concept = concept_mesh.concepts.get(node_id)
            if concept and hasattr(concept, 'charge'):
                total_charge += concept.charge
        
        # Sum charges from edges (if any)
        for u, v, data in graph.edges(data=True):
            edge_charge = data.get('topological_charge', 0)
            total_charge += edge_charge
        
        state.topological_charge = total_charge
        
        # Compute soliton density
        if len(graph.nodes) > 0:
            state.soliton_density = abs(total_charge) / len(graph.nodes)
        
        return total_charge
    
    def apply_phase_sponge(self, mesh_id: str, concept_mesh) -> Dict[str, float]:
        """
        Apply phase sponge damping to boundary nodes
        
        Args:
            mesh_id: Identifier for the mesh
            concept_mesh: The ConceptMesh instance
            
        Returns:
            Dictionary of node dampings
        """
        if not self.enable_phase_sponge:
            return {}
        
        state = self.get_state(mesh_id)
        graph = concept_mesh.graph
        
        if len(graph.nodes) == 0:
            return {}
        
        # Identify boundary nodes (low connectivity)
        avg_degree = sum(dict(graph.degree()).values()) / len(graph.nodes)
        boundary_threshold = avg_degree * 0.5
        
        dampings = {}
        boundary_nodes = []
        
        for node in graph.nodes:
            degree = graph.degree(node)
            
            if degree <= boundary_threshold:
                # This is a boundary node
                boundary_nodes.append(node)
                
                # Calculate damping based on connectivity
                if avg_degree > 0:
                    relative_connectivity = degree / avg_degree
                else:
                    relative_connectivity = 0
                
                # Apply phase sponge profile
                damping = 1.0 - (1.0 - PHASE_SPONGE_DAMPING_FACTOR) * (1.0 - relative_connectivity)
                dampings[node] = damping
            else:
                dampings[node] = 1.0  # No damping for interior nodes
        
        state.boundary_nodes = boundary_nodes
        state.phase_sponge_active = len(boundary_nodes) > 0
        
        if state.phase_sponge_active:
            logger.debug(f"Phase sponge active: {len(boundary_nodes)} boundary nodes identified")
        
        return dampings
    
    def compute_ricci_curvature(self, mesh_id: str, concept_mesh) -> Dict[Tuple[str, str], float]:
        """
        Compute Ollivier-Ricci curvature for edges in the concept graph
        Enhanced with BPS corrections
        
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
        
        # Apply phase sponge if enabled
        dampings = self.apply_phase_sponge(mesh_id, concept_mesh) if self.enable_phase_sponge else {}
        
        for u, v in graph.edges:
            # Get neighbors
            neighbors_u = set(graph.neighbors(u))
            neighbors_v = set(graph.neighbors(v))
            
            # Compute Wasserstein distance approximation
            intersection = len(neighbors_u & neighbors_v)
            union = len(neighbors_u | neighbors_v)
            
            if union > 0:
                jaccard = intersection / union
                # Ricci curvature approximation
                curvature = 2 * jaccard - 1
                
                # Apply phase sponge damping to curvature
                if self.enable_phase_sponge:
                    damping_u = dampings.get(u, 1.0)
                    damping_v = dampings.get(v, 1.0)
                    avg_damping = (damping_u + damping_v) / 2
                    curvature *= avg_damping
                
                # BPS correction: high charge density increases curvature magnitude
                if self.enable_charge_tracking:
                    edge_data = graph[u][v]
                    edge_charge = edge_data.get('topological_charge', 0)
                    if edge_charge != 0:
                        charge_factor = 1.0 + abs(edge_charge) * 0.1
                        curvature *= charge_factor
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
        Enhanced with BPS charge repulsion/attraction
        
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
            positions = {list(graph.nodes)[0]: np.zeros(dimensions)}
        else:
            # Include BPS charge weights if available
            if self.enable_charge_tracking:
                # Create weight dict based on charges
                weights = {}
                for u, v, data in graph.edges(data=True):
                    charge = data.get('topological_charge', 0)
                    strength = data.get('strength', 1.0)
                    # Opposite charges attract, same charges repel
                    weight = strength * (1.0 + abs(charge))
                    weights[(u, v)] = weight
                
                # Use weighted spring layout
                pos_2d = nx.spring_layout(graph, weight='weight', k=1.0, iterations=50)
            else:
                pos_2d = nx.spring_layout(graph, k=1.0, iterations=50)
            
            # Extend to requested dimensions
            positions = {}
            for node, pos in pos_2d.items():
                if dimensions == 2:
                    positions[node] = np.array(pos)
                elif dimensions == 3:
                    # Add z-coordinate based on node importance or charge
                    concept = concept_mesh.concepts.get(node)
                    if self.enable_charge_tracking and concept and hasattr(concept, 'charge'):
                        z = concept.charge * 0.5 + 0.5  # Charge determines height
                    elif concept:
                        z = concept.importance
                    else:
                        z = 0.5
                    positions[node] = np.array([pos[0], pos[1], z])
                else:
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
        mds = MDS(n_components=dimensions, dissimilarity='precomputed', random_state=42)
        positions_array = mds.fit_transform(distances)
        
        # Convert to dictionary
        return {node: positions_array[i] for i, node in enumerate(nodes)}
    
    def compute_phase_coherence(self, mesh_id: str, concept_mesh) -> float:
        """
        Compute phase coherence of the concept mesh
        Measures how well-aligned the concept phases are
        
        Args:
            mesh_id: Identifier for the mesh
            concept_mesh: The ConceptMesh instance
            
        Returns:
            Phase coherence value [0, 1]
        """
        if not self.enable_charge_tracking:
            return 1.0
        
        state = self.get_state(mesh_id)
        graph = concept_mesh.graph
        
        if len(graph.nodes) == 0:
            return 1.0
        
        # Compute complex order parameter
        complex_sum = 0.0 + 0.0j
        count = 0
        
        for node_id in graph.nodes:
            concept = concept_mesh.concepts.get(node_id)
            if concept and hasattr(concept, 'phase'):
                phase = concept.phase
            else:
                # Use position-based phase
                if node_id in state.concept_positions:
                    pos = state.concept_positions[node_id]
                    phase = np.arctan2(pos[1] if len(pos) > 1 else 0, pos[0])
                else:
                    phase = 0
            
            complex_sum += np.exp(1j * phase)
            count += 1
        
        if count > 0:
            coherence = abs(complex_sum) / count
        else:
            coherence = 1.0
        
        state.phase_coherence = coherence
        return coherence
    
    def simulate_dynamics(self, mesh_id: str, concept_mesh, time_step: float = 0.1) -> Dict[str, Any]:
        """
        Simulate one step of cognitive dynamics with BPS physics
        
        Args:
            mesh_id: Identifier for the mesh
            concept_mesh: The ConceptMesh instance  
            time_step: Simulation time step
            
        Returns:
            Dictionary of updated physics properties
        """
        start_time = time.time() if PERFORMANCE_PROFILING_ENABLED else 0
        state = self.get_state(mesh_id)
        
        # Ensure positions are initialized
        if not state.concept_positions:
            self.compute_concept_positions(mesh_id, concept_mesh)
        
        # Apply phase sponge damping
        dampings = self.apply_phase_sponge(mesh_id, concept_mesh) if self.enable_phase_sponge else {}
        
        # Update positions based on forces
        new_positions = {}
        graph = concept_mesh.graph
        
        for node in graph.nodes:
            if node not in state.concept_positions:
                continue
                
            pos = state.concept_positions[node].copy()
            force = np.zeros_like(pos)
            
            # Get node charge if available
            node_charge = 0
            if self.enable_charge_tracking:
                concept = concept_mesh.concepts.get(node)
                if concept and hasattr(concept, 'charge'):
                    node_charge = concept.charge
            
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
                        
                        # BPS charge interaction
                        if self.enable_charge_tracking:
                            neighbor_concept = concept_mesh.concepts.get(neighbor)
                            if neighbor_concept and hasattr(neighbor_concept, 'charge'):
                                neighbor_charge = neighbor_concept.charge
                                # Coulomb force: same sign repels, opposite attracts
                                charge_force = -node_charge * neighbor_charge / (distance ** 2)
                                spring_force += charge_force * delta / distance * ENERGY_PER_Q
                        
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
            
            # Apply phase sponge damping
            damping = dampings.get(node, 1.0) if self.enable_phase_sponge else 1.0
            
            # Update position with damping
            new_pos = pos + force * time_step * self.damping_factor * damping
            new_positions[node] = new_pos
        
        # Update state
        state.concept_positions = new_positions
        state.last_update = time.time()
        
        # Recompute physics properties
        free_energy = self.compute_free_energy(mesh_id, concept_mesh)
        curvatures = self.compute_ricci_curvature(mesh_id, concept_mesh)
        topological_charge = self.compute_topological_charge(mesh_id, concept_mesh)
        phase_coherence = self.compute_phase_coherence(mesh_id, concept_mesh)
        
        # Compute average curvature
        avg_curvature = np.mean(list(curvatures.values())) if curvatures else 0.0
        
        if PERFORMANCE_PROFILING_ENABLED:
            elapsed = time.time() - start_time
            if elapsed > SLOW_OPERATION_THRESHOLD:
                logger.warning(f"Slow dynamics simulation: {elapsed:.3f}s")
        
        return {
            "free_energy": free_energy,
            "entropy": state.entropy,
            "temperature": state.temperature,
            "average_curvature": avg_curvature,
            "topological_charge": topological_charge,
            "bps_energy": state.bps_energy,
            "phase_coherence": phase_coherence,
            "soliton_density": state.soliton_density,
            "boundary_nodes": len(state.boundary_nodes),
            "phase_sponge_active": state.phase_sponge_active,
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
        Enhanced with BPS indicators
        
        Returns indicators of mesh quality and coherence
        """
        state = self.get_state(mesh_id)
        
        health = {
            "mesh_id": mesh_id,
            "free_energy": state.free_energy,
            "entropy": state.entropy,
            "temperature": state.temperature,
            "health_score": 0.0,
            "bps_integrated": self.bps_integrated
        }
        
        # Compute health score (0-100)
        energy_score = np.exp(-abs(state.free_energy) / 10) * 30
        entropy_score = 30 * np.exp(-abs(state.entropy - 2.0))  # Target entropy ~2.0
        
        # BPS contributions to health
        bps_score = 0
        if self.enable_charge_tracking:
            # Check BPS saturation
            if state.topological_charge != 0:
                expected_energy = abs(state.topological_charge) * ENERGY_PER_Q
                if abs(state.bps_energy - expected_energy) < BPS_SATURATION_TOLERANCE:
                    bps_score += 20  # BPS saturated state is healthy
            
            # Phase coherence contribution
            bps_score += state.phase_coherence * 20
            
            health["bps_score"] = bps_score
            health["topological_charge"] = state.topological_charge
            health["phase_coherence"] = state.phase_coherence
            health["soliton_density"] = state.soliton_density
        
        health["health_score"] = min(100, energy_score + entropy_score + bps_score)
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
        
        # Phase sponge status
        if self.enable_phase_sponge:
            health["phase_sponge_active"] = state.phase_sponge_active
            health["boundary_nodes"] = len(state.boundary_nodes)
        
        return health
    
    def inject_soliton(self, mesh_id: str, node_id: str, charge: float, phase: float = 0.0):
        """
        Inject a BPS soliton into a specific concept node
        
        Args:
            mesh_id: Identifier for the mesh
            node_id: Node to inject soliton into
            charge: Topological charge of soliton
            phase: Initial phase of soliton
        """
        if not self.enable_charge_tracking:
            logger.warning("Cannot inject soliton: BPS charge tracking disabled")
            return
        
        state = self.get_state(mesh_id)
        
        # Quantize charge to allowed values
        quantized_charge = min(ALLOWED_Q_VALUES, key=lambda x: abs(x - charge))
        
        logger.info(f"Injecting soliton: mesh={mesh_id}, node={node_id}, Q={quantized_charge}, φ={phase}")
        
        # This would interface with the actual concept mesh
        # For now, we just update our state
        state.topological_charge += quantized_charge
        state.soliton_density = abs(state.topological_charge) / max(len(state.concept_positions), 1)
    
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
                "topological_charge": state.topological_charge,
                "phase_coherence": state.phase_coherence,
                "soliton_density": state.soliton_density,
                "num_positions": len(state.concept_positions),
                "num_curvatures": len(state.ricci_curvature),
                "boundary_nodes": len(state.boundary_nodes),
                "bps_integrated": self.bps_integrated
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

# Export key components
__all__ = [
    'ALBERT',
    'AlbertState',
    'get_albert'
]

if __name__ == "__main__":
    # Test ALBERT with BPS integration
    albert = get_albert()
    logger.info(f"ALBERT initialized with BPS: {albert.bps_integrated}")
    
    # Create a test state
    test_mesh_id = "test_mesh_001"
    state = albert.get_state(test_mesh_id)
    
    logger.info(f"Test state created: {state.mesh_id}")
    logger.info(f"BPS features enabled: charge_tracking={albert.enable_charge_tracking}, "
               f"phase_sponge={albert.enable_phase_sponge}")
    
    # Test soliton injection if BPS is available
    if albert.enable_charge_tracking:
        albert.inject_soliton(test_mesh_id, "concept_1", 1.0, np.pi/4)
        logger.info(f"Soliton injected, charge={state.topological_charge}")
    
    # Get health metrics
    health = albert.get_mesh_health(test_mesh_id)
    logger.info(f"Mesh health: {health}")
    
    logger.info("ALBERT BPS integration test complete!")
