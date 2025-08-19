"""
Bridge between ψ-mesh semantic associations and ALBERT phase encoding
Enables curvature-driven phase propagation through concept networks
"""

import sys
import os
import json
import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path

# JIT compilation support
try:
    import numba
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    njit = jit
    prange = range

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'albert'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ingest_bus', 'workers'))

try:
    from phase_encode import inject_phase_into_concept_mesh
except ImportError:
    logging.warning("Phase encoding module not available - using fallback")
    def inject_phase_into_concept_mesh(concept_id, phase, amplitude, **kwargs):
        logging.info(f"FALLBACK: Phase injection for {concept_id}: φ={phase:.3f}, A={amplitude:.3f}")

from psi_mesh_integration import psi_mesh

logger = logging.getLogger("tori-ingest.psi_phase_bridge")

class PsiPhaseBridge:
    """
    Connects ψ-mesh semantic associations to phase encoding from spacetime curvature
    
    Enables:
    - Phase propagation through semantic networks
    - Curvature-weighted concept associations
    - Topological feature detection in concept space
    - Soliton memory coupling via phase gradients
    """
    
    def __init__(self):
        self.psi_mesh = psi_mesh
        self.phase_cache = {}  # concept_id -> phase data
        self.propagation_history = []  # Track phase propagation events
        self.phase_coherence_threshold = 0.7  # Minimum coherence for strong coupling
        
        logger.info("ψ-Phase Bridge initialized")
    
    def inject_phase_modulation(self, concept_id: str, phase_data: Dict[str, Any]) -> bool:
        """
        Inject curvature-derived phase into ψ-mesh concept
        
        Args:
            concept_id: Target concept identifier
            phase_data: Phase modulation data from ALBERT
                - phase_value: Phase shift (radians)
                - amplitude_value: Amplitude modulation [0,1]
                - curvature_value: Optional source curvature
                - gradient: Optional phase gradient vector
        
        Returns:
            Success status
        """
        try:
            # Validate phase data
            phase_value = phase_data.get('phase_value', 0.0)
            amplitude_value = phase_data.get('amplitude_value', 1.0)
            
            # Ensure phase is in [-π, π]
            phase_value = np.angle(np.exp(1j * phase_value))
            
            # Store in phase cache
            self.phase_cache[concept_id] = {
                'phase': phase_value,
                'amplitude': amplitude_value,
                'curvature': phase_data.get('curvature_value', 0),
                'gradient': phase_data.get('gradient', [0, 0, 0]),
                'timestamp': datetime.now().isoformat(),
                'source': phase_data.get('source', 'direct_injection')
            }
            
            # Update ψ-mesh semantic vectors
            if hasattr(self.psi_mesh, 'semantic_vectors'):
                self.psi_mesh.semantic_vectors[concept_id] = self.phase_cache[concept_id]
            
            # Inject into ConceptMesh
            success = inject_phase_into_concept_mesh(
                concept_id=concept_id,
                phase_value=phase_value,
                amplitude_value=amplitude_value,
                curvature_value=phase_data.get('curvature_value'),
                metadata={
                    'psi_bridge': True,
                    'injection_time': datetime.now().isoformat()
                }
            )
            
            logger.info(f"Injected phase into {concept_id}: φ={phase_value:.3f}, A={amplitude_value:.3f}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to inject phase into {concept_id}: {e}")
            return False
    
    async def propagate_phase_through_mesh(self, 
                                          initial_concept_id: str, 
                                          phase_value: float, 
                                          amplitude_value: float,
                                          curvature_value: Optional[float] = None,
                                          propagation_depth: int = 2,
                                          decay_factor: float = 0.8) -> Dict[str, Any]:
        """
        Propagate phase modulation through semantic associations
        
        Phase propagates with decay based on:
        - Semantic distance (association strength)
        - Propagation depth
        - Phase coherence between concepts
        
        Args:
            initial_concept_id: Starting concept
            phase_value: Initial phase (radians)
            amplitude_value: Initial amplitude [0,1]
            curvature_value: Optional curvature at concept location
            propagation_depth: Maximum hops through network
            decay_factor: Amplitude decay per hop
        
        Returns:
            Propagation results with affected concepts
        """
        logger.info(f"Starting phase propagation from {initial_concept_id}")
        
        # Initial injection
        self.inject_phase_modulation(initial_concept_id, {
            'phase_value': phase_value,
            'amplitude_value': amplitude_value,
            'curvature_value': curvature_value
        })
        
        # Track propagation
        visited = set()
        affected_concepts = []
        phase_field = {}  # Build phase field over concept network
        
        # BFS queue: (concept_id, phase, amplitude, depth, path)
        queue = [(initial_concept_id, phase_value, amplitude_value, 0, [initial_concept_id])]
        
        while queue:
            concept_id, current_phase, current_amplitude, depth, path = queue.pop(0)
            
            if concept_id in visited or depth >= propagation_depth:
                continue
                
            visited.add(concept_id)
            phase_field[concept_id] = (current_phase, current_amplitude)
            
            # Get associations from ψ-mesh
            associations = await self.psi_mesh.get_concept_associations(concept_id)
            
            for assoc in associations:
                target_id = assoc['target']
                similarity = assoc['similarity']
                
                if target_id in visited:
                    continue
                
                # Calculate phase propagation
                target_phase = self.phase_cache.get(target_id, {}).get('phase', 0)
                
                if NUMBA_AVAILABLE:
                    # Use JIT-compiled propagation calculation
                    propagated_phase, propagated_amplitude = self._propagate_phase_jit(
                        current_phase, current_amplitude, target_phase, similarity, decay_factor
                    )
                else:
                    # Original implementation
                    phase_coupling = self._calculate_phase_coupling(
                        current_phase, target_phase, similarity
                    )
                    propagated_amplitude = current_amplitude * decay_factor * np.sqrt(similarity) * phase_coupling
                    phase_shift = similarity * np.pi * (1 - phase_coupling)
                    propagated_phase = current_phase + phase_shift
                
                # Only propagate if amplitude is significant
                if propagated_amplitude > 0.1:
                    # Inject into target concept
                    self.inject_phase_modulation(target_id, {
                        'phase_value': propagated_phase,
                        'amplitude_value': propagated_amplitude,
                        'source': f'propagated_from_{concept_id}'
                    })
                    
                    affected_concepts.append({
                        'concept_id': target_id,
                        'phase': propagated_phase,
                        'amplitude': propagated_amplitude,
                        'depth': depth + 1,
                        'path': path + [target_id]
                    })
                    
                    # Queue for further propagation
                    queue.append((
                        target_id, 
                        propagated_phase, 
                        propagated_amplitude, 
                        depth + 1,
                        path + [target_id]
                    ))
        
        # Record propagation event
        propagation_event = {
            'timestamp': datetime.now().isoformat(),
            'initial_concept': initial_concept_id,
            'initial_phase': phase_value,
            'initial_amplitude': amplitude_value,
            'affected_concepts': len(affected_concepts),
            'max_depth_reached': max([c['depth'] for c in affected_concepts] + [0]),
            'phase_field_size': len(phase_field)
        }
        
        self.propagation_history.append(propagation_event)
        
        logger.info(f"Phase propagation complete: {len(affected_concepts)} concepts affected")
        
        return {
            'initial_concept': initial_concept_id,
            'affected_concepts': affected_concepts,
            'phase_field': phase_field,
            'propagation_metrics': propagation_event
        }
    
    @staticmethod
    @njit(cache=True)
    def _calculate_phase_coupling_jit(phase_a: float, phase_b: float, similarity: float) -> float:
        """
        JIT-compiled phase coupling calculation
        """
        phase_diff = phase_a - phase_b
        phase_coherence = 0.5 * (1 + np.cos(phase_diff))
        coupling = phase_coherence * similarity
        return coupling
    
    @staticmethod
    @njit(cache=True)
    def _propagate_phase_jit(current_phase: float, current_amplitude: float, 
                            target_phase: float, similarity: float, 
                            decay_factor: float) -> Tuple[float, float]:
        """
        JIT-compiled phase propagation calculation
        """
        # Calculate phase coupling
        phase_diff = current_phase - target_phase
        phase_coherence = 0.5 * (1 + np.cos(phase_diff))
        phase_coupling = phase_coherence * similarity
        
        # Amplitude decays with distance and coupling
        propagated_amplitude = current_amplitude * decay_factor * np.sqrt(similarity) * phase_coupling
        
        # Phase shift includes coupling effects
        phase_shift = similarity * np.pi * (1 - phase_coupling)
        propagated_phase = current_phase + phase_shift
        
        # Wrap phase to [-π, π]
        propagated_phase = np.arctan2(np.sin(propagated_phase), np.cos(propagated_phase))
        
        return propagated_phase, propagated_amplitude
    
    def _calculate_phase_coupling(self, phase_a: float, phase_b: float, similarity: float) -> float:
        """
        Calculate phase coupling strength between concepts
        
        High coupling when:
        - Phases are coherent (small phase difference)
        - High semantic similarity
        
        Returns coupling strength [0,1]
        """
        if NUMBA_AVAILABLE:
            return self._calculate_phase_coupling_jit(phase_a, phase_b, similarity)
        else:
            # Original implementation
            phase_diff = phase_a - phase_b
            phase_coherence = 0.5 * (1 + np.cos(phase_diff))
            coupling = phase_coherence * similarity
            return coupling
    
    async def detect_phase_vortices_in_mesh(self, min_loop_size: int = 3) -> List[List[str]]:
        """
        Detect topological defects (vortices) in the phase field over concept network
        
        A vortex occurs when phase winds by 2π around a closed loop in the network
        
        Args:
            min_loop_size: Minimum number of concepts in a loop
        
        Returns:
            List of concept loops that form phase vortices
        """
        vortices = []
        
        # Find loops in the concept association graph
        # This is a simplified version - full implementation would use graph algorithms
        
        for concept_id in self.phase_cache:
            # Try to find loops starting from this concept
            loops = await self._find_loops_from_concept(concept_id, min_loop_size)
            
            for loop in loops:
                if NUMBA_AVAILABLE:
                    # Prepare phase array for JIT
                    phases = []
                    for concept in loop:
                        if concept in self.phase_cache:
                            phases.append(self.phase_cache[concept]['phase'])
                        else:
                            phases.append(0.0)
                    
                    if len(phases) == len(loop):
                        total_phase = self._calculate_phase_winding_jit(np.array(phases))
                        
                        # Check for vortex
                        if abs(total_phase) > 1.5 * np.pi:
                            vortices.append({
                                'loop': loop,
                                'winding_number': int(round(total_phase / (2 * np.pi))),
                                'total_phase': total_phase
                            })
                            logger.info(f"Phase vortex detected: {loop[:3]}... (winding={total_phase:.2f})")
                else:
                    # Original implementation
                    total_phase = 0
                    
                    for i in range(len(loop)):
                        current = loop[i]
                        next_concept = loop[(i + 1) % len(loop)]
                        
                        if current in self.phase_cache and next_concept in self.phase_cache:
                            phase_current = self.phase_cache[current]['phase']
                            phase_next = self.phase_cache[next_concept]['phase']
                            
                            # Add phase difference
                            total_phase += np.angle(np.exp(1j * (phase_next - phase_current)))
                    
                    # Check for vortex (phase winds by 2π)
                    if abs(total_phase) > 1.5 * np.pi:  # Allow some tolerance
                        vortices.append({
                            'loop': loop,
                            'winding_number': int(round(total_phase / (2 * np.pi))),
                            'total_phase': total_phase
                        })
                        
                        logger.info(f"Phase vortex detected: {loop[:3]}... (winding={total_phase:.2f})")
        
        return vortices
    
    async def _find_loops_from_concept(self, start_concept: str, min_size: int) -> List[List[str]]:
        """Find loops in concept graph starting from given concept"""
        # Simplified loop detection - in practice use proper graph algorithms
        loops = []
        
        # DFS to find paths back to start
        async def dfs(current: str, path: List[str], visited: Set[str]):
            if len(path) >= min_size and current == start_concept:
                loops.append(path.copy())
                return
            
            if current in visited or len(path) > min_size + 2:
                return
            
            visited.add(current)
            associations = await self.psi_mesh.get_concept_associations(current)
            
            for assoc in associations:
                await dfs(assoc['target'], path + [assoc['target']], visited.copy())
        
        await dfs(start_concept, [start_concept], set())
        return loops
    
    @staticmethod
    @njit(cache=True)
    def _calculate_phase_winding_jit(phases: np.ndarray) -> float:
        """
        JIT-compiled phase winding calculation around a loop
        """
        total_phase = 0.0
        n = len(phases)
        
        for i in range(n):
            phase_current = phases[i]
            phase_next = phases[(i + 1) % n]
            
            # Add phase difference with proper wrapping
            dphase = phase_next - phase_current
            # Wrap to [-π, π]
            dphase = np.arctan2(np.sin(dphase), np.cos(dphase))
            total_phase += dphase
        
        return total_phase
    
    @staticmethod
    @njit(cache=True)
    def _compute_gradient_components_jit(phase_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        JIT-compiled gradient computation for a single concept
        """
        if len(phase_values) == 0:
            return np.zeros(3)
        
        # Compute weighted phase differences
        grad_sum = 0.0
        total_weight = 0.0
        
        for i in range(len(phase_values)):
            grad_sum += weights[i] * phase_values[i]
            total_weight += weights[i]
        
        if total_weight > 0:
            gradient = grad_sum / total_weight
        else:
            gradient = 0.0
        
        # Convert to 3D vector (simplified)
        return np.array([gradient, 0.0, 0.0])
    
    def compute_phase_gradient_flow(self) -> Dict[str, np.ndarray]:
        """
        Compute gradient of phase field over concept network
        
        This drives soliton drift in memory dynamics
        
        Returns:
            Gradient vectors for each concept
        """
        gradients = {}
        
        for concept_id, phase_data in self.phase_cache.items():
            # Get neighboring concepts
            associations = self.psi_mesh.associations.get(concept_id, [])
            
            if not associations:
                gradients[concept_id] = np.zeros(3)
                continue
            
            if NUMBA_AVAILABLE:
                # Prepare arrays for JIT
                phase_diffs = []
                weights = []
                
                for assoc in associations:
                    neighbor_id = assoc['target']
                    if neighbor_id in self.phase_cache:
                        phase_diffs.append(self.phase_cache[neighbor_id]['phase'] - phase_data['phase'])
                        weights.append(assoc['similarity'])
                
                if phase_diffs:
                    gradients[concept_id] = self._compute_gradient_components_jit(
                        np.array(phase_diffs), np.array(weights)
                    )
                else:
                    gradients[concept_id] = np.zeros(3)
            else:
                # Original implementation
                grad_components = []
                
                for assoc in associations:
                    neighbor_id = assoc['target']
                    if neighbor_id in self.phase_cache:
                        dphase = self.phase_cache[neighbor_id]['phase'] - phase_data['phase']
                        weight = assoc['similarity']
                        grad_components.append(weight * dphase)
                
                if grad_components:
                    gradient = np.mean(grad_components)
                    gradients[concept_id] = np.array([gradient, 0, 0])
                else:
                    gradients[concept_id] = np.zeros(3)
        
        return gradients
    
    def export_phase_field_snapshot(self, output_path: str) -> None:
        """Export current phase field state for visualization"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'phase_cache': self.phase_cache,
            'propagation_history': self.propagation_history[-10:],  # Last 10 events
            'statistics': {
                'total_concepts': len(self.phase_cache),
                'average_amplitude': np.mean([p['amplitude'] for p in self.phase_cache.values()]) if self.phase_cache else 0,
                'phase_variance': np.var([p['phase'] for p in self.phase_cache.values()]) if self.phase_cache else 0
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        logger.info(f"Phase field snapshot exported to {output_path}")

# Global bridge instance
psi_phase_bridge = PsiPhaseBridge()

# Example usage
async def example_curvature_injection():
    """Example: Inject black hole curvature into concept network"""
    
    # Simulate curvature near black hole
    r_horizon = 2.0  # Schwarzschild radius
    r_concept = 3.0  # Concept location
    
    # Kretschmann scalar K ~ M²/r⁶
    curvature = 1.0 / (r_concept ** 6)
    
    # Convert to phase and amplitude
    phase = np.log(curvature + 1e-10) % (2 * np.pi)
    amplitude = 1.0 / (1.0 + curvature * 10)
    
    # Propagate through concept network
    results = await psi_phase_bridge.propagate_phase_through_mesh(
        initial_concept_id="black_hole",
        phase_value=phase,
        amplitude_value=amplitude,
        curvature_value=curvature,
        propagation_depth=3
    )
    
    print(f"Affected {len(results['affected_concepts'])} concepts")
    
    # Detect vortices
    vortices = await psi_phase_bridge.detect_phase_vortices_in_mesh()
    print(f"Found {len(vortices)} phase vortices")

if __name__ == "__main__":
    asyncio.run(example_curvature_injection())
