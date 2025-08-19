"""
🌊 SOLITON COUPLING DRIVER WITH ∇ψ-PHASE GRADIENTS
Enables phase-driven soliton drift, bonding, and topological sync
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

# JIT compilation support
try:
    import numba
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    njit = jit
    prange = range

logger = logging.getLogger("soliton.coupling_driver")


@dataclass
class PhaseGradientField:
    """Phase gradient field data structure"""
    gradient: np.ndarray  # ∇ψ field
    curl: Optional[np.ndarray] = None  # ∇×∇ψ (vorticity)
    divergence: Optional[float] = None  # ∇·∇ψ
    local_tensors: Optional[np.ndarray] = None  # ∂ψ/∂x_i tensors
    coordinates: Optional[Dict[str, np.ndarray]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CouplingZone:
    """Resonance zone where gradients align"""
    center: np.ndarray  # Center position
    radius: float  # Zone radius
    alignment_strength: float  # How well gradients align
    concept_ids: List[str] = field(default_factory=list)
    phase_coherence: float = 0.0


class SolitonCouplingDriver:
    """
    Drives soliton dynamics based on ∇ψ-phase gradients
    
    🔄 Key Features:
    - Multi-axis gradient computation
    - Curl/divergence analysis for vortex detection
    - Resonance zone identification
    - Phase-locked coupling matrix generation
    """
    
    def __init__(self, lattice_size: int = 100):
        self.lattice_size = lattice_size
        self.gradient_fields: Dict[str, PhaseGradientField] = {}
        self.coupling_matrix = np.zeros((lattice_size, lattice_size))
        self.resonance_zones: List[CouplingZone] = []
        self.coupling_strength = 0.1
        
        logger.info("🌊 Soliton Coupling Driver initialized")
    
    @staticmethod
    @njit(cache=True, parallel=True)
    def _compute_gradient_components_jit(psi_phase: np.ndarray, dx: float = 1.0) -> np.ndarray:
        """
        JIT-compiled gradient computation
        Returns gradient field [∂ψ/∂x, ∂ψ/∂y]
        """
        rows, cols = psi_phase.shape
        gradient = np.zeros((rows, cols, 2))
        
        # Parallel computation
        for i in prange(1, rows - 1):
            for j in range(1, cols - 1):
                # Central differences
                gradient[i, j, 0] = (psi_phase[i+1, j] - psi_phase[i-1, j]) / (2 * dx)
                gradient[i, j, 1] = (psi_phase[i, j+1] - psi_phase[i, j-1]) / (2 * dx)
        
        # Handle boundaries
        # Top/bottom
        for j in range(cols):
            gradient[0, j, 0] = (psi_phase[1, j] - psi_phase[0, j]) / dx
            gradient[rows-1, j, 0] = (psi_phase[rows-1, j] - psi_phase[rows-2, j]) / dx
        
        # Left/right
        for i in range(rows):
            gradient[i, 0, 1] = (psi_phase[i, 1] - psi_phase[i, 0]) / dx
            gradient[i, cols-1, 1] = (psi_phase[i, cols-1] - psi_phase[i, cols-2]) / dx
        
        return gradient
    
    @staticmethod
    @njit(cache=True)
    def _compute_curl_2d_jit(gradient: np.ndarray, dx: float = 1.0) -> np.ndarray:
        """
        JIT-compiled 2D curl computation
        For 2D: curl = ∂F_y/∂x - ∂F_x/∂y
        """
        rows, cols = gradient.shape[:2]
        curl = np.zeros((rows, cols))
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # ∂F_y/∂x - ∂F_x/∂y
                dFy_dx = (gradient[i+1, j, 1] - gradient[i-1, j, 1]) / (2 * dx)
                dFx_dy = (gradient[i, j+1, 0] - gradient[i, j-1, 0]) / (2 * dx)
                curl[i, j] = dFy_dx - dFx_dy
        
        return curl
    
    @staticmethod
    @njit(cache=True)
    def _compute_divergence_jit(gradient: np.ndarray, dx: float = 1.0) -> float:
        """
        JIT-compiled divergence computation
        div = ∂F_x/∂x + ∂F_y/∂y
        """
        rows, cols = gradient.shape[:2]
        div_sum = 0.0
        count = 0
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # ∂F_x/∂x + ∂F_y/∂y
                dFx_dx = (gradient[i+1, j, 0] - gradient[i-1, j, 0]) / (2 * dx)
                dFy_dy = (gradient[i, j+1, 1] - gradient[i, j-1, 1]) / (2 * dx)
                div_sum += dFx_dx + dFy_dy
                count += 1
        
        return div_sum / count if count > 0 else 0.0
    
    def compute_phase_gradient(self, 
                              psi_phase: np.ndarray,
                              concept_id: str,
                              coordinates: Optional[Dict[str, np.ndarray]] = None) -> PhaseGradientField:
        """
        📐 Compute full gradient field analysis for phase field
        
        Returns PhaseGradientField with:
        - Gradient vectors ∇ψ
        - Curl (vorticity)
        - Divergence
        - Local tensor components
        """
        logger.info(f"📐 Computing gradient field for concept '{concept_id}'")
        
        # Ensure 2D field
        if psi_phase.ndim == 1:
            # Reshape 1D to 2D square
            size = int(np.sqrt(len(psi_phase)))
            if size * size != len(psi_phase):
                # Pad to perfect square
                pad_size = (size + 1) * (size + 1) - len(psi_phase)
                psi_phase = np.pad(psi_phase, (0, pad_size), mode='edge')
                size = size + 1
            psi_phase = psi_phase.reshape((size, size))
        
        # Compute gradient
        if NUMBA_AVAILABLE:
            gradient = self._compute_gradient_components_jit(psi_phase)
        else:
            # Numpy fallback
            gradient = np.stack(np.gradient(psi_phase), axis=-1)
        
        # Compute curl (vorticity)
        if NUMBA_AVAILABLE:
            curl = self._compute_curl_2d_jit(gradient)
        else:
            # Manual curl computation
            curl = np.zeros_like(psi_phase)
            for i in range(1, psi_phase.shape[0] - 1):
                for j in range(1, psi_phase.shape[1] - 1):
                    curl[i, j] = (gradient[i+1, j, 1] - gradient[i-1, j, 1] - 
                                  gradient[i, j+1, 0] + gradient[i, j-1, 0]) / 2.0
        
        # Compute divergence
        if NUMBA_AVAILABLE:
            divergence = self._compute_divergence_jit(gradient)
        else:
            # Manual divergence
            div_x = np.gradient(gradient[:, :, 0], axis=0)
            div_y = np.gradient(gradient[:, :, 1], axis=1)
            divergence = float(np.mean(div_x + div_y))
        
        # Create gradient field object
        grad_field = PhaseGradientField(
            gradient=gradient,
            curl=curl,
            divergence=divergence,
            local_tensors=gradient,  # For now, same as gradient
            coordinates=coordinates
        )
        
        # Store in cache
        self.gradient_fields[concept_id] = grad_field
        
        # Log statistics
        logger.info(f"   Gradient magnitude: mean={np.mean(np.linalg.norm(gradient, axis=2)):.3f}")
        logger.info(f"   Curl (vorticity): max={np.max(np.abs(curl)):.3f}")
        logger.info(f"   Divergence: {divergence:.3f}")
        
        return grad_field
    
    def update_coupling_matrix(self, soliton_positions: List[Tuple[float, float]]) -> np.ndarray:
        """
        🔄 Update coupling matrix based on gradient alignment
        
        High coupling where:
        - Gradients align (parallel)
        - Phase coherence is high
        - Spatial proximity
        """
        n_solitons = len(soliton_positions)
        if n_solitons == 0:
            return self.coupling_matrix
        
        # Reset matrix
        self.coupling_matrix = np.zeros((n_solitons, n_solitons))
        
        # Get average gradient field
        if not self.gradient_fields:
            return self.coupling_matrix
        
        # Use most recent gradient field
        recent_field = list(self.gradient_fields.values())[-1]
        gradient = recent_field.gradient
        
        # Compute coupling for each soliton pair
        for i in range(n_solitons):
            for j in range(i + 1, n_solitons):
                pos_i = np.array(soliton_positions[i])
                pos_j = np.array(soliton_positions[j])
                
                # Spatial distance factor
                distance = np.linalg.norm(pos_i - pos_j)
                spatial_coupling = np.exp(-distance / 10.0)  # Decay length = 10
                
                # Sample gradients at positions
                try:
                    # Convert to grid indices
                    idx_i = (int(pos_i[0] % gradient.shape[0]), 
                            int(pos_i[1] % gradient.shape[1]))
                    idx_j = (int(pos_j[0] % gradient.shape[0]), 
                            int(pos_j[1] % gradient.shape[1]))
                    
                    grad_i = gradient[idx_i[0], idx_i[1]]
                    grad_j = gradient[idx_j[0], idx_j[1]]
                    
                    # Gradient alignment (dot product)
                    if np.linalg.norm(grad_i) > 0 and np.linalg.norm(grad_j) > 0:
                        alignment = np.dot(grad_i, grad_j) / (
                            np.linalg.norm(grad_i) * np.linalg.norm(grad_j)
                        )
                        alignment = max(0, alignment)  # Only positive alignment
                    else:
                        alignment = 0
                    
                    # Total coupling
                    coupling = self.coupling_strength * spatial_coupling * alignment
                    
                    # Symmetric matrix
                    self.coupling_matrix[i, j] = coupling
                    self.coupling_matrix[j, i] = coupling
                    
                except IndexError:
                    # Out of bounds, no coupling
                    pass
        
        return self.coupling_matrix
    
    def find_resonance_zones(self, min_alignment: float = 0.7) -> List[CouplingZone]:
        """
        🎯 Find resonance zones where gradients align
        
        These are regions where:
        - Multiple gradient vectors point similarly
        - Phase coherence enables coupling
        - Solitons can bond or synchronize
        """
        self.resonance_zones.clear()
        
        if not self.gradient_fields:
            return self.resonance_zones
        
        # Analyze each gradient field
        for concept_id, grad_field in self.gradient_fields.items():
            gradient = grad_field.gradient
            
            # Find regions of aligned gradients
            rows, cols = gradient.shape[:2]
            
            # Sliding window to find alignment
            window_size = 5
            for i in range(0, rows - window_size, 2):
                for j in range(0, cols - window_size, 2):
                    # Extract window
                    window = gradient[i:i+window_size, j:j+window_size]
                    
                    # Compute average gradient direction
                    avg_grad = np.mean(window.reshape(-1, 2), axis=0)
                    
                    if np.linalg.norm(avg_grad) < 0.01:
                        continue  # Skip near-zero gradients
                    
                    # Check alignment within window
                    alignments = []
                    for gi in range(window_size):
                        for gj in range(window_size):
                            local_grad = window[gi, gj]
                            if np.linalg.norm(local_grad) > 0.01:
                                alignment = np.dot(local_grad, avg_grad) / (
                                    np.linalg.norm(local_grad) * np.linalg.norm(avg_grad)
                                )
                                alignments.append(alignment)
                    
                    if alignments:
                        mean_alignment = np.mean(alignments)
                        
                        if mean_alignment >= min_alignment:
                            # Found resonance zone
                            zone = CouplingZone(
                                center=np.array([i + window_size/2, j + window_size/2]),
                                radius=window_size / 2,
                                alignment_strength=mean_alignment,
                                concept_ids=[concept_id],
                                phase_coherence=1.0 - np.std(alignments)
                            )
                            self.resonance_zones.append(zone)
        
        # Merge nearby zones
        self._merge_nearby_zones()
        
        logger.info(f"🎯 Found {len(self.resonance_zones)} resonance zones")
        return self.resonance_zones
    
    def _merge_nearby_zones(self, merge_distance: float = 10.0):
        """Merge resonance zones that are close together"""
        if len(self.resonance_zones) < 2:
            return
        
        merged = []
        used = set()
        
        for i, zone1 in enumerate(self.resonance_zones):
            if i in used:
                continue
                
            # Start new merged zone
            merged_zone = CouplingZone(
                center=zone1.center.copy(),
                radius=zone1.radius,
                alignment_strength=zone1.alignment_strength,
                concept_ids=zone1.concept_ids.copy(),
                phase_coherence=zone1.phase_coherence
            )
            
            # Find nearby zones to merge
            for j, zone2 in enumerate(self.resonance_zones[i+1:], i+1):
                if j in used:
                    continue
                    
                distance = np.linalg.norm(zone1.center - zone2.center)
                if distance < merge_distance:
                    # Merge zones
                    used.add(j)
                    
                    # Update merged zone
                    total_weight = merged_zone.alignment_strength + zone2.alignment_strength
                    merged_zone.center = (
                        merged_zone.center * merged_zone.alignment_strength +
                        zone2.center * zone2.alignment_strength
                    ) / total_weight
                    
                    merged_zone.radius = max(merged_zone.radius, zone2.radius) + distance/2
                    merged_zone.alignment_strength = np.mean([
                        merged_zone.alignment_strength, zone2.alignment_strength
                    ])
                    merged_zone.concept_ids.extend(zone2.concept_ids)
                    merged_zone.phase_coherence = np.mean([
                        merged_zone.phase_coherence, zone2.phase_coherence
                    ])
            
            merged.append(merged_zone)
        
        self.resonance_zones = merged
    
    def compute_drift_forces(self, soliton_position: np.ndarray) -> np.ndarray:
        """
        🌊 Compute drift forces on soliton from gradient fields
        
        Force = -∇ψ × curvature_coupling
        """
        if not self.gradient_fields:
            return np.zeros(2)
        
        # Use average of all gradient fields (weighted by recency)
        total_force = np.zeros(2)
        weights = []
        
        for i, grad_field in enumerate(self.gradient_fields.values()):
            # Recency weight (newer fields have more influence)
            weight = np.exp(-i * 0.1)
            weights.append(weight)
            
            # Sample gradient at soliton position
            try:
                idx = (int(soliton_position[0] % grad_field.gradient.shape[0]),
                      int(soliton_position[1] % grad_field.gradient.shape[1]))
                
                local_gradient = grad_field.gradient[idx[0], idx[1]]
                
                # Force proportional to negative gradient (drift downhill)
                force = -local_gradient * weight
                total_force += force
                
            except IndexError:
                pass
        
        # Normalize by total weight
        if weights:
            total_force /= sum(weights)
        
        return total_force
    
    def export_gradient_data(self, output_path: str):
        """
        📤 Export gradient field data for visualization
        
        Outputs:
        - Gradient vectors as .npy
        - Curl/divergence fields
        - Resonance zones as JSON
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'gradient_fields': {},
            'resonance_zones': []
        }
        
        # Export gradient fields
        for concept_id, grad_field in self.gradient_fields.items():
            # Save numpy arrays
            grad_path = f"{output_path}_{concept_id}_gradient.npy"
            curl_path = f"{output_path}_{concept_id}_curl.npy"
            
            np.save(grad_path, grad_field.gradient)
            if grad_field.curl is not None:
                np.save(curl_path, grad_field.curl)
            
            export_data['gradient_fields'][concept_id] = {
                'gradient_file': grad_path,
                'curl_file': curl_path,
                'divergence': grad_field.divergence,
                'shape': grad_field.gradient.shape,
                'timestamp': grad_field.timestamp.isoformat()
            }
        
        # Export resonance zones
        for zone in self.resonance_zones:
            export_data['resonance_zones'].append({
                'center': zone.center.tolist(),
                'radius': zone.radius,
                'alignment_strength': zone.alignment_strength,
                'concept_ids': zone.concept_ids,
                'phase_coherence': zone.phase_coherence
            })
        
        # Save metadata
        with open(f"{output_path}_metadata.json", 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📤 Exported gradient data to {output_path}")
    
    def visualize_gradients(self) -> Dict[str, Any]:
        """
        🎨 Generate visualization data for gradient fields
        
        Returns dict suitable for plotting with matplotlib/plotly
        """
        viz_data = {
            'gradient_fields': {},
            'resonance_zones': [],
            'coupling_matrix': self.coupling_matrix.tolist() if self.coupling_matrix.size > 0 else []
        }
        
        for concept_id, grad_field in self.gradient_fields.items():
            # Subsample for visualization
            step = max(1, grad_field.gradient.shape[0] // 20)
            
            y, x = np.mgrid[0:grad_field.gradient.shape[0]:step, 
                           0:grad_field.gradient.shape[1]:step]
            u = grad_field.gradient[::step, ::step, 0]
            v = grad_field.gradient[::step, ::step, 1]
            
            viz_data['gradient_fields'][concept_id] = {
                'x': x.tolist(),
                'y': y.tolist(),
                'u': u.tolist(),
                'v': v.tolist(),
                'curl': grad_field.curl[::step, ::step].tolist() if grad_field.curl is not None else None,
                'divergence': grad_field.divergence
            }
        
        # Add resonance zones
        for zone in self.resonance_zones:
            viz_data['resonance_zones'].append({
                'center': zone.center.tolist(),
                'radius': zone.radius,
                'strength': zone.alignment_strength
            })
        
        return viz_data


# Example usage
if __name__ == "__main__":
    # Create coupling driver
    driver = SolitonCouplingDriver(lattice_size=50)
    
    # Example phase field (spiral pattern)
    x = np.linspace(-np.pi, np.pi, 50)
    y = np.linspace(-np.pi, np.pi, 50)
    X, Y = np.meshgrid(x, y)
    
    # Spiral phase with singularity
    psi_phase = np.angle(X + 1j*Y)  # Vortex at origin
    
    # Compute gradient field
    grad_field = driver.compute_phase_gradient(psi_phase, "spiral_vortex")
    
    # Find resonance zones
    zones = driver.find_resonance_zones()
    print(f"Found {len(zones)} resonance zones")
    
    # Example soliton positions
    soliton_positions = [(10, 10), (20, 20), (30, 30), (15, 25)]
    
    # Update coupling matrix
    coupling = driver.update_coupling_matrix(soliton_positions)
    print(f"Coupling matrix shape: {coupling.shape}")
    print(f"Max coupling strength: {np.max(coupling):.3f}")
    
    # Compute drift force at a position
    force = driver.compute_drift_forces(np.array([25, 25]))
    print(f"Drift force at (25,25): {force}")
    
    # Export data
    driver.export_gradient_data("output/gradient_analysis")
