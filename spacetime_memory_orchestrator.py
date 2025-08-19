"""
🎼 SYMBOLIC-TO-NUMERIC ORCHESTRATION
Unified interface for the complete physics-bonded cognitive architecture

This orchestrates:
- Curvature → Phase encoding (ALBERT)
- Phase → Concept injection (ConceptMesh) 
- Gradients → Soliton coupling (CouplingDriver)
- Geodesics → Phase integration (GeodesicIntegrator)
- Feedback → Adaptive optimization (OscillatorFeedback)
"""

import numpy as np
import sympy as sp
from pathlib import Path
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import json
import yaml

# Import all our components
import sys
import os
sys.path.append(os.path.dirname(__file__))

from albert.phase_encode import encode_curvature_to_phase
from albert.inject_curvature import CurvatureInjector
from python.core.concept_mesh import ConceptMesh
from python.core.fractal_soliton_memory import FractalSolitonMemory
from python.core.psi_phase_bridge import psi_phase_bridge
from soliton.coupling_driver import SolitonCouplingDriver
from physics.geodesics import GeodesicIntegrator
from feedback.oscillator_feedback import feedback_loop

logger = logging.getLogger("orchestrator")


class SpacetimeMemoryOrchestrator:
    """
    🌌 Master orchestrator for physics-bonded memory system
    
    This is where spacetime geometry becomes cognitive architecture!
    
    Flow:
    1. Define spacetime metric (symbolic)
    2. Compute curvature fields
    3. Encode as ψ-phase and ψ-amplitude
    4. Inject into concept mesh
    5. Propagate through soliton lattice
    6. Compute geodesic phase twists
    7. Monitor and adapt via feedback
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize all subsystems
        self.concept_mesh = ConceptMesh.instance()
        self.soliton_memory = FractalSolitonMemory.get_instance()
        self.curvature_injector = CurvatureInjector()
        self.coupling_driver = SolitonCouplingDriver()
        self.geodesic_integrator = GeodesicIntegrator()
        
        # Phase 8 feedback (optional)
        self.phase8_enabled = config.get('phase8_enabled', True)
        self.phase8_interval = 5  # Trigger every N evolution steps
        
        # Tracking
        self.active_concepts: Dict[str, Dict[str, Any]] = {}
        self.phase_fields: Dict[str, np.ndarray] = {}
        self.geodesic_cache: Dict[str, Any] = {}
        
        logger.info("🎼 Spacetime Memory Orchestrator initialized")
    
    async def create_memory_from_metric(self,
                                       metric_config: Dict[str, Any],
                                       concept_name: str,
                                       region_config: Optional[Dict[str, Any]] = None) -> str:
        """
        🚀 Main orchestration: Metric → Memory
        
        Args:
            metric_config: Spacetime metric specification
            concept_name: Name for the resulting concept
            region_config: Sampling region configuration
            
        Returns:
            Concept ID with full phase encoding
        """
        logger.info(f"🚀 Creating memory '{concept_name}' from {metric_config.get('type', 'custom')} metric")
        
        # Step 1: Generate curvature from metric
        if 'kretschmann_scalar' in metric_config:
            kretschmann = metric_config['kretschmann_scalar']
        else:
            kretschmann = self._create_metric_scalar(metric_config)
        
        # Default region
        if region_config is None:
            region_config = {
                'r': np.linspace(2, 10, 50),
                'theta': np.linspace(0, np.pi, 30),
                'phi': np.linspace(0, 2*np.pi, 30)
            }
        
        # Get feedback adjustments
        feedback = feedback_loop.get_feedback(concept_name)
        if feedback.requires_reencoding:
            suggestions = feedback_loop.suggest_encoding_parameters(concept_name, region_config)
            if 'suggested_region' in suggestions:
                region_config = suggestions['suggested_region']
                logger.info(f"🔄 Applied feedback adjustments to encoding")
        
        # Step 2: Encode curvature to phase
        coords = list(region_config.keys())
        phase_data = encode_curvature_to_phase(
            kretschmann_scalar=kretschmann,
            coords=coords,
            region_sample=region_config
        )
        
        # Store phase field
        self.phase_fields[concept_name] = phase_data['psi_phase']
        
        # Step 3: Inject into concept mesh
        concept_id = self.concept_mesh.inject_psi_fields(
            concept_id=concept_name,
            psi_phase=phase_data['psi_phase'],
            psi_amplitude=phase_data['psi_amplitude'],
            origin=f"{metric_config.get('type', 'custom')}_metric",
            coordinates=phase_data['coordinates'],
            curvature_value=float(np.mean(np.abs(phase_data['curvature_values']))),
            gradient_field=phase_data.get('phase_gradient'),
            persistence_mode=metric_config.get('persistence', 'persistent')
        )
        
        # Step 4: Inject into soliton memory
        await self._inject_into_soliton_memory(concept_id, phase_data)
        
        # Step 5: Propagate through ψ-mesh
        await self._propagate_phase(concept_id, phase_data)
        
        # Step 6: Compute gradient coupling
        self._update_coupling_matrix(concept_id, phase_data)
        
        # Track active concept
        self.active_concepts[concept_id] = {
            'name': concept_name,
            'metric_config': metric_config,
            'phase_data': {
                'mean_phase': float(np.mean(phase_data['psi_phase'])),
                'mean_amplitude': float(np.mean(phase_data['psi_amplitude'])),
                'vortex_count': len(phase_data.get('vortices', []))
            },
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Successfully created memory '{concept_name}' with ID: {concept_id}")
        
        # Trigger Phase 8 resonance feedback
        self.trigger_phase8_resonance()
        
        return concept_id
    
    def _create_metric_scalar(self, config: Dict[str, Any]) -> sp.Expr:
        """Create symbolic curvature scalar from metric config"""
        metric_type = config.get('type', 'schwarzschild')
        
        if metric_type == 'schwarzschild':
            M = config.get('mass', 1.0)
            r = sp.Symbol('r')
            return 48 * M**2 / r**6
            
        elif metric_type == 'kerr':
            M = config.get('mass', 1.0)
            a = config.get('spin', 0.5)
            r, theta = sp.symbols('r theta')
            rho2 = r**2 + a**2 * sp.cos(theta)**2
            return 48 * M**2 * (r**2 - a**2 * sp.cos(theta)**2) / rho2**3
            
        elif metric_type == 'de_sitter':
            Lambda = config.get('cosmological_constant', 0.1)
            r = sp.Symbol('r')
            return 12 * Lambda  # Constant curvature
            
        elif metric_type == 'custom':
            expr_str = config.get('kretschmann', '1/r**6')
            return sp.sympify(expr_str)
        
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    
    async def _inject_into_soliton_memory(self, 
                                         concept_id: str,
                                         phase_data: Dict[str, Any]):
        """Inject phase-encoded concept into soliton memory"""
        # Create soliton wave
        mean_phase = float(np.mean(phase_data['psi_phase']))
        mean_amplitude = float(np.mean(phase_data['psi_amplitude']))
        
        wave = self.soliton_memory.create_soliton(
            memory_id=concept_id,
            content={
                'concept_id': concept_id,
                'phase_encoded': True,
                'origin': phase_data.get('origin', 'curvature')
            },
            phase=mean_phase,
            curvature=phase_data.get('curvature_value', 1.0)
        )
        
        # Set amplitude
        wave.amplitude = mean_amplitude
        
        logger.info(f"💫 Injected soliton for '{concept_id}' with φ={mean_phase:.3f}")
    
    async def _propagate_phase(self, 
                              concept_id: str,
                              phase_data: Dict[str, Any]):
        """Propagate phase through ψ-mesh"""
        mean_phase = float(np.mean(phase_data['psi_phase']))
        mean_amplitude = float(np.mean(phase_data['psi_amplitude']))
        curvature = phase_data.get('curvature_value', 1.0)
        
        # Propagate through mesh
        result = await psi_phase_bridge.propagate_phase_through_mesh(
            initial_concept_id=concept_id,
            phase_value=mean_phase,
            amplitude_value=mean_amplitude,
            curvature_value=curvature,
            propagation_depth=3,
            decay_factor=0.8
        )
        
        affected = len(result['affected_concepts'])
        logger.info(f"🌊 Phase propagated to {affected} concepts")
    
    def _update_coupling_matrix(self, 
                               concept_id: str,
                               phase_data: Dict[str, Any]):
        """Update soliton coupling based on phase gradients"""
        if 'phase_gradient' not in phase_data:
            return
        
        # Compute gradient field
        grad_field = self.coupling_driver.compute_phase_gradient(
            phase_data['psi_phase'],
            concept_id,
            phase_data.get('coordinates')
        )
        
        # Find resonance zones
        zones = self.coupling_driver.find_resonance_zones()
        if zones:
            logger.info(f"🎯 Found {len(zones)} resonance zones")
    
    async def compute_geodesic_memory_path(self,
                                           start_concept: str,
                                           end_concept: str,
                                           path_type: str = "shortest") -> Optional[float]:
        """
        🌀 Compute geodesic path between concepts and phase twist
        
        Args:
            start_concept: Starting concept ID
            end_concept: Target concept ID  
            path_type: "shortest", "maximal_curvature", or "minimal_twist"
            
        Returns:
            Total phase twist along geodesic
        """
        # Get concept positions in phase space
        start_data = self.concept_mesh.get_psi_fields(start_concept)
        end_data = self.concept_mesh.get_psi_fields(end_concept)
        
        if not start_data or not end_data:
            logger.warning(f"❌ Concepts not found for geodesic")
            return None
        
        # Use coordinates or create synthetic positions
        # For now, use phase values as position proxy
        start_pos = np.array([0, 10, np.pi/2, start_data['psi_phase']])
        end_pos = np.array([0, 5, np.pi/2, end_data['psi_phase']])
        
        # Find geodesic
        if not hasattr(self.geodesic_integrator, 'metric'):
            # Set default metric
            self.geodesic_integrator.set_schwarzschild_metric()
        
        geodesic = self.geodesic_integrator.find_shortest_geodesic(
            start_pos, end_pos
        )
        
        if geodesic is None:
            logger.warning(f"❌ Failed to find geodesic")
            return None
        
        # Compute phase twist if we have phase field
        if start_concept in self.phase_fields:
            phase_field = self.phase_fields[start_concept]
            
            # Simple coordinate mapping
            coords_dict = {
                't': np.array([0]),
                'r': np.linspace(2, 20, 50),
                'theta': np.linspace(0, np.pi, 30),
                'phi': np.linspace(0, 2*np.pi, 40)
            }
            
            total_twist = self.geodesic_integrator.compute_phase_twist_along_geodesic(
                geodesic, phase_field, coords_dict
            )
            
            logger.info(f"🌀 Geodesic phase twist: {total_twist:.3f} rad")
            return total_twist
        
        return geodesic.phase_twist
    
    async def evolve_system(self, time_steps: int = 10, dt: float = 0.1):
        """
        ⏰ Evolve the entire system forward in time
        
        This updates:
        - Soliton positions and phases
        - Phase propagation through mesh
        - Feedback metrics
        """
        logger.info(f"⏰ Evolving system for {time_steps} steps")
        
        for step in range(time_steps):
            # Evolve soliton lattice
            await self.soliton_memory.evolve_lattice(dt=dt, apply_phase_dynamics=True)
            
            # Check for binding events
            for wave_id, wave in self.soliton_memory.waves.items():
                if wave.coherence < 0.3:
                    # Low coherence - binding may have failed
                    feedback_loop.record_activation_event(
                        concept_id=wave_id,
                        event_type="binding_failed",
                        success=False,
                        metadata={
                            'coherence': wave.coherence,
                            'amplitude': wave.amplitude,
                            'phase': wave.phase
                        }
                    )
                elif wave.coherence > 0.8:
                    # High coherence - successful binding
                    feedback_loop.record_activation_event(
                        concept_id=wave_id,
                        event_type="binding_success",
                        success=True,
                        metadata={
                            'coherence': wave.coherence,
                            'amplitude': wave.amplitude,
                            'phase': wave.phase
                        }
                    )
            
            # Update coupling matrix
            positions = [wave.position.tolist() for wave in self.soliton_memory.waves.values()]
            if positions:
                self.coupling_driver.update_coupling_matrix(positions)
            
            # Check for phase drift
            for concept_id in self.active_concepts:
                phase_data = self.concept_mesh.get_psi_fields(concept_id)
                if phase_data and 'phase_stats' in phase_data:
                    # Simple drift detection
                    if step > 0:
                        # Compare with initial phase
                        initial_phase = self.active_concepts[concept_id]['phase_data']['mean_phase']
                        current_phase = phase_data['phase_stats']['mean']
                        drift = abs(current_phase - initial_phase)
                        
                        if drift > 0.5:  # Significant drift
                            feedback_loop.record_activation_event(
                                concept_id=concept_id,
                                event_type="phase_drift",
                                success=False,
                                metadata={'drift_amount': drift}
                            )
            
            # Trigger Phase 8 resonance periodically
            if self.phase8_enabled and step % self.phase8_interval == 0:
                self.trigger_phase8_resonance()
        
        logger.info(f"✅ Evolution complete")
    
    def visualize_system_state(self) -> Dict[str, Any]:
        """
        🎨 Generate visualization data for the entire system
        """
        viz_data = {
            'timestamp': datetime.now().isoformat(),
            'active_concepts': self.active_concepts,
            'soliton_positions': {},
            'phase_statistics': {},
            'coupling_matrix': None,
            'resonance_zones': []
        }
        
        # Soliton positions
        for wave_id, wave in self.soliton_memory.waves.items():
            viz_data['soliton_positions'][wave_id] = {
                'position': wave.position.tolist(),
                'phase': wave.phase,
                'amplitude': wave.amplitude,
                'coherence': wave.coherence
            }
        
        # Phase statistics
        for concept_id in self.active_concepts:
            phase_data = self.concept_mesh.get_psi_fields(concept_id)
            if phase_data and 'phase_stats' in phase_data:
                viz_data['phase_statistics'][concept_id] = phase_data['phase_stats']
        
        # Coupling data
        if hasattr(self.coupling_driver, 'coupling_matrix'):
            viz_data['coupling_matrix'] = self.coupling_driver.coupling_matrix.tolist()
        
        # Resonance zones
        viz_data['resonance_zones'] = [
            {
                'center': zone.center.tolist(),
                'radius': zone.radius,
                'strength': zone.alignment_strength
            }
            for zone in self.coupling_driver.resonance_zones
        ]
        
        return viz_data
    
    def trigger_phase8_resonance(self):
        """Trigger Phase 8 lattice feedback for resonance reinforcement"""
        if not self.phase8_enabled:
            return
        
        try:
            from python.core.phase_8_lattice_feedback import Phase8LatticeFeedback
            feedback = Phase8LatticeFeedback()
            feedback.run_once()
            logger.info("🌐 Phase 8 resonance feedback completed")
        except ImportError:
            logger.warning("Phase 8 module not found, skipping resonance feedback")
            self.phase8_enabled = False
        except Exception as e:
            logger.warning(f"Phase 8 feedback failed: {e}")
    
    async def run_example_workflow(self):
        """
        🎭 Example workflow demonstrating the full system
        """
        logger.info("🎭 Running example workflow")
        
        # 1. Create black hole memory
        bh_config = {
            'type': 'schwarzschild',
            'mass': 1.0,
            'persistence': 'persistent'
        }
        
        bh_id = await self.create_memory_from_metric(
            metric_config=bh_config,
            concept_name="BlackHole_Singularity"
        )
        
        # 2. Create event horizon memory
        horizon_config = {
            'type': 'custom',
            'kretschmann': '48/(r**6) * exp(-((r-2)/0.5)**2)',  # Peak at horizon
            'persistence': 'persistent'
        }
        
        horizon_id = await self.create_memory_from_metric(
            metric_config=horizon_config,
            concept_name="Event_Horizon"
        )
        
        # 3. Compute geodesic between them
        twist = await self.compute_geodesic_memory_path(
            bh_id, horizon_id
        )
        
        if twist is not None:
            logger.info(f"🌀 Phase twist between concepts: {twist:.3f} rad")
        
        # 4. Evolve system
        await self.evolve_system(time_steps=5)
        
        # 5. Check feedback
        bh_feedback = feedback_loop.get_feedback(bh_id)
        if bh_feedback.requires_reencoding:
            logger.info(f"🔄 Black hole concept needs re-encoding")
            feedback_loop.trigger_reencoding(bh_id)
        
        # 6. Generate visualization
        viz_data = self.visualize_system_state()
        
        # 7. Export results
        with open('orchestration_results.json', 'w') as f:
            json.dump(viz_data, f, indent=2)
        
        logger.info("✅ Example workflow complete!")
        
        return viz_data


# Convenience functions for direct usage
async def create_spacetime_memory(metric_type: str, 
                                 concept_name: str,
                                 **kwargs) -> str:
    """
    Quick function to create a memory from a metric type
    
    Examples:
        concept_id = await create_spacetime_memory("schwarzschild", "BlackHole", mass=2.0)
        concept_id = await create_spacetime_memory("kerr", "RotatingBH", mass=1.0, spin=0.9)
    """
    orchestrator = SpacetimeMemoryOrchestrator()
    
    metric_config = {
        'type': metric_type,
        **kwargs
    }
    
    return await orchestrator.create_memory_from_metric(
        metric_config=metric_config,
        concept_name=concept_name
    )


# Main entry point
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    # Run example
    async def main():
        orchestrator = SpacetimeMemoryOrchestrator()
        await orchestrator.run_example_workflow()
    
    # Run
    asyncio.run(main())
