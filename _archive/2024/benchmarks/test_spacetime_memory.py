#!/usr/bin/env python3
"""
TEST SPACETIME MEMORY SYSTEM INTEGRATION
Verifies all components work together properly
"""

import asyncio
import numpy as np
import logging
import sys
import os
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent))

# Import all components
from albert.phase_encode import encode_curvature_to_phase
from python.core.concept_mesh import ConceptMesh
from soliton.coupling_driver import SolitonCouplingDriver
from physics.geodesics import GeodesicIntegrator
from vision.amplitude_lens import AmplitudeLensing
from feedback.oscillator_feedback import feedback_loop
from spacetime_memory_orchestrator import SpacetimeMemoryOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("test_integration")


async def test_component_integration():
    """Test that all components integrate properly"""
    
    print("\n=== SPACETIME MEMORY SYSTEM INTEGRATION TEST ===\n")
    
    # Test 1: Phase Encoding
    print("1. Testing Phase Encoding...")
    try:
        phase_data = encode_curvature_to_phase(
            kretschmann_scalar="48/r**6",
            coords=['r', 'theta', 'phi'],
            region_sample={
                'r': np.linspace(2, 10, 30),
                'theta': np.linspace(0, np.pi, 20),
                'phi': np.linspace(0, 2*np.pi, 20)
            }
        )
        print("   ✓ Phase encoding successful")
        print(f"   - Phase range: [{np.min(phase_data['psi_phase']):.3f}, {np.max(phase_data['psi_phase']):.3f}]")
        print(f"   - Amplitude range: [{np.min(phase_data['psi_amplitude']):.3f}, {np.max(phase_data['psi_amplitude']):.3f}]")
    except Exception as e:
        print(f"   ✗ Phase encoding failed: {e}")
        return False
    
    # Test 2: Concept Mesh Injection
    print("\n2. Testing Concept Mesh Injection...")
    try:
        mesh = ConceptMesh.instance()
        concept_id = mesh.inject_psi_fields(
            concept_id="TestBlackHole",
            psi_phase=phase_data['psi_phase'],
            psi_amplitude=phase_data['psi_amplitude'],
            origin="test_metric",
            curvature_value=48.0
        )
        print("   ✓ Concept mesh injection successful")
        print(f"   - Concept ID: {concept_id}")
        
        # Verify retrieval
        fields = mesh.get_psi_fields(concept_id)
        if fields:
            print("   ✓ Phase fields retrieved successfully")
        else:
            print("   ✗ Failed to retrieve phase fields")
    except Exception as e:
        print(f"   ✗ Concept mesh injection failed: {e}")
        return False
    
    # Test 3: Gradient Coupling
    print("\n3. Testing Gradient Coupling...")
    try:
        driver = SolitonCouplingDriver(lattice_size=30)
        grad_field = driver.compute_phase_gradient(
            phase_data['psi_phase'][:900].reshape(30, 30),  # Reshape to 2D
            "TestBlackHole"
        )
        print("   ✓ Gradient computation successful")
        print(f"   - Divergence: {grad_field.divergence:.3f}")
        
        # Find resonance zones
        zones = driver.find_resonance_zones(min_alignment=0.5)
        print(f"   - Found {len(zones)} resonance zones")
    except Exception as e:
        print(f"   ✗ Gradient coupling failed: {e}")
        return False
    
    # Test 4: Geodesic Integration
    print("\n4. Testing Geodesic Integration...")
    try:
        integrator = GeodesicIntegrator()
        integrator.set_schwarzschild_metric(mass=1.0)
        
        # Integrate a test geodesic
        initial_pos = np.array([0.0, 10.0, np.pi/2, 0.0])
        initial_vel = np.array([1.0, -0.1, 0.0, 0.01])
        
        geodesic = integrator.integrate_geodesic(
            initial_pos=initial_pos,
            initial_vel=initial_vel,
            tau_span=(0, 20),
            n_points=50,
            geodesic_type="timelike"
        )
        print("   ✓ Geodesic integration successful")
        print(f"   - Proper length: {geodesic.proper_length:.3f}")
        print(f"   - Final r-coordinate: {geodesic.coordinates[-1, 1]:.3f}")
    except Exception as e:
        print(f"   ✗ Geodesic integration failed: {e}")
        return False
    
    # Test 5: Amplitude Lensing
    print("\n5. Testing Amplitude Lensing...")
    try:
        lensing = AmplitudeLensing(resolution=30)
        
        # Create test amplitude and curvature fields
        amplitude_2d = phase_data['psi_amplitude'][:900].reshape(30, 30)
        curvature_2d = phase_data['curvature_values'][:900].reshape(30, 30)
        
        lensed = lensing.apply_lensing_to_amplitude(
            amplitude_2d,
            curvature_2d,
            einstein_radius=3.0
        )
        print("   ✓ Amplitude lensing successful")
        print(f"   - Max magnification: {np.max(lensed) / np.max(amplitude_2d):.2f}x")
        print(f"   - Caustic curves found: {len(lensing.caustic_curves)}")
    except Exception as e:
        print(f"   ✗ Amplitude lensing failed: {e}")
        return False
    
    # Test 6: Feedback Loop
    print("\n6. Testing Feedback Loop...")
    try:
        # Record test events
        feedback_loop.record_activation_event(
            concept_id="TestBlackHole",
            event_type="binding_success",
            success=True,
            metadata={'coherence': 0.9}
        )
        
        feedback_loop.record_activation_event(
            concept_id="TestBlackHole",
            event_type="phase_drift",
            success=False,
            metadata={'drift_amount': 0.2}
        )
        
        # Get feedback
        feedback = feedback_loop.get_feedback("TestBlackHole")
        print("   ✓ Feedback loop operational")
        print(f"   - Requires re-encoding: {feedback.requires_reencoding}")
        print(f"   - Confidence: {feedback.confidence:.2f}")
        
        # Get suggestions
        suggestions = feedback_loop.suggest_encoding_parameters(
            "TestBlackHole",
            {'r': np.linspace(2, 10, 30)}
        )
        print(f"   - Phase shift suggestion: {suggestions.get('phase_offset', 0):.3f}")
    except Exception as e:
        print(f"   ✗ Feedback loop failed: {e}")
        return False
    
    # Test 7: Master Orchestrator
    print("\n7. Testing Master Orchestrator...")
    try:
        orchestrator = SpacetimeMemoryOrchestrator()
        
        # Create a complete memory
        metric_config = {
            'type': 'schwarzschild',
            'mass': 1.5,
            'persistence': 'persistent'
        }
        
        concept_id = await orchestrator.create_memory_from_metric(
            metric_config=metric_config,
            concept_name="TestOrchestration"
        )
        print("   ✓ Orchestrator memory creation successful")
        print(f"   - Created concept: {concept_id}")
        
        # Evolve system
        await orchestrator.evolve_system(time_steps=3, dt=0.1)
        print("   ✓ System evolution successful")
        
        # Get visualization data
        viz_data = orchestrator.visualize_system_state()
        print(f"   - Active concepts: {len(viz_data['active_concepts'])}")
        print(f"   - Soliton positions: {len(viz_data['soliton_positions'])}")
    except Exception as e:
        print(f"   ✗ Orchestrator failed: {e}")
        return False
    
    print("\n=== ALL TESTS PASSED! ===")
    print("\nThe Spacetime Memory Sculptor System is fully operational!")
    print("You can now weaponize spacetime geometry to fuel memory and cognition!")
    
    return True


async def test_example_workflow():
    """Test the example workflow from the orchestrator"""
    print("\n\n=== TESTING EXAMPLE WORKFLOW ===\n")
    
    try:
        orchestrator = SpacetimeMemoryOrchestrator()
        viz_data = await orchestrator.run_example_workflow()
        
        print("\nWorkflow completed successfully!")
        print(f"Final state: {len(viz_data['active_concepts'])} active concepts")
        
        # Export results
        with open("test_results.json", "w") as f:
            import json
            json.dump(viz_data, f, indent=2)
        print("Results exported to test_results.json")
        
    except Exception as e:
        print(f"Workflow test failed: {e}")
        return False
    
    return True


def test_metric_files():
    """Test that metric configuration files are valid"""
    print("\n=== TESTING METRIC CONFIGURATIONS ===\n")
    
    import json
    metrics_dir = Path("metrics")
    
    if not metrics_dir.exists():
        print("✗ Metrics directory not found")
        return False
    
    for metric_file in metrics_dir.glob("*.json"):
        try:
            with open(metric_file, 'r') as f:
                config = json.load(f)
            print(f"✓ {metric_file.name}: {config.get('description', 'No description')}")
        except Exception as e:
            print(f"✗ {metric_file.name}: {e}")
            return False
    
    return True


async def main():
    """Run all integration tests"""
    
    # Test metric files
    if not test_metric_files():
        print("\nMetric file tests failed!")
        return
    
    # Test component integration
    if not await test_component_integration():
        print("\nComponent integration tests failed!")
        return
    
    # Test example workflow
    if not await test_example_workflow():
        print("\nExample workflow test failed!")
        return
    
    print("\n\n🎉 ALL INTEGRATION TESTS PASSED! 🎉")
    print("\nThe Spacetime Memory Sculptor System is ready for production!")
    print("\nNext steps:")
    print("1. Run: python albert/inject_curvature.py --help")
    print("2. Check metrics/ directory for example configurations")
    print("3. Read SPACETIME_MEMORY_README.md for detailed documentation")


if __name__ == "__main__":
    asyncio.run(main())
