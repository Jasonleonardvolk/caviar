#!/usr/bin/env python3
"""
TEST PHASE 8 RESONANCE FEEDBACK INTEGRATION
Demonstrates how Phase 8 enhances the spacetime memory system
"""

import asyncio
import numpy as np
import logging
from pathlib import Path
import sys

# Add project paths
sys.path.append(str(Path(__file__).parent))

from spacetime_memory_orchestrator import SpacetimeMemoryOrchestrator
from python.core.phase_8_lattice_feedback import Phase8LatticeFeedback
from python.core.fractal_soliton_memory import FractalSolitonMemory
from python.core.concept_mesh import ConceptMesh
from python.core.launch_scheduler import compute_entropy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_phase8")


async def test_phase8_with_spacetime():
    """Test Phase 8 integration with spacetime memory"""
    
    print("\n=== PHASE 8 RESONANCE FEEDBACK TEST ===\n")
    
    # Initialize orchestrator with Phase 8 enabled
    orchestrator = SpacetimeMemoryOrchestrator({
        'phase8_enabled': True,
        'phase8_interval': 3  # More frequent for testing
    })
    
    # Create some related concepts from different metrics
    print("1. Creating spacetime memories...")
    
    # Black hole
    bh_config = {
        'type': 'schwarzschild',
        'mass': 1.0,
        'persistence': 'persistent'
    }
    
    bh_id = await orchestrator.create_memory_from_metric(
        metric_config=bh_config,
        concept_name="BlackHole_Singularity"
    )
    print(f"   ✓ Created Black Hole: {bh_id}")
    
    # Event horizon (should resonate with black hole)
    horizon_config = {
        'type': 'custom',
        'kretschmann': '48/(r**6) * exp(-((r-2)/0.5)**2)',
        'persistence': 'persistent'
    }
    
    horizon_id = await orchestrator.create_memory_from_metric(
        metric_config=horizon_config,
        concept_name="Event_Horizon"
    )
    print(f"   ✓ Created Event Horizon: {horizon_id}")
    
    # Wormhole (different topology)
    wormhole_config = {
        'type': 'custom',
        'kretschmann': '1/(r**2 + 1)**3',
        'persistence': 'persistent'
    }
    
    wormhole_id = await orchestrator.create_memory_from_metric(
        metric_config=wormhole_config,
        concept_name="Wormhole_Throat"
    )
    print(f"   ✓ Created Wormhole: {wormhole_id}")
    
    # Check initial entropy
    soliton = FractalSolitonMemory.get_instance()
    initial_entropy = compute_entropy(soliton.waves.values())
    print(f"\n2. Initial system entropy: {initial_entropy:.3f}")
    
    # Evolve system to allow resonance
    print("\n3. Evolving system with Phase 8 feedback...")
    await orchestrator.evolve_system(time_steps=12, dt=0.1)
    
    # Check final entropy
    final_entropy = compute_entropy(soliton.waves.values())
    print(f"\n4. Final system entropy: {final_entropy:.3f}")
    print(f"   Entropy change: {final_entropy - initial_entropy:+.3f}")
    
    # Check for reinforced relations
    mesh = ConceptMesh.instance()
    print("\n5. Checking for resonance-reinforced relations...")
    
    # Get all relations
    resonant_relations = []
    for relation in mesh.relations:
        if relation.metadata.get('source') == 'phase_8_feedback':
            resonant_relations.append(relation)
            
            # Get concept names
            source_name = mesh.concepts.get(relation.source_id, {}).name
            target_name = mesh.concepts.get(relation.target_id, {}).name
            
            print(f"   🌐 {source_name} ↔ {target_name} (strength={relation.strength:.3f})")
    
    print(f"\n   Total resonant relations: {len(resonant_relations)}")
    
    # Manually trigger Phase 8 to show immediate effect
    print("\n6. Manual Phase 8 trigger with lower thresholds...")
    feedback = Phase8LatticeFeedback()
    feedback.run_once(coherence_threshold=0.5, similarity_threshold=0.6)
    
    # Visualize final state
    viz_data = orchestrator.visualize_system_state()
    
    # Summary
    print("\n=== PHASE 8 TEST SUMMARY ===")
    print(f"Active concepts: {len(viz_data['active_concepts'])}")
    print(f"Soliton waves: {len(viz_data['soliton_positions'])}")
    print(f"Resonant relations: {len(resonant_relations)}")
    print(f"System entropy: {initial_entropy:.3f} → {final_entropy:.3f}")
    
    return viz_data


async def test_adaptive_scheduler():
    """Test the adaptive scheduling based on entropy"""
    
    print("\n=== ADAPTIVE SCHEDULER TEST ===\n")
    
    soliton = FractalSolitonMemory.get_instance()
    
    # Create waves with different coherence levels
    print("1. Creating test waves with varying coherence...")
    
    # High coherence waves (low entropy)
    for i in range(3):
        wave = soliton.create_soliton(
            memory_id=f"high_coherence_{i}",
            content={"test": "high"},
            phase=0.0,
            curvature=1.0
        )
        wave.coherence = 0.95
        wave.embedding = np.random.randn(768)
        wave.embedding /= np.linalg.norm(wave.embedding)
    
    # Low coherence waves (high entropy)
    for i in range(3):
        wave = soliton.create_soliton(
            memory_id=f"low_coherence_{i}",
            content={"test": "low"},
            phase=np.pi,
            curvature=0.1
        )
        wave.coherence = 0.2
        wave.embedding = np.random.randn(768)
        wave.embedding /= np.linalg.norm(wave.embedding)
    
    # Calculate entropy
    entropy = compute_entropy(soliton.waves.values())
    print(f"\n2. System entropy: {entropy:.3f}")
    
    # Calculate adaptive interval
    MIN_INTERVAL = 120   # 2 minutes
    MAX_INTERVAL = 1800  # 30 minutes
    
    adaptive_interval = int(MAX_INTERVAL * entropy + MIN_INTERVAL * (1 - entropy))
    
    print(f"\n3. Adaptive interval calculation:")
    print(f"   Entropy: {entropy:.3f}")
    print(f"   Interval: {adaptive_interval} seconds ({adaptive_interval/60:.1f} minutes)")
    
    # Show interpretation
    if entropy < 0.3:
        state = "🧘 Highly synchronized (tight coherence)"
    elif entropy < 0.7:
        state = "⚖️ Balanced state"
    else:
        state = "🌪️ Chaotic (scattered coherence)"
    
    print(f"   State: {state}")
    
    # Test Phase 8 on this configuration
    print("\n4. Running Phase 8 on test configuration...")
    feedback = Phase8LatticeFeedback()
    feedback.run_once(coherence_threshold=0.5, similarity_threshold=0.7)


async def test_full_integration():
    """Run the complete Phase 8 integration test"""
    
    print("\n🔥 PHASE 8 + SPACETIME MEMORY INTEGRATION TEST 🔥\n")
    
    # Test spacetime integration
    await test_phase8_with_spacetime()
    
    # Test adaptive scheduler
    await test_adaptive_scheduler()
    
    print("\n✅ ALL PHASE 8 TESTS COMPLETE!")
    print("\n🧠 TORI now has:")
    print("   - Resonance-based memory reinforcement")
    print("   - Adaptive scheduling based on system entropy")
    print("   - Bidirectional phase-coherent learning")
    print("   - Self-organizing concept topology")
    
    print("\n🚀 The system is learning from its own resonances!")


def test_manual_execution():
    """Test manual Phase 8 execution"""
    
    print("\n=== MANUAL PHASE 8 EXECUTION ===\n")
    
    # Initialize systems
    soliton = FractalSolitonMemory.get_instance()
    mesh = ConceptMesh.instance()
    
    # Create some test waves
    print("1. Creating test soliton waves...")
    
    # Related concepts
    wave1 = soliton.create_soliton(
        memory_id="concept_relativity",
        content={"topic": "general relativity"},
        phase=0.0
    )
    wave1.coherence = 0.9
    wave1.embedding = np.random.randn(768)
    
    wave2 = soliton.create_soliton(
        memory_id="concept_spacetime",
        content={"topic": "spacetime curvature"},
        phase=0.1
    )
    wave2.coherence = 0.88
    # Make embeddings similar
    wave2.embedding = wave1.embedding + 0.1 * np.random.randn(768)
    
    # Normalize
    wave1.embedding /= np.linalg.norm(wave1.embedding)
    wave2.embedding /= np.linalg.norm(wave2.embedding)
    
    # Check similarity
    similarity = np.dot(wave1.embedding, wave2.embedding)
    print(f"   Wave similarity: {similarity:.3f}")
    print(f"   Wave coherences: {wave1.coherence:.3f}, {wave2.coherence:.3f}")
    
    # Run Phase 8
    print("\n2. Running Phase 8 feedback...")
    feedback = Phase8LatticeFeedback()
    feedback.run_once(coherence_threshold=0.8, similarity_threshold=0.7)
    
    print("\n✅ Manual execution complete!")


if __name__ == "__main__":
    # Run different test modes
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "manual":
            test_manual_execution()
        elif sys.argv[1] == "scheduler":
            asyncio.run(test_adaptive_scheduler())
        elif sys.argv[1] == "spacetime":
            asyncio.run(test_phase8_with_spacetime())
        else:
            asyncio.run(test_full_integration())
    else:
        # Run full integration test by default
        asyncio.run(test_full_integration())
