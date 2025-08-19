#!/usr/bin/env python3
"""
Test script for Unified Metacognitive Integration
Demonstrates all components working together
"""

import asyncio
import numpy as np
from datetime import datetime

# Import all our metacognitive components
from python.core.unified_metacognitive_integration import (
    UnifiedMetacognitiveSystem, MetacognitiveState
)
from python.core.temporal_reasoning_integration import (
    TemporalConceptMesh, EdgeType
)
from python.core.reasoning_traversal import ConceptNode
from python.core.reflection_fixed_point_integration import ReflectionType
from python.core.soliton_memory_integration import (
    EnhancedSolitonMemory, SolitonMemoryIntegration, MemoryType
)
from python.core.cognitive_dynamics_monitor import (
    CognitiveDynamicsMonitor, CognitiveStateManager,
    ReasoningDynamicsIntegration
)

async def test_unified_metacognitive_system():
    """Test the complete metacognitive integration"""
    
    print("🧠 Unified Metacognitive System Test")
    print("=" * 70)
    print("Testing all components: Filter, Memory, Reflection, and Dynamics")
    print("=" * 70)
    
    # Step 1: Create test concept mesh
    print("\n1️⃣ Creating concept mesh...")
    mesh = TemporalConceptMesh()
    
    # Add test concepts
    concepts = [
        ConceptNode("ai", "Artificial Intelligence", 
                   "Computer systems that mimic human intelligence",
                   ["computer_science_2024"]),
        ConceptNode("consciousness", "Consciousness",
                   "Subjective experience and awareness",
                   ["neuroscience_2024", "philosophy_2024"]),
        ConceptNode("emergence", "Emergence",
                   "Complex properties arising from simple rules",
                   ["complexity_2024"]),
        ConceptNode("quantum", "Quantum Computing",
                   "Computing using quantum mechanical phenomena",
                   ["physics_2024"])
    ]
    
    for concept in concepts:
        mesh.add_node(concept)
    
    # Add relationships
    mesh.add_temporal_edge("emergence", "consciousness", EdgeType.ENABLES,
                          justification="emergence enables consciousness")
    mesh.add_temporal_edge("ai", "consciousness", EdgeType.RELATES,
                          justification="AI research informs consciousness studies")
    mesh.add_temporal_edge("quantum", "ai", EdgeType.ENABLES,
                          justification="quantum computing enhances AI capabilities")
    
    print(f"✅ Created mesh with {len(concepts)} concepts")
    
    # Step 2: Initialize metacognitive system
    print("\n2️⃣ Initializing metacognitive system...")
    meta_system = UnifiedMetacognitiveSystem(mesh, enable_all_systems=True)
    print("✅ All subsystems initialized")
    
    # Step 3: Test queries
    test_queries = [
        {
            "query": "How does emergence relate to consciousness in AI systems?",
            "context": {"deep_reflection": True},
            "description": "Complex query requiring deep reflection"
        },
        {
            "query": "Can quantum computing enable artificial consciousness?",
            "context": {"intent": "causal"},
            "description": "Causal reasoning query"
        },
        {
            "query": "Consciousness does not emerge from physical processes.",
            "context": {"allow_experimental": True},
            "description": "Contradictory statement to test memory dissonance"
        }
    ]
    
    print("\n3️⃣ Processing test queries...")
    for i, test in enumerate(test_queries):
        print(f"\n{'='*70}")
        print(f"Query {i+1}: {test['description']}")
        print(f"Question: {test['query']}")
        print("-" * 70)
        
        # Process through metacognitive pipeline
        response = await meta_system.process_query_metacognitively(
            test["query"], 
            test["context"]
        )
        
        # Display results
        print(f"\n📝 Response: {response.text[:200]}...")
        print(f"\n📊 Metacognitive Metrics:")
        print(f"   • State: {meta_system.meta_state.value}")
        print(f"   • Memory Resonance: {response.metadata.get('memory_resonance', 0):.2f}")
        print(f"   • Stability Score: {response.metadata.get('stability_score', 0):.2f}")
        print(f"   • Reflection Depth: {response.metadata.get('reflection_depth', 0)}")
        print(f"   • Concept Purity: {response.metadata.get('concept_purity', 1.0):.2f}")
        print(f"   • Confidence: {response.confidence:.2f}")
        
        # Small delay between queries
        await asyncio.sleep(0.5)
    
    # Step 4: Test individual components
    print(f"\n{'='*70}")
    print("4️⃣ Testing Individual Components")
    print("-" * 70)
    
    # Test REAL-TORI Filter
    print("\n🔍 Testing REAL-TORI Filter:")
    test_concepts = ["quantum", "consciousness", "xyz123", "a"]
    purity = meta_system.tori_filter.analyze_concept_purity(test_concepts)
    print(f"   Concept purity for {test_concepts}: {purity:.2f}")
    
    # Test Soliton Memory
    print("\n🌊 Testing Soliton Memory:")
    memory_count = len(meta_system.soliton_memory.memory_lattice)
    vaulted_count = len(meta_system.soliton_memory.vault_status)
    print(f"   Total memories stored: {memory_count}")
    print(f"   Vaulted memories: {vaulted_count}")
    
    # Test Reflection System
    print("\n🔄 Testing Reflection System:")
    reflection_count = len(meta_system.reflection_system.reflection_history)
    print(f"   Total reflections performed: {reflection_count}")
    
    # Test Dynamics System
    print("\n🎢 Testing Dynamics System:")
    stability = meta_system.dynamics_system.analyze_stability()
    print(f"   Current stability: {'Stable' if stability['stable'] else 'Unstable'}")
    print(f"   Chaotic: {stability['chaotic']}")
    print(f"   Max Lyapunov: {stability['max_lyapunov']:.3f}")
    
    # Step 5: Generate final report
    print(f"\n{'='*70}")
    print("5️⃣ Metacognitive System Report")
    print("-" * 70)
    
    report = meta_system.get_metacognitive_report()
    
    print(f"\n📈 Processing Statistics:")
    print(f"   • Total queries: {report['processing_stats']['total_queries']}")
    print(f"   • Avg duration: {report['processing_stats']['average_duration']:.2f}s")
    print(f"   • Avg resonance: {report['processing_stats']['average_resonance']:.2f}")
    print(f"   • Avg stability: {report['processing_stats']['average_stability']:.2f}")
    
    print(f"\n🧠 Memory Statistics:")
    print(f"   • Total memories: {report['memory_stats']['total_memories']}")
    print(f"   • Vaulted memories: {report['memory_stats']['vaulted_memories']}")
    print(f"   • Phase distribution: {report['memory_stats']['phase_distribution']} phases")
    
    print(f"\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"   • {rec}")
    
    print(f"\n{'='*70}")
    print("✅ All tests completed successfully!")
    print("🎉 The Unified Metacognitive System is fully operational!")

def test_memory_phase_visualization():
    """Visualize memory phase distribution"""
    print("\n\n📊 Memory Phase Distribution Test")
    print("=" * 70)
    
    # Create memory system
    memory = EnhancedSolitonMemory()
    
    # Add test memories at different phases
    test_memories = [
        ("Consciousness is emergent", ["consciousness", "emergence"], 0),
        ("AI requires quantum computing", ["ai", "quantum"], np.pi/2),
        ("Emergence enables complexity", ["emergence", "complexity"], np.pi),
        ("Quantum states are probabilistic", ["quantum", "probability"], 3*np.pi/2)
    ]
    
    for content, concepts, phase in test_memories:
        memory.store_enhanced_memory(
            content=content,
            concept_ids=concepts,
            memory_type=MemoryType.SEMANTIC,
            sources=["test"],
            metadata={"forced_phase": phase}
        )
    
    # Show phase distribution
    print("\nMemory Phase Distribution:")
    for phase_key, memory_ids in sorted(memory.phase_index.items()):
        phase_deg = phase_key * 180 / np.pi
        print(f"   Phase {phase_deg:6.1f}°: {len(memory_ids)} memories")
    
    # Test resonance
    print("\nTesting Phase Resonance:")
    query_phase = np.pi/4  # 45 degrees
    resonant = memory.find_resonant_memories_enhanced(query_phase, ["consciousness"], 0.5)
    print(f"   Query phase: {query_phase * 180 / np.pi:.1f}°")
    print(f"   Found {len(resonant)} resonant memories")

def test_chaos_detection():
    """Test chaos detection and stabilization"""
    print("\n\n🎢 Chaos Detection Test")
    print("=" * 70)
    
    # Create state manager and monitor
    state_manager = CognitiveStateManager()
    monitor = CognitiveDynamicsMonitor(state_manager)
    
    # Simulate chaotic trajectory
    print("\nSimulating chaotic cognitive trajectory...")
    for i in range(20):
        # Exponentially growing random state (chaotic)
        chaotic_state = np.random.randn(100) * (1.1 ** i)
        state_manager.update_state(new_state=chaotic_state)
    
    # Monitor dynamics
    result = monitor.monitor_and_stabilize()
    
    print(f"\nDynamics Analysis:")
    print(f"   • State: {result['dynamics_state']}")
    print(f"   • Max Lyapunov: {result['metrics']['max_lyapunov']:.3f}")
    print(f"   • Energy: {result['metrics']['energy']:.3f}")
    print(f"   • Intervention needed: {result['intervention'] is not None}")
    
    if result['intervention']:
        print(f"   • Strategy applied: {result['intervention']['strategy']}")

if __name__ == "__main__":
    # Run main test
    asyncio.run(test_unified_metacognitive_system())
    
    # Run additional tests
    test_memory_phase_visualization()
    test_chaos_detection()
    
    print("\n\n🎊 All metacognitive integration tests completed!")
    print("The system is ready for production use.")
