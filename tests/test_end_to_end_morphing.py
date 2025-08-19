#!/usr/bin/env python3
"""
Integration test for end-to-end topology morphing
Tests the complete flow from decision to morphing to completion
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
import time

from python.core.oscillator_lattice import OscillatorLattice, get_global_lattice
from python.core.soliton_memory_integration import EnhancedSolitonMemory, MemoryType
from python.core.hot_swap_laplacian import HotSwapLaplacian
from python.core.topology_policy import TopologyPolicy
from python.core.nightly_growth_engine import NightlyGrowthEngine
from python.core.blowup_harness import BlowupHarness, SafetyConfig


class TestEndToEndMorphing:
    """Test complete topology morphing workflow"""
    
    @pytest.fixture
    async def setup_complete_system(self):
        """Set up the complete system with all components"""
        # Clear global lattice
        lattice = get_global_lattice()
        lattice.clear()
        
        # Initialize components
        memory_system = EnhancedSolitonMemory(lattice_size=1000)
        hot_swap = HotSwapLaplacian(initial_topology="kagome")
        policy = TopologyPolicy()  # Will load from config
        growth_engine = NightlyGrowthEngine(memory_system, hot_swap)
        
        # Safety harness
        safety_config = SafetyConfig(
            max_amplitude=5.0,
            max_energy=500.0,
            max_oscillator_count=10000
        )
        harness = BlowupHarness(safety_config)
        
        return {
            'memory_system': memory_system,
            'hot_swap': hot_swap,
            'policy': policy,
            'growth_engine': growth_engine,
            'harness': harness,
            'lattice': lattice
        }
    
    @pytest.mark.asyncio
    async def test_policy_driven_morphing(self, setup_complete_system):
        """Test policy-driven topology morphing"""
        system = await setup_complete_system
        memory_system = system['memory_system']
        hot_swap = system['hot_swap']
        policy = system['policy']
        lattice = system['lattice']
        
        # Phase 1: Start in idle state with kagome topology
        assert hot_swap.current_topology == "kagome"
        assert policy.current_state == "idle"
        
        # Phase 2: Add memories to trigger state change
        print("\n=== Phase 2: Adding memories to trigger active state ===")
        
        # Add enough memories to exceed idle thresholds
        for i in range(100):
            memory_system.store_enhanced_memory(
                content=f"Test memory {i}",
                concept_ids=[f"concept_{i % 10}"],
                memory_type=MemoryType.SEMANTIC,
                sources=[f"source_{i}"]
            )
        
        # Update metrics
        metrics = policy.update_metrics(lattice)
        print(f"Metrics after adding memories: {metrics}")
        
        # Check for topology decision
        new_topology = policy.decide_topology(hot_swap.current_topology, metrics)
        
        # Should transition to active state and suggest hexagonal
        assert policy.current_state == "active"
        assert new_topology == "hexagonal"
        
        # Phase 3: Execute morphing
        print("\n=== Phase 3: Executing topology morph ===")
        
        # Start morphing
        hot_swap.initiate_morph("hexagonal", blend_rate=0.1)
        assert hot_swap.is_morphing
        
        # Simulate morphing steps
        steps = 0
        energy_harvested = 0
        
        while hot_swap.is_morphing and steps < 20:
            result = hot_swap.step_blend()
            energy_harvested += result.get('energy_harvested', 0)
            steps += 1
            
            # Check progress
            if steps % 5 == 0:
                print(f"Morph progress: {result.get('progress', 0)*100:.1f}%")
            
            await asyncio.sleep(0.01)  # Small delay
        
        # Verify completion
        assert not hot_swap.is_morphing
        assert hot_swap.current_topology == "hexagonal"
        assert energy_harvested > 0
        print(f"Morphing complete in {steps} steps, harvested {energy_harvested:.2f} energy")
        
        # Phase 4: Trigger intensive state
        print("\n=== Phase 4: Triggering intensive state ===")
        
        # Add many more memories
        for i in range(500):
            memory_system.store_enhanced_memory(
                content=f"Intensive memory {i}",
                concept_ids=[f"intensive_{i % 20}"],
                memory_type=MemoryType.SEMANTIC,
                sources=[f"source_{i}"]
            )
        
        # Force metric update
        metrics = policy.update_metrics(lattice)
        new_topology = policy.decide_topology(hot_swap.current_topology, metrics)
        
        # Should suggest square topology
        assert new_topology == "square"
        
        # Execute switch (instant this time)
        hot_swap.switch_topology("square")
        assert hot_swap.current_topology == "square"
        
        # Phase 5: Test emergency conditions
        print("\n=== Phase 5: Testing emergency conditions ===")
        
        # Simulate high stress
        for memory in list(memory_system.memory_entries.values())[:50]:
            memory.comfort_metrics.stress = 0.9
        
        metrics = policy.update_metrics(lattice)
        
        # Force emergency state
        policy.force_state("emergency")
        new_topology = policy.decide_topology(hot_swap.current_topology, metrics)
        
        # Should fall back to kagome
        assert new_topology == "kagome"
        
        print("\n✅ Policy-driven morphing test passed!")
    
    @pytest.mark.asyncio
    async def test_morphing_with_safety_harness(self, setup_complete_system):
        """Test morphing with safety harness active"""
        system = await setup_complete_system
        memory_system = system['memory_system']
        hot_swap = system['hot_swap']
        harness = system['harness']
        
        # Start safety monitoring
        harness.start_monitoring(memory_system)
        
        try:
            # Add memories with increasing amplitude
            for i in range(20):
                memory_system.store_enhanced_memory(
                    content=f"High energy memory {i}",
                    concept_ids=[f"energy_{i}"],
                    memory_type=MemoryType.SEMANTIC,
                    sources=["high_energy"],
                    metadata={"amplitude_boost": 2.0 + i * 0.5}
                )
            
            # Start morphing
            hot_swap.initiate_morph("hexagonal", blend_rate=0.05)
            
            # Run morphing with safety checks
            steps = 0
            violations = []
            
            while hot_swap.is_morphing and steps < 30:
                # Step morphing
                result = hot_swap.step_blend()
                
                # Check harness status
                status = harness.get_status()
                if status['recent_violations'] > 0:
                    violations.extend(status['recent_actions'])
                
                # Emergency brake check
                if harness.emergency_brake_active:
                    print("Emergency brake activated!")
                    break
                
                steps += 1
                await asyncio.sleep(0.05)  # Slower due to safety checks
            
            # Verify safety measures were taken
            print(f"\nSafety violations handled: {len(violations)}")
            print(f"Throttle factor: {harness.throttle_factor}")
            print(f"Emergency brake: {harness.emergency_brake_active}")
            
            # System should have handled high amplitudes
            assert len(violations) > 0 or harness.throttle_factor > 1.0
            
        finally:
            harness.stop_monitoring()
        
        print("\n✅ Safety harness morphing test passed!")
    
    @pytest.mark.asyncio
    async def test_comfort_feedback_during_morphing(self, setup_complete_system):
        """Test comfort-based feedback during morphing"""
        system = await setup_complete_system
        memory_system = system['memory_system']
        hot_swap = system['hot_swap']
        
        # Add memories
        memory_ids = []
        for i in range(50):
            mem_id = memory_system.store_enhanced_memory(
                content=f"Comfort test memory {i}",
                concept_ids=[f"comfort_{i % 5}"],
                memory_type=MemoryType.SEMANTIC,
                sources=["comfort_test"]
            )
            memory_ids.append(mem_id)
        
        # Start morphing
        hot_swap.initiate_morph("small_world", blend_rate=0.02)
        
        # Track comfort metrics during morphing
        comfort_history = []
        
        while hot_swap.is_morphing:
            # Collect comfort metrics
            total_stress = 0
            total_perturbation = 0
            count = 0
            
            for mem_id in memory_ids[:10]:  # Sample first 10
                if mem_id in memory_system.memory_entries:
                    memory = memory_system.memory_entries[mem_id]
                    total_stress += memory.comfort_metrics.stress
                    total_perturbation += memory.comfort_metrics.perturbation
                    count += 1
            
            if count > 0:
                avg_stress = total_stress / count
                avg_perturbation = total_perturbation / count
                
                comfort_history.append({
                    'progress': hot_swap.morph_progress,
                    'stress': avg_stress,
                    'perturbation': avg_perturbation
                })
            
            # Step morphing
            hot_swap.step_blend()
            await asyncio.sleep(0.05)
        
        # Analyze comfort evolution
        print("\n=== Comfort metrics during morphing ===")
        for i in range(0, len(comfort_history), 5):
            h = comfort_history[i]
            print(f"Progress {h['progress']*100:.0f}%: "
                  f"stress={h['stress']:.3f}, perturbation={h['perturbation']:.3f}")
        
        # Perturbation should increase during morphing
        mid_perturbation = comfort_history[len(comfort_history)//2]['perturbation']
        initial_perturbation = comfort_history[0]['perturbation']
        assert mid_perturbation > initial_perturbation
        
        print("\n✅ Comfort feedback test passed!")
    
    @pytest.mark.asyncio
    async def test_optimal_path_morphing(self, setup_complete_system):
        """Test optimal path calculation for energy harvesting"""
        system = await setup_complete_system
        hot_swap = system['hot_swap']
        
        # Test different paths
        paths_to_test = [
            ("kagome", "small_world"),
            ("kagome", "square"),
            ("hexagonal", "small_world")
        ]
        
        print("\n=== Optimal morphing paths ===")
        
        for start, end in paths_to_test:
            # Set starting topology
            hot_swap.current_topology = start
            
            # Find optimal path
            optimal_path = hot_swap.optimize_transition_path(start, end)
            
            # Calculate total harvest
            total_harvest = 0
            for i in range(len(optimal_path) - 1):
                harvest = hot_swap._calculate_transition_harvest(
                    optimal_path[i], optimal_path[i+1]
                )
                total_harvest += harvest
            
            print(f"{start} -> {end}: {' -> '.join(optimal_path)} "
                  f"(harvest: {total_harvest:.1f})")
        
        print("\n✅ Optimal path test passed!")


class TestMemoryLifecycle:
    """Test complete memory lifecycle from creation to crystallization"""
    
    @pytest.fixture
    def setup_memory_system(self):
        """Set up memory system for lifecycle testing"""
        from python.core.memory_crystallization import MemoryCrystallizer, CrystallizationConfig
        
        # Clear global lattice
        lattice = get_global_lattice()
        lattice.clear()
        
        # Initialize components
        memory_system = EnhancedSolitonMemory(lattice_size=1000)
        
        # Custom crystallization config for testing
        config = CrystallizationConfig(
            hot_threshold=0.6,
            cold_threshold=0.2,
            fusion_threshold=0.6,
            fusion_similarity_threshold=0.6
        )
        crystallizer = MemoryCrystallizer(config)
        
        return {
            'memory_system': memory_system,
            'crystallizer': crystallizer,
            'lattice': lattice
        }
    
    def test_memory_lifecycle_complete(self, setup_memory_system):
        """Test complete memory lifecycle"""
        memory_system = setup_memory_system['memory_system']
        crystallizer = setup_memory_system['crystallizer']
        lattice = setup_memory_system['lattice']
        
        print("\n=== Testing Complete Memory Lifecycle ===")
        
        # Phase 1: Creation
        print("\n--- Phase 1: Memory Creation ---")
        
        # Create diverse memories
        memory_ids = {
            'frequently_accessed': [],
            'similar_content': [],
            'large_memories': [],
            'dark_memories': []
        }
        
        # Frequently accessed memories
        for i in range(5):
            mem_id = memory_system.store_enhanced_memory(
                content=f"Important fact {i}",
                concept_ids=["important", "fact"],
                memory_type=MemoryType.SEMANTIC,
                sources=["reference"]
            )
            memory_ids['frequently_accessed'].append(mem_id)
        
        # Similar content memories (fusion candidates)
        for i in range(10):
            mem_id = memory_system.store_enhanced_memory(
                content=f"The quantum state is {i % 3}",
                concept_ids=["quantum", "physics"],
                memory_type=MemoryType.SEMANTIC,
                sources=["textbook"]
            )
            memory_ids['similar_content'].append(mem_id)
        
        # Large memories (fission candidates)
        for i in range(3):
            mem_id = memory_system.store_enhanced_memory(
                content="X" * 1500 + f" Large content {i}",
                concept_ids=[f"large_{i}"],
                memory_type=MemoryType.SEMANTIC,
                sources=["document"]
            )
            # Boost amplitude to trigger fission
            memory_system.memory_entries[mem_id].amplitude = 1.8
            memory_ids['large_memories'].append(mem_id)
        
        # Dark memories (suppression)
        for i in range(2):
            mem_id = memory_system.store_enhanced_memory(
                content=f"Suppress quantum info",
                concept_ids=["quantum"],
                memory_type=MemoryType.TRAUMATIC,
                sources=["suppression"]
            )
            memory_ids['dark_memories'].append(mem_id)
        
        initial_count = len(memory_system.memory_entries)
        print(f"Created {initial_count} memories")
        
        # Phase 2: Access patterns (heat generation)
        print("\n--- Phase 2: Simulating Access Patterns ---")
        
        # Frequently access some memories
        for _ in range(10):
            for mem_id in memory_ids['frequently_accessed']:
                # Simulate access via resonance search
                phase = memory_system._calculate_concept_phase(["important", "fact"])
                results = memory_system.find_resonant_memories_enhanced(
                    phase, ["important", "fact"], threshold=0.5
                )
        
        # Check heat levels
        hot_count = sum(1 for m in memory_system.memory_entries.values() if m.heat > 0.5)
        print(f"Hot memories: {hot_count}")
        
        # Phase 3: Test suppression
        print("\n--- Phase 3: Testing Dark Soliton Suppression ---")
        
        # Try to recall quantum memories
        phase = memory_system._calculate_concept_phase(["quantum"])
        quantum_results = memory_system.find_resonant_memories_enhanced(
            phase, ["quantum"], threshold=0.5
        )
        
        # Should be suppressed by dark solitons
        assert len(quantum_results) == 0
        print("✓ Dark solitons successfully suppressing quantum memories")
        
        # Phase 4: Crystallization
        print("\n--- Phase 4: Running Crystallization ---")
        
        report = crystallizer.crystallize(memory_system, lattice)
        
        print(f"Crystallization results:")
        print(f"  Migrations: {report['migrations']}")
        print(f"  Fusions: {report['fusions']}")
        print(f"  Decayed: {report['decayed']}")
        
        # Verify expectations
        assert report['migrations'] > 0  # Hot memories should migrate
        assert report['fusions'] > 0     # Similar memories should fuse
        
        # Phase 5: Verify final state
        print("\n--- Phase 5: Verifying Final State ---")
        
        final_count = len(memory_system.memory_entries)
        print(f"Final memory count: {final_count} (was {initial_count})")
        
        # Check fission occurred
        split_memories = [m for m in memory_system.memory_entries.values() 
                         if "split_from" in m.metadata]
        print(f"Split memories: {len(split_memories)}")
        
        # Verify no orphan oscillators
        active_memories = len(memory_system.memory_entries)
        active_oscillators = len([o for o in lattice.oscillators if o.get('active', True)])
        
        # Account for dark solitons (2 oscillators each)
        dark_count = len([m for m in memory_system.memory_entries.values() 
                         if m.polarity == "dark"])
        expected_oscillators = active_memories + dark_count
        
        print(f"Active oscillators: {active_oscillators}, Expected: {expected_oscillators}")
        
        # Allow small discrepancy due to pruning
        assert abs(active_oscillators - expected_oscillators) < 5
        
        print("\n✅ Complete memory lifecycle test passed!")
    
    def test_memory_vaulting_lifecycle(self, setup_memory_system):
        """Test memory vaulting and phase shifts"""
        memory_system = setup_memory_system['memory_system']
        
        print("\n=== Testing Memory Vaulting ===")
        
        # Create a memory
        mem_id = memory_system.store_enhanced_memory(
            content="Sensitive information",
            concept_ids=["sensitive", "private"],
            memory_type=MemoryType.SEMANTIC,
            sources=["confidential"]
        )
        
        memory = memory_system.memory_entries[mem_id]
        original_phase = memory.phase
        
        # Test different vault levels
        from python.core.soliton_memory_integration import VaultStatus
        
        vault_tests = [
            (VaultStatus.PHASE_45, np.pi/4, "Light vault"),
            (VaultStatus.PHASE_90, np.pi/2, "Medium vault"),
            (VaultStatus.PHASE_180, np.pi, "Deep vault"),
            (VaultStatus.ACTIVE, 0, "Unvault")
        ]
        
        for vault_status, expected_shift, description in vault_tests:
            memory_system.vault_memory_with_phase_shift(
                mem_id, vault_status, f"Testing {description}"
            )
            
            # Check phase shift
            if vault_status == VaultStatus.ACTIVE:
                # Should return to original phase (approximately)
                phase_diff = abs(memory.phase - original_phase)
                assert phase_diff < 0.1 or phase_diff > 2*np.pi - 0.1
            else:
                # Check approximate phase shift
                expected_phase = (original_phase + expected_shift) % (2 * np.pi)
                phase_diff = abs(memory.phase - expected_phase)
                assert phase_diff < 0.1
            
            print(f"✓ {description}: phase shifted correctly")
        
        print("\n✅ Memory vaulting test passed!")


if __name__ == "__main__":
    # Run tests
    import asyncio
    
    # Test morphing
    morphing_test = TestEndToEndMorphing()
    
    async def run_morphing_tests():
        system = await morphing_test.setup_complete_system()
        await morphing_test.test_policy_driven_morphing(system)
        
        system = await morphing_test.setup_complete_system()
        await morphing_test.test_morphing_with_safety_harness(system)
        
        system = await morphing_test.setup_complete_system()
        await morphing_test.test_comfort_feedback_during_morphing(system)
        
        system = await morphing_test.setup_complete_system()
        await morphing_test.test_optimal_path_morphing(system)
    
    asyncio.run(run_morphing_tests())
    
    # Test memory lifecycle
    lifecycle_test = TestMemoryLifecycle()
    system = lifecycle_test.setup_memory_system()
    lifecycle_test.test_memory_lifecycle_complete(system)
    
    system = lifecycle_test.setup_memory_system()
    lifecycle_test.test_memory_vaulting_lifecycle(system)
