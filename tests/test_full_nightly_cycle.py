#!/usr/bin/env python3
"""
Integration test for full nightly cycle with all fixes
Tests the complete flow: memory creation, dark solitons, topology morphing,
crystallization with fusion/fission, and oscillator cleanup
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np

from python.core.soliton_memory_integration import EnhancedSolitonMemory, MemoryType
from python.core.hot_swap_laplacian import HotSwapLaplacian
from python.core.nightly_growth_engine import NightlyGrowthEngine
from python.core.topology_policy import TopologyPolicy
from python.core.oscillator_lattice import get_global_lattice


class TestFullNightlyCycle:
    """Test complete nightly maintenance cycle"""
    
    @pytest.fixture
    def setup_system(self):
        """Set up complete system with all components"""
        # Initialize memory system
        memory_system = EnhancedSolitonMemory(lattice_size=1000)
        
        # Initialize lattice
        lattice = get_global_lattice()
        lattice.clear()  # Start fresh
        
        # Initialize hot-swap system
        hot_swap = HotSwapLaplacian(initial_topology="kagome")
        
        # Initialize growth engine
        growth_engine = NightlyGrowthEngine(memory_system, hot_swap)
        
        return {
            'memory_system': memory_system,
            'lattice': lattice,
            'hot_swap': hot_swap,
            'growth_engine': growth_engine
        }
    
    def test_complete_nightly_cycle(self, setup_system):
        """Test full nightly cycle with all operations"""
        memory_system = setup_system['memory_system']
        lattice = setup_system['lattice']
        hot_swap = setup_system['hot_swap']
        
        # Phase 1: Populate system with various memories
        print("\n=== Phase 1: Populating System ===")
        
        # Add bright memories
        bright_ids = []
        for i in range(10):
            mem_id = memory_system.store_enhanced_memory(
                content=f"Test memory {i} with some content that makes it interesting",
                concept_ids=[f"concept_{i%3}"],  # 3 different concepts
                memory_type=MemoryType.SEMANTIC,
                sources=[f"source_{i}"]
            )
            bright_ids.append(mem_id)
        
        # Add dark memories to suppress some concepts
        dark_ids = []
        for i in range(2):
            mem_id = memory_system.store_enhanced_memory(
                content=f"Suppress concept_{i}",
                concept_ids=[f"concept_{i}"],
                memory_type=MemoryType.TRAUMATIC,  # Creates dark soliton
                sources=["suppression"]
            )
            dark_ids.append(mem_id)
        
        # Add some large memories that should be split
        large_ids = []
        for i in range(2):
            mem_id = memory_system.store_enhanced_memory(
                content="X" * 1500,  # Very long content
                concept_ids=[f"large_concept_{i}"],
                memory_type=MemoryType.SEMANTIC,
                sources=["large_source"],
                metadata={"amplitude_boost": 2.0}  # Will set high amplitude
            )
            # Manually boost amplitude to trigger splitting
            memory_system.memory_entries[mem_id].amplitude = 1.8
            large_ids.append(mem_id)
        
        initial_memory_count = len(memory_system.memory_entries)
        initial_oscillator_count = len([o for o in lattice.oscillators if o.get('active', True)])
        
        print(f"Initial state:")
        print(f"  Memories: {initial_memory_count} (10 bright, 2 dark, 2 large)")
        print(f"  Oscillators: {initial_oscillator_count}")
        print(f"  Current topology: {hot_swap.current_topology}")
        
        # Phase 2: Simulate access patterns to heat up some memories
        print("\n=== Phase 2: Simulating Access Patterns ===")
        
        # Access some memories multiple times to heat them up
        for i in range(5):
            memory_system.find_resonant_memories_enhanced(
                query_phase=memory_system._calculate_concept_phase(["concept_0"]),
                concept_ids=["concept_0"],
                threshold=0.5
            )
        
        # Check that dark memories are suppressing
        suppressed_results = memory_system.find_resonant_memories_enhanced(
            query_phase=memory_system._calculate_concept_phase(["concept_1"]),
            concept_ids=["concept_1"],
            threshold=0.5
        )
        assert len(suppressed_results) == 0, "Dark soliton should suppress bright memories"
        
        # Phase 3: Run nightly crystallization
        print("\n=== Phase 3: Running Nightly Crystallization ===")
        
        # Switch to consolidating topology
        hot_swap.switch_topology("small_world")
        assert hot_swap.current_topology == "small_world"
        
        # Run crystallization
        report = memory_system.nightly_crystallization()
        
        print(f"Crystallization report:")
        print(f"  Migrated: {report['migrated']}")
        print(f"  Decayed: {report['decayed']}")
        print(f"  Fused: {report['fused']}")
        print(f"  Split: {report['split']}")
        
        # Phase 4: Verify results
        print("\n=== Phase 4: Verifying Results ===")
        
        final_memory_count = len(memory_system.memory_entries)
        final_oscillator_count = len([o for o in lattice.oscillators if o.get('active', True)])
        
        print(f"Final state:")
        print(f"  Memories: {final_memory_count}")
        print(f"  Oscillators: {final_oscillator_count}")
        
        # Assertions
        # 1. Some memories should have been fused (same concept)
        assert report['fused'] > 0, "Should have fused some memories with same concept"
        
        # 2. Large memories should have been split
        assert report['split'] == 2, "Should have split 2 large memories"
        
        # 3. Memory count should reflect fusions and splits
        expected_memories = initial_memory_count - report['fused'] + report['split']
        assert final_memory_count == expected_memories, \
            f"Memory count mismatch: {final_memory_count} != {expected_memories}"
        
        # 4. Oscillator count should match (no orphans)
        # Each memory has 1 oscillator, except dark memories have 2
        expected_oscillators = final_memory_count + len([m for m in memory_system.memory_entries.values() 
                                                        if m.polarity == "dark"])
        assert final_oscillator_count == expected_oscillators, \
            f"Oscillator count mismatch: {final_oscillator_count} != {expected_oscillators}"
        
        # 5. Check that split memories exist and have correct properties
        split_memories = [m for m in memory_system.memory_entries.values() 
                         if "split_from" in m.metadata]
        assert len(split_memories) == 4, "Should have 4 memories from splitting 2 large ones"
        
        for split_mem in split_memories:
            assert split_mem.amplitude < 1.5, "Split memories should have reduced amplitude"
            assert "oscillator_idx" in split_mem.metadata, "Split memories should have oscillators"
        
        # Phase 5: Return to stable topology
        print("\n=== Phase 5: Returning to Stable State ===")
        
        hot_swap.switch_topology("kagome")
        assert hot_swap.current_topology == "kagome"
        
        print("\n✅ Full nightly cycle test passed!")
        
    @pytest.mark.asyncio
    async def test_topology_morphing_gradual(self, setup_system):
        """Test gradual topology morphing with blending"""
        hot_swap = setup_system['hot_swap']
        
        # Start morphing from kagome to hexagonal
        hot_swap.initiate_morph("hexagonal", blend_rate=0.1)
        
        # Simulate multiple blend steps
        blend_steps = 0
        while hot_swap.is_morphing and blend_steps < 15:
            hot_swap.step_blend()
            blend_steps += 1
            await asyncio.sleep(0.01)  # Small delay to simulate time passing
        
        # Should complete in ~10 steps with blend_rate=0.1
        assert not hot_swap.is_morphing, "Morphing should be complete"
        assert hot_swap.current_topology == "hexagonal", "Should be at target topology"
        assert blend_steps >= 10, "Should take at least 10 steps"
        
    def test_comfort_metrics_update(self, setup_system):
        """Test that comfort metrics are properly calculated"""
        memory_system = setup_system['memory_system']
        
        # Create a memory
        mem_id = memory_system.store_enhanced_memory(
            content="Test comfort metrics",
            concept_ids=["comfort_test"],
            memory_type=MemoryType.SEMANTIC,
            sources=["test"]
        )
        
        memory = memory_system.memory_entries[mem_id]
        
        # Initial comfort metrics
        assert memory.comfort_metrics.energy > 0
        assert memory.comfort_metrics.stress == 1.0 - memory.stability
        
        # Update comfort (this would normally be called by the system)
        # Since we can't directly call Rust methods from Python test,
        # we verify the structure exists and is ready for updates
        assert hasattr(memory.comfort_metrics, 'flux')
        assert hasattr(memory.comfort_metrics, 'perturbation')
        
    def test_policy_config_loading(self):
        """Test that TopologyPolicy loads from config correctly"""
        policy = TopologyPolicy("conf/soliton_memory_config_consolidated.yaml")
        
        # Check that config was loaded
        assert policy.min_switch_interval == 60
        assert policy.comfort_weight == 0.3
        assert policy.fallback_topology == "kagome"
        
        # Check that rules were created from config
        assert "idle" in policy.rules
        assert "active" in policy.rules
        assert "intensive" in policy.rules
        assert policy.rules["idle"]["preferred_topology"] == "kagome"


if __name__ == "__main__":
    # Run the test
    test = TestFullNightlyCycle()
    setup = test.setup_system()
    test.test_complete_nightly_cycle(setup)
