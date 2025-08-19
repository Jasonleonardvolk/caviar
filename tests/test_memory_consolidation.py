# tests/test_memory_consolidation.py

import pytest
import asyncio
from datetime import datetime, timedelta

from python.core.soliton_memory_integration import (
    EnhancedSolitonMemory, MemoryType, VaultStatus
)
from python.core.memory_crystallization import MemoryCrystallizer
from python.core.nightly_growth_engine import NightlyGrowthEngine
from python.core.hot_swap_laplacian import HotSwappableLaplacian

class TestMemoryConsolidation:
    """Test suite for memory consolidation and growth cycles"""
    
    @pytest.fixture
    def memory_system(self):
        """Create memory system"""
        return EnhancedSolitonMemory(lattice_size=1000)
        
    @pytest.fixture
    def hot_swap(self):
        """Create hot-swap system"""
        return HotSwappableLaplacian(lattice_size=(20, 20))
        
    @pytest.fixture
    def crystallizer(self, memory_system, hot_swap):
        """Create crystallizer"""
        return MemoryCrystallizer(memory_system, hot_swap)
        
    @pytest.fixture
    def growth_engine(self, memory_system, hot_swap):
        """Create growth engine"""
        return NightlyGrowthEngine(memory_system, hot_swap)
        
    def test_memory_heat_tracking(self, memory_system):
        """Test heat-based memory tracking"""
        # Store memory
        mem_id = memory_system.store_enhanced_memory(
            "Test memory",
            ["test"],
            MemoryType.SEMANTIC,
            ["source"]
        )
        
        entry = memory_system.memory_entries[mem_id]
        initial_heat = entry.heat
        
        # Access memory (should increase heat)
        memory_system.find_resonant_memories_enhanced(
            entry.phase,
            ["test"]
        )
        
        assert entry.heat > initial_heat
        assert entry.access_count == 1
        
    @pytest.mark.asyncio
    async def test_memory_fusion(self, memory_system):
        """Test fusion of similar memories"""
        # Store duplicate memories
        id1 = memory_system.store_enhanced_memory(
            "Paris is the capital of France",
            ["Paris"],
            MemoryType.SEMANTIC,
            ["source1"]
        )
        
        id2 = memory_system.store_enhanced_memory(
            "France's capital is Paris",
            ["Paris"],
            MemoryType.SEMANTIC,
            ["source2"]
        )
        
        # Initial count
        initial_count = len(memory_system.memory_entries)
        
        # Perform fusion
        fused = memory_system._perform_memory_fusion()
        
        assert fused > 0
        assert len(memory_system.memory_entries) < initial_count
        
        # One should remain with combined properties
        remaining = [e for e in memory_system.memory_entries.values() 
                    if "Paris" in e.concept_ids]
        assert len(remaining) == 1
        assert remaining[0].amplitude > 1.0  # Strengthened
        
    def test_memory_fission(self, memory_system):
        """Test splitting of complex memories"""
        # Store complex memory
        complex_id = memory_system.store_enhanced_memory(
            "Quantum computing uses superposition and entanglement for parallel processing",
            ["quantum", "computing", "physics"],
            MemoryType.SEMANTIC,
            ["textbook"]
        )
        
        # Make it appear complex
        entry = memory_system.memory_entries[complex_id]
        entry.frequency = 0.9  # High complexity
        
        # Check if should split
        from python.core.memory_crystallization import MemoryCrystallizer
        crystallizer = MemoryCrystallizer(memory_system, None)
        
        should_split = crystallizer._should_split(entry)
        assert should_split == True
        
        # Perform split
        new_memories = crystallizer._perform_split(entry)
        assert len(new_memories) == 3  # One per concept
        
        # Each should have one concept
        for mem in new_memories:
            assert len(mem.concept_ids) == 1
            assert mem.amplitude < entry.amplitude
            
    @pytest.mark.asyncio
    async def test_crystallization_cycle(self, crystallizer, memory_system):
        """Test full crystallization cycle"""
        # Add various memories with different heat levels
        hot_memory = memory_system.store_enhanced_memory(
            "Frequently accessed fact",
            ["hot"],
            MemoryType.SEMANTIC,
            ["source"]
        )
        memory_system.memory_entries[hot_memory].heat = 0.9
        
        cold_memory = memory_system.store_enhanced_memory(
            "Rarely accessed fact",
            ["cold"],
            MemoryType.SEMANTIC,
            ["source"]
        )
        memory_system.memory_entries[cold_memory].heat = 0.05
        
        # Run crystallization
        report = await crystallizer.crystallize()
        
        assert report["migrated"] >= 0
        assert report["decayed"] >= 0
        assert "error" not in report
        
        # Check cold memory decayed
        cold_entry = memory_system.memory_entries.get(cold_memory)
        if cold_entry:  # Might be removed
            assert cold_entry.amplitude < 1.0
            
    @pytest.mark.asyncio
    async def test_nightly_growth_cycle(self, growth_engine):
        """Test complete nightly growth cycle"""
        # Configure for immediate run
        growth_engine.config["tasks"] = [
            "crystallization",
            "soliton_voting",
            "topology_optimization"
        ]
        
        # Run cycle
        report = await growth_engine.run_nightly_cycle()
        
        assert "error" not in report
        assert len(report["tasks_run"]) > 0
        assert "crystallization" in report["results"]
        
    def test_growth_engine_scheduling(self, growth_engine):
        """Test growth engine scheduling logic"""
        # Test should_run logic
        now = datetime.now()
        
        # Set to run at current hour
        growth_engine.nightly_hour = now.hour
        growth_engine.last_run = None
        
        assert growth_engine._should_run(now) == True
        
        # Already run today
        growth_engine.last_run = now
        assert growth_engine._should_run(now) == False
        
        # Different hour
        growth_engine.nightly_hour = (now.hour + 1) % 24
        assert growth_engine._should_run(now) == False
        
    @pytest.mark.asyncio
    async def test_soliton_voting(self, growth_engine, memory_system):
        """Test soliton voting mechanism"""
        # Create conflicting memories
        bright_id = memory_system.store_enhanced_memory(
            "Fact: The sun is hot",
            ["sun"],
            MemoryType.SEMANTIC,
            ["science"]
        )
        memory_system.memory_entries[bright_id].amplitude = 1.0
        
        dark_id = memory_system.store_enhanced_memory(
            "Forget about the sun",
            ["sun"],
            MemoryType.TRAUMATIC,
            ["suppression"]
        )
        memory_system.memory_entries[dark_id].polarity = "dark"
        memory_system.memory_entries[dark_id].amplitude = 1.2
        
        # Run voting
        result = await growth_engine._run_soliton_voting()
        
        assert result["concepts_evaluated"] > 0
        assert result["conflicts_resolved"] > 0
        
        # Bright memory should be suppressed
        assert memory_system.memory_entries[bright_id].vault_status == VaultStatus.QUARANTINE
        
    def test_comfort_analysis(self, memory_system, growth_engine):
        """Test comfort metric analysis"""
        # Add some memories with varying stability
        for i in range(10):
            mem_id = memory_system.store_enhanced_memory(
                f"Memory {i}",
                [f"concept_{i}"],
                MemoryType.SEMANTIC,
                ["test"]
            )
            # Vary stability
            memory_system.memory_entries[mem_id].stability = 0.5 + (i * 0.05)
            
        # Analyze comfort
        comfort = asyncio.run(growth_engine._analyze_comfort())
        
        assert comfort["total_memories"] == 10
        assert 0 <= comfort["average_stress"] <= 1
        assert comfort["stressed_count"] >= 0
