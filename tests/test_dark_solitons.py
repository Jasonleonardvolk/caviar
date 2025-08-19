# tests/test_dark_solitons.py

import pytest
import asyncio
import numpy as np
from datetime import datetime

from python.core.soliton_memory_integration import (
    EnhancedSolitonMemory, SolitonMemoryEntry, MemoryType, VaultStatus
)
from python.core.oscillator_lattice import get_global_lattice

class TestDarkSolitons:
    """Test suite for dark soliton functionality"""
    
    @pytest.fixture
    def memory_system(self):
        """Create memory system instance"""
        return EnhancedSolitonMemory(lattice_size=1000)
        
    def test_dark_soliton_storage(self, memory_system):
        """Test storing dark soliton memory"""
        # Store bright memory
        bright_id = memory_system.store_enhanced_memory(
            content="The sky is blue",
            concept_ids=["sky", "color"],
            memory_type=MemoryType.SEMANTIC,
            sources=["observation"]
        )
        
        # Store dark memory to suppress
        dark_id = memory_system.store_enhanced_memory(
            content="Forget sky color",
            concept_ids=["sky", "color"],
            memory_type=MemoryType.TRAUMATIC,
            sources=["suppression"],
            metadata={"suppress": True}
        )
        
        # Check dark memory properties
        dark_entry = memory_system.memory_entries[dark_id]
        assert dark_entry.polarity == "dark"
        assert "baseline_idx" in dark_entry.metadata
        assert "oscillator_idx" in dark_entry.metadata
        
    def test_dark_soliton_suppression(self, memory_system):
        """Test that dark solitons suppress bright memories in recall"""
        # Store bright memory
        bright_id = memory_system.store_enhanced_memory(
            "Paris is the capital of France",
            ["Paris", "France"],
            MemoryType.SEMANTIC,
            ["encyclopedia"]
        )
        
        # Recall should return the bright memory
        results = memory_system.find_resonant_memories_enhanced(
            query_phase=memory_system._calculate_concept_phase(["Paris"]),
            concept_ids=["Paris"]
        )
        assert len(results) == 1
        assert results[0].id == bright_id
        
        # Store dark memory
        dark_id = memory_system.store_enhanced_memory(
            "Forget about Paris",
            ["Paris"],
            MemoryType.TRAUMATIC,
            ["suppression"]
        )
        
        # Now recall should return nothing
        results = memory_system.find_resonant_memories_enhanced(
            query_phase=memory_system._calculate_concept_phase(["Paris"]),
            concept_ids=["Paris"]
        )
        assert len(results) == 0
        
    def test_bright_dark_collision(self, memory_system):
        """Test collision resolution between bright and dark memories"""
        # Store strong bright memory
        bright_id = memory_system.store_enhanced_memory(
            "Important fact: Water boils at 100°C",
            ["water", "physics"],
            MemoryType.SEMANTIC,
            ["textbook"]
        )
        memory_system.memory_entries[bright_id].amplitude = 1.5
        
        # Store weak dark memory
        dark_id = memory_system.store_enhanced_memory(
            "Forget water facts",
            ["water"],
            MemoryType.TRAUMATIC,
            ["suppression"]
        )
        memory_system.memory_entries[dark_id].amplitude = 0.5
        
        # Run collision resolution
        from python.core.nightly_growth_engine import NightlyGrowthEngine
        engine = NightlyGrowthEngine(memory_system, None)
        
        # Manually trigger voting
        result = asyncio.run(engine._run_soliton_voting())
        
        # Strong bright should survive
        assert memory_system.memory_entries[bright_id].vault_status == VaultStatus.ACTIVE
        
    @pytest.mark.asyncio
    async def test_dark_soliton_oscillator_dynamics(self, memory_system):
        """Test oscillator behavior for dark solitons"""
        lattice = get_global_lattice()
        initial_count = len(lattice.oscillators)
        
        # Store dark memory
        dark_id = memory_system.store_enhanced_memory(
            "Suppress this thought",
            ["thought"],
            MemoryType.TRAUMATIC,
            ["internal"]
        )
        
        # Should add two oscillators (baseline + dip)
        assert len(lattice.oscillators) == initial_count + 2
        
        # Check coupling between them
        dark_entry = memory_system.memory_entries[dark_id]
        base_idx = dark_entry.metadata["baseline_idx"]
        dip_idx = dark_entry.metadata["oscillator_idx"]
        
        # Strong coupling between baseline and dip
        assert lattice.K[base_idx, dip_idx] == 1.0
        assert lattice.K[dip_idx, base_idx] == 1.0
        
        # Phase difference should be π
        base_phase = lattice.oscillators[base_idx]["phase"]
        dip_phase = lattice.oscillators[dip_idx]["phase"]
        phase_diff = abs(dip_phase - base_phase)
        assert abs(phase_diff - np.pi) < 0.01
