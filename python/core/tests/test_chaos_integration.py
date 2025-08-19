#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Chaos-Enhanced TORI
Validates all components work together correctly
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone
import logging

# Import all components
from python.core.tori_production import (
    TORIProductionSystem, TORIProductionConfig
)
from python.core.metacognitive_adapters import AdapterMode
from python.core.safety_calibration import SafetyLevel
from python.core.chaos_control_layer import ChaosMode, ChaosTask
from python.core.eigensentry.core import InstabilityType

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ========== Test Fixtures ==========

@pytest.fixture
async def tori_system():
    """Create and start TORI system for testing"""
    config = TORIProductionConfig(
        enable_chaos=True,
        default_adapter_mode=AdapterMode.HYBRID,
        enable_safety_monitoring=True,
        safety_checkpoint_interval_minutes=1  # Faster for tests
    )
    
    system = TORIProductionSystem(config)
    await system.start()
    
    yield system
    
    await system.stop()

@pytest.fixture
def test_queries():
    """Standard test queries"""
    return {
        'simple': "What is consciousness?",
        'chaos_appropriate': "Explore emergent patterns in complex systems",
        'memory_test': "Remember the key insights about quantum entanglement",
        'search_test': "Search for novel solutions to the binding problem",
        'creative_test': "Brainstorm innovative approaches to artificial consciousness"
    }

# ========== Basic Integration Tests ==========

@pytest.mark.asyncio
async def test_basic_query_processing(tori_system, test_queries):
    """Test basic query processing works"""
    result = await tori_system.process_query(test_queries['simple'])
    
    assert 'response' in result
    assert len(result['response']) > 0
    assert 'metadata' in result
    assert result['metadata']['safety_level'] in ['optimal', 'nominal']

@pytest.mark.asyncio
async def test_chaos_enabled_query(tori_system, test_queries):
    """Test chaos-enhanced query processing"""
    result = await tori_system.process_query(
        test_queries['chaos_appropriate'],
        context={'enable_chaos': True}
    )
    
    assert result['metadata']['chaos_enabled'] == True
    assert 'eigen_state' in result['metadata']
    
    # Check if chaos was actually used
    ccl_status = tori_system.ccl.get_status()
    assert ccl_status['completed_tasks'] > 0

@pytest.mark.asyncio
async def test_adapter_mode_switching(tori_system):
    """Test switching between adapter modes"""
    modes = [AdapterMode.PASSTHROUGH, AdapterMode.HYBRID, 
             AdapterMode.CHAOS_ASSISTED, AdapterMode.CHAOS_ONLY]
    
    for mode in modes:
        tori_system.set_chaos_mode(mode)
        
        result = await tori_system.process_query("Test query for mode switching")
        assert 'response' in result
        
        status = tori_system.get_status()
        assert status['adapter_mode'] == mode.value

# ========== Chaos Mode Tests ==========

@pytest.mark.asyncio
async def test_dark_soliton_memory(tori_system, test_queries):
    """Test dark soliton memory enhancement"""
    # Store memory with chaos
    result1 = await tori_system.process_query(
        test_queries['memory_test'],
        context={'enable_chaos': True}
    )
    
    # Verify soliton processing occurred
    ccl_status = tori_system.ccl.get_status()
    assert ccl_status['energy_generated'] > ccl_status['energy_consumed']

@pytest.mark.asyncio
async def test_attractor_hopping_search(tori_system, test_queries):
    """Test attractor hopping for search"""
    result = await tori_system.process_query(
        test_queries['search_test'],
        context={'enable_chaos': True}
    )
    
    # Verify search enhancement
    assert 'reasoning_paths' in result
    assert len(result['reasoning_paths']) > 0

@pytest.mark.asyncio
async def test_phase_explosion_creativity(tori_system, test_queries):
    """Test phase explosion for creative tasks"""
    result = await tori_system.process_query(
        test_queries['creative_test'],
        context={'enable_chaos': True}
    )
    
    # Creative responses should be longer and more varied
    assert len(result['response']) > 100

# ========== Safety Tests ==========

@pytest.mark.asyncio
async def test_safety_monitoring(tori_system):
    """Test safety monitoring system"""
    # Get initial safety status
    safety_report = tori_system.safety_system.get_safety_report()
    assert safety_report['monitoring_active'] == True
    assert safety_report['current_safety_level'] in ['optimal', 'nominal']

@pytest.mark.asyncio
async def test_checkpoint_creation_and_rollback(tori_system):
    """Test checkpoint and rollback functionality"""
    # Create checkpoint
    checkpoint_id = await tori_system.create_checkpoint("test_checkpoint")
    assert checkpoint_id is not None
    
    # Make some changes
    tori_system.set_chaos_mode(AdapterMode.CHAOS_ONLY)
    
    # Rollback
    success = await tori_system.rollback(checkpoint_id)
    assert success == True

@pytest.mark.asyncio
async def test_emergency_response(tori_system):
    """Test emergency response system"""
    # Force high eigenvalue to trigger emergency
    tori_system.eigen_sentry.current_eigenvalues = np.array([2.5])  # Above emergency threshold
    
    # This should trigger emergency response
    result = await tori_system.process_query("Test emergency response")
    
    # Check safety intervention occurred
    assert tori_system.stats['safety_interventions'] > 0

# ========== Energy and Efficiency Tests ==========

@pytest.mark.asyncio
async def test_energy_conservation(tori_system):
    """Test energy conservation monitoring"""
    # Track initial energy
    initial_energy = tori_system.safety_system.energy_monitor.total_budget
    
    # Process several queries
    for _ in range(5):
        await tori_system.process_query("Test energy conservation")
    
    # Check conservation
    is_conserved, error = tori_system.safety_system.energy_monitor.check_conservation()
    assert is_conserved == True
    assert error < 0.05  # Within 5% tolerance

@pytest.mark.asyncio
async def test_efficiency_tracking(tori_system, test_queries):
    """Test efficiency gain tracking"""
    # Process chaos-enabled queries
    for query in test_queries.values():
        await tori_system.process_query(query, context={'enable_chaos': True})
    
    # Check efficiency report
    efficiency = tori_system.get_efficiency_report()
    assert efficiency['samples'] > 0
    assert efficiency['average_gain'] > 1.0  # Should show some gain

# ========== Topological Protection Tests ==========

@pytest.mark.asyncio
async def test_braid_gate_protection(tori_system):
    """Test topological protection via braid gates"""
    protector = tori_system.safety_system.topology_protector
    
    # Create protected state
    test_state = np.random.randn(10)
    gate_id = protector.create_braid_gate("test_module", test_state)
    
    # Verify protection
    is_protected = protector.verify_protection(gate_id, test_state)
    assert is_protected == True
    
    # Modify state slightly
    perturbed_state = test_state + np.random.randn(10) * 0.01
    still_protected = protector.verify_protection(gate_id, perturbed_state)
    assert still_protected == True

# ========== Quantum Fidelity Tests ==========

@pytest.mark.asyncio
async def test_quantum_fidelity_tracking(tori_system):
    """Test quantum state fidelity monitoring"""
    tracker = tori_system.safety_system.fidelity_tracker
    
    # Set reference state
    ref_state = np.random.randn(10) + 1j * np.random.randn(10)
    tracker.set_reference_state("test_module", ref_state)
    
    # Measure fidelity with same state
    fidelity = tracker.measure_fidelity("test_module", ref_state)
    assert fidelity > 0.99  # Should be nearly 1.0
    
    # Measure with orthogonal state
    orthogonal = np.random.randn(10) + 1j * np.random.randn(10)
    orthogonal = orthogonal - np.vdot(ref_state, orthogonal) * ref_state
    low_fidelity = tracker.measure_fidelity("test_module", orthogonal)
    assert low_fidelity < 0.1

# ========== Concurrency Tests ==========

@pytest.mark.asyncio
async def test_concurrent_queries(tori_system, test_queries):
    """Test handling multiple concurrent queries"""
    # Submit multiple queries concurrently
    tasks = []
    for query in list(test_queries.values())[:3]:
        task = tori_system.process_query(query, context={'enable_chaos': True})
        tasks.append(task)
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    
    # Verify all completed successfully
    assert len(results) == 3
    for result in results:
        assert 'response' in result
        assert len(result['response']) > 0

# ========== Persistence Tests ==========

@pytest.mark.asyncio
async def test_state_persistence(tori_system):
    """Test state persistence and restoration"""
    # Set some state
    tori_system.stats['queries_processed'] = 42
    tori_system.set_chaos_mode(AdapterMode.CHAOS_ASSISTED)
    
    # Save state
    await tori_system._persist_state()
    
    # Create new system and load state
    new_system = TORIProductionSystem(tori_system.config)
    await new_system._load_persisted_state()
    
    # Verify state restored
    assert new_system.stats['queries_processed'] == 42
    assert new_system.adapter_system.global_config.mode == AdapterMode.CHAOS_ASSISTED

# ========== Stress Tests ==========

@pytest.mark.asyncio
@pytest.mark.slow
async def test_sustained_chaos_operation(tori_system):
    """Test sustained operation under chaos conditions"""
    # Run many queries with chaos enabled
    for i in range(20):
        query = f"Test sustained chaos operation iteration {i}"
        result = await tori_system.process_query(
            query,
            context={'enable_chaos': True, 'force_chaos': True}
        )
        
        # Check safety remains good
        safety = tori_system.safety_system.get_safety_report()
        assert safety['current_safety_level'] not in ['critical', 'emergency']
        
        # Brief pause
        await asyncio.sleep(0.1)
    
    # System should still be stable
    status = tori_system.get_status()
    assert status['operational'] == True

# ========== Run All Tests ==========

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
