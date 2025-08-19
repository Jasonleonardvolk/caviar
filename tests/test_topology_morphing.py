# tests/test_topology_morphing.py

import pytest
import asyncio
import numpy as np

from python.core.hot_swap_laplacian import HotSwappableLaplacian
from python.core.topology_policy import TopologyPolicy, TopologyState

class TestTopologyMorphing:
    """Test suite for dynamic topology morphing"""
    
    @pytest.fixture
    def hot_swap(self):
        """Create hot-swap instance"""
        return HotSwappableLaplacian(
            initial_topology='kagome',
            lattice_size=(10, 10)
        )
        
    @pytest.fixture
    def policy(self, hot_swap):
        """Create policy instance"""
        return TopologyPolicy(hot_swap)
        
    def test_topology_initialization(self, hot_swap):
        """Test initial topology setup"""
        assert hot_swap.current_topology == 'kagome'
        assert len(hot_swap.topologies) == 4
        assert 'kagome' in hot_swap.topologies
        assert 'hexagonal' in hot_swap.topologies
        
    @pytest.mark.asyncio
    async def test_topology_switching(self, hot_swap):
        """Test switching between topologies"""
        # Record initial state
        initial_topology = hot_swap.current_topology
        initial_swap_count = hot_swap.swap_count
        
        # Switch to hexagonal
        await hot_swap.hot_swap_laplacian_with_safety('hexagonal')
        
        assert hot_swap.current_topology == 'hexagonal'
        assert hot_swap.swap_count == initial_swap_count + 1
        assert len(hot_swap.swap_history) > 0
        
        # Check swap record
        last_swap = hot_swap.swap_history[-1]
        assert last_swap['from'] == 'kagome'
        assert last_swap['to'] == 'hexagonal'
        assert last_swap['success'] == True
        
    @pytest.mark.asyncio
    async def test_energy_harvesting_during_swap(self, hot_swap):
        """Test energy harvesting when switching under high load"""
        # Simulate high energy scenario
        hot_swap.active_solitons = [
            {'amplitude': 10.0, 'phase': i * 0.1, 'index': i}
            for i in range(100)
        ]
        
        # Create mock high-energy lattice
        class MockLattice:
            def __init__(self):
                self.total_energy = 1500.0  # Above threshold
                self.psi = np.ones(100, dtype=complex)
            def step(self):
                pass
                
        hot_swap.lattice = MockLattice()
        
        # Switch topology
        await hot_swap.hot_swap_laplacian_with_safety('small_world')
        
        # Energy should have been harvested
        assert hot_swap.energy_harvested_total > 0
        
    def test_topology_recommendation(self, hot_swap):
        """Test topology recommendation for different problems"""
        assert hot_swap.recommend_topology_for_problem('pattern_recognition') == 'kagome'
        assert hot_swap.recommend_topology_for_problem('global_search') == 'small_world'
        assert hot_swap.recommend_topology_for_problem('optimization') == 'triangular'
        assert hot_swap.recommend_topology_for_problem('unknown') == 'kagome'  # Default
        
    @pytest.mark.asyncio
    async def test_adaptive_complexity_switching(self, hot_swap):
        """Test automatic topology switching based on complexity"""
        # Start with kagome
        assert hot_swap.current_topology == 'kagome'
        
        # Trigger O(n²) complexity
        await hot_swap.adaptive_swap_for_complexity("O(n²)")
        assert hot_swap.current_topology == 'small_world'
        
        # Trigger dense computation
        await hot_swap.adaptive_swap_for_complexity("dense_matrix")
        assert hot_swap.current_topology == 'triangular'
        
    def test_policy_state_transitions(self, policy):
        """Test policy state machine"""
        # Initial state
        assert policy.state == TopologyState.ACTIVE
        
        # High access rate -> intensive
        policy.metrics["access_rate"] = 150
        policy._update_state()
        assert policy.state == TopologyState.INTENSIVE
        
        # Low soliton count -> idle
        policy.metrics["access_rate"] = 0
        policy.metrics["soliton_count"] = 50
        policy._update_state()
        assert policy.state == TopologyState.IDLE
        
    @pytest.mark.asyncio
    async def test_policy_recommendations(self, policy):
        """Test policy topology recommendations"""
        # Set to idle state
        policy.state = TopologyState.IDLE
        recommendation = await policy.evaluate_topology()
        # Should recommend kagome for idle (if not already)
        
        # Set high soliton count
        policy.state = TopologyState.ACTIVE
        policy.metrics["soliton_count"] = 3000
        recommendation = await policy.evaluate_topology()
        # Should recommend switching
        
    def test_laplacian_matrix_properties(self, hot_swap):
        """Test generated Laplacian matrices"""
        for topology_name in ['kagome', 'hexagonal', 'square']:
            topology = hot_swap.topologies[topology_name]
            L = hot_swap._build_laplacian(topology_name)
            
            # Check basic Laplacian properties
            assert L.shape[0] == L.shape[1]  # Square
            assert np.allclose(L.sum(axis=1).A1, 0)  # Row sums = 0
            assert (L.diagonal() >= 0).all()  # Non-negative diagonal
            
    def test_shadow_trace_generation(self, hot_swap):
        """Test shadow trace creation for stability"""
        bright_soliton = {
            'amplitude': 1.0,
            'phase': np.pi/4,
            'topological_charge': 1,
            'index': 0
        }
        
        shadow = hot_swap.create_shadow_trace(bright_soliton)
        
        assert shadow.amplitude == -0.1  # 10% negative
        assert abs(shadow.phaseTag - (np.pi/4 + np.pi)) < 0.01  # π shift
        assert shadow.polarity == 'dark'
        
    def test_swap_metrics(self, hot_swap):
        """Test metrics reporting"""
        metrics = hot_swap.get_swap_metrics()
        
        assert 'current_topology' in metrics
        assert 'total_swaps' in metrics
        assert 'available_topologies' in metrics
        assert metrics['current_topology'] == 'kagome'
        assert metrics['total_swaps'] == 0
        assert set(metrics['available_topologies']) == {'kagome', 'honeycomb', 'triangular', 'small_world'}
