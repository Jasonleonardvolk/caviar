#!/usr/bin/env python3
"""
Test suite for Hot-Swappable Graph Laplacian
Validates O(n²) mitigation and topology switching
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
import sys
sys.path.append('../../')

from python.core.hot_swap_laplacian import (
    HotSwappableLaplacian, 
    TopologyConfig,
    ShadowTrace,
    integrate_hot_swap_with_ccl
)
from python.core.blowup_harness import induce_blowup

class TestHotSwapLaplacian:
    """Test suite for hot-swappable Laplacian functionality"""
    
    @pytest.fixture
    def hot_swap(self):
        """Create hot-swap instance"""
        return HotSwappableLaplacian(
            initial_topology='kagome',
            lattice_size=(10, 10)
        )
        
    @pytest.fixture
    def mock_lattice(self):
        """Create mock lattice for testing"""
        class MockLattice:
            def __init__(self):
                self.psi = np.random.randn(100) + 1j * np.random.randn(100)
                self.total_energy = 500.0
            def step(self):
                self.psi *= 1.1  # Simulate energy growth
                
        return MockLattice()
        
    def test_initialization(self, hot_swap):
        """Test proper initialization"""
        assert hot_swap.current_topology == 'kagome'
        assert hot_swap.lattice_size == (10, 10)
        assert hot_swap.swap_count == 0
        assert len(hot_swap.topologies) == 4
        
    def test_topology_configs(self, hot_swap):
        """Test topology configurations"""
        kagome = hot_swap.topologies['kagome']
        assert kagome.chern_number == 1
        assert kagome.coordination_number == 4
        assert 'pattern_recognition' in kagome.optimal_for
        
        small_world = hot_swap.topologies['small_world']
        assert small_world.chern_number == 0
        assert 'global_search' in small_world.optimal_for
        
    def test_laplacian_construction(self, hot_swap):
        """Test Laplacian matrix construction"""
        for topology in ['kagome', 'honeycomb', 'triangular', 'small_world']:
            L = hot_swap._build_laplacian(topology)
            
            # Check basic properties
            assert L.shape[0] == L.shape[1]  # Square matrix
            assert np.allclose(L.sum(axis=1), 0)  # Row sums = 0
            assert (L.diagonal() >= 0).all()  # Non-negative diagonal
            
    def test_shadow_trace_creation(self, hot_swap):
        """Test shadow trace generation"""
        bright_soliton = {
            'amplitude': 1.0,
            'phase': np.pi/4,
            'topological_charge': 1,
            'index': 0
        }
        
        shadow = hot_swap.create_shadow_trace(bright_soliton)
        
        assert shadow.amplitude == -0.1  # 10% negative
        assert shadow.phaseTag == np.pi/4 + np.pi  # π phase shift
        assert shadow.polarity == 'dark'
        assert shadow.topological_charge == 1
        
    @pytest.mark.asyncio
    async def test_hot_swap_basic(self, hot_swap):
        """Test basic hot-swap functionality"""
        # Add some active solitons
        hot_swap.active_solitons = [
            {'amplitude': 1.0, 'phase': 0, 'index': 0},
            {'amplitude': 0.5, 'phase': np.pi/2, 'index': 1}
        ]
        
        # Perform swap
        await hot_swap.hot_swap_laplacian_with_safety('honeycomb')
        
        assert hot_swap.current_topology == 'honeycomb'
        assert hot_swap.swap_count == 1
        assert len(hot_swap.swap_history) == 1
        
    @pytest.mark.asyncio
    async def test_energy_harvesting(self, hot_swap, mock_lattice):
        """Test energy harvesting during high-energy swap"""
        hot_swap.lattice = mock_lattice
        hot_swap.lattice.total_energy = 1500.0  # Above critical threshold
        
        # Track energy before
        initial_psi = hot_swap.lattice.psi.copy()
        
        # Perform swap
        await hot_swap.hot_swap_laplacian_with_safety('small_world')
        
        # Energy should be harvested
        assert hot_swap.energy_harvested_total > 0
        assert hot_swap.lattice.psi is not None
        
    @pytest.mark.asyncio
    async def test_stability_verification(self, hot_swap):
        """Test swap stability verification"""
        # Create stable configuration
        hot_swap.active_solitons = [
            {'amplitude': 0.5, 'phase': 0, 'index': 0}
        ]
        
        # Verify should pass
        is_stable = await hot_swap.verify_swap_stability()
        assert is_stable == True
        
        # Create unstable configuration
        hot_swap.active_solitons = [
            {'amplitude': 100.0, 'phase': 0, 'index': 0}  # Too high energy
        ]
        
        # Verify should fail
        is_stable = await hot_swap.verify_swap_stability()
        assert is_stable == False
        
    @pytest.mark.asyncio
    async def test_adaptive_complexity_swap(self, hot_swap):
        """Test adaptive swapping based on complexity"""
        # Start with kagome
        assert hot_swap.current_topology == 'kagome'
        
        # Detect O(n²) complexity
        await hot_swap.adaptive_swap_for_complexity("O(n²)")
        
        # Should switch to small-world
        assert hot_swap.current_topology == 'small_world'
        
    def test_topology_recommendation(self, hot_swap):
        """Test topology recommendation for problems"""
        assert hot_swap.recommend_topology_for_problem('pattern_recognition') == 'kagome'
        assert hot_swap.recommend_topology_for_problem('global_search') == 'small_world'
        assert hot_swap.recommend_topology_for_problem('optimization') == 'triangular'
        
    @pytest.mark.asyncio
    async def test_ccl_integration(self):
        """Test integration with CCL"""
        # Mock CCL
        mock_ccl = Mock()
        mock_ccl.energy_broker = Mock()
        mock_ccl.energy_broker.request = Mock(return_value=True)
        mock_ccl.config = Mock(max_lyapunov=0.05)
        
        # Integrate
        hot_swap = await integrate_hot_swap_with_ccl(mock_ccl)
        
        assert hasattr(mock_ccl, 'hot_swap_laplacian')
        assert mock_ccl.hot_swap_laplacian == hot_swap
        
    def test_swap_metrics(self, hot_swap):
        """Test metrics reporting"""
        metrics = hot_swap.get_swap_metrics()
        
        assert 'current_topology' in metrics
        assert 'total_swaps' in metrics
        assert 'available_topologies' in metrics
        assert metrics['current_topology'] == 'kagome'
        assert metrics['total_swaps'] == 0
        
    @pytest.mark.asyncio
    async def test_rollback_on_failure(self, hot_swap):
        """Test rollback mechanism on swap failure"""
        # Mock a failing stability check
        with patch.object(hot_swap, 'verify_swap_stability', return_value=False):
            # Attempt swap
            await hot_swap.hot_swap_laplacian_with_safety('triangular')
            
            # Should remain on original topology
            assert hot_swap.current_topology == 'kagome'
            
            # Check rollback recorded
            if hot_swap.swap_history:
                assert hot_swap.swap_history[-1].get('rollback') == True

class TestO2Mitigation:
    """Test O(n²) complexity mitigation strategies"""
    
    @pytest.mark.asyncio
    async def test_complexity_detection_and_mitigation(self):
        """Test full O(n²) detection and mitigation flow"""
        hot_swap = HotSwappableLaplacian()
        
        # Simulate O(n²) workload
        n = 100
        hot_swap.active_solitons = [
            {'amplitude': 1.0, 'phase': i*0.1, 'index': i}
            for i in range(n)
        ]
        
        # Energy grows quadratically
        class QuadraticLattice:
            def __init__(self, n):
                self.n = n
                self.psi = np.ones(n*n, dtype=complex)
                self.total_energy = n*n  # O(n²)
            def step(self):
                pass
                
        hot_swap.lattice = QuadraticLattice(n)
        
        # Should trigger harvest due to high energy
        await hot_swap.hot_swap_laplacian_with_safety('small_world')
        
        # Verify mitigation
        assert hot_swap.current_topology == 'small_world'
        assert hot_swap.energy_harvested_total > 0
        
        # Small-world should have O(log n) properties
        config = hot_swap.topologies['small_world']
        assert config.optimal_for == ['global_search', 'fast_mixing']
        
    def test_shadow_interference_for_redundancy(self):
        """Test shadow traces eliminate redundant computations"""
        hot_swap = HotSwappableLaplacian()
        
        # Create redundant bright solitons
        redundant_solitons = [
            {'amplitude': 1.0, 'phase': 0, 'index': 0},
            {'amplitude': 1.0, 'phase': 0, 'index': 1},  # Duplicate
            {'amplitude': 1.0, 'phase': 0, 'index': 2},  # Duplicate
        ]
        
        # Create shadows
        shadows = [hot_swap.create_shadow_trace(s) for s in redundant_solitons]
        
        # Shadows should have opposite phase for cancellation
        for i in range(1, len(shadows)):
            phase_diff = abs(shadows[i].phaseTag - shadows[0].phaseTag)
            assert np.isclose(phase_diff, np.pi)  # π phase difference
            
    @pytest.mark.asyncio
    async def test_topology_sequence_for_algorithms(self):
        """Test topology switching sequence for different algorithms"""
        hot_swap = HotSwappableLaplacian()
        
        # All-pairs shortest path
        await hot_swap.adaptive_swap_for_complexity("O(n²)")
        assert hot_swap.current_topology == 'small_world'
        
        # Dense matrix multiplication
        await hot_swap.adaptive_swap_for_complexity("dense_matrix")
        assert hot_swap.current_topology == 'triangular'
        
        # Sparse search
        await hot_swap.adaptive_swap_for_complexity("sparse_search")
        assert hot_swap.current_topology == 'honeycomb'

# Performance benchmarks
class BenchmarkHotSwap:
    """Benchmark hot-swap performance"""
    
    def test_swap_speed(self, benchmark):
        """Benchmark swap execution time"""
        hot_swap = HotSwappableLaplacian(lattice_size=(50, 50))
        
        async def do_swap():
            await hot_swap.hot_swap_laplacian_with_safety('triangular')
            
        result = benchmark(lambda: asyncio.run(do_swap()))
        
        # Should complete in reasonable time
        assert result < 1.0  # Less than 1 second
        
    def test_memory_overhead(self):
        """Test memory efficiency of topology storage"""
        import sys
        
        hot_swap = HotSwappableLaplacian(lattice_size=(100, 100))
        
        # Measure size
        laplacian_size = sys.getsizeof(hot_swap.graph_laplacian.data)
        
        # Sparse matrix should be efficient
        n_nodes = 100 * 100 * 3  # Kagome has 3 sites per cell
        dense_size = n_nodes * n_nodes * 8  # 8 bytes per float64
        
        assert laplacian_size < 0.01 * dense_size  # Less than 1% of dense

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
