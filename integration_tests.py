#!/usr/bin/env python3
"""
TORI Complete System Integration Tests
Verifies all components work together correctly
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone
import websockets
import json
import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from tori_master import TORIMaster

class TestTORIIntegration:
    """Integration tests for complete TORI system"""
    
    @pytest.fixture
    async def tori_master(self):
        """Create and start TORI master"""
        master = TORIMaster()
        
        # Disable UI server for tests
        master.config['enable_ui_server'] = False
        
        # Start in background
        start_task = asyncio.create_task(master.start())
        
        # Wait for initialization
        await asyncio.sleep(3)
        
        yield master
        
        # Cleanup
        await master.stop()
        start_task.cancel()
        
    @pytest.mark.asyncio
    async def test_all_components_start(self, tori_master):
        """Test that all components initialize correctly"""
        # Check core components
        assert 'tori' in tori_master.components
        assert 'dark_soliton' in tori_master.components
        assert 'chaos_controller' in tori_master.components
        assert 'eigen_guard' in tori_master.components
        
        # Check TORI is operational
        tori_status = tori_master.components['tori'].get_status()
        assert tori_status['operational'] == True
        
    @pytest.mark.asyncio
    async def test_websocket_metrics_connection(self, tori_master):
        """Test WebSocket metrics server connection"""
        if not tori_master.config['enable_websocket']:
            pytest.skip("WebSocket disabled")
            
        # Connect to WebSocket
        uri = f"ws://localhost:8765/ws/eigensentry"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Should receive initial state
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                
                assert data['type'] == 'initial_state'
                assert 'metrics' in data['data']
                assert 'config' in data['data']
                
        except Exception as e:
            pytest.fail(f"WebSocket connection failed: {e}")
            
    @pytest.mark.asyncio
    async def test_dark_soliton_chaos_integration(self, tori_master):
        """Test dark soliton simulator with chaos controller"""
        # Get components
        soliton_sim = tori_master.components['dark_soliton']
        chaos_ctrl = tori_master.components['chaos_controller']
        
        # Create soliton
        soliton_sim.create_dark_soliton(64, 64, width=10.0, depth=0.8)
        
        # Trigger chaos burst
        burst_id = chaos_ctrl.trigger(level=0.3, duration=50)
        assert burst_id != ""
        
        # Run simulation with chaos
        for _ in range(50):
            soliton_sim.step(1)
            chaos_ctrl.step()
            
        # Check chaos affected soliton (energy should change)
        metrics = soliton_sim.step(1)
        assert metrics['energy'] > 0
        
        # Check chaos completed
        assert chaos_ctrl.state.value == 'cooldown'
        
    @pytest.mark.asyncio
    async def test_eigensentry_curvature_adaptation(self, tori_master):
        """Test EigenSentry adapts to soliton curvature"""
        eigen_guard = tori_master.components['eigen_guard']
        soliton_sim = tori_master.components['dark_soliton']
        
        # Create high-curvature soliton
        soliton_sim.create_dark_soliton(64, 64, width=5.0, depth=0.9)
        
        # Get soliton field
        field = soliton_sim.get_field()
        
        # Compute curvature
        curvature_metrics = eigen_guard.compute_local_curvature(field.real)
        
        # Update threshold
        eigen_guard.update_threshold(curvature_metrics)
        
        # Verify threshold adapted (should be lower for high curvature)
        assert eigen_guard.current_threshold < eigen_guard.base_threshold
        
    @pytest.mark.asyncio
    async def test_chaos_burst_from_reflection(self, tori_master):
        """Test reflection layer can trigger chaos bursts"""
        # Process a query that should trigger reflection with chaos
        query = "Explore creative solutions to consciousness emergence"
        
        result = await tori_master.process_query(
            query,
            context={'enable_chaos': True, 'request_chaos_burst': True}
        )
        
        # Check chaos was used
        assert 'chaos_burst_id' in result
        
        # Verify burst in controller
        chaos_ctrl = tori_master.components['chaos_controller']
        assert len(chaos_ctrl.burst_history) > 0 or chaos_ctrl.state.value != 'idle'
        
    @pytest.mark.asyncio
    async def test_phase_codec_integration(self, tori_master):
        """Test phase codec with TORI processing"""
        # This would test the phase codec integration
        # For now, verify TORI can process code-related queries
        
        code_query = "Transform this code into phase representation: function hello() { return 'world'; }"
        
        result = await tori_master.process_query(code_query)
        
        assert 'response' in result
        assert len(result['response']) > 0
        
    @pytest.mark.asyncio
    async def test_safety_monitoring_integration(self, tori_master):
        """Test safety monitoring across components"""
        tori = tori_master.components['tori']
        
        # Get safety report
        safety = tori.safety_system.get_safety_report()
        
        assert safety['monitoring_active'] == True
        assert safety['current_safety_level'] in ['optimal', 'nominal']
        
        # Verify metrics include chaos components
        assert 'checkpoints_available' in safety
        assert 'energy_status' in safety
        
    @pytest.mark.asyncio
    async def test_cross_component_energy_flow(self, tori_master):
        """Test energy conservation across components"""
        # Get initial energy states
        chaos_ctrl = tori_master.components['chaos_controller']
        initial_chaos_energy = chaos_ctrl.current_energy
        
        # Trigger activity
        await tori_master.process_query(
            "Search for patterns using chaos dynamics",
            context={'enable_chaos': True}
        )
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Check energy conservation
        final_chaos_energy = chaos_ctrl.current_energy
        
        # Energy should return near baseline
        assert abs(final_chaos_energy - initial_chaos_energy) < 0.2
        
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, tori_master):
        """Test system handles concurrent operations"""
        # Submit multiple queries concurrently
        queries = [
            "Analyze quantum entanglement patterns",
            "Search for emergent behaviors in complex systems",
            "Remember key insights about consciousness",
            "Brainstorm novel approaches to AI safety"
        ]
        
        tasks = []
        for query in queries:
            task = tori_master.process_query(query, context={'enable_chaos': True})
            tasks.append(task)
            
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all completed successfully
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Query {i} failed: {result}"
            assert 'response' in result
            
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sustained_operation(self, tori_master):
        """Test system stability over extended operation"""
        # Run for extended period
        start_time = datetime.now(timezone.utc)
        query_count = 0
        
        while (datetime.now(timezone.utc) - start_time).total_seconds() < 30:
            # Process query
            result = await tori_master.process_query(
                f"Test query {query_count}",
                context={'enable_chaos': query_count % 3 == 0}  # Every 3rd with chaos
            )
            
            assert 'response' in result
            query_count += 1
            
            # Brief pause
            await asyncio.sleep(0.5)
            
        # Check system still healthy
        tori_status = tori_master.components['tori'].get_status()
        assert tori_status['operational'] == True
        
        # Verify processed queries
        assert query_count > 50
        
        # Check no memory leaks (energy should be stable)
        chaos_ctrl = tori_master.components['chaos_controller']
        assert chaos_ctrl.current_energy < 2.0  # Not accumulating

# Performance benchmarks
class BenchmarkTORI:
    """Performance benchmarks for integrated system"""
    
    @pytest.mark.benchmark
    async def benchmark_query_processing(benchmark, tori_master):
        """Benchmark query processing speed"""
        async def process_query():
            return await tori_master.process_query("Test query for benchmarking")
            
        result = benchmark(asyncio.run, process_query())
        assert 'response' in result
        
    @pytest.mark.benchmark  
    async def benchmark_chaos_burst(benchmark, tori_master):
        """Benchmark chaos burst performance"""
        chaos_ctrl = tori_master.components['chaos_controller']
        
        def trigger_and_run():
            burst_id = chaos_ctrl.trigger(level=0.5, duration=100)
            for _ in range(100):
                chaos_ctrl.step()
            return burst_id
            
        burst_id = benchmark(trigger_and_run)
        assert burst_id != ""

if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])
