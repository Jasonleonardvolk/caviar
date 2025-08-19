#!/usr/bin/env python3
"""
Modified braid_aggregator.py - Enhanced with TorusCells integration
Production-ready implementation with comprehensive error handling
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging
import json
from pathlib import Path
import sys
from collections import deque

# Add parent paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from python.core.braid_buffers import (
    TemporalBraidingEngine, BraidEvent, TimeScale, get_braiding_engine
)
from alan_backend.origin_sentry import OriginSentry

# NEW IMPORTS for TorusCells integration
from python.core.torus_cells import get_torus_cells, betti_update
from python.core.observer_synthesis import emit_token

logger = logging.getLogger(__name__)

class BraidAggregator:
    """
    Aggregates temporal braid data and computes spectral summaries
    Now using TorusCells for topology computation and persistence
    """
    
    def __init__(self, braiding_engine: Optional[TemporalBraidingEngine] = None,
                 origin_sentry: Optional[OriginSentry] = None,
                 novelty_threshold: float = 0.7,
                 logger: Optional[logging.Logger] = None):
        self.engine = braiding_engine or get_braiding_engine()
        self.origin_sentry = origin_sentry or OriginSentry()
        self.novelty_threshold = novelty_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Schedule intervals (seconds)
        self.schedule = {
            TimeScale.MICRO: 0.1,    # 100ms
            TimeScale.MESO: 10.0,    # 10s
            TimeScale.MACRO: 300.0   # 5 min
        }
        
        # Lookback configuration
        self.lookback_map = {
            TimeScale.MICRO: 1000,        # 1ms
            TimeScale.MESO: 10_000_000,   # 10s
            TimeScale.MACRO: 600_000_000  # 10min
        }
        
        # Running tasks
        self.tasks = {}
        self.running = False
        
        # Spectral summary cache with bounded deques
        self.spectral_summaries = {
            scale: deque(maxlen=1000) for scale in TimeScale
        }
        
        # Track last seen timestamps for efficient filtering
        self._last_seen_timestamps = {scale: 0 for scale in TimeScale}
        
        # TorusCells integration
        self.torus_cells = get_torus_cells()
        
        # Metrics
        self.metrics = {
            'aggregations_performed': 0,
            'spectral_computations': 0,
            'retro_updates': 0,
            'novelty_spikes': 0,
            'betti_computations': 0,
            'errors': 0,
            'tokens_emitted': 0
        }
        
        self.logger.info("BraidAggregator initialized with TorusCells integration")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
        return False  # Don't suppress exceptions
    
    async def start(self):
        """Start aggregation tasks"""
        self.running = True
        
        # Start aggregation tasks for each timescale
        for scale, interval in self.schedule.items():
            task = asyncio.create_task(self._aggregation_loop(scale, interval))
            self.tasks[scale] = task
            
        self.logger.info("BraidAggregator started with 3 aggregation loops")
    
    async def stop(self):
        """Stop all aggregation tasks"""
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks.values():
            task.cancel()
            
        # Wait for cancellation
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        
        # Clear tasks
        self.tasks.clear()
        
        # Clear summaries to prevent memory leak
        for summaries in self.spectral_summaries.values():
            summaries.clear()
            
        self.logger.info("BraidAggregator stopped")
    
    async def _aggregation_loop(self, scale: TimeScale, interval: float):
        """Main aggregation loop for a timescale"""
        self.logger.info(f"Starting {scale.value} aggregation loop (interval: {interval}s)")
        
        while self.running:
            try:
                # Wait for interval
                await asyncio.sleep(interval)
                
                # Perform aggregation
                await self._aggregate_scale(scale)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Aggregation error for {scale.value}: {e}")
                self.metrics['errors'] += 1
                await asyncio.sleep(interval)  # Continue after error
    
    async def _aggregate_scale(self, scale: TimeScale):
        """Aggregate data for specific timescale with TorusCells integration"""
        buffer = self.engine.buffers[scale]
        
        # Get all events and filter only new ones
        all_events = buffer.get_window()
        last_ts = self._last_seen_timestamps[scale]
        
        # Filter only new events
        events = [e for e in all_events if e.t_epoch_us > last_ts]
        
        if not events:
            return
            
        # Update last seen timestamp
        self._last_seen_timestamps[scale] = events[-1].t_epoch_us
        
        # Compute spectral summary
        summary = self._compute_spectral_summary(events, scale)
        
        # Store summary
        self.spectral_summaries[scale].append(summary)
        
        # Check for novelty spikes
        if summary['lambda_max'] > 0.0:
            try:
                # Get Origin classification with proper Betti numbers
                test_eigenvalues = self._reconstruct_eigenvalues(summary)
                
                # Use actual Betti numbers from summary or empty list
                betti_numbers = summary.get('betti_max', [])
                
                classification = self.origin_sentry.classify(
                    test_eigenvalues,
                    betti_numbers=betti_numbers
                )
                
                # Handle novelty spike
                if classification['novelty_score'] > self.novelty_threshold:
                    await self._handle_novelty_spike(scale, summary, classification)
                    
            except Exception as e:
                self.logger.error(f"Origin classification failed: {e}")
                self.metrics['errors'] += 1
        
        self.metrics['aggregations_performed'] += 1
        
        # Log summary
        self.logger.debug(f"{scale.value} summary: λ_max={summary['lambda_max']:.3f}, "
                         f"events={len(events)}")
    
    def _compute_spectral_summary(self, events: List[BraidEvent], scale: TimeScale) -> Dict[str, Any]:
        """
        Compute spectral summary from events
        Now uses TorusCells for Betti computation
        """
        # Extract spectral data
        lambda_values = [float(e.lambda_max) for e in events if e.lambda_max is not None]
        
        # Compute statistics
        summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_count': len(events),
            'scale': scale.value,
            'lambda_max': float(max(lambda_values)) if lambda_values else 0.0,
            'lambda_mean': float(np.mean(lambda_values)) if lambda_values else 0.0,
            'lambda_std': float(np.std(lambda_values)) if lambda_values else 0.0,
            'lambda_trajectory': [float(x) for x in lambda_values[-10:]] if lambda_values else [],
        }
        
        # Use TorusCells for Betti computation
        if len(events) > 0 and lambda_values:
            try:
                # Create point cloud from spectral data
                # This is a simplified representation - in production, you might
                # use the actual state vectors from events
                n_points = min(len(lambda_values), 20)
                points = np.array([[λ, i*0.1] for i, λ in enumerate(lambda_values[-n_points:])])
                
                # Compute Betti numbers using TorusCells
                idea_id = f"{scale.value}_{summary['timestamp']}"
                b0, b1 = self.torus_cells.betti_update(
                    idea_id=idea_id,
                    vertices=points,
                    coherence_band=self._get_coherence_band(summary['lambda_max']),
                    metadata={
                        'scale': scale.value,
                        'event_count': len(events),
                        'lambda_max': summary['lambda_max']
                    }
                )
                
                summary['betti_max'] = [float(b0), float(b1)]
                summary['betti_computed'] = True
                self.metrics['betti_computations'] += 1
                
                # Check if idea became topologically protected
                protected_ideas = self.torus_cells.get_protected_ideas()
                if idea_id in protected_ideas:
                    summary['topologically_protected'] = True
                    self.logger.info(f"Idea {idea_id} is topologically protected")
                    
            except Exception as e:
                self.logger.warning(f"Failed to compute Betti numbers: {e}")
                summary['betti_max'] = [1.0, 0.0]  # Default values
                summary['betti_computed'] = False
        else:
            summary['betti_max'] = [0.0, 0.0]
            summary['betti_computed'] = False
        
        self.metrics['spectral_computations'] += 1
        
        # Emit observer token for spectral summary
        try:
            token = emit_token({
                "type": "braid_spectral_summary",
                "source": "braid_aggregator",
                "scale": scale.value,
                "lambda_max": summary['lambda_max'],
                "betti": summary['betti_max'],
                "event_count": summary['event_count'],
                "timestamp": summary['timestamp']
            })
            self.metrics['tokens_emitted'] += 1
            self.logger.debug(f"Emitted braid summary token: {token[:8]}...")
        except Exception as e:
            self.logger.warning(f"Failed to emit summary token: {e}")
        
        return summary
    
    def _get_coherence_band(self, lambda_max: float) -> str:
        """Determine coherence band from lambda_max"""
        if lambda_max > 0.04:
            return 'critical'
        elif lambda_max > 0.01:
            return 'global'
        else:
            return 'local'
    
    def _reconstruct_eigenvalues(self, summary: Dict[str, Any]) -> np.ndarray:
        """Reconstruct approximate eigenvalue array from summary"""
        # Use actual trajectory if available
        trajectory = summary.get('lambda_trajectory', [])
        
        if trajectory:
            # Ensure all values are Python floats
            return np.asarray([float(x) for x in trajectory])
        else:
            # Synthetic spectrum with exponential decay
            lambda_max = summary.get('lambda_max', 0.0)
            n_modes = 8
            eigenvalues = lambda_max * np.exp(-np.arange(n_modes) * 0.5)
            return eigenvalues
    
    async def _handle_novelty_spike(self, scale: TimeScale, 
                                   summary: Dict[str, Any],
                                   classification: Dict[str, Any]):
        """Handle detected novelty spike"""
        self.logger.info(f"🌟 Novelty spike in {scale.value}: score={classification['novelty_score']:.3f}")
        
        self.metrics['novelty_spikes'] += 1
        
        # Apply retro-coherence based on scale
        try:
            self.engine.apply_retro_coherence(
                int(datetime.now(timezone.utc).timestamp() * 1_000_000),
                f"novelty_spike_{scale.value}",
                lookback_us=self.lookback_map[scale]
            )
            
            self.metrics['retro_updates'] += 1
        except Exception as e:
            self.logger.error(f"Failed to apply retro-coherence: {e}")
            self.metrics['errors'] += 1
        
        # Emit event for downstream systems
        await self._emit_novelty_event(scale, classification, summary)
    
    async def _emit_novelty_event(self, scale: TimeScale, 
                                 classification: Dict[str, Any],
                                 summary: Dict[str, Any]):
        """Emit novelty event for other systems"""
        event_data = {
            'type': 'novelty_spike',
            'scale': scale.value,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'classification': classification,
            'summary_metrics': {
                'lambda_max': summary.get('lambda_max', 0.0),
                'betti': summary.get('betti_max', []),
                'topologically_protected': summary.get('topologically_protected', False)
            },
            'suggested_action': 'entropy_injection' if classification['novelty_score'] > 0.8 else 'monitor'
        }
        
        # Emit observer token
        try:
            token = emit_token({
                "type": "novelty_spike",
                "source": "braid_aggregator",
                "scale": scale.value,
                "novelty_score": classification['novelty_score'],
                "lambda_max": summary.get('lambda_max', 0.0),
                "betti": summary.get('betti_max', []),
                "action": event_data['suggested_action'],
                "timestamp": event_data['timestamp']
            })
            self.metrics['tokens_emitted'] += 1
            self.logger.debug(f"Emitted novelty spike token: {token[:8]}...")
        except Exception as e:
            self.logger.warning(f"Failed to emit novelty token: {e}")
        
        # Log for now - would emit to message queue in production
        self.logger.info(f"Novelty event: {json.dumps(event_data, indent=2)}")
        
        # TODO: When message bus is available, replace with:
        # await self.message_bus.publish('novelty.spike', event_data)
    
    def get_spectral_timeline(self, scale: TimeScale, 
                            window_size: int = 100) -> List[Dict[str, Any]]:
        """Get spectral timeline for a scale"""
        summaries = self.spectral_summaries[scale]
        
        if not summaries:
            return []
        
        # Return last window_size summaries
        return list(summaries)[-window_size:]
    
    def get_cross_scale_coherence(self) -> Dict[str, float]:
        """Compute coherence metrics across scales"""
        coherence = {}
        
        # Get recent summaries from each scale
        recent = {}
        for scale in TimeScale:
            timeline = self.get_spectral_timeline(scale, window_size=10)
            if timeline:
                recent[scale] = timeline
        
        # Compute cross-scale correlations
        if len(recent) >= 2:
            scales = list(recent.keys())
            
            for i in range(len(scales)):
                for j in range(i+1, len(scales)):
                    scale1, scale2 = scales[i], scales[j]
                    
                    # Extract lambda_max sequences
                    seq1 = [s['lambda_max'] for s in recent[scale1]]
                    seq2 = [s['lambda_max'] for s in recent[scale2]]
                    
                    # Compute correlation (handle different lengths)
                    min_len = min(len(seq1), len(seq2))
                    if min_len > 1:
                        # Ensure arrays for correlation
                        arr1 = np.array(seq1[-min_len:])
                        arr2 = np.array(seq2[-min_len:])
                        
                        # Check for zero variance
                        if np.std(arr1) > 0 and np.std(arr2) > 0:
                            corr = np.corrcoef(arr1, arr2)[0, 1]
                            coherence[f"{scale1.value}_{scale2.value}"] = float(corr)
                        else:
                            coherence[f"{scale1.value}_{scale2.value}"] = 0.0
        
        return coherence
    
    def get_topological_summary(self) -> Dict[str, Any]:
        """Get summary of topological features across scales"""
        protected_ideas = self.torus_cells.get_protected_ideas()
        
        # Analyze protected ideas by scale
        scale_topology = {scale.value: {'protected': 0, 'total': 0} for scale in TimeScale}
        
        for idea_id, idea_data in protected_ideas.items():
            # Extract scale from idea_id (format: "scale_timestamp")
            parts = idea_id.split('_')
            if parts[0] in scale_topology:
                scale_topology[parts[0]]['protected'] += 1
        
        # Count total ideas per scale
        for scale in TimeScale:
            scale_topology[scale.value]['total'] = len(self.spectral_summaries[scale])
        
        return {
            'total_protected': len(protected_ideas),
            'by_scale': scale_topology,
            'betti_computations': self.metrics['betti_computations']
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get aggregator status"""
        return {
            'running': self.running,
            'metrics': self.metrics.copy(),
            'spectral_cache_sizes': {
                scale.value: len(self.spectral_summaries[scale])
                for scale in TimeScale
            },
            'cross_scale_coherence': self.get_cross_scale_coherence(),
            'topological_summary': self.get_topological_summary(),
            'torus_cells_backend': self.torus_cells.backend,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

# Convenience functions for integration
async def start_braid_aggregator():
    """Start the global braid aggregator"""
    aggregator = BraidAggregator()
    await aggregator.start()
    return aggregator

if __name__ == "__main__":
    # Test the aggregator with TorusCells
    async def test_aggregator():
        print("Testing BraidAggregator with TorusCells integration...")
        
        # Get braiding engine
        engine = get_braiding_engine()
        
        # Create aggregator with context manager
        async with BraidAggregator(engine) as aggregator:
            # Simulate some events
            print("Simulating events...")
            for i in range(50):
                engine.record_event(
                    kind="test",
                    lambda_max=0.02 + (i % 5) * 0.01,
                    betti=[1.0, float(i % 3)],
                    data={'test_id': i}
                )
                await asyncio.sleep(0.02)
            
            # Wait for aggregation
            print("Waiting for aggregation...")
            await asyncio.sleep(2)
            
            # Get status
            status = aggregator.get_status()
            print("\nAggregator Status:", json.dumps(status, indent=2))
            
            # Get spectral timeline
            micro_timeline = aggregator.get_spectral_timeline(TimeScale.MICRO)
            print(f"\nMicro timeline ({len(micro_timeline)} summaries):")
            for summary in micro_timeline[-5:]:
                print(f"  λ_max={summary['lambda_max']:.3f}, "
                      f"betti={summary.get('betti_max', [])}, "
                      f"events={summary['event_count']}")
            
            # Check topological summary
            topo_summary = aggregator.get_topological_summary()
            print(f"\nTopological Summary:")
            print(f"  Total protected ideas: {topo_summary['total_protected']}")
            print(f"  Betti computations: {topo_summary['betti_computations']}")
            
        print("\n✅ Test completed!")
    
    # Run test
    asyncio.run(test_aggregator())
