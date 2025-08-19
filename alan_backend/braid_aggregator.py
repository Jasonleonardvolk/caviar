#!/usr/bin/env python3
"""
Braid Aggregator - Scheduled processing of temporal buffers
Computes spectral summaries and manages cross-scale propagation
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging
import json
from pathlib import Path
import sys

# Add parent paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from python.core.braid_buffers import (
    TemporalBraidingEngine, BraidEvent, TimeScale, get_braiding_engine
)
from alan_backend.origin_sentry import OriginSentry

logger = logging.getLogger(__name__)

class BraidAggregator:
    """
    Aggregates temporal braid data and computes spectral summaries
    """
    
    def __init__(self, braiding_engine: Optional[TemporalBraidingEngine] = None,
                 origin_sentry: Optional[OriginSentry] = None):
        self.engine = braiding_engine or get_braiding_engine()
        self.origin_sentry = origin_sentry or OriginSentry()
        
        # Schedule intervals (seconds)
        self.schedule = {
            TimeScale.MICRO: 0.1,    # 100ms
            TimeScale.MESO: 10.0,    # 10s
            TimeScale.MACRO: 300.0   # 5 min
        }
        
        # Running tasks
        self.tasks = {}
        self.running = False
        
        # Spectral summary cache
        self.spectral_summaries = {
            scale: [] for scale in TimeScale
        }
        
        # Metrics
        self.metrics = {
            'aggregations_performed': 0,
            'spectral_computations': 0,
            'retro_updates': 0,
            'novelty_spikes': 0
        }
        
        logger.info("BraidAggregator initialized")
    
    async def start(self):
        """Start aggregation tasks"""
        self.running = True
        
        # Start aggregation tasks for each timescale
        for scale, interval in self.schedule.items():
            task = asyncio.create_task(self._aggregation_loop(scale, interval))
            self.tasks[scale] = task
            
        logger.info("BraidAggregator started with 3 aggregation loops")
    
    async def stop(self):
        """Stop all aggregation tasks"""
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks.values():
            task.cancel()
            
        # Wait for cancellation
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        
        logger.info("BraidAggregator stopped")
    
    async def _aggregation_loop(self, scale: TimeScale, interval: float):
        """Main aggregation loop for a timescale"""
        logger.info(f"Starting {scale.value} aggregation loop (interval: {interval}s)")
        
        while self.running:
            try:
                # Wait for interval
                await asyncio.sleep(interval)
                
                # Perform aggregation
                await self._aggregate_scale(scale)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aggregation error for {scale.value}: {e}")
                await asyncio.sleep(interval)  # Continue after error
    
    async def _aggregate_scale(self, scale: TimeScale):
        """Aggregate data for specific timescale"""
        buffer = self.engine.buffers[scale]
        
        # Get recent events
        events = buffer.get_window()
        
        if not events:
            return
        
        # Compute spectral summary
        summary = self._compute_spectral_summary(events)
        
        # Store summary
        self.spectral_summaries[scale].append(summary)
        
        # Limit history
        if len(self.spectral_summaries[scale]) > 1000:
            self.spectral_summaries[scale] = self.spectral_summaries[scale][-1000:]
        
        # Check for novelty spikes
        if summary['lambda_max'] > 0.0:
            # Get Origin classification
            test_eigenvalues = self._reconstruct_eigenvalues(summary)
            classification = self.origin_sentry.classify(
                test_eigenvalues,
                betti_numbers=summary.get('betti_max')
            )
            
            # Handle novelty spike
            if classification['novelty_score'] > 0.7:
                await self._handle_novelty_spike(scale, summary, classification)
        
        self.metrics['aggregations_performed'] += 1
        
        # Log summary
        logger.debug(f"{scale.value} summary: λ_max={summary['lambda_max']:.3f}, "
                    f"events={len(events)}")
    
    def _compute_spectral_summary(self, events: List[BraidEvent]) -> Dict[str, Any]:
        """Compute spectral summary from events"""
        # Extract spectral data
        lambda_values = [e.lambda_max for e in events if e.lambda_max is not None]
        betti_arrays = [e.betti for e in events if e.betti]
        
        # Compute statistics
        summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_count': len(events),
            'lambda_max': max(lambda_values) if lambda_values else 0.0,
            'lambda_mean': float(np.mean(lambda_values)) if lambda_values else 0.0,
            'lambda_std': float(np.std(lambda_values)) if lambda_values else 0.0,
            'lambda_trajectory': lambda_values[-10:] if lambda_values else [],  # Last 10
        }
        
        # Aggregate Betti numbers
        if betti_arrays:
            max_dim = max(len(b) for b in betti_arrays)
            betti_max = []
            for i in range(max_dim):
                values = [b[i] for b in betti_arrays if i < len(b)]
                betti_max.append(max(values) if values else 0.0)
            summary['betti_max'] = betti_max
            summary['betti_mean'] = [
                float(np.mean([b[i] for b in betti_arrays if i < len(b)]))
                for i in range(max_dim)
            ]
        
        self.metrics['spectral_computations'] += 1
        
        return summary
    
    def _reconstruct_eigenvalues(self, summary: Dict[str, Any]) -> np.ndarray:
        """Reconstruct approximate eigenvalue array from summary"""
        # Simple reconstruction using max and trajectory
        lambda_max = summary['lambda_max']
        trajectory = summary.get('lambda_trajectory', [])
        
        if trajectory:
            # Use actual trajectory values
            return np.array(trajectory)
        else:
            # Synthetic spectrum with exponential decay
            n_modes = 8
            eigenvalues = lambda_max * np.exp(-np.arange(n_modes) * 0.5)
            return eigenvalues
    
    async def _handle_novelty_spike(self, scale: TimeScale, 
                                   summary: Dict[str, Any],
                                   classification: Dict[str, Any]):
        """Handle detected novelty spike"""
        logger.info(f"🌟 Novelty spike in {scale.value}: score={classification['novelty_score']:.3f}")
        
        self.metrics['novelty_spikes'] += 1
        
        # Apply retro-coherence based on scale
        lookback_map = {
            TimeScale.MICRO: 1000,        # 1ms
            TimeScale.MESO: 10_000_000,   # 10s
            TimeScale.MACRO: 600_000_000  # 10min
        }
        
        self.engine.apply_retro_coherence(
            int(datetime.now(timezone.utc).timestamp() * 1_000_000),
            f"novelty_spike_{scale.value}",
            lookback_us=lookback_map[scale]
        )
        
        self.metrics['retro_updates'] += 1
        
        # Emit event for downstream systems
        await self._emit_novelty_event(scale, classification)
    
    async def _emit_novelty_event(self, scale: TimeScale, classification: Dict[str, Any]):
        """Emit novelty event for other systems"""
        event_data = {
            'type': 'novelty_spike',
            'scale': scale.value,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'classification': classification,
            'suggested_action': 'entropy_injection' if classification['novelty_score'] > 0.8 else 'monitor'
        }
        
        # Log for now - would emit to message queue in production
        logger.info(f"Novelty event: {json.dumps(event_data, indent=2)}")
    
    def get_spectral_timeline(self, scale: TimeScale, 
                            window_size: int = 100) -> List[Dict[str, Any]]:
        """Get spectral timeline for a scale"""
        summaries = self.spectral_summaries[scale]
        
        if not summaries:
            return []
        
        # Return last window_size summaries
        return summaries[-window_size:]
    
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
                        corr = np.corrcoef(seq1[-min_len:], seq2[-min_len:])[0, 1]
                        coherence[f"{scale1.value}_{scale2.value}"] = float(corr)
        
        return coherence
    
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
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

# Convenience functions for integration
async def start_braid_aggregator():
    """Start the global braid aggregator"""
    aggregator = BraidAggregator()
    await aggregator.start()
    return aggregator

if __name__ == "__main__":
    # Test the aggregator
    async def test_aggregator():
        # Get braiding engine
        engine = get_braiding_engine()
        
        # Create aggregator
        aggregator = BraidAggregator(engine)
        
        # Start aggregation
        await aggregator.start()
        
        # Simulate some events
        for i in range(50):
            engine.record_event(
                kind="test",
                lambda_max=0.02 + (i % 5) * 0.01,
                betti=[1.0, float(i % 3)],
                data={'test_id': i}
            )
            await asyncio.sleep(0.02)
        
        # Wait for aggregation
        await asyncio.sleep(2)
        
        # Get status
        status = aggregator.get_status()
        print("Aggregator Status:", json.dumps(status, indent=2))
        
        # Get spectral timeline
        micro_timeline = aggregator.get_spectral_timeline(TimeScale.MICRO)
        print(f"\nMicro timeline ({len(micro_timeline)} summaries):")
        for summary in micro_timeline[-5:]:
            print(f"  λ_max={summary['lambda_max']:.3f}, events={summary['event_count']}")
        
        # Stop aggregator
        await aggregator.stop()
    
    # Run test
    asyncio.run(test_aggregator())
