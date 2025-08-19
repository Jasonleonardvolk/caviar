#!/usr/bin/env python3
"""
Temporal Braiding Buffers - Multi-timescale cognitive traces
μ-intuition (≤1ms) ↔ meso-planning (10s-10min) ↔ macro-vision (hours-days)
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque
import json
import logging
import threading
import time
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class TimeScale(Enum):
    """Temporal scales for cognitive traces"""
    MICRO = "micro"      # ≤ 1ms - intuition bursts
    MESO = "meso"        # 10s-10min - planning cycles  
    MACRO = "macro"      # hours-days - long-term vision

@dataclass
class BraidEvent:
    """Single event in the temporal braid"""
    t_epoch_us: int  # Microsecond timestamp
    kind: str        # 'user', 'ai', 'ghost', 'eigenmode'
    lambda_max: Optional[float] = None
    betti: List[float] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            't_epoch_us': self.t_epoch_us,
            'kind': self.kind,
            'lambda_max': self.lambda_max,
            'betti': self.betti,
            'data': self.data
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BraidEvent':
        return cls(
            t_epoch_us=d['t_epoch_us'],
            kind=d['kind'],
            lambda_max=d.get('lambda_max'),
            betti=d.get('betti', []),
            data=d.get('data', {})
        )

class TemporalBuffer:
    """Ring buffer for a specific timescale"""
    
    def __init__(self, scale: TimeScale, capacity: int, 
                 window_us: int, persist_path: Optional[Path] = None):
        self.scale = scale
        self.capacity = capacity
        self.window_us = window_us  # Time window in microseconds
        self.buffer = deque(maxlen=capacity)
        self.persist_path = persist_path
        self.lock = threading.RLock()
        
        # Downsample callbacks
        self.downsample_callbacks = []
        
        # Load persisted data
        if persist_path and persist_path.exists():
            self._load()
    
    def append(self, event: BraidEvent):
        """Add event to buffer"""
        with self.lock:
            self.buffer.append(event)
            
            # Check if we need to trigger downsampling
            if len(self.buffer) >= self.capacity * 0.9:
                self._trigger_downsample()
    
    def get_window(self, window_us: Optional[int] = None) -> List[BraidEvent]:
        """Get events within time window"""
        with self.lock:
            if not self.buffer:
                return []
            
            window = window_us or self.window_us
            now_us = int(time.time() * 1_000_000)
            cutoff_us = now_us - window
            
            return [e for e in self.buffer if e.t_epoch_us >= cutoff_us]
    
    def downsample(self, factor: int = 10) -> List[BraidEvent]:
        """
        Downsample buffer by factor, aggregating spectral data
        """
        with self.lock:
            if len(self.buffer) < factor:
                return list(self.buffer)
            
            downsampled = []
            
            # Process in chunks
            events = list(self.buffer)
            for i in range(0, len(events), factor):
                chunk = events[i:i+factor]
                
                # Aggregate spectral data
                lambda_maxes = [e.lambda_max for e in chunk if e.lambda_max is not None]
                betti_arrays = [e.betti for e in chunk if e.betti]
                
                # Create aggregated event
                agg_event = BraidEvent(
                    t_epoch_us=chunk[len(chunk)//2].t_epoch_us,  # Median time
                    kind=f"{self.scale.value}_aggregate",
                    lambda_max=max(lambda_maxes) if lambda_maxes else None,
                    betti=self._aggregate_betti(betti_arrays),
                    data={
                        'source_events': len(chunk),
                        'kinds': [e.kind for e in chunk],
                        'lambda_mean': np.mean(lambda_maxes) if lambda_maxes else None,
                        'lambda_std': np.std(lambda_maxes) if lambda_maxes else None
                    }
                )
                
                downsampled.append(agg_event)
            
            return downsampled
    
    def _aggregate_betti(self, betti_arrays: List[List[float]]) -> List[float]:
        """Aggregate Betti numbers (max across arrays)"""
        if not betti_arrays:
            return []
        
        max_len = max(len(b) for b in betti_arrays)
        aggregated = []
        
        for i in range(max_len):
            values = [b[i] for b in betti_arrays if i < len(b)]
            aggregated.append(max(values) if values else 0.0)
        
        return aggregated
    
    def register_downsample_callback(self, callback: Callable[[List[BraidEvent]], None]):
        """Register callback for downsample events"""
        self.downsample_callbacks.append(callback)
    
    def _trigger_downsample(self):
        """Trigger downsampling and notify callbacks"""
        downsampled = self.downsample()
        for callback in self.downsample_callbacks:
            try:
                callback(downsampled)
            except Exception as e:
                logger.error(f"Downsample callback error: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get buffer summary statistics"""
        with self.lock:
            events = list(self.buffer)
            
            if not events:
                return {
                    'scale': self.scale.value,
                    'count': 0,
                    'capacity': self.capacity,
                    'fill_ratio': 0.0
                }
            
            lambda_values = [e.lambda_max for e in events if e.lambda_max is not None]
            
            return {
                'scale': self.scale.value,
                'count': len(events),
                'capacity': self.capacity,
                'fill_ratio': len(events) / self.capacity,
                'time_span_us': events[-1].t_epoch_us - events[0].t_epoch_us,
                'lambda_stats': {
                    'max': max(lambda_values) if lambda_values else None,
                    'mean': np.mean(lambda_values) if lambda_values else None,
                    'std': np.std(lambda_values) if lambda_values else None
                },
                'kind_distribution': self._count_kinds(events)
            }
    
    def _count_kinds(self, events: List[BraidEvent]) -> Dict[str, int]:
        """Count event kinds"""
        counts = {}
        for e in events:
            counts[e.kind] = counts.get(e.kind, 0) + 1
        return counts
    
    def persist(self):
        """Save buffer to disk"""
        if not self.persist_path:
            return
        
        with self.lock:
            data = {
                'scale': self.scale.value,
                'capacity': self.capacity,
                'window_us': self.window_us,
                'events': [e.to_dict() for e in self.buffer]
            }
            
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def _load(self):
        """Load buffer from disk"""
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
            
            for event_dict in data.get('events', []):
                self.buffer.append(BraidEvent.from_dict(event_dict))
                
            logger.info(f"Loaded {len(self.buffer)} events for {self.scale.value} buffer")
        except Exception as e:
            logger.error(f"Failed to load buffer: {e}")

class TemporalBraidingEngine:
    """
    Main temporal braiding engine with retro-coherent updates
    """
    
    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir) if persist_dir else Path("braid_buffers")
        
        # Create multi-scale buffers
        self.buffers = {
            TimeScale.MICRO: TemporalBuffer(
                TimeScale.MICRO,
                capacity=10000,  # 10k events
                window_us=1000,  # 1ms window
                persist_path=self.persist_dir / "micro_buffer.json"
            ),
            TimeScale.MESO: TemporalBuffer(
                TimeScale.MESO,
                capacity=1000,   # 1k events
                window_us=60_000_000,  # 1 minute window
                persist_path=self.persist_dir / "meso_buffer.json"
            ),
            TimeScale.MACRO: TemporalBuffer(
                TimeScale.MACRO,
                capacity=100,    # 100 events
                window_us=3600_000_000,  # 1 hour window
                persist_path=self.persist_dir / "macro_buffer.json"
            )
        }
        
        # Set up cascading downsamples
        self.buffers[TimeScale.MICRO].register_downsample_callback(
            self._cascade_to_meso
        )
        self.buffers[TimeScale.MESO].register_downsample_callback(
            self._cascade_to_macro
        )
        
        # Retro-coherence queue
        self.retro_queue = deque(maxlen=100)
        
        # Causal DAG for meso-scale
        self.causal_dag = {}
        
        # Start background persistence
        self._start_persistence_thread()
        
        logger.info("Temporal Braiding Engine initialized with 3 timescales")
    
    def record_event(self, kind: str, lambda_max: Optional[float] = None,
                    betti: Optional[List[float]] = None, 
                    data: Optional[Dict[str, Any]] = None):
        """Record event at micro scale"""
        event = BraidEvent(
            t_epoch_us=int(time.time() * 1_000_000),
            kind=kind,
            lambda_max=lambda_max,
            betti=betti or [],
            data=data or {}
        )
        
        self.buffers[TimeScale.MICRO].append(event)
    
    def _cascade_to_meso(self, micro_events: List[BraidEvent]):
        """Cascade downsampled micro events to meso buffer"""
        for event in micro_events:
            self.buffers[TimeScale.MESO].append(event)
            
            # Build causal links
            self._update_causal_dag(event)
    
    def _cascade_to_macro(self, meso_events: List[BraidEvent]):
        """Cascade downsampled meso events to macro buffer"""
        for event in meso_events:
            self.buffers[TimeScale.MACRO].append(event)
    
    def _update_causal_dag(self, event: BraidEvent):
        """Update causal DAG for meso-scale planning"""
        # Simple causality: high lambda_max events cause subsequent instabilities
        if event.lambda_max and event.lambda_max > 0.03:
            # Mark as causal source
            self.causal_dag[event.t_epoch_us] = {
                'event': event,
                'effects': [],
                'strength': event.lambda_max
            }
            
            # Link to recent events
            cutoff = event.t_epoch_us - 10_000_000  # 10 seconds
            for ts, node in self.causal_dag.items():
                if cutoff < ts < event.t_epoch_us:
                    node['effects'].append(event.t_epoch_us)
    
    def apply_retro_coherence(self, event_time_us: int, label: str, 
                            lookback_us: int = 1000):
        """
        Apply retro-coherent update - backpropagate labels to past events
        """
        # Find events to label
        micro_events = self.buffers[TimeScale.MICRO].get_window(lookback_us)
        
        labeled_count = 0
        for event in micro_events:
            if event.t_epoch_us < event_time_us:
                event.data['retro_label'] = label
                event.data['retro_time'] = event_time_us
                labeled_count += 1
        
        logger.info(f"Retro-coherence: labeled {labeled_count} events with '{label}'")
        
        # Queue for higher-scale propagation
        self.retro_queue.append({
            'time_us': event_time_us,
            'label': label,
            'lookback_us': lookback_us
        })
    
    def get_multi_scale_context(self, time_us: Optional[int] = None) -> Dict[str, Any]:
        """Get context across all timescales"""
        if time_us is None:
            time_us = int(time.time() * 1_000_000)
        
        context = {}
        
        # Get windows from each scale
        for scale in TimeScale:
            buffer = self.buffers[scale]
            events = buffer.get_window()
            
            context[scale.value] = {
                'events': [e.to_dict() for e in events[-10:]],  # Last 10
                'summary': buffer.get_summary()
            }
        
        # Add causal analysis
        context['causal_sources'] = self._identify_causal_sources(time_us)
        
        return context
    
    def _identify_causal_sources(self, time_us: int, window_us: int = 60_000_000) -> List[Dict]:
        """Identify causal sources within window"""
        sources = []
        
        cutoff = time_us - window_us
        for ts, node in self.causal_dag.items():
            if cutoff < ts < time_us:
                sources.append({
                    'time_us': ts,
                    'strength': node['strength'],
                    'effects_count': len(node['effects'])
                })
        
        return sorted(sources, key=lambda x: x['strength'], reverse=True)[:5]
    
    def _start_persistence_thread(self):
        """Start background thread for periodic persistence"""
        def persist_loop():
            while True:
                time.sleep(60)  # Persist every minute
                try:
                    for buffer in self.buffers.values():
                        buffer.persist()
                except Exception as e:
                    logger.error(f"Persistence error: {e}")
        
        thread = threading.Thread(target=persist_loop, daemon=True)
        thread.start()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive braiding status"""
        return {
            'buffers': {
                scale.value: buffer.get_summary() 
                for scale, buffer in self.buffers.items()
            },
            'causal_dag_size': len(self.causal_dag),
            'retro_queue_size': len(self.retro_queue),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

# Global instance
_engine = None

def get_braiding_engine() -> TemporalBraidingEngine:
    """Get or create global braiding engine"""
    global _engine
    if _engine is None:
        _engine = TemporalBraidingEngine()
    return _engine

if __name__ == "__main__":
    # Test the braiding engine
    engine = get_braiding_engine()
    
    # Simulate some events
    for i in range(100):
        engine.record_event(
            kind="eigenmode",
            lambda_max=0.01 + (i % 10) * 0.005,
            betti=[1.0, 0.5] if i % 3 == 0 else [1.0],
            data={'iteration': i}
        )
        time.sleep(0.001)  # 1ms between events
    
    # Apply retro-coherence
    engine.apply_retro_coherence(
        int(time.time() * 1_000_000),
        "creative_spike",
        lookback_us=10000  # 10ms lookback
    )
    
    # Get multi-scale context
    context = engine.get_multi_scale_context()
    print(json.dumps(context, indent=2))
    
    # Get status
    status = engine.get_status()
    print("\nBraiding Status:", json.dumps(status, indent=2))
