#!/usr/bin/env python3
"""
Advanced patches for BraidAggregator - implements Future Ideas:
1. gRPC/WebSocket emitter for novelty events
2. Adaptive scheduling based on activity
3. Circuit breaker for error protection
"""

import re
import sys
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BraidAggregatorAdvancedPatcher:
    """Applies advanced feature patches to braid_aggregator.py"""
    
    def __init__(self):
        self.file_path = Path(__file__).parent / "braid_aggregator.py"
        self.backup_path = self.file_path.with_suffix('.py.advanced_bak')
        self.patches = []
        
    def create_patches(self):
        """Create patches for advanced features"""
        
        # Feature 1: Add WebSocket/Message Bus Support
        self.patches.append({
            'name': 'Add asyncio Queue for event emission',
            'old': """import asyncio
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging
import json
from pathlib import Path
import sys
from collections import deque""",
            'new': """import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
import logging
import json
from pathlib import Path
import sys
from collections import deque
from asyncio import Queue"""
        })
        
        self.patches.append({
            'name': 'Add event emitter initialization',
            'old': """    def __init__(self, braiding_engine: Optional[TemporalBraidingEngine] = None,
                 origin_sentry: Optional[OriginSentry] = None,
                 novelty_threshold: float = 0.7,
                 logger: Optional[logging.Logger] = None):""",
            'new': """    def __init__(self, braiding_engine: Optional[TemporalBraidingEngine] = None,
                 origin_sentry: Optional[OriginSentry] = None,
                 novelty_threshold: float = 0.7,
                 logger: Optional[logging.Logger] = None,
                 event_emitter: Optional[Callable] = None,
                 event_queue_size: int = 1000):"""
        })
        
        self.patches.append({
            'name': 'Initialize event queue and emitter',
            'old': """        self.novelty_threshold = novelty_threshold
        self.logger = logger or logging.getLogger(__name__)""",
            'new': """        self.novelty_threshold = novelty_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Event emission
        self.event_emitter = event_emitter
        self.event_queue = Queue(maxsize=event_queue_size)
        self._emitter_task = None"""
        })
        
        # Feature 2: Add Circuit Breaker
        self.patches.append({
            'name': 'Add circuit breaker state',
            'old': """        # Performance optimization: track last seen timestamps
        self._last_seen_timestamps = {
            scale: 0 for scale in TimeScale
        }""",
            'new': """        # Performance optimization: track last seen timestamps
        self._last_seen_timestamps = {
            scale: 0 for scale in TimeScale
        }
        
        # Circuit breaker
        self.circuit_breaker = {
            'enabled': True,
            'consecutive_errors': 0,
            'max_consecutive_errors': 5,
            'tripped': False,
            'trip_time': None
        }"""
        })
        
        # Feature 3: Add Adaptive Scheduling
        self.patches.append({
            'name': 'Add adaptive schedule state',
            'old': """        # Circuit breaker
        self.circuit_breaker = {
            'enabled': True,
            'consecutive_errors': 0,
            'max_consecutive_errors': 5,
            'tripped': False,
            'trip_time': None
        }""",
            'new': """        # Circuit breaker
        self.circuit_breaker = {
            'enabled': True,
            'consecutive_errors': 0,
            'max_consecutive_errors': 5,
            'tripped': False,
            'trip_time': None
        }
        
        # Adaptive scheduling
        self.adaptive_schedule = {
            TimeScale.MICRO: {
                'base_interval': 0.1,
                'current_interval': 0.1,
                'min_interval': 0.05,
                'max_interval': 1.0,
                'activity_threshold': 0.5,
                'last_activity': 0.0
            },
            TimeScale.MESO: {
                'base_interval': 10.0,
                'current_interval': 10.0,
                'min_interval': 5.0,
                'max_interval': 30.0,
                'activity_threshold': 0.3,
                'last_activity': 0.0
            },
            TimeScale.MACRO: {
                'base_interval': 300.0,
                'current_interval': 300.0,
                'min_interval': 60.0,
                'max_interval': 600.0,
                'activity_threshold': 0.2,
                'last_activity': 0.0
            }
        }"""
        })
        
        # Replace start method to include emitter task
        self.patches.append({
            'name': 'Start emitter task in start()',
            'old': """    async def start(self):
        \"\"\"Start aggregation tasks\"\"\"
        self.running = True
        
        # Start aggregation tasks for each timescale
        for scale, interval in self.schedule.items():
            task = asyncio.create_task(self._aggregation_loop(scale, interval))
            self.tasks[scale] = task
            
        self.logger.info("BraidAggregator started with 3 aggregation loops")""",
            'new': """    async def start(self):
        \"\"\"Start aggregation tasks\"\"\"
        self.running = True
        
        # Start aggregation tasks for each timescale
        for scale in TimeScale:
            interval = self.adaptive_schedule[scale]['current_interval']
            task = asyncio.create_task(self._adaptive_aggregation_loop(scale))
            self.tasks[scale] = task
        
        # Start event emitter task if configured
        if self.event_emitter:
            self._emitter_task = asyncio.create_task(self._event_emitter_loop())
            
        self.logger.info("BraidAggregator started with adaptive scheduling")"""
        })
        
        # Update stop method
        self.patches.append({
            'name': 'Stop emitter task in stop()',
            'old': """        # Clear tasks and summaries to prevent memory leak
        self.tasks.clear()
        for summaries in self.spectral_summaries.values():
            summaries.clear()
        
        self.logger.info("BraidAggregator stopped")""",
            'new': """        # Stop emitter task
        if self._emitter_task:
            self._emitter_task.cancel()
            try:
                await self._emitter_task
            except asyncio.CancelledError:
                pass
        
        # Clear tasks and summaries to prevent memory leak
        self.tasks.clear()
        for summaries in self.spectral_summaries.values():
            summaries.clear()
        
        self.logger.info("BraidAggregator stopped")"""
        })
        
        # Replace aggregation loop with adaptive version
        self.patches.append({
            'name': 'Replace aggregation loop with adaptive version',
            'old': """    async def _aggregation_loop(self, scale: TimeScale, interval: float):
        \"\"\"Main aggregation loop for a timescale\"\"\"
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
                await asyncio.sleep(interval)  # Continue after error""",
            'new': """    async def _adaptive_aggregation_loop(self, scale: TimeScale):
        \"\"\"Adaptive aggregation loop with dynamic intervals\"\"\"
        schedule = self.adaptive_schedule[scale]
        self.logger.info(f"Starting adaptive {scale.value} loop (base: {schedule['base_interval']}s)")
        
        while self.running:
            try:
                # Check circuit breaker
                if self.circuit_breaker['tripped']:
                    self.logger.warning(f"Circuit breaker tripped, skipping {scale.value}")
                    await asyncio.sleep(schedule['current_interval'])
                    continue
                
                # Wait for adaptive interval
                await asyncio.sleep(schedule['current_interval'])
                
                # Perform aggregation
                await self._aggregate_scale(scale)
                
                # Reset error count on success
                self.circuit_breaker['consecutive_errors'] = 0
                
                # Adapt schedule based on activity
                self._adapt_schedule(scale)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Aggregation error for {scale.value}: {e}")
                self._handle_circuit_breaker_error()
                await asyncio.sleep(schedule['current_interval'])"""
        })
        
        # Add circuit breaker error handling
        self.patches.append({
            'name': 'Add circuit breaker error handling method',
            'old': """                self.logger.error(f"Origin classification failed for {scale.value}: {e}")
                self.metrics['errors'] = self.metrics.get('errors', 0) + 1
        
        self.metrics['aggregations_performed'] += 1""",
            'new': """                self.logger.error(f"Origin classification failed for {scale.value}: {e}")
                self.metrics['errors'] = self.metrics.get('errors', 0) + 1
                self._handle_circuit_breaker_error()
        
        self.metrics['aggregations_performed'] += 1"""
        })
        
        # Update emit novelty event to use queue
        self.patches.append({
            'name': 'Update novelty event emission',
            'old': """    async def _emit_novelty_event(self, scale: TimeScale, classification: Dict[str, Any]):
        \"\"\"Emit novelty event for other systems\"\"\"
        event_data = {
            'type': 'novelty_spike',
            'scale': scale.value,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'classification': classification,
            'suggested_action': 'entropy_injection' if classification['novelty_score'] > 0.8 else 'monitor'
        }
        
        # Log for now - would emit to message queue in production
        self.logger.info(f"Novelty event: {json.dumps(event_data, indent=2)}")""",
            'new': """    async def _emit_novelty_event(self, scale: TimeScale, classification: Dict[str, Any]):
        \"\"\"Emit novelty event through async queue\"\"\"
        event_data = {
            'type': 'novelty_spike',
            'scale': scale.value,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'classification': classification,
            'suggested_action': 'entropy_injection' if classification['novelty_score'] > 0.8 else 'monitor'
        }
        
        # Queue event for emission
        try:
            self.event_queue.put_nowait(event_data)
        except asyncio.QueueFull:
            self.logger.warning("Event queue full, dropping novelty event")
            
        # Also log for debugging
        self.logger.debug(f"Queued novelty event: {event_data['type']} for {scale.value}")"""
        })
        
        # Add new helper methods at the end
        self.patches.append({
            'name': 'Add adaptive scheduling and circuit breaker methods',
            'old': """            'cross_scale_coherence': self.get_cross_scale_coherence(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }""",
            'new': """            'cross_scale_coherence': self.get_cross_scale_coherence(),
            'circuit_breaker': self.circuit_breaker.copy(),
            'adaptive_intervals': {
                scale.value: self.adaptive_schedule[scale]['current_interval']
                for scale in TimeScale
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _adapt_schedule(self, scale: TimeScale):
        \"\"\"Adapt aggregation interval based on activity\"\"\"
        schedule = self.adaptive_schedule[scale]
        
        # Calculate activity from recent lambda_max values
        recent = self.spectral_summaries[scale]
        if len(recent) >= 2:
            # Activity = max lambda in recent window
            recent_lambdas = [s['lambda_max'] for s in list(recent)[-10:]]
            activity = max(recent_lambdas) if recent_lambdas else 0.0
            
            # Update activity tracking
            schedule['last_activity'] = activity
            
            # Adapt interval
            if activity > schedule['activity_threshold']:
                # High activity - speed up
                schedule['current_interval'] = max(
                    schedule['min_interval'],
                    schedule['current_interval'] * 0.8
                )
                self.logger.debug(f"{scale.value} high activity ({activity:.3f}), interval -> {schedule['current_interval']:.2f}s")
            else:
                # Low activity - slow down
                schedule['current_interval'] = min(
                    schedule['max_interval'],
                    schedule['current_interval'] * 1.2
                )
    
    def _handle_circuit_breaker_error(self):
        \"\"\"Handle circuit breaker logic on errors\"\"\"
        if not self.circuit_breaker['enabled']:
            return
            
        self.circuit_breaker['consecutive_errors'] += 1
        
        if self.circuit_breaker['consecutive_errors'] >= self.circuit_breaker['max_consecutive_errors']:
            self.circuit_breaker['tripped'] = True
            self.circuit_breaker['trip_time'] = datetime.now(timezone.utc)
            self.logger.error(f"Circuit breaker TRIPPED after {self.circuit_breaker['consecutive_errors']} errors")
            
            # Schedule reset after 60 seconds
            asyncio.create_task(self._reset_circuit_breaker())
    
    async def _reset_circuit_breaker(self):
        \"\"\"Reset circuit breaker after cooldown\"\"\"
        await asyncio.sleep(60)  # 1 minute cooldown
        self.circuit_breaker['tripped'] = False
        self.circuit_breaker['consecutive_errors'] = 0
        self.logger.info("Circuit breaker reset")
    
    async def _event_emitter_loop(self):
        \"\"\"Process queued events for emission\"\"\"
        self.logger.info("Event emitter loop started")
        
        while self.running:
            try:
                # Get event with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(), 
                    timeout=1.0
                )
                
                # Emit via configured emitter
                if self.event_emitter:
                    try:
                        await self.event_emitter(event)
                    except Exception as e:
                        self.logger.error(f"Event emission failed: {e}")
                        
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Event emitter error: {e}")
        
        self.logger.info("Event emitter loop stopped")"""
        })
        
    def apply_patches(self, dry_run=False):
        """Apply all patches to the file"""
        if not self.file_path.exists():
            logger.error(f"File not found: {self.file_path}")
            return False
            
        # Create backup
        if not dry_run:
            shutil.copy2(self.file_path, self.backup_path)
            logger.info(f"Created backup: {self.backup_path}")
        
        # Read content
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        applied_patches = []
        failed_patches = []
        
        # Apply patches
        for patch in self.patches:
            if patch['old'] in content:
                if not dry_run:
                    content = content.replace(patch['old'], patch['new'])
                applied_patches.append(patch['name'])
                logger.info(f"✓ Applied: {patch['name']}")
            else:
                failed_patches.append(patch['name'])
                logger.warning(f"✗ Pattern not found: {patch['name']}")
        
        # Write back
        if not dry_run and content != original_content:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Wrote patched file: {self.file_path}")
        
        # Summary
        logger.info(f"\nAdvanced Patch Summary:")
        logger.info(f"  Total patches: {len(self.patches)}")
        logger.info(f"  Applied: {len(applied_patches)}")
        logger.info(f"  Failed: {len(failed_patches)}")
        
        return len(failed_patches) == 0
    
    def create_example_integration(self):
        """Create example integration showing advanced features"""
        example_content = '''#!/usr/bin/env python3
"""
Example integration with advanced BraidAggregator features
Shows WebSocket emission, adaptive scheduling, and circuit breaker
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from python.core.braid_buffers import get_braiding_engine
from alan_backend.braid_aggregator import BraidAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketEventEmitter:
    """Example WebSocket event emitter"""
    
    def __init__(self, uri="ws://localhost:8765"):
        self.uri = uri
        self.websocket = None
        
    async def connect(self):
        """Connect to WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.uri)
            logger.info(f"Connected to WebSocket: {self.uri}")
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            
    async def emit(self, event):
        """Emit event over WebSocket"""
        if self.websocket and not self.websocket.closed:
            try:
                await self.websocket.send(json.dumps(event))
                logger.info(f"Emitted: {event['type']} for {event.get('scale', 'unknown')}")
            except Exception as e:
                logger.error(f"WebSocket send failed: {e}")
        else:
            logger.warning("WebSocket not connected")
            
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()


async def demonstrate_advanced_features():
    """Demonstrate advanced BraidAggregator features"""
    
    # Create WebSocket emitter
    ws_emitter = WebSocketEventEmitter()
    await ws_emitter.connect()
    
    # Create aggregator with advanced features
    engine = get_braiding_engine()
    aggregator = BraidAggregator(
        braiding_engine=engine,
        novelty_threshold=0.6,
        event_emitter=ws_emitter.emit  # Pass WebSocket emitter
    )
    
    logger.info("Starting BraidAggregator with advanced features...")
    
    async with aggregator as agg:
        # Monitor adaptive scheduling
        logger.info("Initial intervals:")
        for scale, schedule in agg.adaptive_schedule.items():
            logger.info(f"  {scale.value}: {schedule['current_interval']}s")
        
        # Generate varying activity levels
        logger.info("\\nGenerating activity patterns...")
        
        # Phase 1: Low activity
        logger.info("Phase 1: Low activity")
        for i in range(20):
            engine.record_event(
                kind="low_activity",
                lambda_max=0.1 + (i % 3) * 0.01  # Low values
            )
            await asyncio.sleep(0.1)
        
        await asyncio.sleep(2)
        
        # Check adapted intervals
        logger.info("\\nIntervals after low activity:")
        for scale, schedule in agg.adaptive_schedule.items():
            logger.info(f"  {scale.value}: {schedule['current_interval']}s (activity: {schedule['last_activity']:.3f})")
        
        # Phase 2: High activity spike
        logger.info("\\nPhase 2: High activity spike")
        for i in range(30):
            engine.record_event(
                kind="high_activity",
                lambda_max=2.0 + (i % 5) * 0.5,  # High values
                betti=[float(i), float(i % 3), float(i % 5)]
            )
            await asyncio.sleep(0.05)
        
        await asyncio.sleep(2)
        
        # Check adapted intervals
        logger.info("\\nIntervals after high activity:")
        for scale, schedule in agg.adaptive_schedule.items():
            logger.info(f"  {scale.value}: {schedule['current_interval']}s (activity: {schedule['last_activity']:.3f})")
        
        # Phase 3: Test circuit breaker
        logger.info("\\nPhase 3: Testing circuit breaker")
        
        # Inject errors by creating invalid origin sentry
        class FailingSentry:
            def __init__(self):
                self.fail_count = 0
                
            def classify(self, *args, **kwargs):
                self.fail_count += 1
                if self.fail_count > 3:
                    raise RuntimeError(f"Simulated failure #{self.fail_count}")
                return {'novelty_score': 0.5}
        
        # Replace with failing sentry
        agg.origin_sentry = FailingSentry()
        
        # Generate events to trigger errors
        for i in range(10):
            engine.record_event(
                kind="error_test",
                lambda_max=10.0  # High value to trigger classification
            )
            await asyncio.sleep(0.1)
        
        await asyncio.sleep(1)
        
        # Check circuit breaker status
        status = agg.get_status()
        cb_status = status['circuit_breaker']
        logger.info(f"\\nCircuit breaker status: {cb_status}")
        
        if cb_status['tripped']:
            logger.warning("Circuit breaker is TRIPPED!")
            logger.info("Waiting for automatic reset...")
            
            # Wait a bit
            await asyncio.sleep(5)
    
    # Cleanup
    await ws_emitter.close()
    
    logger.info("\\n✅ Advanced features demonstration complete!")


# Mock WebSocket server for testing
async def mock_websocket_server():
    """Simple WebSocket server for testing"""
    async def handler(websocket, path):
        logger.info("Client connected to mock WebSocket server")
        try:
            async for message in websocket:
                data = json.loads(message)
                logger.info(f"Server received: {data['type']} event")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
    
    server = await websockets.serve(handler, "localhost", 8765)
    logger.info("Mock WebSocket server running on ws://localhost:8765")
    await server.wait_closed()


async def main():
    """Run demonstration with mock server"""
    # Start mock server
    server_task = asyncio.create_task(mock_websocket_server())
    
    # Give server time to start
    await asyncio.sleep(1)
    
    # Run demonstration
    try:
        await demonstrate_advanced_features()
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
'''
        
        example_path = self.file_path.parent / "example_advanced_aggregator.py"
        with open(example_path, 'w', encoding='utf-8') as f:
            f.write(example_content)
        
        logger.info(f"Created example: {example_path}")
        return example_path


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply advanced patches to braid_aggregator.py")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed')
    parser.add_argument('--create-example', action='store_true', help='Create example integration')
    parser.add_argument('--rollback', action='store_true', help='Restore from backup')
    
    args = parser.parse_args()
    
    patcher = BraidAggregatorAdvancedPatcher()
    
    if args.rollback:
        if patcher.backup_path.exists():
            shutil.copy2(patcher.backup_path, patcher.file_path)
            logger.info(f"Restored from backup: {patcher.backup_path}")
        else:
            logger.error("No backup found")
        return
    
    # Note: This assumes the basic patches have already been applied
    logger.info("Note: This patch assumes basic patches are already applied!")
    
    # Create patches
    patcher.create_patches()
    logger.info(f"Created {len(patcher.patches)} advanced patches")
    
    # Apply patches
    success = patcher.apply_patches(dry_run=args.dry_run)
    
    # Create example
    if args.create_example and success:
        patcher.create_example_integration()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
