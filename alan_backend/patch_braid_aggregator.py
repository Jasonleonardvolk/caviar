#!/usr/bin/env python3
"""
Patch script for BraidAggregator based on red-pen review.
Addresses correctness, safety, performance, and API issues.
"""

import re
import sys
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BraidAggregatorPatcher:
    """Applies patches to braid_aggregator.py"""
    
    def __init__(self):
        self.file_path = Path(__file__).parent / "braid_aggregator.py"
        self.backup_path = self.file_path.with_suffix('.py.bak')
        self.patches = []
        
    def create_patches(self):
        """Create all patches organized by priority"""
        
        # PRIORITY 1: Correctness & Safety Fixes
        
        # Fix 1: JSON serialization of np.float64
        self.patches.append({
            'name': 'Fix lambda_max JSON serialization',
            'old': "'lambda_max': max(lambda_values) if lambda_values else 0.0,",
            'new': "'lambda_max': float(max(lambda_values)) if lambda_values else 0.0,"
        })
        
        # Fix 2: Handle None betti_numbers
        self.patches.append({
            'name': 'Fix None betti_numbers TypeError',
            'old': "betti_numbers=summary.get('betti_max')",
            'new': "betti_numbers=summary.get('betti_max', [])"
        })
        
        # Fix 3: Clear spectral_summaries on stop
        self.patches.append({
            'name': 'Clear spectral_summaries to prevent memory leak',
            'old': """        # Wait for cancellation
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        
        logger.info("BraidAggregator stopped")""",
            'new': """        # Wait for cancellation
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        
        # Clear tasks and summaries to prevent memory leak
        self.tasks.clear()
        for summaries in self.spectral_summaries.values():
            summaries.clear()
        
        logger.info("BraidAggregator stopped")"""
        })
        
        # Fix 4: Cast eigenvalues in _reconstruct_eigenvalues
        self.patches.append({
            'name': 'Fix eigenvalue JSON serialization in reconstruction',
            'old': """        if trajectory:
            # Use actual trajectory values
            return np.array(trajectory)""",
            'new': """        if trajectory:
            # Use actual trajectory values with proper casting
            return np.asarray([float(x) for x in trajectory])"""
        })
        
        # Fix 5: Add error tracking in _aggregate_scale
        self.patches.append({
            'name': 'Add error tracking for silent failures',
            'old': """        # Check for novelty spikes
        if summary['lambda_max'] > 0.0:
            # Get Origin classification
            test_eigenvalues = self._reconstruct_eigenvalues(summary)
            classification = self.origin_sentry.classify(
                test_eigenvalues,
                betti_numbers=summary.get('betti_max', [])
            )
            
            # Handle novelty spike
            if classification['novelty_score'] > 0.7:
                await self._handle_novelty_spike(scale, summary, classification)""",
            'new': """        # Check for novelty spikes
        if summary['lambda_max'] > 0.0:
            try:
                # Get Origin classification
                test_eigenvalues = self._reconstruct_eigenvalues(summary)
                classification = self.origin_sentry.classify(
                    test_eigenvalues,
                    betti_numbers=summary.get('betti_max', [])
                )
                
                # Handle novelty spike
                if classification['novelty_score'] > self.novelty_threshold:
                    await self._handle_novelty_spike(scale, summary, classification)
                    
            except Exception as e:
                logger.error(f"Origin classification failed for {scale.value}: {e}")
                self.metrics['errors'] = self.metrics.get('errors', 0) + 1"""
        })
        
        # PRIORITY 2: Performance Improvements
        
        # Fix 6: Add last timestamp tracking for efficient aggregation
        self.patches.append({
            'name': 'Add timestamp tracking for performance',
            'old': """        # Metrics
        self.metrics = {
            'aggregations_performed': 0,
            'spectral_computations': 0,
            'retro_updates': 0,
            'novelty_spikes': 0
        }""",
            'new': """        # Metrics
        self.metrics = {
            'aggregations_performed': 0,
            'spectral_computations': 0,
            'retro_updates': 0,
            'novelty_spikes': 0,
            'errors': 0
        }
        
        # Performance optimization: track last seen timestamps
        self._last_seen_timestamps = {
            scale: 0 for scale in TimeScale
        }"""
        })
        
        # Fix 7: Optimize event filtering
        self.patches.append({
            'name': 'Optimize event window retrieval',
            'old': """        # Get recent events
        events = buffer.get_window()""",
            'new': """        # Get recent events efficiently
        all_events = buffer.get_window()
        last_ts = self._last_seen_timestamps.get(scale, 0)
        
        # Filter only new events
        events = [e for e in all_events if e.t_epoch_us > last_ts]
        
        # Update last seen timestamp
        if events:
            self._last_seen_timestamps[scale] = events[-1].t_epoch_us"""
        })
        
        # Fix 8: Use deque for spectral_summaries
        self.patches.append({
            'name': 'Use deque for O(1) operations',
            'old': """from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging
import json
from pathlib import Path
import sys""",
            'new': """from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging
import json
from pathlib import Path
import sys
from collections import deque"""
        })
        
        self.patches.append({
            'name': 'Replace list with deque for spectral_summaries',
            'old': """        # Spectral summary cache
        self.spectral_summaries = {
            scale: [] for scale in TimeScale
        }""",
            'new': """        # Spectral summary cache with bounded size
        self.spectral_summaries = {
            scale: deque(maxlen=1000) for scale in TimeScale
        }"""
        })
        
        # Remove the manual limiting since deque handles it
        self.patches.append({
            'name': 'Remove manual list limiting',
            'old': """        # Store summary
        self.spectral_summaries[scale].append(summary)
        
        # Limit history
        if len(self.spectral_summaries[scale]) > 1000:
            self.spectral_summaries[scale] = self.spectral_summaries[scale][-1000:]""",
            'new': """        # Store summary (deque handles size limiting)
        self.spectral_summaries[scale].append(summary)"""
        })
        
        # PRIORITY 3: API & Style Improvements
        
        # Fix 9: Add configurable thresholds
        self.patches.append({
            'name': 'Add configurable thresholds',
            'old': """class BraidAggregator:
    \"\"\"
    Aggregates temporal braid data and computes spectral summaries
    \"\"\"
    
    def __init__(self, braiding_engine: Optional[TemporalBraidingEngine] = None,
                 origin_sentry: Optional[OriginSentry] = None):""",
            'new': """class BraidAggregator:
    \"\"\"Aggregate temporal braid data and compute spectral summaries.
    
    Provides scheduled processing of temporal buffers with configurable
    thresholds and cross-scale coherence computation.
    \"\"\"
    
    def __init__(self, braiding_engine: Optional[TemporalBraidingEngine] = None,
                 origin_sentry: Optional[OriginSentry] = None,
                 novelty_threshold: float = 0.7,
                 logger: Optional[logging.Logger] = None):"""
        })
        
        self.patches.append({
            'name': 'Initialize configurable attributes',
            'old': """        self.engine = braiding_engine or get_braiding_engine()
        self.origin_sentry = origin_sentry or OriginSentry()""",
            'new': """        self.engine = braiding_engine or get_braiding_engine()
        self.origin_sentry = origin_sentry or OriginSentry()
        self.novelty_threshold = novelty_threshold
        self.logger = logger or logging.getLogger(__name__)"""
        })
        
        # Fix 10: Replace logger with self.logger
        self.patches.append({
            'name': 'Use instance logger',
            'old': 'logger.info',
            'new': 'self.logger.info'
        })
        
        self.patches.append({
            'name': 'Use instance logger for debug',
            'old': 'logger.debug',
            'new': 'self.logger.debug'
        })
        
        self.patches.append({
            'name': 'Use instance logger for error',
            'old': 'logger.error',
            'new': 'self.logger.error'
        })
        
        # Fix 11: Add context manager support
        self.patches.append({
            'name': 'Add context manager support',
            'old': """        logger.info("BraidAggregator stopped")
        
    async def _aggregation_loop(self, scale: TimeScale, interval: float):""",
            'new': """        self.logger.info("BraidAggregator stopped")
    
    async def __aenter__(self):
        \"\"\"Async context manager entry\"\"\"
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        \"\"\"Async context manager exit\"\"\"
        await self.stop()
        return False
    
    async def _aggregation_loop(self, scale: TimeScale, interval: float):"""
        })
        
        # Fix 12: Add lookback configuration
        self.patches.append({
            'name': 'Make lookback configurable',
            'old': """        # Apply retro-coherence based on scale
        lookback_map = {
            TimeScale.MICRO: 1000,        # 1ms
            TimeScale.MESO: 10_000_000,   # 10s
            TimeScale.MACRO: 600_000_000  # 10min
        }""",
            'new': """        # Apply retro-coherence based on scale (configurable)
        lookback_map = getattr(self, 'lookback_map', {
            TimeScale.MICRO: 1000,        # 1ms
            TimeScale.MESO: 10_000_000,   # 10s
            TimeScale.MACRO: 600_000_000  # 10min
        })"""
        })
        
        # Fix 13: Return list from deque in get_spectral_timeline
        self.patches.append({
            'name': 'Convert deque to list for timeline',
            'old': """        # Return last window_size summaries
        return summaries[-window_size:]""",
            'new': """        # Return last window_size summaries as list
        return list(summaries)[-window_size:]"""
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
        logger.info(f"\nPatch Summary:")
        logger.info(f"  Total patches: {len(self.patches)}")
        logger.info(f"  Applied: {len(applied_patches)}")
        logger.info(f"  Failed: {len(failed_patches)}")
        
        if failed_patches:
            logger.warning(f"\nFailed patches:")
            for name in failed_patches:
                logger.warning(f"  - {name}")
        
        return len(failed_patches) == 0
    
    def create_test_file(self):
        """Create a test file for the patched aggregator"""
        test_content = '''#!/usr/bin/env python3
"""
Test script for patched BraidAggregator
"""

import asyncio
import numpy as np
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from python.core.braid_buffers import TimeScale, get_braiding_engine
from alan_backend.braid_aggregator import BraidAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_patched_aggregator():
    """Test the patched BraidAggregator"""
    
    logger.info("Testing patched BraidAggregator...")
    
    # Test 1: Context manager support
    logger.info("\\nTest 1: Context manager support")
    engine = get_braiding_engine()
    
    async with BraidAggregator(engine, novelty_threshold=0.6) as agg:
        # Verify it started
        assert agg.running, "Aggregator should be running"
        logger.info("✓ Context manager entry successful")
        
        # Add some events
        for i in range(10):
            engine.record_event(
                kind="test",
                lambda_max=np.float64(0.1 + i * 0.01),  # Test float64 handling
                betti=[1.0, 2.0, float(i % 3)],
                data={'test': i}
            )
        
        await asyncio.sleep(0.5)
    
    # Verify it stopped
    assert not agg.running, "Aggregator should be stopped"
    assert len(agg.tasks) == 0, "Tasks should be cleared"
    logger.info("✓ Context manager exit successful")
    
    # Test 2: Error handling
    logger.info("\\nTest 2: Error handling")
    
    # Create aggregator with mocked failing origin_sentry
    class FailingSentry:
        def classify(self, *args, **kwargs):
            raise RuntimeError("Test error")
    
    agg2 = BraidAggregator(engine, origin_sentry=FailingSentry())
    await agg2.start()
    
    # Generate high-novelty events
    for i in range(5):
        engine.record_event(
            kind="high_novelty",
            lambda_max=10.0,  # High value to trigger classification
            betti=None  # Test None handling
        )
    
    await asyncio.sleep(0.2)
    await agg2.stop()
    
    # Check error tracking
    assert agg2.metrics.get('errors', 0) > 0, "Errors should be tracked"
    logger.info(f"✓ Error tracking working: {agg2.metrics['errors']} errors")
    
    # Test 3: Performance with large dataset
    logger.info("\\nTest 3: Performance test")
    
    agg3 = BraidAggregator(engine)
    await agg3.start()
    
    # Generate many events
    start_time = asyncio.get_event_loop().time()
    for i in range(1000):
        engine.record_event(
            kind="perf_test",
            lambda_max=np.random.random(),
            betti=[float(x) for x in np.random.randint(0, 5, 3)]
        )
    
    await asyncio.sleep(0.5)
    
    # Check performance
    elapsed = asyncio.get_event_loop().time() - start_time
    status = agg3.get_status()
    
    logger.info(f"✓ Processed {status['metrics']['aggregations_performed']} aggregations in {elapsed:.2f}s")
    logger.info(f"✓ Cache sizes: {status['spectral_cache_sizes']}")
    
    await agg3.stop()
    
    # Test 4: JSON serialization
    logger.info("\\nTest 4: JSON serialization")
    
    try:
        json_status = json.dumps(status)
        logger.info("✓ Status is JSON serializable")
    except TypeError as e:
        logger.error(f"✗ JSON serialization failed: {e}")
        raise
    
    logger.info("\\n✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_patched_aggregator())
'''
        
        test_path = self.file_path.parent / "test_braid_aggregator_patched.py"
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        logger.info(f"Created test file: {test_path}")
        return test_path


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Patch braid_aggregator.py")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed')
    parser.add_argument('--create-test', action='store_true', help='Create test file')
    parser.add_argument('--rollback', action='store_true', help='Restore from backup')
    
    args = parser.parse_args()
    
    patcher = BraidAggregatorPatcher()
    
    if args.rollback:
        if patcher.backup_path.exists():
            shutil.copy2(patcher.backup_path, patcher.file_path)
            logger.info(f"Restored from backup: {patcher.backup_path}")
        else:
            logger.error("No backup found")
        return
    
    # Create patches
    patcher.create_patches()
    logger.info(f"Created {len(patcher.patches)} patches")
    
    # Apply patches
    success = patcher.apply_patches(dry_run=args.dry_run)
    
    # Create test file
    if args.create_test and success:
        patcher.create_test_file()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
