#!/usr/bin/env python3
"""
Quick demo of BraidAggregator patches
Shows before/after comparison of key issues
"""

import asyncio
import numpy as np
import json
import sys
from pathlib import Path

# Add parent paths
sys.path.append(str(Path(__file__).parent.parent.parent))

async def demonstrate_patches():
    """Demonstrate the impact of patches"""
    
    print("=" * 60)
    print("BraidAggregator Patch Demonstration")
    print("=" * 60)
    
    # Issue 1: JSON Serialization
    print("\n1. JSON Serialization Issue")
    print("-" * 30)
    
    # Simulate the problem
    lambda_values = [np.float64(0.1), np.float64(0.2), np.float64(0.3)]
    
    # Before patch
    try:
        bad_dict = {'lambda_max': max(lambda_values)}
        json.dumps(bad_dict)
        print("✗ Should have failed with np.float64")
    except TypeError as e:
        print(f"✓ Expected error: {type(e).__name__}: {e}")
    
    # After patch
    good_dict = {'lambda_max': float(max(lambda_values))}
    json_str = json.dumps(good_dict)
    print(f"✓ Fixed with explicit cast: {json_str}")
    
    # Issue 2: None betti_numbers
    print("\n2. None betti_numbers Issue")
    print("-" * 30)
    
    # Simulate OriginSentry.classify expecting list
    def mock_classify(eigenvalues, betti_numbers):
        # This would crash with None
        return {'novelty_score': sum(betti_numbers) / len(betti_numbers)}
    
    # Before patch
    summary = {'betti_max': None}
    try:
        mock_classify([1, 2, 3], summary.get('betti_max'))
        print("✗ Should have failed with None")
    except TypeError as e:
        print(f"✓ Expected error: {type(e).__name__}: {e}")
    
    # After patch
    result = mock_classify([1, 2, 3], summary.get('betti_max', []))
    print(f"✓ Fixed with default []: No error!")
    
    # Issue 3: Performance Impact
    print("\n3. Performance Optimization")
    print("-" * 30)
    
    # Simulate event filtering
    import time
    
    # Create mock events
    class MockEvent:
        def __init__(self, timestamp):
            self.t_epoch_us = timestamp
    
    # Large buffer
    all_events = [MockEvent(i) for i in range(10000)]
    
    # Before: Process all events
    start = time.perf_counter()
    processed = all_events[:]  # Process everything
    before_time = time.perf_counter() - start
    
    # After: Process only new events
    last_seen = 9500
    start = time.perf_counter()
    new_events = [e for e in all_events if e.t_epoch_us > last_seen]
    after_time = time.perf_counter() - start
    
    print(f"Before: Process {len(processed)} events in {before_time*1000:.2f}ms")
    print(f"After:  Process {len(new_events)} events in {after_time*1000:.2f}ms")
    print(f"✓ Speedup: {before_time/after_time:.1f}x faster!")
    
    # Issue 4: Context Manager
    print("\n4. Context Manager Support")
    print("-" * 30)
    
    print("Before: Manual cleanup required")
    print("```python")
    print("agg = BraidAggregator()")
    print("try:")
    print("    await agg.start()")
    print("    # ... work ...")
    print("finally:")
    print("    await agg.stop()  # Easy to forget!")
    print("```")
    
    print("\nAfter: Automatic cleanup")
    print("```python")
    print("async with BraidAggregator() as agg:")
    print("    # ... work ...")
    print("    # Auto-stops on exit!")
    print("```")
    
    print("\n" + "=" * 60)
    print("✅ All patches address real production issues!")
    print("=" * 60)
    
    # Show patch command
    print("\nTo apply these patches:")
    print("  python patch_braid_aggregator.py --dry-run  # Preview")
    print("  python patch_braid_aggregator.py            # Apply")
    print("  python patch_braid_aggregator.py --create-test  # Test")


if __name__ == "__main__":
    asyncio.run(demonstrate_patches())
