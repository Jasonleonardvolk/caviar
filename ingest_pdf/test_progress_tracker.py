#!/usr/bin/env python3
"""
Test script for the improved ProgressTracker.
Demonstrates usage with and without tqdm.
"""

import time
import sys
import asyncio

# Simulate the ProgressTracker (normally imported from pipeline)
import threading
from typing import Dict, Any, Optional

# Try to import tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("tqdm not available - install with: pip install tqdm")

class ProgressTracker:
    """Simplified version of the improved ProgressTracker for testing"""
    
    def __init__(self, total: int, min_change: float = 1.0, min_seconds: float = 0.0,
                 description: str = "Progress", use_tqdm: bool = None):
        self.total = total
        self.current = 0
        self.last_reported_pct = -1
        self.last_reported_time = 0.0
        self.min_change = min_change
        self.min_seconds = min_seconds
        self.description = description
        
        self._lock = threading.RLock()
        
        if use_tqdm is None:
            self.use_tqdm = TQDM_AVAILABLE and sys.stdout.isatty()
        else:
            self.use_tqdm = use_tqdm and TQDM_AVAILABLE
            
        self.tqdm_bar = None
        if self.use_tqdm:
            self.tqdm_bar = tqdm(total=total, desc=description, unit="items")
    
    def update(self, increment: int = 1) -> Optional[float]:
        with self._lock:
            self.current += increment
            if self.total <= 0:
                return None
                
            pct = (self.current / self.total) * 100
            current_time = time.time()
            
            if self.tqdm_bar and increment > 0:
                self.tqdm_bar.update(increment)
            
            pct_change_ok = abs(pct - self.last_reported_pct) >= self.min_change
            time_change_ok = (current_time - self.last_reported_time) >= self.min_seconds
            
            if pct_change_ok and (self.min_seconds == 0 or time_change_ok):
                self.last_reported_pct = pct
                self.last_reported_time = current_time
                
                if not self.tqdm_bar:
                    print(f"{self.description}: {pct:.0f}%")
                
                return pct
            return None
    
    def close(self):
        if self.tqdm_bar:
            self.tqdm_bar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    async def __aenter__(self):
        return self.__enter__()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


def test_sync_progress():
    """Test synchronous progress tracking"""
    print("\n=== Testing Synchronous Progress ===")
    
    # Test with automatic tqdm detection
    print("\n1. Automatic tqdm detection:")
    with ProgressTracker(total=50, description="Processing files") as tracker:
        for i in range(50):
            time.sleep(0.02)  # Simulate work
            tracker.update(1)
    
    # Test with forced logging (no tqdm)
    print("\n2. Forced logging mode:")
    with ProgressTracker(total=20, description="Analyzing", use_tqdm=False, min_change=25) as tracker:
        for i in range(20):
            time.sleep(0.05)
            tracker.update(1)
    
    # Test with time-based throttling
    print("\n3. Time-based throttling (updates every 0.5 seconds):")
    with ProgressTracker(total=100, description="Time throttled", use_tqdm=False, min_seconds=0.5) as tracker:
        for i in range(100):
            time.sleep(0.01)
            tracker.update(1)


async def test_async_progress():
    """Test asynchronous progress tracking"""
    print("\n=== Testing Asynchronous Progress ===")
    
    print("\n1. Async context manager:")
    async with ProgressTracker(total=30, description="Async processing") as tracker:
        for i in range(30):
            await asyncio.sleep(0.03)  # Simulate async work
            tracker.update(1)  # Note: no await needed!
    
    print("\n2. Mixed sync/async operations:")
    tracker = ProgressTracker(total=40, description="Mixed mode", use_tqdm=True)
    
    # Some sync updates
    for i in range(20):
        time.sleep(0.02)
        tracker.update(1)
    
    # Some async updates
    for i in range(20):
        await asyncio.sleep(0.02)
        tracker.update(1)  # Same method works!
    
    tracker.close()


def test_concurrent_updates():
    """Test thread-safe concurrent updates"""
    print("\n=== Testing Concurrent Updates ===")
    
    import concurrent.futures
    
    tracker = ProgressTracker(total=100, description="Concurrent", use_tqdm=True)
    
    def worker(n):
        for i in range(n):
            time.sleep(0.01)
            tracker.update(1)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, 25) for _ in range(4)]
        concurrent.futures.wait(futures)
    
    tracker.close()
    print("Concurrent updates completed successfully!")


def main():
    """Run all tests"""
    print(f"tqdm available: {TQDM_AVAILABLE}")
    print(f"Running in TTY: {sys.stdout.isatty()}")
    
    # Run sync tests
    test_sync_progress()
    
    # Run async tests
    print("\n" + "="*50)
    asyncio.run(test_async_progress())
    
    # Run concurrent tests
    print("\n" + "="*50)
    test_concurrent_updates()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
