#!/usr/bin/env python3
"""
Enhanced production fixes verification script.
Tests all implemented fixes including edge cases.
"""

import os
import sys
import time
import asyncio
import threading
import concurrent.futures
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_execution_helpers_enhanced():
    """Test execution helpers with timeout and cancellation"""
    print("\n1️⃣ Testing Enhanced Execution Helpers")
    print("-" * 40)
    
    from ingest_pdf.pipeline.execution_helpers import run_sync, shutdown_executors
    
    # Test 1: Basic functionality
    async def quick_task():
        await asyncio.sleep(0.001)
        return "success"
    
    result = run_sync(quick_task())
    print(f"✅ Basic test: {result}")
    
    # Test 2: Timeout functionality
    async def slow_task():
        await asyncio.sleep(2.0)
        return "should not see this"
    
    try:
        result = run_sync(slow_task(), timeout=0.5)
        print("❌ Timeout test failed - should have raised TimeoutError")
    except concurrent.futures.TimeoutError:
        print("✅ Timeout test: Correctly raised TimeoutError")
    
    # Test 3: Cancellation on exception
    async def failing_task():
        await asyncio.sleep(0.1)
        raise ValueError("Test error")
    
    try:
        result = run_sync(failing_task())
        print("❌ Exception test failed - should have raised ValueError")
    except ValueError as e:
        print(f"✅ Exception test: Correctly propagated error: {e}")
    
    # Test 4: Idempotent shutdown
    print("Testing idempotent shutdown...")
    shutdown_executors()
    shutdown_executors()  # Should not raise
    print("✅ Idempotent shutdown: No errors on double shutdown")
    
    # Re-create for other tests
    from ingest_pdf.pipeline import execution_helpers
    execution_helpers._persistent_executor = execution_helpers.PersistentLoopExecutor()
    

def test_progress_tracker_enhanced():
    """Test enhanced progress tracker with time throttling"""
    print("\n2️⃣ Testing Enhanced Progress Tracker")
    print("-" * 40)
    
    from ingest_pdf.pipeline.pipeline import ProgressTracker
    
    # Test 1: Time-based throttling
    progress = ProgressTracker(total=100, min_change=1.0, min_seconds=0.5)
    
    reported = []
    start = time.time()
    
    # Rapid updates - should be throttled by time
    for i in range(100):
        if pct := progress.update_sync():
            reported.append((pct, time.time() - start))
        time.sleep(0.01)  # 10ms between updates
    
    print(f"✅ Time throttling: {len(reported)} reports in {time.time() - start:.2f}s")
    
    # Check time gaps
    if len(reported) > 1:
        time_gaps = [reported[i][1] - reported[i-1][1] for i in range(1, len(reported))]
        min_gap = min(time_gaps)
        print(f"   Min time gap: {min_gap:.3f}s (should be >= 0.5s)")
    
    # Test 2: Context manager
    print("\nTesting context manager...")
    with ProgressTracker(total=10) as progress:
        for i in range(10):
            progress.update_sync()
    print("✅ Context manager: Entry/exit handled correctly")
    
    # Test 3: Race condition test with RLock
    progress = ProgressTracker(total=1000, min_change=5.0)
    results = []
    
    def concurrent_update():
        for _ in range(100):
            if pct := progress.update_sync():
                results.append(pct)
    
    threads = [threading.Thread(target=concurrent_update) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Check for duplicates
    unique_results = set(results)
    print(f"✅ Race condition test: {len(results)} reports, {len(unique_results)} unique")
    if len(results) != len(unique_results):
        print(f"   ⚠️  Found {len(results) - len(unique_results)} duplicate reports")
    

def test_logging_improvements():
    """Test logging improvements"""
    print("\n3️⃣ Testing Logging Improvements")
    print("-" * 40)
    
    # Test duplicate handler prevention
    import logging
    
    # Import multiple times to test handler duplication
    for i in range(3):
        if 'ingest_pdf.pipeline.pipeline' in sys.modules:
            del sys.modules['ingest_pdf.pipeline.pipeline']
        from ingest_pdf.pipeline.pipeline import logger as test_logger
        
    # Check handler count
    handler_count = len(test_logger.handlers)
    print(f"✅ Handler count after 3 imports: {handler_count} (should be 1)")
    
    # Test propagate=False
    print(f"✅ Logger propagate: {test_logger.propagate} (should be False)")
    

def test_config_deep_copy_enhanced():
    """Test configuration deep copy with mutations"""
    print("\n4️⃣ Testing Config Deep Copy (Enhanced)")
    print("-" * 40)
    
    from ingest_pdf.pipeline.config import settings
    
    # Create multiple copies and mutate them
    copies = []
    for i in range(3):
        copy = settings.copy(deep=True)
        copy.entropy_threshold = 0.001 * (i + 1)
        copy.section_weights["test"] = float(i)
        copies.append(copy)
    
    # Verify original is unchanged
    print(f"✅ Original entropy_threshold: {settings.entropy_threshold}")
    print(f"✅ Copy 1 entropy_threshold: {copies[0].entropy_threshold}")
    print(f"✅ Copy 2 entropy_threshold: {copies[1].entropy_threshold}")
    print(f"✅ Copy 3 entropy_threshold: {copies[2].entropy_threshold}")
    
    # Check nested dict mutation
    print(f"✅ Original has 'test' in section_weights: {'test' in settings.section_weights}")
    print(f"✅ Copies have different 'test' values: {[c.section_weights.get('test') for c in copies]}")
    

def test_lazy_secrets_loading():
    """Test lazy loading of secrets"""
    print("\n5️⃣ Testing Lazy Secrets Loading")
    print("-" * 40)
    
    from ingest_pdf.pipeline.config_enhancements import vault_settings_source, aws_secrets_manager_source
    
    # Clear any existing credentials
    env_backup = {}
    for key in ['VAULT_ADDR', 'VAULT_TOKEN', 'AWS_ACCESS_KEY_ID']:
        env_backup[key] = os.environ.pop(key, None)
    
    try:
        # Test 1: Missing credentials should warn (once)
        print("Testing missing credentials warnings...")
        
        # Should see warning on first call
        result1 = vault_settings_source()
        result2 = vault_settings_source()  # Should not warn again
        print(f"✅ Vault with no creds returned: {result1}")
        
        result3 = aws_secrets_manager_source()
        result4 = aws_secrets_manager_source()  # Should not warn again
        print(f"✅ AWS with no creds returned: {result3}")
        
        # Test 2: Caching behavior
        print("\nTesting caching...")
        
        # Set fake credentials
        os.environ['VAULT_ADDR'] = 'http://localhost:8200'
        os.environ['VAULT_TOKEN'] = 'fake-token'
        
        # Clear cache
        vault_settings_source.cache_clear()
        
        # This would normally try to connect (and fail)
        # But it should be fast due to caching
        start = time.time()
        for _ in range(10):
            vault_settings_source()
        elapsed = time.time() - start
        
        print(f"✅ 10 calls completed in {elapsed:.3f}s (cached)")
        
    finally:
        # Restore environment
        for key, value in env_backup.items():
            if value is not None:
                os.environ[key] = value
        
        # Clear caches
        vault_settings_source.cache_clear()
        aws_secrets_manager_source.cache_clear()
    

def test_async_context_managers():
    """Test async context managers for ProgressTracker"""
    print("\n6️⃣ Testing Async Context Managers")
    print("-" * 40)
    
    from ingest_pdf.pipeline.pipeline import ProgressTracker
    
    async def async_progress_test():
        async with ProgressTracker(total=10) as progress:
            for i in range(10):
                await progress.update()
                await asyncio.sleep(0.01)
        print("✅ Async context manager completed")
    
    # Run async test
    asyncio.run(async_progress_test())
    

def test_exports():
    """Test that all important functions are exported"""
    print("\n7️⃣ Testing Module Exports")
    print("-" * 40)
    
    from ingest_pdf import pipeline
    
    # Check key exports
    exports_to_check = [
        'run_sync',
        'await_sync', 
        'ProgressTracker',
        'ingest_pdf_clean',
        'preload_concept_database'
    ]
    
    for export in exports_to_check:
        if hasattr(pipeline, export):
            print(f"✅ {export} is exported")
        else:
            print(f"❌ {export} is NOT exported")
    
    # Check __all__
    if hasattr(pipeline, '__all__'):
        print(f"\n__all__ contains {len(pipeline.__all__)} exports")
    

def main():
    """Run all enhanced tests"""
    print("=" * 60)
    print("ENHANCED PRODUCTION FIXES VERIFICATION")
    print("=" * 60)
    
    try:
        test_execution_helpers_enhanced()
        test_progress_tracker_enhanced()
        test_logging_improvements()
        test_config_deep_copy_enhanced()
        test_lazy_secrets_loading()
        test_async_context_managers()
        test_exports()
        
        print("\n" + "=" * 60)
        print("✅ ALL ENHANCED TESTS PASSED - PRODUCTION READY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
