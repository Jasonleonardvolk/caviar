"""
Test script for graceful shutdown functionality
Run this and press Ctrl+C to test shutdown behavior
"""

import asyncio
import signal
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.graceful_shutdown import GracefulShutdownHandler, AsyncioGracefulShutdown, delayed_keyboard_interrupt

def test_basic_shutdown():
    """Test basic shutdown functionality"""
    print("=== Testing Basic Shutdown ===")
    print("Press Ctrl+C to test graceful shutdown")
    
    handler = GracefulShutdownHandler()
    handler.setup_signal_handlers()
    
    def cleanup():
        print("🧹 Running cleanup callback")
        time.sleep(1)  # Simulate cleanup work
        print("✅ Cleanup completed")
    
    handler.add_cleanup_callback(cleanup)
    
    try:
        while True:
            print(".", end="", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n✅ Graceful shutdown completed!")


def test_delayed_interrupt():
    """Test delayed keyboard interrupt"""
    print("\n=== Testing Delayed Interrupt ===")
    print("Try pressing Ctrl+C during initialization...")
    
    with delayed_keyboard_interrupt():
        print("🚀 Starting critical initialization (5 seconds)...")
        for i in range(5):
            print(f"  Initializing... {i+1}/5")
            time.sleep(1)
        print("✅ Initialization complete!")
    
    print("Now you can interrupt normally")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n✅ Normal interrupt handled")


async def test_asyncio_shutdown():
    """Test asyncio graceful shutdown"""
    print("\n=== Testing Asyncio Shutdown ===")
    print("Press Ctrl+C to test asyncio graceful shutdown")
    
    async_handler = AsyncioGracefulShutdown()
    async_handler.setup_async_signal_handlers()
    
    # Protect the main task
    async_handler.protect_current_task()
    
    # Create some tasks
    async def worker(n):
        try:
            print(f"Worker {n} started")
            await asyncio.sleep(10)
            print(f"Worker {n} completed")
        except asyncio.CancelledError:
            print(f"Worker {n} cancelled gracefully")
            raise
    
    # Start workers
    tasks = [asyncio.create_task(worker(i)) for i in range(3)]
    
    try:
        await asyncio.sleep(3600)  # Run for a long time
    except asyncio.CancelledError:
        print("Main task received cancellation")
        # Wait for workers to finish
        await asyncio.gather(*tasks, return_exceptions=True)
        print("✅ All workers finished")


def main():
    """Run all tests"""
    print("🧪 Graceful Shutdown Test Suite")
    print("=" * 40)
    
    tests = [
        ("Basic Shutdown", test_basic_shutdown),
        ("Delayed Interrupt", test_delayed_interrupt),
        ("Asyncio Shutdown", lambda: asyncio.run(test_asyncio_shutdown()))
    ]
    
    for name, test_func in tests:
        try:
            test_func()
        except KeyboardInterrupt:
            print(f"\n{name} test interrupted")
        except Exception as e:
            print(f"\n❌ {name} test failed: {e}")
        
        print("\n" + "=" * 40)
        
        # Ask if user wants to continue
        if len(tests) > 1:
            try:
                response = input("Continue with next test? (y/n): ")
                if response.lower() != 'y':
                    break
            except KeyboardInterrupt:
                print("\nTest suite interrupted")
                break
    
    print("\n✅ Test suite completed!")


if __name__ == "__main__":
    main()
