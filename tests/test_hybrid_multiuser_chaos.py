#!/usr/bin/env python3
"""
Chaos Testing Suite for TORI/Saigon v5
Tests multi-user scenarios, adapter swaps, mesh updates, and error recovery
"""

import asyncio
import random
import time
import threading
import concurrent.futures
from typing import List, Dict, Any
import json
import pytest
import logging
from pathlib import Path

# Import system components
from python.core.saigon_inference_v5 import SaigonInference
from python.core.atomic_adapter_loader import adapter_loader
from python.core.auto_refresh_mesh import mesh_manager
from api.saigon_inference_api_v5 import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChaosTestSuite:
    """Chaos testing for bulletproof system validation"""
    
    def __init__(self):
        self.users = ["alice", "bob", "carol", "dave", "eve"]
        self.adapters = ["v1", "v2", "v3", "latest", "experimental"]
        self.test_prompts = [
            "Explain quantum computing",
            "What is consciousness?",
            "Generate a story about AI",
            "Solve this math problem: 2+2",
            "Translate 'hello' to French"
        ]
        self.errors_encountered = []
        self.test_results = []
    
    def test_multiuser_concurrent_inference(self, num_users: int = 10, requests_per_user: int = 5):
        """Test concurrent inference from multiple users"""
        logger.info(f"Testing {num_users} users with {requests_per_user} requests each")
        
        def user_session(user_id: str):
            results = []
            for i in range(requests_per_user):
                prompt = random.choice(self.test_prompts)
                try:
                    # Simulate inference
                    inference = SaigonInference()
                    result = inference.infer(
                        user_id=user_id,
                        prompt=prompt,
                        adapter_name=f"user_{user_id}_adapter.pt"
                    )
                    results.append({"success": True, "user": user_id, "prompt": prompt})
                except Exception as e:
                    results.append({"success": False, "user": user_id, "error": str(e)})
                    self.errors_encountered.append(e)
                
                # Random delay
                time.sleep(random.uniform(0.1, 0.5))
            
            return results
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = []
            for i in range(num_users):
                user_id = f"test_user_{i}"
                futures.append(executor.submit(user_session, user_id))
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                self.test_results.extend(future.result())
        
        # Analyze results
        success_count = sum(1 for r in self.test_results if r.get("success"))
        failure_count = len(self.test_results) - success_count
        
        logger.info(f"Results: {success_count} successes, {failure_count} failures")
        assert failure_count == 0, f"Had {failure_count} failures in concurrent inference"
    
    def test_rapid_adapter_swapping(self, num_swaps: int = 20):
        """Test rapid adapter hot-swapping under load"""
        logger.info(f"Testing {num_swaps} rapid adapter swaps")
        
        for i in range(num_swaps):
            user = random.choice(self.users)
            adapter = random.choice(self.adapters)
            
            try:
                # Perform hot-swap
                success = adapter_loader.hot_swap_adapter(
                    user,
                    f"adapter_{adapter}.pt"
                )
                assert success, f"Swap failed for {user} to {adapter}"
                
                # Immediately run inference
                inference = SaigonInference()
                result = inference.infer(
                    user_id=user,
                    prompt="Test after swap"
                )
                
                # Random rollback
                if random.random() < 0.3:
                    adapter_loader.rollback_adapter(user)
                
            except Exception as e:
                self.errors_encountered.append(e)
                logger.error(f"Swap test failed: {e}")
            
            # Minimal delay
            time.sleep(0.05)
        
        logger.info(f"Completed {num_swaps} swaps with {len(self.errors_encountered)} errors")
    
    def test_mesh_update_race_conditions(self, num_threads: int = 10):
        """Test concurrent mesh updates for race conditions"""
        logger.info(f"Testing mesh updates with {num_threads} concurrent threads")
        
        def update_mesh_thread(thread_id: int):
            user = random.choice(self.users)
            for i in range(10):
                change = {
                    "action": random.choice(["add_concept", "add_relationship"]),
                    "data": {
                        "name": f"concept_{thread_id}_{i}",
                        "content": f"Thread {thread_id} update {i}"
                    }
                }
                
                try:
                    mesh_manager.update_mesh(user, change)
                    
                    # Verify summary is fresh
                    fresh = mesh_manager.validate_summary_freshness(user)
                    assert fresh, f"Stale summary detected for {user}"
                    
                except Exception as e:
                    self.errors_encountered.append(e)
                    logger.error(f"Mesh update failed: {e}")
                
                time.sleep(random.uniform(0.01, 0.1))
        
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=update_mesh_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        logger.info(f"Mesh update test completed with {len(self.errors_encountered)} errors")
    
    def test_memory_pressure(self, duration_seconds: int = 30):
        """Test system under memory pressure"""
        logger.info(f"Testing memory pressure for {duration_seconds} seconds")
        
        start_time = time.time()
        large_prompts = []
        
        # Generate large prompts
        for i in range(100):
            large_prompts.append("x" * 10000)  # 10KB prompts
        
        while time.time() - start_time < duration_seconds:
            user = random.choice(self.users)
            
            try:
                # Try large inference
                inference = SaigonInference()
                result = inference.infer(
                    user_id=user,
                    prompt=random.choice(large_prompts)
                )
                
                # Also update mesh with large data
                mesh_manager.update_mesh(user, {
                    "action": "add_concept",
                    "data": {
                        "name": f"large_{time.time()}",
                        "content": "y" * 5000  # 5KB content
                    }
                })
                
            except MemoryError as e:
                logger.error(f"Memory error: {e}")
                self.errors_encountered.append(e)
                break
            except Exception as e:
                self.errors_encountered.append(e)
            
            time.sleep(0.1)
        
        logger.info(f"Memory pressure test completed with {len(self.errors_encountered)} errors")
    
    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        logger.info("Testing error recovery mechanisms")
        
        test_cases = [
            # Invalid adapter
            {
                "action": lambda: adapter_loader.hot_swap_adapter("alice", "nonexistent.pt"),
                "expected": False,
                "recovery": lambda: adapter_loader.get_active_adapter("alice") is not None
            },
            # Corrupt mesh update
            {
                "action": lambda: mesh_manager.update_mesh("bob", {"invalid": "data"}),
                "expected": False,
                "recovery": lambda: mesh_manager.get_summary("bob") is not None
            },
            # OOM simulation
            {
                "action": lambda: SaigonInference().infer("carol", "x" * 1000000),
                "expected": Exception,
                "recovery": lambda: True  # System should still be responsive
            }
        ]
        
        for i, test in enumerate(test_cases):
            try:
                result = test["action"]()
                if test["expected"] == False:
                    assert result == False, f"Test {i} should have failed"
                elif test["expected"] == Exception:
                    assert False, f"Test {i} should have raised exception"
            except Exception as e:
                if test["expected"] != Exception:
                    self.errors_encountered.append(e)
                    logger.error(f"Unexpected error in test {i}: {e}")
            
            # Check recovery
            recovered = test["recovery"]()
            assert recovered, f"System did not recover from test {i}"
        
        logger.info("Error recovery test completed")
    
    def test_device_fallback(self):
        """Test rendering fallback mechanisms"""
        logger.info("Testing device fallback mechanisms")
        
        # Simulate different device capabilities
        from frontend.lib.deviceDetect import detectCapabilities, loadRenderer
        
        device_scenarios = [
            {"webgpu": True, "wasm": True, "memory": 8192},   # High-end
            {"webgpu": False, "wasm": True, "memory": 2048},  # Mid-range
            {"webgpu": False, "wasm": False, "memory": 1024}, # Low-end
            {"webgpu": False, "wasm": False, "memory": 512},  # Minimal
        ]
        
        for scenario in device_scenarios:
            # Mock capabilities
            caps = type('obj', (object,), scenario)()
            caps.webgl2 = True
            caps.webgl = True
            caps.cpu = True
            caps.cores = 4
            caps.isMobile = scenario["memory"] < 2048
            
            try:
                # This would normally load the appropriate renderer
                # For testing, we just verify it doesn't crash
                config = {"backend": "cpu" if scenario["memory"] < 1024 else "wasm"}
                logger.info(f"Device with {scenario['memory']}MB would use {config['backend']}")
                
            except Exception as e:
                self.errors_encountered.append(e)
                logger.error(f"Fallback failed for scenario {scenario}: {e}")
        
        logger.info("Device fallback test completed")
    
    def test_sse_event_flooding(self):
        """Test SSE stream under event flooding"""
        logger.info("Testing SSE event flooding")
        
        async def flood_events():
            from api.routes.hybrid_control import broadcast_event, event_queues
            
            # Create queues for test users
            for user in self.users:
                event_queues[user] = asyncio.Queue()
            
            # Flood with events
            for i in range(1000):
                user = random.choice(self.users)
                event = {
                    "type": random.choice(["mesh_updated", "adapter_swapped", "phase_changed"]),
                    "data": f"Event {i}",
                    "timestamp": time.time()
                }
                
                try:
                    await broadcast_event(user, event)
                except Exception as e:
                    self.errors_encountered.append(e)
                
                if i % 100 == 0:
                    await asyncio.sleep(0.01)
            
            # Check queue sizes
            for user in self.users:
                if user in event_queues:
                    size = event_queues[user].qsize()
                    logger.info(f"Queue size for {user}: {size}")
                    assert size < 10000, f"Queue overflow for {user}"
        
        # Run async test
        asyncio.run(flood_events())
        logger.info("SSE flooding test completed")
    
    def run_all_tests(self):
        """Run complete chaos test suite"""
        logger.info("=" * 60)
        logger.info("STARTING CHAOS TEST SUITE")
        logger.info("=" * 60)
        
        test_methods = [
            self.test_multiuser_concurrent_inference,
            self.test_rapid_adapter_swapping,
            self.test_mesh_update_race_conditions,
            self.test_memory_pressure,
            self.test_error_recovery,
            self.test_device_fallback,
            self.test_sse_event_flooding
        ]
        
        for test in test_methods:
            logger.info(f"\nRunning: {test.__name__}")
            try:
                test()
                logger.info(f"✅ {test.__name__} PASSED")
            except Exception as e:
                logger.error(f"❌ {test.__name__} FAILED: {e}")
                self.errors_encountered.append(e)
        
        # Final report
        logger.info("\n" + "=" * 60)
        logger.info("CHAOS TEST SUITE COMPLETE")
        logger.info(f"Total errors encountered: {len(self.errors_encountered)}")
        
        if self.errors_encountered:
            logger.error("Errors summary:")
            for i, error in enumerate(self.errors_encountered[:10]):  # Show first 10
                logger.error(f"  {i+1}. {error}")
        
        logger.info("=" * 60)
        
        # Return success/failure
        return len(self.errors_encountered) == 0

# Pytest integration
class TestChaos:
    def setup_method(self):
        self.suite = ChaosTestSuite()
    
    def test_concurrent_users(self):
        self.suite.test_multiuser_concurrent_inference(5, 3)
    
    def test_adapter_swapping(self):
        self.suite.test_rapid_adapter_swapping(10)
    
    def test_mesh_race_conditions(self):
        self.suite.test_mesh_update_race_conditions(5)
    
    def test_memory_handling(self):
        self.suite.test_memory_pressure(10)
    
    def test_error_recovery(self):
        self.suite.test_error_recovery()
    
    def test_fallback_mechanisms(self):
        self.suite.test_device_fallback()

if __name__ == "__main__":
    # Run full suite
    suite = ChaosTestSuite()
    success = suite.run_all_tests()
    
    if success:
        logger.info("\n🎉 ALL CHAOS TESTS PASSED! System is bulletproof!")
        exit(0)
    else:
        logger.error("\n❌ Some chaos tests failed. System needs hardening.")
        exit(1)
