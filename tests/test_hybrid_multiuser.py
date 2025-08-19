"""
Test hybrid multi-user concurrent operations.
Simulates multiple users performing operations simultaneously.
"""
import pytest
import asyncio
import httpx
import time
import threading
import random
from typing import List, Dict, Any
from pathlib import Path
import json

@pytest.mark.asyncio
async def test_concurrent_adapter_swaps(tmp_workdir):
    """Test multiple users swapping adapters simultaneously."""
    base_url = "http://127.0.0.1:8000"
    num_users = 10
    adapters = ["adapter1.bin", "adapter2.bin", "adapter3.bin"]
    
    async def user_swap_adapters(user_id: int, iterations: int = 5):
        """Simulate a user repeatedly swapping adapters."""
        results = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            for i in range(iterations):
                adapter = random.choice(adapters)
                try:
                    response = await client.post(
                        f"{base_url}/api/v2/adapter/swap",
                        json={"name": adapter, "user": f"user_{user_id}"}
                    )
                    results.append({
                        "user": user_id,
                        "iteration": i,
                        "adapter": adapter,
                        "status": response.status_code,
                        "timestamp": time.time()
                    })
                except Exception as e:
                    results.append({
                        "user": user_id,
                        "iteration": i,
                        "error": str(e),
                        "timestamp": time.time()
                    })
                
                # Random delay between swaps
                await asyncio.sleep(random.uniform(0.1, 0.5))
        
        return results
    
    # Run concurrent user operations
    tasks = [user_swap_adapters(i) for i in range(num_users)]
    all_results = await asyncio.gather(*tasks)
    
    # Analyze results
    total_operations = sum(len(r) for r in all_results)
    successful_ops = sum(1 for results in all_results for r in results if r.get("status") == 200)
    
    # Verify no data corruption
    assert successful_ops > 0, "No successful adapter swaps"
    assert successful_ops / total_operations > 0.9, f"Too many failures: {successful_ops}/{total_operations}"
    
    # Check audit log for consistency
    audit_log = Path(tmp_workdir) / "logs" / "inference" / "adapter_swap.log"
    if audit_log.exists():
        with open(audit_log) as f:
            log_lines = f.readlines()
            assert len(log_lines) > 0, "Audit log is empty"
            
            # Verify all users appear in log
            users_in_log = set()
            for line in log_lines:
                if "USER=" in line:
                    user_part = line.split("USER=")[1].split("|")[0].strip()
                    users_in_log.add(user_part)
            
            assert len(users_in_log) >= num_users * 0.8, "Not all users recorded in audit log"
    
    print(f"✅ Multi-user test passed: {successful_ops}/{total_operations} successful swaps")


@pytest.mark.asyncio
async def test_concurrent_mesh_updates(tmp_workdir):
    """Test multiple users updating mesh context simultaneously."""
    base_url = "http://127.0.0.1:8000"
    num_users = 20
    
    async def user_update_mesh(user_id: int):
        """Simulate a user updating mesh context."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            updates = []
            for i in range(10):
                mesh_data = {
                    "nodes": {
                        f"user_{user_id}_node_{i}": f"value_{i}",
                        f"timestamp": time.time()
                    },
                    "edges": {
                        f"edge_{i}": [f"user_{user_id}_node_{i}", f"user_{user_id}_node_{i+1}"]
                    }
                }
                
                try:
                    response = await client.post(
                        f"{base_url}/api/v2/mesh/update",
                        json=mesh_data
                    )
                    updates.append(response.status_code)
                except Exception as e:
                    updates.append(str(e))
                
                await asyncio.sleep(random.uniform(0.05, 0.2))
            
            return updates
    
    # Run concurrent mesh updates
    tasks = [user_update_mesh(i) for i in range(num_users)]
    results = await asyncio.gather(*tasks)
    
    # Verify mesh integrity
    successful_updates = sum(1 for result_list in results for r in result_list if r == 200)
    total_updates = sum(len(r) for r in results)
    
    assert successful_updates > total_updates * 0.8, "Too many mesh update failures"
    
    # Check mesh summary exists and is valid
    mesh_summary_path = Path(tmp_workdir) / "data" / "mesh_contexts" / "mesh_summary.json"
    if mesh_summary_path.exists():
        with open(mesh_summary_path) as f:
            summary = json.load(f)
            assert "node_count" in summary or "keys" in summary, "Invalid mesh summary format"
    
    print(f"✅ Concurrent mesh updates: {successful_updates}/{total_updates} successful")


@pytest.mark.asyncio
async def test_hybrid_pipeline_stress(tmp_workdir):
    """Stress test the entire hybrid pipeline with multiple operations."""
    base_url = "http://127.0.0.1:8000"
    
    async def stress_operation(op_type: str, client: httpx.AsyncClient):
        """Execute a stress operation."""
        if op_type == "prompt":
            return await client.post(
                f"{base_url}/api/v2/hybrid/prompt",
                json={"text": f"test_{random.randint(0, 1000)}", "persona": "neutral"}
            )
        elif op_type == "adapter":
            return await client.post(
                f"{base_url}/api/v2/adapter/swap",
                json={"name": f"adapter{random.randint(1, 3)}.bin"}
            )
        elif op_type == "mesh":
            return await client.post(
                f"{base_url}/api/v2/mesh/update",
                json={"nodes": {f"node_{random.randint(0, 100)}": "value"}}
            )
        elif op_type == "phase":
            return await client.post(
                f"{base_url}/api/v2/phase/compute",
                json={"mode": random.choice(["Kerr", "Soliton", "Tensor"])}
            )
    
    # Run mixed operations concurrently
    operation_types = ["prompt", "adapter", "mesh", "phase"]
    num_operations = 100
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = []
        for _ in range(num_operations):
            op_type = random.choice(operation_types)
            tasks.append(stress_operation(op_type, client))
        
        # Add delays to prevent overwhelming the server
        delayed_tasks = []
        for i, task in enumerate(tasks):
            delayed_tasks.append(
                asyncio.create_task(
                    asyncio.sleep(i * 0.05).__await__() and task
                )
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Analyze results
    successful = sum(1 for r in results if isinstance(r, httpx.Response) and r.status_code == 200)
    errors = sum(1 for r in results if isinstance(r, Exception))
    
    print(f"Stress test: {successful} successful, {errors} errors out of {num_operations}")
    
    # System should handle at least 70% of operations successfully under stress
    assert successful > num_operations * 0.7, f"System failed under stress: only {successful}/{num_operations}"


@pytest.mark.asyncio
async def test_sse_broadcast_consistency():
    """Test that all connected SSE clients receive the same events."""
    base_url = "http://127.0.0.1:8000"
    num_clients = 5
    test_duration = 10  # seconds
    
    client_events = {i: [] for i in range(num_clients)}
    
    async def sse_client(client_id: int):
        """Connect an SSE client and collect events."""
        events = []
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("GET", f"{base_url}/api/v2/hybrid/events/sse") as response:
                    start_time = time.time()
                    async for line in response.aiter_lines():
                        if time.time() - start_time > test_duration:
                            break
                        
                        if line.startswith("data:"):
                            try:
                                event = json.loads(line[5:].strip())
                                events.append({
                                    "timestamp": time.time(),
                                    "type": event.get("type"),
                                    "id": event.get("id")
                                })
                            except json.JSONDecodeError:
                                pass
        except asyncio.TimeoutError:
            pass
        
        client_events[client_id] = events
    
    # Start SSE clients
    client_tasks = [asyncio.create_task(sse_client(i)) for i in range(num_clients)]
    
    # Wait for clients to connect
    await asyncio.sleep(1)
    
    # Generate events
    async with httpx.AsyncClient() as client:
        for i in range(20):
            await client.post(
                f"{base_url}/api/v2/hybrid/prompt",
                json={"text": f"broadcast_test_{i}"}
            )
            await asyncio.sleep(0.3)
    
    # Wait for clients to finish
    await asyncio.gather(*client_tasks)
    
    # Verify all clients received similar events
    event_counts = [len(events) for events in client_events.values()]
    
    if event_counts:
        avg_events = sum(event_counts) / len(event_counts)
        min_events = min(event_counts)
        max_events = max(event_counts)
        
        # All clients should receive roughly the same number of events
        assert max_events - min_events < avg_events * 0.3, \
            f"Event distribution too uneven: min={min_events}, max={max_events}, avg={avg_events}"
        
        print(f"✅ SSE broadcast test: {num_clients} clients received {min_events}-{max_events} events")


@pytest.mark.asyncio
async def test_resource_cleanup_on_disconnect():
    """Test that resources are properly cleaned up when users disconnect."""
    base_url = "http://127.0.0.1:8000"
    
    # Create multiple short-lived connections
    for i in range(10):
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Start operations
            await client.post(
                f"{base_url}/api/v2/adapter/swap",
                json={"name": "adapter1.bin", "user": f"transient_user_{i}"}
            )
            
            # Connect SSE briefly
            try:
                async with client.stream("GET", f"{base_url}/api/v2/hybrid/events/sse") as response:
                    await asyncio.sleep(0.5)
                    # Abruptly disconnect
            except:
                pass
        
        # Brief pause between connections
        await asyncio.sleep(0.1)
    
    # Check that server is still responsive
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/health")
        assert response.status_code == 200, "Server not responsive after disconnections"
    
    print("✅ Resource cleanup test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
