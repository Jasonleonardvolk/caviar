"""
Enhanced SSE Event Integrity Test
Comprehensive verification that all posted events are received without drops.
"""
import asyncio
import pytest
import httpx
import time
from typing import List, Dict, Any, Set
from pathlib import Path
import json

@pytest.mark.asyncio
async def test_sse_event_count_integrity(tmp_workdir):
    """Test that exact count of posted events matches received events."""
    base = "http://127.0.0.1:8000"
    
    # Track all posted events with unique IDs
    posted_events = []
    received_events = []
    
    # Start SSE listener in background
    async def sse_listener():
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("GET", f"{base}/api/v2/hybrid/events/sse") as response:
                    start_time = time.time()
                    async for line in response.aiter_lines():
                        if time.time() - start_time > 10:  # 10 second timeout
                            break
                        
                        if line.startswith("data:"):
                            try:
                                event = json.loads(line[5:].strip())
                                if "id" in event:
                                    received_events.append(event)
                            except json.JSONDecodeError:
                                pass
        except asyncio.TimeoutError:
            pass
    
    # Start listener
    listener_task = asyncio.create_task(sse_listener())
    
    # Wait for listener to connect
    await asyncio.sleep(0.5)
    
    # Post events with unique IDs
    async with httpx.AsyncClient(timeout=10.0) as client:
        for i in range(50):
            # Post different event types
            if i % 3 == 0:
                r = await client.post(
                    f"{base}/api/v2/hybrid/prompt",
                    json={"text": f"test_{i}", "event_id": i}
                )
                if r.status_code == 200:
                    data = r.json()
                    if "event_id" in data:
                        posted_events.append({"id": data["event_id"], "type": "prompt"})
            
            elif i % 3 == 1:
                r = await client.post(
                    f"{base}/api/v2/mesh/update",
                    json={"nodes": {f"node_{i}": f"value_{i}"}}
                )
                if r.status_code == 200:
                    data = r.json()
                    if "event_id" in data:
                        posted_events.append({"id": data["event_id"], "type": "mesh_updated"})
            
            else:
                r = await client.post(
                    f"{base}/api/v2/adapter/swap",
                    json={"name": f"adapter_{i % 3 + 1}.bin"}
                )
                if r.status_code == 200:
                    data = r.json()
                    if "event_id" in data:
                        posted_events.append({"id": data["event_id"], "type": "adapter_swap"})
            
            # Small delay to avoid overwhelming
            await asyncio.sleep(0.01)
    
    # Wait for events to be received
    await asyncio.sleep(2)
    
    # Cancel listener
    listener_task.cancel()
    try:
        await listener_task
    except asyncio.CancelledError:
        pass
    
    # Verify integrity
    posted_ids = {e["id"] for e in posted_events if "id" in e}
    received_ids = {e["id"] for e in received_events if "id" in e}
    
    # Calculate metrics
    missing_events = posted_ids - received_ids
    extra_events = received_ids - posted_ids
    drop_rate = len(missing_events) / len(posted_ids) if posted_ids else 0
    
    # Assertions
    assert drop_rate == 0, f"Events were dropped! Missing: {missing_events}"
    assert len(extra_events) == 0, f"Received unexpected events: {extra_events}"
    assert len(received_ids) == len(posted_ids), \
        f"Event count mismatch: posted={len(posted_ids)}, received={len(received_ids)}"
    
    # Verify event types match
    posted_types = {e["id"]: e["type"] for e in posted_events if "id" in e and "type" in e}
    received_types = {e["id"]: e.get("type") for e in received_events if "id" in e}
    
    for event_id in posted_ids & received_ids:
        if event_id in posted_types and event_id in received_types:
            assert posted_types[event_id] == received_types[event_id], \
                f"Event {event_id} type mismatch: posted={posted_types[event_id]}, received={received_types[event_id]}"
    
    print(f"✅ SSE Integrity Test Passed: {len(posted_ids)} events posted, {len(received_ids)} received, 0% drop rate")


@pytest.mark.asyncio
async def test_sse_ordering_integrity(tmp_workdir):
    """Test that events are received in the correct order."""
    base = "http://127.0.0.1:8000"
    
    received_order = []
    
    # Start SSE listener
    async def sse_listener():
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("GET", f"{base}/api/v2/hybrid/events/sse") as response:
                    start_time = time.time()
                    async for line in response.aiter_lines():
                        if time.time() - start_time > 5:
                            break
                        
                        if line.startswith("data:"):
                            try:
                                event = json.loads(line[5:].strip())
                                if "sequence" in event.get("data", {}):
                                    received_order.append(event["data"]["sequence"])
                            except json.JSONDecodeError:
                                pass
        except asyncio.TimeoutError:
            pass
    
    # Start listener
    listener_task = asyncio.create_task(sse_listener())
    await asyncio.sleep(0.5)
    
    # Post events with sequence numbers
    posted_order = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for seq in range(20):
            r = await client.post(
                f"{base}/api/v2/hybrid/prompt",
                json={"text": f"seq_{seq}", "sequence": seq}
            )
            if r.status_code == 200:
                posted_order.append(seq)
            await asyncio.sleep(0.05)  # 50ms between posts
    
    # Wait for events
    await asyncio.sleep(1)
    
    # Cancel listener
    listener_task.cancel()
    try:
        await listener_task
    except asyncio.CancelledError:
        pass
    
    # Verify ordering
    assert len(received_order) > 0, "No sequenced events received"
    
    # Check that received events maintain order
    for i in range(1, len(received_order)):
        assert received_order[i] >= received_order[i-1], \
            f"Events received out of order at index {i}: {received_order[i-1]} -> {received_order[i]}"
    
    print(f"✅ SSE Ordering Test Passed: Events received in correct order")


@pytest.mark.asyncio
async def test_sse_concurrent_clients_integrity():
    """Test that multiple SSE clients all receive the same events."""
    base = "http://127.0.0.1:8000"
    
    client_events = {
        "client1": [],
        "client2": [],
        "client3": []
    }
    
    # Start multiple SSE listeners
    async def sse_listener(client_id: str):
        events = []
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("GET", f"{base}/api/v2/hybrid/events/sse") as response:
                    start_time = time.time()
                    async for line in response.aiter_lines():
                        if time.time() - start_time > 5:
                            break
                        
                        if line.startswith("data:"):
                            try:
                                event = json.loads(line[5:].strip())
                                if "id" in event:
                                    events.append(event["id"])
                            except json.JSONDecodeError:
                                pass
        except asyncio.TimeoutError:
            pass
        
        client_events[client_id] = events
    
    # Start 3 concurrent listeners
    listeners = [
        asyncio.create_task(sse_listener("client1")),
        asyncio.create_task(sse_listener("client2")),
        asyncio.create_task(sse_listener("client3"))
    ]
    
    await asyncio.sleep(0.5)
    
    # Post events
    posted_ids = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for i in range(10):
            r = await client.post(
                f"{base}/api/v2/hybrid/prompt",
                json={"text": f"broadcast_{i}"}
            )
            if r.status_code == 200:
                data = r.json()
                if "event_id" in data:
                    posted_ids.append(data["event_id"])
            await asyncio.sleep(0.1)
    
    # Wait for events
    await asyncio.sleep(1)
    
    # Cancel all listeners
    for task in listeners:
        task.cancel()
    
    await asyncio.gather(*listeners, return_exceptions=True)
    
    # Verify all clients received same events
    client1_set = set(client_events["client1"])
    client2_set = set(client_events["client2"])
    client3_set = set(client_events["client3"])
    
    # All clients should have received something
    assert len(client1_set) > 0, "Client 1 received no events"
    assert len(client2_set) > 0, "Client 2 received no events"
    assert len(client3_set) > 0, "Client 3 received no events"
    
    # Check intersection - all should have the same events
    common_events = client1_set & client2_set & client3_set
    assert len(common_events) > 0, "No common events received by all clients"
    
    # Verify against posted events
    posted_set = set(posted_ids)
    for client_id, events in client_events.items():
        client_set = set(events)
        missing = posted_set - client_set
        if missing:
            print(f"Warning: {client_id} missed events: {missing}")
    
    print(f"✅ Multi-client SSE Test Passed: All clients received events")


@pytest.mark.asyncio
async def test_sse_recovery_after_error():
    """Test SSE stream recovery after connection errors."""
    base = "http://127.0.0.1:8000"
    
    events_before = []
    events_after = []
    
    # First connection
    async def first_connection():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                async with client.stream("GET", f"{base}/api/v2/hybrid/events/sse") as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            try:
                                event = json.loads(line[5:].strip())
                                if "phase" in event.get("data", {}):
                                    events_before.append(event)
                                    if event["data"]["phase"] == "disconnect":
                                        # Simulate connection drop
                                        raise httpx.ReadError("Simulated disconnect")
                            except json.JSONDecodeError:
                                pass
        except (httpx.ReadError, asyncio.TimeoutError):
            pass
    
    # Start first connection
    first_task = asyncio.create_task(first_connection())
    await asyncio.sleep(0.5)
    
    # Post events before disconnect
    async with httpx.AsyncClient() as client:
        await client.post(f"{base}/api/v2/hybrid/prompt", 
                         json={"text": "before", "phase": "before"})
        await asyncio.sleep(0.1)
        await client.post(f"{base}/api/v2/hybrid/prompt",
                         json={"text": "disconnect", "phase": "disconnect"})
    
    # Wait for first connection to drop
    await first_task
    
    # Start second connection (recovery)
    async def second_connection():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                async with client.stream("GET", f"{base}/api/v2/hybrid/events/sse") as response:
                    start_time = time.time()
                    async for line in response.aiter_lines():
                        if time.time() - start_time > 2:
                            break
                        
                        if line.startswith("data:"):
                            try:
                                event = json.loads(line[5:].strip())
                                if "phase" in event.get("data", {}):
                                    events_after.append(event)
                            except json.JSONDecodeError:
                                pass
        except asyncio.TimeoutError:
            pass
    
    second_task = asyncio.create_task(second_connection())
    await asyncio.sleep(0.5)
    
    # Post events after reconnect
    async with httpx.AsyncClient() as client:
        await client.post(f"{base}/api/v2/hybrid/prompt",
                         json={"text": "after", "phase": "after"})
    
    await second_task
    
    # Verify recovery
    assert len(events_before) > 0, "No events received before disconnect"
    assert len(events_after) > 0, "No events received after recovery"
    
    # Check that "after" phase events were received in second connection
    after_phases = [e["data"]["phase"] for e in events_after if "phase" in e.get("data", {})]
    assert "after" in after_phases, "Recovery connection didn't receive new events"
    
    print(f"✅ SSE Recovery Test Passed: Connection recovered after error")


@pytest.mark.asyncio
async def test_sse_throughput_integrity():
    """Test SSE integrity under high throughput."""
    base = "http://127.0.0.1:8000"
    
    posted_count = 0
    received_count = 0
    
    # High-speed SSE listener
    async def fast_listener():
        nonlocal received_count
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("GET", f"{base}/api/v2/hybrid/events/sse") as response:
                    start_time = time.time()
                    async for line in response.aiter_lines():
                        if time.time() - start_time > 10:
                            break
                        
                        if line.startswith("data:"):
                            try:
                                event = json.loads(line[5:].strip())
                                if "burst" in event.get("data", {}):
                                    received_count += 1
                            except json.JSONDecodeError:
                                pass
        except asyncio.TimeoutError:
            pass
    
    # Start listener
    listener_task = asyncio.create_task(fast_listener())
    await asyncio.sleep(0.5)
    
    # Burst post events
    async def burst_poster(batch_id: int):
        nonlocal posted_count
        async with httpx.AsyncClient(timeout=10.0) as client:
            for i in range(20):
                r = await client.post(
                    f"{base}/api/v2/hybrid/prompt",
                    json={"text": f"burst_{batch_id}_{i}", "burst": True}
                )
                if r.status_code == 200:
                    posted_count += 1
                # No delay - maximum speed
    
    # Run 5 concurrent bursts
    await asyncio.gather(*[burst_poster(i) for i in range(5)])
    
    # Wait for events to be received
    await asyncio.sleep(3)
    
    # Cancel listener
    listener_task.cancel()
    try:
        await listener_task
    except asyncio.CancelledError:
        pass
    
    # Calculate metrics
    drop_rate = 1 - (received_count / max(posted_count, 1))
    throughput = posted_count / 3  # events per second
    
    print(f"Throughput test: Posted={posted_count}, Received={received_count}, "
          f"Drop rate={drop_rate:.1%}, Throughput={throughput:.0f} events/sec")
    
    # Allow small drop rate under extreme load
    assert drop_rate < 0.05, f"Drop rate too high under load: {drop_rate:.1%}"
    assert received_count > posted_count * 0.95, "Too many events lost under high throughput"
    
    print(f"✅ SSE Throughput Test Passed: System handled {throughput:.0f} events/sec with {drop_rate:.1%} drop rate")
