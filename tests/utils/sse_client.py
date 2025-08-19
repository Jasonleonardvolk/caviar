"""
SSE client utilities for testing event streaming.
"""
import httpx, asyncio, re, json
from typing import Set, List, Dict, Any

async def read_sse(url: str, expect_types: set, timeout=5.0) -> bool:
    """Read SSE stream and check for expected event types."""
    found = set()
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", url) as r:
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    try:
                        event = json.loads(data)
                        if "type" in event:
                            found.add(event["type"])
                    except json.JSONDecodeError:
                        # Try regex fallback
                        m = re.search(r'"type"\s*:\s*"([^"]+)"', data)
                        if m:
                            found.add(m.group(1))
                    
                    if expect_types.issubset(found):
                        return True
    return False

async def collect_sse_events(url: str, count: int, timeout=10.0) -> List[Dict[str, Any]]:
    """Collect a specific number of SSE events for analysis."""
    events = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", url) as r:
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    try:
                        event = json.loads(data)
                        events.append(event)
                        if len(events) >= count:
                            return events
                    except json.JSONDecodeError:
                        pass
    return events

async def verify_sse_integrity(url: str, posted_events: List[int], timeout=10.0) -> Dict[str, Any]:
    """
    Verify SSE stream contains all posted events without drops.
    Returns integrity report.
    """
    received_ids = set()
    events_received = []
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream("GET", url) as r:
                start_time = asyncio.get_event_loop().time()
                async for line in r.aiter_lines():
                    if asyncio.get_event_loop().time() - start_time > timeout:
                        break
                    
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        try:
                            event = json.loads(data)
                            if "id" in event:
                                received_ids.add(event["id"])
                                events_received.append(event)
                        except json.JSONDecodeError:
                            pass
        except asyncio.TimeoutError:
            pass
    
    posted_set = set(posted_events)
    missing = posted_set - received_ids
    extra = received_ids - posted_set
    
    return {
        "posted_count": len(posted_events),
        "received_count": len(received_ids),
        "missing_events": list(missing),
        "extra_events": list(extra),
        "drop_rate": len(missing) / len(posted_events) if posted_events else 0,
        "integrity": len(missing) == 0 and len(extra) == 0
    }
