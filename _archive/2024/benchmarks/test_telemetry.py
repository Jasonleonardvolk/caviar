"""
Test script for HolographicTelemetry system
Simulates client events and verifies server reception
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path

# Configuration
TELEMETRY_URL = "http://localhost:8003/api/telemetry"
STATS_URL = "http://localhost:8003/api/telemetry/stats"
SESSIONS_URL = "http://localhost:8003/api/telemetry/sessions"

def generate_test_events():
    """Generate sample telemetry events"""
    session_id = f"test-{int(time.time())}"
    
    events = [
        # Session start
        {
            "event": "session_start",
            "data": {
                "url": "http://localhost:3000/test",
                "referrer": ""
            },
            "ts": int(time.time() * 1000),
            "sessionId": session_id
        },
        
        # Device profile
        {
            "event": "device_profile",
            "data": {
                "tier": "high",
                "gpu": "NVIDIA RTX 4090",
                "memory": 32,
                "cores": 16,
                "screen": {"width": 2560, "height": 1440, "dpr": 2},
                "webgpu": True,
                "browser": "Chrome",
                "os": "Windows"
            },
            "ts": int(time.time() * 1000),
            "sessionId": session_id
        },
        
        # Performance metrics
        {
            "event": "performance",
            "data": {
                "fps": 58,
                "frameTime": 17.2,
                "jitter": 1.3,
                "droppedFrames": 2,
                "propagationTime": 5.4,
                "encodingTime": 3.2,
                "renderTime": 8.6
            },
            "ts": int(time.time() * 1000),
            "sessionId": session_id
        },
        
        # Holographic metrics
        {
            "event": "holographic",
            "data": {
                "mode": "standard",
                "renderMode": "hologram",
                "viewCount": 45,
                "fieldSize": 1024,
                "propagationDistance": 0.5,
                "qualityScale": 1.0,
                "parallaxEnabled": True,
                "sensorType": "looking_glass"
            },
            "ts": int(time.time() * 1000),
            "sessionId": session_id
        },
        
        # User interaction
        {
            "event": "interaction",
            "data": {
                "action": "quality_change",
                "from": "medium",
                "to": "high",
                "totalInteractions": 5
            },
            "ts": int(time.time() * 1000),
            "sessionId": session_id
        },
        
        # Head tracking
        {
            "event": "head_tracking",
            "data": {
                "avgMovement": 0.234,
                "maxMovement": 0.567,
                "samples": 100
            },
            "ts": int(time.time() * 1000),
            "sessionId": session_id
        },
        
        # Engagement metrics
        {
            "event": "engagement",
            "data": {
                "sessionDuration": 120,
                "interactionCount": 15,
                "qualityChanges": 3,
                "modeChanges": 1,
                "headMovementIntensity": 0.456
            },
            "ts": int(time.time() * 1000),
            "sessionId": session_id
        }
    ]
    
    return events, session_id

async def send_telemetry(events, session_id):
    """Send telemetry events to server"""
    payload = {
        "events": events,
        "sessionId": session_id
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(TELEMETRY_URL, json=payload) as resp:
                result = await resp.json()
                return resp.status, result
        except Exception as e:
            return None, str(e)

async def check_stats():
    """Check telemetry statistics"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(STATS_URL) as resp:
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}

async def check_sessions():
    """Check active sessions"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(SESSIONS_URL) as resp:
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}

async def main():
    """Run telemetry test"""
    print("=" * 60)
    print("HolographicTelemetry Test Suite")
    print("=" * 60)
    
    # Generate test events
    print("\n1. Generating test events...")
    events, session_id = generate_test_events()
    print(f"   Generated {len(events)} events for session {session_id}")
    
    # Send telemetry
    print("\n2. Sending telemetry to server...")
    status, result = await send_telemetry(events, session_id)
    if status == 200:
        print(f"   ✅ Success: {result}")
    else:
        print(f"   ❌ Failed: {result}")
        if status is None:
            print("   Make sure telemetry server is running:")
            print("   python api\\telemetry_server.py")
            return
    
    # Wait for processing
    await asyncio.sleep(1)
    
    # Check stats
    print("\n3. Checking telemetry stats...")
    stats = await check_stats()
    if "error" not in stats:
        print(f"   Active sessions: {stats.get('active_sessions', 0)}")
        print(f"   Total sessions: {stats.get('total_sessions', 0)}")
        
        if "log_files" in stats:
            print("   Log files:")
            for name, info in stats["log_files"].items():
                print(f"     - {name}: {info['events']} events, {info['size_kb']:.2f} KB")
        
        if "performance_summary" in stats:
            perf = stats["performance_summary"]
            print("   Performance summary:")
            print(f"     - Avg FPS: {perf.get('avg_fps', 0)}")
            print(f"     - Avg Jitter: {perf.get('avg_jitter', 0)}")
            print(f"     - Dropped frames: {perf.get('total_dropped_frames', 0)}")
        
        if "device_tiers" in stats:
            print("   Device tiers:")
            for tier, count in stats["device_tiers"].items():
                print(f"     - {tier}: {count}")
    else:
        print(f"   ❌ Error: {stats['error']}")
    
    # Check sessions
    print("\n4. Checking active sessions...")
    sessions = await check_sessions()
    if "error" not in sessions:
        print(f"   Found {sessions.get('active_count', 0)} active sessions")
        for session in sessions.get("sessions", []):
            print(f"   - {session['session_id']}: {session['device_tier']} ({session['duration_seconds']:.1f}s)")
    else:
        print(f"   ❌ Error: {sessions['error']}")
    
    # Verify log files
    print("\n5. Verifying log files...")
    log_dir = Path("logs/telemetry")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.jsonl"))
        print(f"   Found {len(log_files)} log files:")
        for log_file in log_files:
            size_kb = log_file.stat().st_size / 1024
            print(f"   - {log_file.name}: {size_kb:.2f} KB")
            
            # Check if our session is in the logs
            if log_file.name == "events.jsonl":
                with log_file.open('r') as f:
                    for line in f:
                        try:
                            event = json.loads(line)
                            if event.get("session_id") == session_id:
                                print(f"   ✅ Found our test session in {log_file.name}")
                                break
                        except:
                            pass
    else:
        print("   ❌ Log directory not found")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
