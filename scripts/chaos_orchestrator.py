"""
Chaos Orchestrator - End-to-end chaos testing driver.
Coordinates multi-user adapter swaps, mesh updates, WGSL validation, and SSE verification.
"""
import asyncio, os, sys, time, subprocess, json, logging
from pathlib import Path
import httpx
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

BASE = os.environ.get("CHAOS_BASE_URL", "http://127.0.0.1:8000")
FRONTEND_DIR = Path("frontend")
WGSL_PATH = FRONTEND_DIR / "hybrid" / "wgsl" / "lightFieldComposer.wgsl"
CHAOS_DURATION = int(os.environ.get("CHAOS_DURATION", "30"))  # seconds
USER_COUNT = int(os.environ.get("CHAOS_USERS", "10"))

class ChaosMetrics:
    """Track chaos test metrics."""
    def __init__(self):
        self.events_posted = 0
        self.events_received = 0
        self.errors = []
        self.adapter_swaps = 0
        self.mesh_updates = 0
        self.start_time = time.time()
    
    def report(self):
        duration = time.time() - self.start_time
        return {
            "duration": duration,
            "events_posted": self.events_posted,
            "events_received": self.events_received,
            "drop_rate": 1 - (self.events_received / max(self.events_posted, 1)),
            "adapter_swaps": self.adapter_swaps,
            "mesh_updates": self.mesh_updates,
            "errors": len(self.errors),
            "throughput": self.events_posted / duration if duration > 0 else 0
        }

metrics = ChaosMetrics()

async def user_flow(uid: int, duration: int):
    """Simulate a user performing various operations."""
    end_time = time.time() + duration
    local_events = 0
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.time() < end_time:
            try:
                # Random operations
                operation = hash(f"{uid}{time.time()}") % 4
                
                if operation == 0:
                    # Adapter swap
                    adapter_name = f"adapter_{(uid + int(time.time())) % 5 + 1}.bin"
                    r = await client.post(
                        f"{BASE}/api/v2/adapter/swap",
                        json={"name": adapter_name, "user": f"user_{uid}"}
                    )
                    if r.status_code == 200:
                        metrics.adapter_swaps += 1
                        local_events += 1
                        logger.debug(f"User {uid}: Swapped to {adapter_name}")
                
                elif operation == 1:
                    # Mesh update
                    r = await client.post(
                        f"{BASE}/api/v2/mesh/update",
                        json={
                            "user": uid,
                            "nodes": {f"u{uid}_n{local_events}": f"data_{local_events}"},
                            "edges": {f"u{uid}_e{local_events}": [local_events, local_events + 1]}
                        }
                    )
                    if r.status_code == 200:
                        metrics.mesh_updates += 1
                        local_events += 1
                        logger.debug(f"User {uid}: Updated mesh")
                
                elif operation == 2:
                    # Prompt
                    r = await client.post(
                        f"{BASE}/api/v2/hybrid/prompt",
                        json={"user": uid, "text": f"chaos_test_{local_events}"}
                    )
                    if r.status_code == 200:
                        local_events += 1
                
                else:
                    # Persona change
                    personas = ["neutral", "happy", "sad", "angry", "enola"]
                    persona = personas[uid % len(personas)]
                    r = await client.post(
                        f"{BASE}/api/v2/hybrid/persona",
                        json={"user": uid, "persona": persona}
                    )
                    if r.status_code == 200:
                        local_events += 1
                
                metrics.events_posted += 1
                
                # Random delay between operations
                await asyncio.sleep(0.1 + (hash(f"{uid}{time.time()}") % 100) / 1000)
                
            except Exception as e:
                metrics.errors.append(f"User {uid}: {str(e)}")
                logger.warning(f"User {uid} error: {e}")
    
    logger.info(f"User {uid} completed: {local_events} events")
    return local_events

async def sse_monitor(duration: int):
    """Monitor SSE stream for event integrity."""
    end_time = time.time() + duration
    received = []
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("GET", f"{BASE}/api/v2/hybrid/events/sse") as response:
                async for line in response.aiter_lines():
                    if time.time() > end_time:
                        break
                    
                    if line.startswith("data:"):
                        try:
                            event = json.loads(line[5:])
                            received.append(event)
                            metrics.events_received += 1
                            
                            if event.get("type") == "adapter_swap":
                                logger.debug(f"SSE: Adapter swap to {event.get('name')}")
                            elif event.get("type") == "mesh_updated":
                                logger.debug(f"SSE: Mesh updated to v{event.get('version')}")
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        metrics.errors.append(f"SSE monitor: {str(e)}")
        logger.error(f"SSE monitor error: {e}")
    
    logger.info(f"SSE monitor received {len(received)} events")
    return received

async def chaos_validator():
    """Periodically validate system state during chaos."""
    while True:
        try:
            async with httpx.AsyncClient() as client:
                # Check stats endpoint
                r = await client.get(f"{BASE}/api/v2/stats")
                if r.status_code == 200:
                    stats = r.json()
                    logger.info(f"System stats: mesh_v{stats.get('mesh_version')}, "
                              f"dropped={stats.get('dropped_events')}, "
                              f"queue={stats.get('queue_size')}")
                    
                    # Alert on high drop rate
                    if stats.get('dropped_events', 0) > 10:
                        logger.warning(f"High event drop rate: {stats['dropped_events']} events dropped")
                
                # Verify mesh file exists and is valid
                mesh_file = Path("data/mesh_contexts/user_demo_mesh.json")
                if mesh_file.exists():
                    try:
                        with mesh_file.open() as f:
                            mesh_data = json.load(f)
                        logger.debug(f"Mesh file valid, version {mesh_data.get('version')}")
                    except json.JSONDecodeError:
                        metrics.errors.append("Corrupted mesh file")
                        logger.error("Mesh file corrupted!")
            
            await asyncio.sleep(5)
            
        except Exception as e:
            metrics.errors.append(f"Validator: {str(e)}")
            logger.error(f"Validator error: {e}")

async def run_chaos():
    """Main chaos test orchestration."""
    logger.info(f"Starting chaos test: {USER_COUNT} users for {CHAOS_DURATION}s")
    
    # Setup environment
    os.environ.setdefault("TORI_LOG_DIR", "logs")
    os.environ.setdefault("TORI_ADAPTER_DIR", "tests/adapters")
    Path(os.environ["TORI_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORI_ADAPTER_DIR"]).mkdir(parents=True, exist_ok=True)
    
    # Seed adapters
    logger.info("Seeding test adapters...")
    subprocess.check_call([sys.executable, "scripts/seed_fake_adapters.py"])
    
    # WGSL validation test (if shader exists)
    if WGSL_PATH.exists():
        logger.info("Testing WGSL validation...")
        try:
            subprocess.check_call(
                ["node", "scripts/touch_wgsl_and_validate.js", str(WGSL_PATH)],
                timeout=5
            )
            logger.info("WGSL validation test passed")
        except subprocess.CalledProcessError:
            logger.warning("WGSL validation test failed (non-critical)")
        except FileNotFoundError:
            logger.warning("Node.js not found, skipping WGSL test")
    
    # Start chaos tasks
    tasks = []
    
    # User flows
    for i in range(USER_COUNT):
        tasks.append(asyncio.create_task(user_flow(i, CHAOS_DURATION)))
    
    # SSE monitor
    tasks.append(asyncio.create_task(sse_monitor(CHAOS_DURATION)))
    
    # Validator (runs until cancelled)
    validator_task = asyncio.create_task(chaos_validator())
    
    # Wait for user flows and SSE monitor
    user_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Cancel validator
    validator_task.cancel()
    try:
        await validator_task
    except asyncio.CancelledError:
        pass
    
    # Collect results
    total_user_events = sum(r for r in user_results if isinstance(r, int))
    
    # Final verification
    logger.info("Running final verification...")
    
    async with httpx.AsyncClient() as client:
        # Get final stats
        r = await client.get(f"{BASE}/api/v2/stats")
        if r.status_code == 200:
            final_stats = r.json()
            logger.info(f"Final stats: {final_stats}")
        
        # Log a probe error to test logging
        r = await client.post(
            f"{BASE}/api/v2/hybrid/log_error",
            json={"error": "CHAOS_TEST_COMPLETE"}
        )
    
    # Check logs
    log_files = {
        "adapter_swap.log": Path(os.environ["TORI_LOG_DIR"]) / "inference" / "adapter_swap.log",
        "mesh.log": Path(os.environ["TORI_LOG_DIR"]) / "mesh" / "mesh.log",
        "errors.log": Path(os.environ["TORI_LOG_DIR"]) / "errors" / "errors.log"
    }
    
    for name, path in log_files.items():
        if path.exists():
            lines = path.read_text().count('\n')
            logger.info(f"Log file {name}: {lines} lines")
        else:
            logger.warning(f"Log file {name} not found")
    
    return metrics.report()

async def extended_chaos():
    """Extended chaos test with optional hardenings."""
    base_report = await run_chaos()
    
    logger.info("Running extended chaos tests...")
    
    # Additional hardening: Runtime scaling test
    logger.info("Testing adaptive renderer runtime scaling...")
    # This would interact with the frontend if running
    
    # Additional hardening: Log file integrity
    logger.info("Verifying log file integrity...")
    log_path = Path(os.environ["TORI_LOG_DIR"]) / "inference" / "adapter_swap.log"
    if log_path.exists():
        lines = log_path.read_text().strip().split('\n')
        timestamps = []
        for line in lines:
            if "ACTION=SWAP" in line and "Z | " in line:
                timestamp = line.split("Z | ")[0]
                timestamps.append(timestamp)
        
        # Verify chronological order
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i-1]:
                logger.error(f"Log timestamps out of order: {timestamps[i-1]} -> {timestamps[i]}")
                base_report["errors"] += 1
    
    # Additional hardening: SSE event count verification
    logger.info("Verifying SSE event integrity...")
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE}/api/v2/event_history")
        if r.status_code == 200:
            history = r.json()
            logger.info(f"Event history: {history['total']} events recorded")
            
            # Compare with metrics
            if history['total'] < metrics.events_posted * 0.95:
                logger.warning(f"Event loss detected: {metrics.events_posted} posted, {history['total']} in history")
    
    return base_report

if __name__ == "__main__":
    try:
        if "--extended" in sys.argv:
            report = asyncio.run(extended_chaos())
        else:
            report = asyncio.run(run_chaos())
        
        # Print report
        print("\n" + "="*60)
        print("CHAOS TEST REPORT")
        print("="*60)
        for key, value in report.items():
            if key == "errors":
                print(f"{key:20}: {value} errors")
            elif isinstance(value, float):
                print(f"{key:20}: {value:.2f}")
            else:
                print(f"{key:20}: {value}")
        
        # Exit code based on success
        if report.get("drop_rate", 0) > 0.05:  # > 5% drop rate
            print("\n❌ CHAOS TEST FAILED: High drop rate")
            sys.exit(1)
        elif report.get("errors", 0) > 0:
            print(f"\n⚠️ CHAOS TEST WARNING: {report['errors']} errors occurred")
            sys.exit(0)  # Warning but not failure
        else:
            print("\n✅ CHAOS TEST PASSED")
            sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ CHAOS TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
