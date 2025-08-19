"""
Minimal FastAPI backend for chaos testing.
Provides just enough endpoints to test adapter swaps, mesh updates, SSE, and logging.
"""
import os, time, json, asyncio, hashlib
from pathlib import Path
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any, Optional
from datetime import datetime
import logging

app = FastAPI()
STATE: Dict[str, Any] = {
    "mesh_version": 0,
    "events": asyncio.Queue(),
    "active_adapter": None,
    "mesh": {"nodes": {}, "edges": {}, "last_updated": None},
    "event_count": 0,
    "dropped_events": 0
}

# Track event integrity
EVENT_HISTORY = []
MAX_HISTORY = 1000

def log_line(kind: str, msg: str):
    root = Path(os.environ.get("TORI_LOG_DIR", ".")) / kind
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{kind}.log"
    with p.open("a", encoding="utf-8") as f:
        timestamp = datetime.utcnow().isoformat()
        f.write(f"{timestamp}Z | {msg}\n")
        f.flush()  # Ensure immediate write

@app.post("/api/v2/hybrid/persona")
async def set_persona(payload: Dict[str, Any]):
    event = {"type":"persona","data":payload,"t":time.time(), "id": STATE["event_count"]}
    STATE["event_count"] += 1
    EVENT_HISTORY.append(event)
    if len(EVENT_HISTORY) > MAX_HISTORY:
        EVENT_HISTORY.pop(0)
    
    try:
        await asyncio.wait_for(STATE["events"].put(event), timeout=0.1)
    except asyncio.TimeoutError:
        STATE["dropped_events"] += 1
        log_line("errors", f"EVENT_DROPPED type=persona id={event['id']}")
    
    return {"ok": True, "event_id": event["id"]}

@app.post("/api/v2/hybrid/prompt")
async def post_prompt(payload: Dict[str, Any]):
    event = {"type":"prompt","data":payload,"t":time.time(), "id": STATE["event_count"]}
    STATE["event_count"] += 1
    EVENT_HISTORY.append(event)
    if len(EVENT_HISTORY) > MAX_HISTORY:
        EVENT_HISTORY.pop(0)
    
    try:
        await asyncio.wait_for(STATE["events"].put(event), timeout=0.1)
    except asyncio.TimeoutError:
        STATE["dropped_events"] += 1
        log_line("errors", f"EVENT_DROPPED type=prompt id={event['id']}")
    
    return {"ok": True, "event_id": event["id"]}

@app.get("/api/v2/hybrid/events/sse")
async def events_sse():
    """SSE endpoint with event integrity tracking."""
    client_id = os.urandom(8).hex()
    log_line("sse", f"CLIENT_CONNECT id={client_id}")
    
    async def gen():
        sent_count = 0
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(STATE["events"].get(), timeout=30.0)
                    sent_count += 1
                    yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f": keepalive\n\n"
        finally:
            log_line("sse", f"CLIENT_DISCONNECT id={client_id} sent={sent_count}")
    
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/api/v2/hybrid/log_error")
async def log_error(payload: Dict[str, Any]):
    error_msg = payload.get('error', 'UNKNOWN')
    log_line("errors", f"FRONTEND_ERROR {error_msg}")
    return {"ok": True}

@app.post("/api/v2/mesh/update")
async def mesh_update(payload: Dict[str, Any]):
    STATE["mesh"].update(payload or {})
    STATE["mesh"]["last_updated"] = time.time()
    STATE["mesh_version"] += 1
    
    # Event for SSE
    event = {"type":"mesh_updated","version":STATE["mesh_version"], "id": STATE["event_count"]}
    STATE["event_count"] += 1
    EVENT_HISTORY.append(event)
    
    try:
        await asyncio.wait_for(STATE["events"].put(event), timeout=0.1)
    except asyncio.TimeoutError:
        STATE["dropped_events"] += 1
        log_line("errors", f"EVENT_DROPPED type=mesh_updated id={event['id']}")
    
    # Simulate atomic exporter
    out = Path("data/mesh_contexts")
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": STATE["mesh_version"], 
        "keys": list(STATE["mesh"].keys()),
        "node_count": len(STATE["mesh"].get("nodes", {})),
        "edge_count": len(STATE["mesh"].get("edges", {})),
        "last_updated": STATE["mesh"]["last_updated"]
    }
    tmp = out / "user_demo_mesh.json.tmp"
    final = out / "user_demo_mesh.json"
    
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    
    tmp.replace(final)
    log_line("mesh", f"EXPORT user_demo_mesh.json v={STATE['mesh_version']}")
    
    return {"ok": True, "mesh_version": STATE["mesh_version"], "event_id": event["id"]}

@app.post("/api/v2/adapter/swap")
async def adapter_swap(payload: Dict[str, Any]):
    name = payload.get("name")
    prev = STATE["active_adapter"]
    STATE["active_adapter"] = name
    
    # Audit log
    log_line("inference", f"ADAPTER_SWAP to={name} from={prev}")
    
    # Event for SSE
    event = {"type":"adapter_swap","name":name, "prev": prev, "id": STATE["event_count"]}
    STATE["event_count"] += 1
    EVENT_HISTORY.append(event)
    
    try:
        await asyncio.wait_for(STATE["events"].put(event), timeout=0.1)
    except asyncio.TimeoutError:
        STATE["dropped_events"] += 1
        log_line("errors", f"EVENT_DROPPED type=adapter_swap id={event['id']}")
    
    return {"ok": True, "active": name, "previous": prev, "event_id": event["id"]}

@app.get("/api/v2/stats")
async def get_stats():
    """Diagnostic endpoint for test verification."""
    return {
        "event_count": STATE["event_count"],
        "dropped_events": STATE["dropped_events"],
        "mesh_version": STATE["mesh_version"],
        "active_adapter": STATE["active_adapter"],
        "event_history_size": len(EVENT_HISTORY),
        "queue_size": STATE["events"].qsize()
    }

@app.get("/api/v2/event_history")
async def get_event_history():
    """Return recent event history for integrity checking."""
    return {"events": EVENT_HISTORY[-100:], "total": len(EVENT_HISTORY)}
