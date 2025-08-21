# D:\Dev\kha\api\routes\memory_vault_routes.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Telemetry & health
try:
    from api.state.memory_trace_store import TRACE  # ring buffer
    from api.state.health_rules import evaluate     # status evaluator
except Exception as e:
    raise RuntimeError("Missing api.state.{memory_trace_store|health_rules}") from e

# Import FSM from core
from python.core.fractal_soliton_memory import FractalSolitonMemory
from fastapi import Request

def _get_fsm(request: Request) -> FractalSolitonMemory:
    fsm = getattr(request.app.state, "fsm", None)
    if fsm is None:
        raise HTTPException(status_code=503, detail="FSM not initialized")
    return fsm

router = APIRouter()

class MemoryState(BaseModel):
    user_id: str
    coherence: Optional[float] = None
    energy: Optional[float] = None
    laplacian_version: str = "unset"
    size: int = 0
    status: str = "unknown"
    status_reasons: List[str] = []
    reports: Optional[Dict[str, str]] = None

@router.get("/api/memory/state/{user_id}", response_model=MemoryState)
async def get_memory_state(user_id: str, request: Request) -> MemoryState:
    """
    Returns physics invariants (energy, coherence) and Laplacian version for admin ops.
    Assumes a single FSM instance (multi-tenant selection is your higher-level concern).
    """
    fsm = _get_fsm(request)
    # If you need per-user psi, select/swap here; otherwise snapshot current lattice:
    snap = fsm.snapshot_invariants(lambda_=request.app.state.get("fsm_lambda", 0.0))
    
    # Define report paths
    reports = {
        "resonance": "reports/resonance_bench/topk_pr.json",
        "coherence": "reports/coherence_bench/coherence_vs_edit.json",
        "energy":    "reports/energy_bench/energy_traces.json",
    }
    
    # Record to trace & evaluate if available
    try:
        E = snap["energy"]
        C = snap["coherence"]
        Lver_int = 0  # Convert string version to int for trace compatibility
        if E is not None and C is not None:
            TRACE.record(user_id, E, C, Lver_int)
        stats = TRACE.stats(user_id, window=50)
        health = evaluate(stats)
    except:
        health = {"status": "unknown", "reasons": []}
    
    return MemoryState(
        user_id=user_id,
        coherence=snap["coherence"],
        energy=snap["energy"],
        laplacian_version=snap["laplacian_version"],
        size=snap["size"],
        status=health["status"],
        status_reasons=health["reasons"],
        reports=reports,
    )

class TraceRow(BaseModel):
    t: float
    E: float
    C: float
    Lver: int

class TracePayload(BaseModel):
    rows: List[TraceRow]

@router.get("/api/memory/trace/{user_id}", response_model=TracePayload)
async def get_memory_trace(user_id: str, n: int = Query(200, ge=1, le=2048)) -> TracePayload:
    rows = TRACE.last_n(user_id, n=n)
    return TracePayload(rows=[TraceRow(t=t, E=E, C=C, Lver=L) for (t, E, C, L) in rows])
