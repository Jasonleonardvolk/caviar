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

# FSM access (flexible: memory_vault.get_fsm(user) OR FractalSolitonMemory.get())
FSM = None
def _get_fsm(user_id: str):
    global FSM
    if FSM is not None:
        return FSM  # allow hot-injection in your app
    try:
        # Preferred: a per-user FSM from your vault or service container
        import python.core.memory_vault as memory_vault  # type: ignore
        if hasattr(memory_vault, "get_fsm"):
            return memory_vault.get_fsm(user_id)
    except Exception:
        pass
    try:
        from python.core.fractal_soliton_memory import FractalSolitonMemory as _FSM
        return _FSM.get() if hasattr(_FSM, "get") else _FSM()
    except Exception:
        return None

router = APIRouter()

class MemoryState(BaseModel):
    user_id: str
    coherence: Optional[float] = None
    energy: Optional[float] = None
    laplacian_version: int = 0
    status: str = "unknown"
    status_reasons: List[str] = []

@router.get("/api/memory/state/{user_id}", response_model=MemoryState)
async def get_memory_state(user_id: str) -> MemoryState:
    """
    Returns live FSM metrics and pushes a point into the telemetry trace.
    """
    fsm = _get_fsm(user_id)
    if fsm is None:
        raise HTTPException(status_code=503, detail="FSM unavailable")

    # Compute invariants
    try:
        C = float(fsm.coherence())
    except Exception:
        C = None
    try:
        E = float(fsm.energy())
    except Exception:
        E = None
    Lver = int(getattr(fsm, "laplacian_version", 0))

    # Record to trace & evaluate
    if E is not None and C is not None:
        TRACE.record(user_id, E, C, Lver)
    stats = TRACE.stats(user_id, window=50)
    health = evaluate(stats)

    return MemoryState(
        user_id=user_id,
        coherence=C,
        energy=E,
        laplacian_version=Lver,
        status=health["status"],
        status_reasons=health["reasons"],
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
