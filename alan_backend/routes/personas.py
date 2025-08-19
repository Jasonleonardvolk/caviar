from fastapi import APIRouter, HTTPException, WebSocket
from pydantic import BaseModel, Field
from uuid import uuid4
from typing import Dict
from pathlib import Path
import json, asyncio

router = APIRouter()

# ─────────────────────────── Data models ────────────────────────────
class TraitBlock(BaseModel):
    role: str = ""
    tone: str = ""
    big5: Dict[str, float] = Field(default_factory=dict)
    values: Dict[str, float] = Field(default_factory=dict)

class Persona(BaseModel):
    id: str
    displayName: str
    avatar: str = "/assets/personas/default.svg"
    traits: TraitBlock

# ─────────────────────────── In-memory store ─────────────────────────
_personas: Dict[str, Persona] = {}

DATA_PATH = Path(__file__).parents[2] / "data" / "personas.json"
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─────────── helpers ───────────
def _save():
    DATA_PATH.write_text(json.dumps([p.dict() for p in _personas.values()], indent=2))

def _load():
    if DATA_PATH.exists():
        for obj in json.loads(DATA_PATH.read_text()):
            _personas[obj["id"]] = Persona.parse_obj(obj)

# ─────────── bootstrap ─────────
def _init_builtins():
    from datetime import datetime
    _personas["refactor-guru"] = Persona(
        id="refactor-guru",
        displayName="Refactor Guru",
        traits=TraitBlock(role="software-architect", tone="concise",
                          big5={"O":0.7,"C":0.9,"E":0.4,"A":0.6,"N":0.2})
    )
    print(f"[{datetime.now():%H:%M:%S}] Personas bootstrap complete")

_load()                 # load file first
if not _personas:       # then fall back to built-ins
    _init_builtins()
    _save()

# ─────────────────────────── REST endpoints ─────────────────────────
@router.get("/api/personas")
async def list_personas() -> Dict[str, Persona]:
    return list(_personas.values())

@router.post("/api/personas", status_code=201)
async def create_persona(p: Persona):
    if not p.id:
        p.id = str(uuid4())
    _personas[p.id] = p
    _save()
    await _broadcast({"type":"persona-updated","id":p.id,"data":p.dict()})
    return p

@router.patch("/api/personas/{pid}")
async def update_persona(pid: str, patch: dict):
    if pid not in _personas:
        raise HTTPException(404, "Persona not found")
    stored = _personas[pid].dict()
    stored.update(patch)
    _personas[pid] = Persona.parse_obj(stored)
    _save()
    await _broadcast({"type":"persona-updated","id":pid,"data":patch})
    return _personas[pid]

@router.delete("/api/personas/{pid}", status_code=204)
async def delete_persona(pid: str):
    _personas.pop(pid, None)
    _save()
    await _broadcast({"type":"persona-deleted","id":pid})

# ────────────────────────── WebSocket stream ────────────────────────
_active_ws = set()

@router.websocket("/ws/personas")
async def personas_ws(sock: WebSocket):
    await sock.accept()
    _active_ws.add(sock)
    try:
        # send current list once on connect
        await sock.send_json({"type":"snapshot","data":list(_personas.values())})
        while True:
            await sock.receive_text()           # keepalive; ignore content
    except Exception:
        pass
    finally:
        _active_ws.discard(sock)

async def _broadcast(msg):
    for ws in list(_active_ws):
        try:
            await ws.send_json(msg)
        except Exception:
            _active_ws.discard(ws)
