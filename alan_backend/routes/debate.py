from fastapi import APIRouter, WebSocket
from uuid import uuid4
from .llm_tools import run_debate_loop   # ⬅ write or stub separately

router = APIRouter()
active_debates = {}          # debate_id → asyncio.Task

@router.post("/api/debate")
async def start_debate(payload: dict):
    debate_id = str(uuid4())
    task      = run_debate_loop(payload, debate_id)
    active_debates[debate_id] = task
    return {"id": debate_id}

@router.websocket("/ws/debate/{debate_id}")
async def debate_ws(ws: WebSocket, debate_id: str):
    await ws.accept()
    task = active_debates.get(debate_id)
    if not task:
        await ws.close(code=4000)
        return
    async for update in task:
        await ws.send_json(update)
    await ws.close()
