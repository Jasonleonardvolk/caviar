"""Phase 6: ScholarSphere diff endpoint for concept mesh changes."""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import aiofiles
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/concept-mesh", tags=["concept-mesh"])

# Queue directory for diffs
PSI_ARCHIVE_DIR = Path("data/psi_archive")
PSI_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

class DiffRequest(BaseModel):
    """Request to record a concept mesh diff."""
    record_id: str
    concept_id: str
    operation: str = "update"  # create, update, delete
    metadata: Optional[Dict[str, Any]] = None

class DiffResponse(BaseModel):
    """Response after queuing a diff."""
    status: str
    diff_id: str
    queued_at: str

@router.post("/record_diff", response_model=DiffResponse)
async def record_concept_diff(request: DiffRequest):
    """Record a concept mesh diff to the psi_archive queue."""
    try:
        # Generate diff ID
        timestamp = datetime.utcnow()
        diff_id = f"{request.concept_id}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create diff record
        diff_record = {
            "diff_id": diff_id,
            "record_id": request.record_id,
            "concept_id": request.concept_id,
            "operation": request.operation,
            "timestamp": timestamp.isoformat(),
            "metadata": request.metadata or {},
        }
        
        # Write to JSONL queue
        queue_file = PSI_ARCHIVE_DIR / f"diff_queue_{timestamp.strftime('%Y%m%d')}.jsonl"
        async with aiofiles.open(queue_file, "a", encoding="utf-8") as f:
            await f.write(json.dumps(diff_record) + "\n")
        
        logger.info(f"📤 Diff queued: {diff_id} for concept {request.concept_id}")
        
        return DiffResponse(
            status="queued",
            diff_id=diff_id,
            queued_at=timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to queue diff: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue diff: {str(e)}")

@router.get("/diff_status/{diff_id}")
async def get_diff_status(diff_id: str):
    """Check the status of a queued diff."""
    # TODO: Implement actual status checking from processing pipeline
    return {
        "diff_id": diff_id,
        "status": "processing",
        "message": "Diff is being processed"
    }
