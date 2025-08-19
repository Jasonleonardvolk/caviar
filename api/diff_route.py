"""
/api/concept-mesh/record_diff
─────────────────────────────
POST  { "record_id": "abc123" }
Writes a diff JSONL file to data/psi_archive/diffs/YYYY-MM-DD.jsonl
and returns { "status": "queued", "path": "<outfile>" }.

Updated to use ConceptMesh dependency injection pattern.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import uuid
from deepdiff import DeepDiff
import asyncio
import logging

logger = logging.getLogger(__name__)

# Import ConceptMesh and dependency injection
import sys
sys.path.append(str(Path(__file__).parent.parent))
from python.core.concept_mesh import ConceptMesh, ConceptDiff, get_mesh

router = APIRouter(prefix="/api/concept-mesh", tags=["concept-mesh"])
DATA_DIR = Path("data/psi_archive/diffs")
DATA_DIR.mkdir(parents=True, exist_ok=True)

class DiffRequest(BaseModel):
    record_id: str

class ConceptDiffRequest(BaseModel):
    diff_type: str  # "add", "remove", "modify", "relate", "unrelate"
    concepts: List[str]  # Concept IDs involved
    new_value: Optional[Dict[str, Any]] = None
    old_value: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

def snapshot_path(record_id: str) -> Path:
    """Return the snapshot file path for this record."""
    return Path("data/mesh_snapshots") / f"{record_id}.json"

@router.post("/record_diff")
async def record_diff(
    req: DiffRequest,
    mesh: ConceptMesh = Depends(get_mesh)
):
    """
    Record a diff between current mesh state and previous snapshot.
    Uses ConceptMesh singleton for current state.
    """
    # 1. Get current mesh state
    current_mesh = mesh.export_to_json(Path("temp_export.json"))
    if current_mesh:
        with open("temp_export.json", "r") as f:
            current_data = json.load(f)
        Path("temp_export.json").unlink()  # Clean up temp file
    else:
        current_data = {"concepts": [], "relations": []}
    
    # 2. Load previous snapshot (if any)
    snap_path = snapshot_path(req.record_id)
    if snap_path.exists():
        previous_data = json.loads(snap_path.read_text())
    else:
        previous_data = {"concepts": [], "relations": []}
    
    # 3. Compute diff
    diff = DeepDiff(previous_data, current_data, verbose_level=2).to_json()
    if diff == "{}":
        raise HTTPException(status_code=204, detail="no changes")
    
    # 4. Persist diff to archive
    ts = datetime.utcnow().strftime("%Y-%m-%d")
    out_file = DATA_DIR / f"{ts}.jsonl"
    with out_file.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps({
            "id": str(uuid.uuid4()), 
            "record": req.record_id, 
            "diff": json.loads(diff),
            "timestamp": datetime.utcnow().isoformat(),
            "concept_count": len(current_data.get("concepts", [])),
            "relation_count": len(current_data.get("relations", []))
        }) + "\n")
    
    # 5. Save new snapshot
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps(current_data, indent=2))
    
    # 6. Auto-upload to ScholarSphere
    try:
        from api.scholarsphere_upload import upload_concepts_to_scholarsphere
        concepts = current_data.get("concepts", [])
        
        # Upload immediately
        diff_id = await upload_concepts_to_scholarsphere(concepts, source="record_diff")
        logger.info(f"Auto-uploaded diff to ScholarSphere: {diff_id}")
    except Exception as e:
        logger.warning(f"Auto-upload failed: {e}")
    
    return {
        "status": "queued", 
        "path": str(out_file),
        "concept_count": len(current_data.get("concepts", [])),
        "relation_count": len(current_data.get("relations", []))
    }

@router.post("/diff")
async def record_concept_diff(
    request: ConceptDiffRequest,
    mesh: ConceptMesh = Depends(get_mesh)
):
    """
    Record a concept diff directly to the ConceptMesh.
    This is used by the Rust extension and other external sources.
    """
    try:
        diff = ConceptDiff(
            id=f"diff_{datetime.now().timestamp()}",
            diff_type=request.diff_type,
            concepts=request.concepts,
            old_value=request.old_value,
            new_value=request.new_value,
            metadata=request.metadata or {}
        )
        
        # Record the diff (this will also apply it to the mesh)
        mesh.record_diff(diff)
        
        return {
            "status": "recorded",
            "diff_id": diff.id,
            "timestamp": diff.timestamp.isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to record diff: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/diffs")
async def get_diffs(
    limit: int = 100,
    mesh: ConceptMesh = Depends(get_mesh)
):
    """Get recent diffs from the ConceptMesh"""
    return mesh.get_diff_history(limit)

@router.get("/stats")
async def get_mesh_stats(mesh: ConceptMesh = Depends(get_mesh)):
    """Get comprehensive ConceptMesh statistics"""
    return mesh.get_statistics()

@router.get("/concepts/{concept_id}")
async def get_concept(
    concept_id: str,
    mesh: ConceptMesh = Depends(get_mesh)
):
    """Get a specific concept by ID"""
    if concept_id not in mesh.concepts:
        raise HTTPException(status_code=404, detail="Concept not found")
    
    concept = mesh.concepts[concept_id]
    return {
        "id": concept.id,
        "name": concept.name,
        "description": concept.description,
        "category": concept.category,
        "importance": concept.importance,
        "created_at": concept.created_at.isoformat(),
        "last_accessed": concept.last_accessed.isoformat(),
        "access_count": concept.access_count,
        "metadata": concept.metadata
    }

@router.get("/concepts/{concept_id}/related")
async def get_related_concepts(
    concept_id: str,
    max_depth: int = 1,
    relation_type: Optional[str] = None,
    mesh: ConceptMesh = Depends(get_mesh)
):
    """Get concepts related to a specific concept"""
    related = mesh.get_related_concepts(concept_id, relation_type, max_depth)
    
    results = []
    for related_id, rel_type, strength in related:
        if related_id in mesh.concepts:
            concept = mesh.concepts[related_id]
            results.append({
                "id": related_id,
                "name": concept.name,
                "category": concept.category,
                "relation_type": rel_type,
                "strength": strength
            })
    
    return results

@router.get("/concepts/{concept_id}/similar")
async def get_similar_concepts(
    concept_id: str,
    threshold: float = 0.7,
    max_results: int = 10,
    mesh: ConceptMesh = Depends(get_mesh)
):
    """Find concepts similar to a specific concept"""
    similar = mesh.find_similar_concepts(concept_id, threshold, max_results)
    
    results = []
    for similar_id, similarity in similar:
        if similar_id in mesh.concepts:
            concept = mesh.concepts[similar_id]
            results.append({
                "id": similar_id,
                "name": concept.name,
                "category": concept.category,
                "similarity": similarity
            })
    
    return results

@router.post("/sync_to_scholarsphere")
async def sync_to_scholarsphere(mesh: ConceptMesh = Depends(get_mesh)):
    """Sync current concept mesh to ScholarSphere"""
    try:
        # Export mesh to get concepts
        export_path = Path("temp_scholarsphere_export.json")
        success = mesh.export_to_json(export_path)
        
        if not success:
            return {"status": "export_failed", "message": "Failed to export mesh"}
        
        # Read the exported data
        with open(export_path, "r") as f:
            mesh_data = json.load(f)
        
        # Clean up temp file
        export_path.unlink()
        
        # Extract concepts
        concepts = mesh_data.get("concepts", [])
        
        if not concepts:
            return {"status": "no_concepts", "message": "No concepts to sync"}
        
        # Import uploader
        from api.scholarsphere_upload import upload_concepts_to_scholarsphere
        
        # Upload to ScholarSphere
        diff_id = await upload_concepts_to_scholarsphere(concepts, source="manual_sync")
        
        if diff_id:
            logger.info(f"Successfully synced to ScholarSphere: {diff_id}")
            return {
                "status": "success",
                "diff_id": diff_id,
                "concept_count": len(concepts),
                "relation_count": len(mesh_data.get("relations", [])),
                "message": f"Synced {len(concepts)} concepts to ScholarSphere"
            }
        else:
            return {"status": "failed", "message": "Upload failed"}
            
    except Exception as e:
        logger.error(f"ScholarSphere sync failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@router.post("/export")
async def export_mesh(
    file_path: str,
    mesh: ConceptMesh = Depends(get_mesh)
):
    """Export the entire mesh to a JSON file"""
    try:
        path = Path(file_path)
        success = mesh.export_to_json(path)
        
        if success:
            return {
                "status": "success",
                "path": str(path),
                "concept_count": len(mesh.concepts),
                "relation_count": len(mesh.relations)
            }
        else:
            raise HTTPException(status_code=500, detail="Export failed")
            
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
