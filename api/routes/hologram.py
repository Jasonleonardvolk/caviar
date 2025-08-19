"""
Hologram API Routes
Endpoints for uploading and querying holographic memories
"""

from fastapi import APIRouter, HTTPException, status, File, UploadFile, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import shutil
import time
import os

# Import holographic components
try:
    from hott_integration.holographic_orchestrator import get_orchestrator
    from hott_integration.psi_morphon import HolographicMemory, ModalityType
    from python.core.scoped_concept_mesh import ScopedConceptMesh
    HOLOGRAM_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import hologram components: {e}")
    HOLOGRAM_COMPONENTS_AVAILABLE = False

# Import auth dependency
from api.routes.mesh import get_mesh_scope

logger = logging.getLogger("hologram_api")

router = APIRouter(prefix="/api/hologram", tags=["hologram"])

# Request/Response models
class HologramMetadata(BaseModel):
    title: Optional[str] = Field(None, description="Title for the hologram")
    description: Optional[str] = Field(None, description="Description")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")
    generate_proofs: bool = Field(True, description="Generate proofs for morphons")

class HologramJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    
class MorphonResponse(BaseModel):
    id: str
    modality: str
    content: Any
    salience: float
    temporal_index: Optional[float]
    metadata: Dict[str, Any]
    connections: List[str]  # Connected morphon IDs

# Upload endpoint
@router.post("/upload", response_model=HologramJobResponse)
async def upload_hologram(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    scope_info: tuple[str, str] = Depends(get_mesh_scope),
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[str] = None  # Comma-separated
):
    """
    Upload a media file for holographic ingestion
    
    Supported formats:
    - Images: jpg, png, gif, bmp, webp
    - Audio: mp3, wav, ogg, m4a, flac
    - Video: mp4, avi, mov, mkv, webm
    """
    if not HOLOGRAM_COMPONENTS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hologram components not available"
        )
    
    scope, scope_id = scope_info
    
    # Validate file size (100MB limit)
    max_size = 100 * 1024 * 1024
    if file.size and file.size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {max_size/1024/1024:.0f}MB"
        )
    
    # Save uploaded file
    upload_dir = Path("data/uploads") / scope / scope_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    timestamp = int(time.time() * 1000)
    file_ext = Path(file.filename).suffix
    saved_filename = f"{timestamp}_{file.filename}"
    file_path = upload_dir / saved_filename
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved upload to {file_path}")
        
        # Prepare metadata
        metadata = {
            "title": title or file.filename,
            "description": description,
            "tags": tags.split(",") if tags else [],
            "original_filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": file_path.stat().st_size
        }
        
        # Get orchestrator and queue ingestion
        orchestrator = get_orchestrator()
        
        # Queue ingestion in background
        job_id = await orchestrator.ingest_file(
            file_path,
            scope,
            scope_id,
            metadata
        )
        
        return HologramJobResponse(
            job_id=job_id,
            status="queued",
            message=f"File '{file.filename}' queued for holographic ingestion"
        )
        
    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )

# Job status endpoint
@router.get("/job/{job_id}")
async def get_job_status(job_id: str, scope_info: tuple[str, str] = Depends(get_mesh_scope)):
    """Get status of a holographic ingestion job"""
    if not HOLOGRAM_COMPONENTS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hologram components not available"
        )
    
    scope, scope_id = scope_info
    orchestrator = get_orchestrator()
    
    job = orchestrator.get_job_status(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Verify job belongs to tenant
    if job["tenant_scope"] != scope or job["tenant_id"] != scope_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this job"
        )
    
    return job

# List memories endpoint
@router.get("/memories")
async def list_memories(scope_info: tuple[str, str] = Depends(get_mesh_scope)):
    """List all holographic memories for the tenant"""
    if not HOLOGRAM_COMPONENTS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hologram components not available"
        )
    
    scope, scope_id = scope_info
    orchestrator = get_orchestrator()
    
    memories = await orchestrator.list_memories(scope, scope_id)
    
    return {
        "scope": scope,
        "scope_id": scope_id,
        "memories": memories,
        "total": len(memories)
    }

# Get specific memory
@router.get("/memory/{memory_id}")
async def get_memory(memory_id: str, scope_info: tuple[str, str] = Depends(get_mesh_scope)):
    """Get a specific holographic memory with all morphons and strands"""
    if not HOLOGRAM_COMPONENTS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hologram components not available"
        )
    
    scope, scope_id = scope_info
    orchestrator = get_orchestrator()
    
    memory = await orchestrator.get_memory(memory_id, scope, scope_id)
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found"
        )
    
    return memory.to_dict()

# Get morphons for visualization
@router.get("/morphons")
async def get_morphons(
    scope_info: tuple[str, str] = Depends(get_mesh_scope),
    memory_id: Optional[str] = None,
    modality: Optional[str] = None,
    limit: int = 100
):
    """
    Get morphons for visualization
    
    Can filter by memory_id or modality type
    """
    if not HOLOGRAM_COMPONENTS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hologram components not available"
        )
    
    scope, scope_id = scope_info
    morphons = []
    
    if memory_id:
        # Get morphons from specific memory
        orchestrator = get_orchestrator()
        memory = await orchestrator.get_memory(memory_id, scope, scope_id)
        
        if memory:
            for morphon in memory.morphons[:limit]:
                # Find connected morphons
                connections = memory.get_connected_morphons(morphon.id)
                
                morphon_data = MorphonResponse(
                    id=morphon.id,
                    modality=morphon.modality.value,
                    content=morphon.content,
                    salience=morphon.salience,
                    temporal_index=morphon.temporal_index,
                    metadata=morphon.metadata,
                    connections=connections
                )
                morphons.append(morphon_data)
    else:
        # Get recent morphons from all memories
        orchestrator = get_orchestrator()
        memories_list = await orchestrator.list_memories(scope, scope_id)
        
        # Load most recent memories
        for mem_info in memories_list[:10]:  # Last 10 memories
            memory = await orchestrator.get_memory(mem_info["id"], scope, scope_id)
            if memory:
                for morphon in memory.morphons:
                    if modality and morphon.modality.value != modality:
                        continue
                    
                    connections = memory.get_connected_morphons(morphon.id)
                    
                    morphon_data = MorphonResponse(
                        id=morphon.id,
                        modality=morphon.modality.value,
                        content=morphon.content,
                        salience=morphon.salience,
                        temporal_index=morphon.temporal_index,
                        metadata=morphon.metadata,
                        connections=connections
                    )
                    morphons.append(morphon_data)
                    
                    if len(morphons) >= limit:
                        break
            
            if len(morphons) >= limit:
                break
    
    return {
        "morphons": morphons,
        "total": len(morphons),
        "filters": {
            "memory_id": memory_id,
            "modality": modality
        }
    }

# Get visualization graph
@router.get("/graph")
async def get_hologram_graph(
    scope_info: tuple[str, str] = Depends(get_mesh_scope),
    memory_id: Optional[str] = None
):
    """
    Get graph structure for visualization
    
    Returns nodes (morphons + concepts) and edges (strands)
    """
    if not HOLOGRAM_COMPONENTS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hologram components not available"
        )
    
    scope, scope_id = scope_info
    nodes = []
    edges = []
    
    # Get concept mesh
    mesh = ScopedConceptMesh.get_instance(scope, scope_id)
    
    if memory_id:
        # Get specific memory
        orchestrator = get_orchestrator()
        memory = await orchestrator.get_memory(memory_id, scope, scope_id)
        
        if memory:
            # Add morphon nodes
            for morphon in memory.morphons:
                nodes.append({
                    "id": morphon.id,
                    "type": "morphon",
                    "modality": morphon.modality.value,
                    "label": morphon.metadata.get("title", morphon.id[:8]),
                    "salience": morphon.salience,
                    "x": morphon.temporal_index or 0,
                    "y": hash(morphon.modality.value) % 100
                })
            
            # Add strand edges
            for strand in memory.strands:
                edges.append({
                    "source": strand.source_morphon_id,
                    "target": strand.target_morphon_id,
                    "type": strand.strand_type.value,
                    "strength": strand.strength,
                    "bidirectional": strand.bidirectional
                })
            
            # Add connected concepts
            concept_ids = set()
            for strand in memory.strands:
                if strand.source_morphon_id.startswith("concept_"):
                    concept_ids.add(strand.source_morphon_id[8:])
                if strand.target_morphon_id.startswith("concept_"):
                    concept_ids.add(strand.target_morphon_id[8:])
            
            for concept_id in concept_ids:
                if concept_id in mesh.concepts:
                    concept = mesh.concepts[concept_id]
                    nodes.append({
                        "id": f"concept_{concept_id}",
                        "type": "concept",
                        "label": concept.name,
                        "category": concept.category,
                        "importance": concept.importance
                    })
    else:
        # Get recent concepts with morphon connections
        # This is a simplified version - in production, query more efficiently
        for concept in list(mesh.concepts.values())[:50]:
            if concept.metadata.get("source_morphon"):
                nodes.append({
                    "id": f"concept_{concept.id}",
                    "type": "concept",
                    "label": concept.name,
                    "category": concept.category,
                    "importance": concept.importance
                })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": {
                "morphons": sum(1 for n in nodes if n["type"] == "morphon"),
                "concepts": sum(1 for n in nodes if n["type"] == "concept")
            }
        }
    }

# Health check
@router.get("/health")
async def hologram_health():
    """Check hologram subsystem health"""
    return {
        "status": "operational" if HOLOGRAM_COMPONENTS_AVAILABLE else "degraded",
        "components": {
            "orchestrator": HOLOGRAM_COMPONENTS_AVAILABLE,
            "handlers": {
                "image": HOLOGRAM_COMPONENTS_AVAILABLE,
                "audio": HOLOGRAM_COMPONENTS_AVAILABLE,
                "video": HOLOGRAM_COMPONENTS_AVAILABLE
            },
            "synthesizer": HOLOGRAM_COMPONENTS_AVAILABLE
        }
    }
