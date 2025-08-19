"""
Concept Mesh API Router
Handles concept mesh operations including recording diffs and syncing with ScholarSphere
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
import json
import os
from datetime import datetime
import asyncio

# Set up module logger
logger = logging.getLogger("concept_mesh_api")

router = APIRouter(prefix="/api/concept-mesh", tags=["concept-mesh"])

# Import concept mesh if available
CONCEPT_MESH_AVAILABLE = False
concept_mesh_module = None

try:
    # Try production Rust wheel first
    import concept_mesh_rs as cm
    CONCEPT_MESH_AVAILABLE = True
    USING_RUST = True
    logger.info("✅ Using concept_mesh_rs (Rust wheel)")
except ImportError:
    try:
        # Fall back to Python stub
        from python.core import ConceptMeshStub as cm
        CONCEPT_MESH_AVAILABLE = True
        USING_RUST = False
        logger.info("✅ Using concept_mesh Python stub")
    except ImportError:
        logger.warning("⚠️ No concept mesh implementation available")
        cm = None
        USING_RUST = False

# Pydantic models
class ConceptDiff(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    embedding: Optional[List[float]] = None
    strength: float = 1.0
    metadata: Optional[Dict[str, Any]] = {}

class RecordDiffRequest(BaseModel):
    concepts: List[ConceptDiff]
    source: Optional[str] = "pdf_upload"
    user_id: Optional[str] = "default"

class RecordDiffResponse(BaseModel):
    success: bool
    recorded_count: int
    mesh_state: Optional[Dict[str, Any]] = None
    message: str

# Global mesh instance (singleton)
_mesh_instance = None

def get_mesh_instance():
    """Get or create the global concept mesh instance"""
    global _mesh_instance
    if _mesh_instance is None and CONCEPT_MESH_AVAILABLE:
        try:
            _mesh_instance = cm.ConceptMesh()
            logger.info("Created new concept mesh instance")
        except Exception as e:
            logger.error(f"Failed to create concept mesh: {e}")
    return _mesh_instance

@router.post("/record_diff", response_model=RecordDiffResponse)
async def record_concept_diff(
    request: RecordDiffRequest,
    background_tasks: BackgroundTasks
):
    """
    Record a concept diff to the mesh and trigger ScholarSphere sync
    """
    logger.info(f"Recording diff with {len(request.concepts)} concepts from {request.source}")
    
    if not CONCEPT_MESH_AVAILABLE:
        logger.warning("Concept mesh not available, returning mock response")
        return RecordDiffResponse(
            success=True,
            recorded_count=len(request.concepts),
            mesh_state={"mock": True, "total_concepts": len(request.concepts)},
            message="Concept mesh not available - diff recorded in mock mode"
        )
    
    mesh = get_mesh_instance()
    if not mesh:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Concept mesh service unavailable"
        )
    
    recorded = 0
    errors = []
    
    # Record each concept
    for concept in request.concepts:
        try:
            # Add to mesh
            mesh.add_concept(
                concept_id=concept.id,
                name=concept.name,
                embedding=concept.embedding,
                strength=concept.strength,
                metadata={
                    **concept.metadata,
                    "source": request.source,
                    "user_id": request.user_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            recorded += 1
            
            # Inject into oscillator lattice (fixes the zero oscillator issue)
            if hasattr(mesh, 'inject_oscillator'):
                phase_vector = [concept.strength] * 8  # 8-dimensional phase space
                mesh.inject_oscillator(concept.id, phase_vector)
                
        except Exception as e:
            logger.error(f"Failed to record concept {concept.id}: {e}")
            errors.append(str(e))
    
    # Get current mesh state
    mesh_state = None
    try:
        if hasattr(mesh, 'get_stats'):
            mesh_state = mesh.get_stats()
        else:
            mesh_state = {
                "total_concepts": recorded,
                "using_rust": USING_RUST
            }
    except Exception as e:
        logger.error(f"Failed to get mesh stats: {e}")
    
    # Schedule ScholarSphere sync in background
    if recorded > 0:
        background_tasks.add_task(
            sync_to_scholarsphere,
            request.concepts[:recorded],
            request.source
        )
    
    return RecordDiffResponse(
        success=recorded > 0,
        recorded_count=recorded,
        mesh_state=mesh_state,
        message=f"Recorded {recorded}/{len(request.concepts)} concepts" + 
                (f", errors: {errors[:3]}" if errors else "")
    )

async def sync_to_scholarsphere(concepts: List[ConceptDiff], source: str):
    """
    Background task to sync concepts to ScholarSphere
    """
    logger.info(f"Starting ScholarSphere sync for {len(concepts)} concepts")
    
    # Create JSONL file
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    diff_filename = f"mesh_diff_{timestamp}.jsonl"
    diff_path = os.path.join("data", "scholarsphere", diff_filename)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(diff_path), exist_ok=True)
    
    # Write JSONL
    with open(diff_path, 'w') as f:
        for concept in concepts:
            diff_entry = {
                "id": concept.id,
                "name": concept.name,
                "strength": concept.strength,
                "metadata": concept.metadata,
                "source": source,
                "timestamp": datetime.utcnow().isoformat()
            }
            f.write(json.dumps(diff_entry) + '\n')
    
    logger.info(f"Created diff file: {diff_path}")
    
    # TODO: Upload to ScholarSphere when API is available
    # For now, just log that it's ready
    logger.info(f"ScholarSphere diff ready for upload: {diff_filename}")

@router.get("/stats")
async def get_mesh_stats():
    """Get current concept mesh statistics"""
    if not CONCEPT_MESH_AVAILABLE:
        return {
            "available": False,
            "message": "Concept mesh not available"
        }
    
    mesh = get_mesh_instance()
    if not mesh:
        return {
            "available": False,
            "message": "Failed to initialize mesh"
        }
    
    try:
        stats = mesh.get_stats() if hasattr(mesh, 'get_stats') else {}
        return {
            "available": True,
            "using_rust": USING_RUST,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get mesh stats: {e}")
        return {
            "available": True,
            "error": str(e)
        }

@router.get("/health")
async def health_check():
    """Check concept mesh service health"""
    return {
        "status": "healthy" if CONCEPT_MESH_AVAILABLE else "degraded",
        "concept_mesh_available": CONCEPT_MESH_AVAILABLE,
        "using_rust": USING_RUST if CONCEPT_MESH_AVAILABLE else None,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/concepts")
async def get_all_concept_mesh_concepts():
    """
    Return all concepts in the mesh.
    Always returns a valid response even if mesh is unavailable.
    """
    try:
        if CONCEPT_MESH_AVAILABLE:
            mesh = get_mesh_instance()
            if mesh and hasattr(mesh, "concepts"):
                # Try to get concepts from the mesh
                concepts = []
                if hasattr(mesh.concepts, "values"):
                    # Dictionary-like structure
                    for concept in mesh.concepts.values():
                        concept_dict = {
                            "id": getattr(concept, "id", "unknown"),
                            "name": getattr(concept, "name", "Unknown"),
                            "strength": getattr(concept, "strength", 1.0),
                            "metadata": getattr(concept, "metadata", {})
                        }
                        concepts.append(concept_dict)
                elif hasattr(mesh.concepts, "__iter__"):
                    # List-like structure
                    for concept in mesh.concepts:
                        if hasattr(concept, "dict"):
                            concepts.append(concept.dict())
                        elif isinstance(concept, dict):
                            concepts.append(concept)
                        else:
                            concepts.append({
                                "id": str(concept),
                                "name": str(concept),
                                "strength": 1.0,
                                "metadata": {}
                            })
                
                return {
                    "concepts": concepts,
                    "count": len(concepts),
                    "status": "ok",
                    "source": "concept_mesh"
                }
            else:
                # Mesh exists but no concepts
                return {
                    "concepts": [],
                    "count": 0,
                    "status": "empty",
                    "message": "Concept mesh is empty"
                }
        else:
            # Return mock concepts when mesh unavailable
            mock_concepts = [
                {"id": "consciousness", "name": "Consciousness", "strength": 0.95, "metadata": {"category": "philosophy"}},
                {"id": "cognition", "name": "Cognition", "strength": 0.92, "metadata": {"category": "psychology"}},
                {"id": "intelligence", "name": "Intelligence", "strength": 0.89, "metadata": {"category": "ai"}},
                {"id": "awareness", "name": "Awareness", "strength": 0.87, "metadata": {"category": "philosophy"}},
                {"id": "learning", "name": "Learning", "strength": 0.85, "metadata": {"category": "education"}}
            ]
            return {
                "concepts": mock_concepts,
                "count": len(mock_concepts),
                "status": "mock",
                "message": "Concept mesh unavailable - returning mock data"
            }
    except Exception as e:
        logger.error(f"Error getting concepts: {e}")
        return {
            "concepts": [],
            "count": 0,
            "status": "error",
            "error": str(e)
        }
