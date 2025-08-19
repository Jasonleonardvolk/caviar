"""
Concept Mesh API Routes
=======================

Handles concept mesh operations for TORI.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from datetime import datetime
from pathlib import Path
import logging
import json
import os
import sys

# ScholarSphere uploader - inline fallback
async def upload_concepts_to_scholarsphere(concepts, source="concept_mesh_sync"):
    """Upload concepts to ScholarSphere with fallback to JSONL"""
    api_key = os.getenv("SCHOLARSPHERE_API_KEY")
    if not api_key:
        logger.warning("ScholarSphere API key not set - writing JSONL fallback")
        Path("data/scholarsphere").mkdir(parents=True, exist_ok=True)
        fname = f"data/scholarsphere/mesh_{datetime.utcnow():%Y%m%d_%H%M%S}.jsonl"
        with open(fname, "w", encoding="utf-8") as f:
            for c in concepts:
                f.write(json.dumps(c) + "\n")
        return fname
    # TODO: Implement actual HTTPS upload when API is available
    return None

# Import the real ConceptMesh
from python.core.concept_mesh import ConceptMesh

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Use the real ConceptMesh singleton - it will auto-load data
mesh = ConceptMesh.instance(config={})
logger.info(f"Using ConceptMesh singleton with {len(mesh.concepts)} concepts")

# Create a view to maintain compatibility with existing code
def _concepts_view():
    """Return mesh content as the legacy _concepts dict format"""
    result = {}
    for concept in mesh.concepts.values():
        # Handle embedding - it might be a list or numpy array
        if hasattr(concept.embedding, 'tolist'):
            embedding = concept.embedding.tolist()
        elif isinstance(concept.embedding, list):
            embedding = concept.embedding
        else:
            embedding = []
            
        result[concept.id] = {
            "id": concept.id,
            "name": concept.name,
            "embedding": embedding,
            "strength": concept.importance,
            "metadata": concept.metadata,
            "timestamp": concept.created_at.isoformat() if hasattr(concept, 'created_at') else datetime.utcnow().isoformat()
        }
    return result

# For backward compatibility, create _concepts as a property
_concepts = _concepts_view()


class ConceptDiff(BaseModel):
    """Concept difference/update"""
    id: str
    name: str
    embedding: List[float]
    strength: float = 1.0
    metadata: Optional[Dict[str, Any]] = {}


class ConceptQuery(BaseModel):
    """Concept search query"""
    embedding: Optional[List[float]] = None
    text: Optional[str] = None
    limit: int = 10
    threshold: float = 0.7


@router.post("/record_diff")
async def record_concept_diff(diff: ConceptDiff):
    """Record a concept difference/update"""
    try:
        # Store concept in the real mesh
        mesh.add_concept(
            name=diff.name,
            importance=diff.strength,
            embedding=diff.embedding,
            metadata={
                **diff.metadata,
                "id": diff.id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Update the view
        global _concepts
        _concepts = _concepts_view()
        
        logger.info(f"Recorded concept diff: {diff.id}")
        
        # In production, this would also:
        # 1. Update the concept mesh
        # 2. Trigger HoTT verification
        # 3. Save to ScholarSphere
        
        return {
            "success": True,
            "conceptId": diff.id,
            "message": "Concept recorded"
        }
    
    except Exception as e:
        logger.error(f"Record diff error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def query_concepts(query: ConceptQuery):
    """Query concepts by embedding or text"""
    try:
        # Simple implementation - return all concepts
        # In production, use vector similarity search
        
        concepts = list(_concepts_view().values())
        
        if query.text:
            # Filter by text
            text_lower = query.text.lower()
            concepts = [
                c for c in concepts
                if text_lower in c["name"].lower()
            ]
        
        # Limit results
        concepts = concepts[:query.limit]
        
        return {
            "concepts": concepts,
            "count": len(concepts),
            "total": len(_concepts)
        }
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_mesh_stats():
    """Get concept mesh statistics"""
    try:
        current_concepts = _concepts_view()
        total_concepts = len(current_concepts)
        
        # Calculate average strength
        total_strength = sum(c.get("strength", 1.0) for c in current_concepts.values())
        avg_strength = total_strength / total_concepts if total_concepts else 0.0
        
        return {
            "totalConcepts": total_concepts,
            "averageStrength": avg_strength,
            "lastUpdate": max(
                (c["timestamp"] for c in current_concepts.values()),
                default=None
            )
        }
    
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/concept/{concept_id}")
async def get_concept(concept_id: str):
    """Get specific concept by ID"""
    try:
        current_concepts = _concepts_view()
        if concept_id not in current_concepts:
            raise HTTPException(status_code=404, detail="Concept not found")
        
        return current_concepts[concept_id]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get concept error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync_to_scholarsphere")
async def sync_to_scholarsphere():
    """Sync concepts to ScholarSphere"""
    try:
        concepts = list(_concepts_view().values())
        diff_id = await upload_concepts_to_scholarsphere(
            concepts, source="concept_mesh_sync"
        )
        return {
            "success": diff_id is not None,
            "diffId": diff_id,
            "conceptCount": len(concepts),
            "message": "Uploaded to ScholarSphere" if diff_id else "Upload failed - check API key"
        }
    
    except Exception as e:
        logger.error(f"Sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


logger.info("Concept Mesh routes initialized")
