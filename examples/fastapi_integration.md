# FastAPI Lifecycle Hooks for ConceptMesh

## Add to your main.py:

```python
from fastapi import FastAPI, Depends
from python.core.concept_mesh import ConceptMesh, get_mesh
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Load concept mesh on startup"""
    try:
        # Get mesh instance (automatically loads persisted data)
        mesh = ConceptMesh.instance()
        logger.info(f"ConceptMesh initialized with {mesh.count()} concepts")
        
        # Ensure we have at least some concepts
        if mesh.count() == 0:
            logger.warning("ConceptMesh is empty, attempting to load seeds...")
            mesh.ensure_populated()
            
    except Exception as e:
        logger.error(f"Failed to initialize ConceptMesh: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Save concept mesh on shutdown"""
    try:
        mesh = ConceptMesh.instance()
        mesh.shutdown()  # This calls _save_mesh()
        logger.info("ConceptMesh saved successfully")
    except Exception as e:
        logger.error(f"Failed to save ConceptMesh: {e}")

# Example endpoint using dependency injection
@app.get("/api/concepts/stats")
async def get_concept_stats(mesh: ConceptMesh = Depends(get_mesh)):
    """Get concept mesh statistics"""
    return mesh.get_statistics()

@app.post("/api/concepts")
async def add_concept(
    name: str,
    description: str = "",
    category: str = "general",
    mesh: ConceptMesh = Depends(get_mesh)
):
    """Add a new concept"""
    concept_id = mesh.add_concept(name, description, category)
    return {"id": concept_id, "status": "created"}
```

## For diff_route.py:

```python
from fastapi import APIRouter, Depends, HTTPException
from python.core.concept_mesh import ConceptMesh, ConceptDiff, get_mesh
from datetime import datetime

router = APIRouter(prefix="/api/concept-mesh", tags=["concept-mesh"])

@router.post("/diff")
async def record_diff(
    diff_type: str,
    concepts: list[str],
    new_value: dict = None,
    old_value: dict = None,
    mesh: ConceptMesh = Depends(get_mesh)
):
    """Record a concept diff"""
    try:
        diff = ConceptDiff(
            id=f"diff_{datetime.now().timestamp()}",
            diff_type=diff_type,
            concepts=concepts,
            old_value=old_value,
            new_value=new_value
        )
        mesh.record_diff(diff)
        return {"status": "recorded", "diff_id": diff.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/diffs")
async def get_diffs(
    limit: int = 100,
    mesh: ConceptMesh = Depends(get_mesh)
):
    """Get recent diffs"""
    return mesh.get_diff_history(limit)
```

## Create seed file at data/concept_mesh/seed_concepts.json:

```json
[
    {
        "name": "artificial intelligence",
        "description": "The simulation of human intelligence in machines",
        "category": "technology",
        "importance": 0.9
    },
    {
        "name": "machine learning", 
        "description": "A subset of AI that enables systems to learn from data",
        "category": "technology",
        "importance": 0.85
    },
    {
        "name": "neural networks",
        "description": "Computing systems inspired by biological neural networks",
        "category": "technology",
        "importance": 0.8
    },
    {
        "name": "knowledge representation",
        "description": "How knowledge is structured and stored in AI systems",
        "category": "concepts",
        "importance": 0.75
    },
    {
        "name": "natural language processing",
        "description": "AI's ability to understand and generate human language",
        "category": "technology",
        "importance": 0.85
    }
]
```
