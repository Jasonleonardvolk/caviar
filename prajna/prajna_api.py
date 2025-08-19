"""
Prajna API Application
======================

Prajna sub-application that integrates with main API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Import from kha.api package
try:
    from kha.api import app as main_api_app
    # Extract routers from the main app
    routers_available = True
    logger.info("Successfully imported kha.api app")
except ImportError as e:
    logger.warning(f"Could not import kha.api: {e}")
    routers_available = False
    main_api_app = None

# Create Prajna FastAPI app
app = FastAPI(
    title="Prajna API",
    description="Prajna language model integration with TORI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the main API app as a sub-application if available
if routers_available and main_api_app:
    # Mount the entire main API under /main
    app.mount("/main", main_api_app)
    logger.info("Main API mounted at /main")
    
    # Or alternatively, include specific routers
    try:
        from kha.api.soliton import router as soliton_router
        from kha.api.concept_mesh import router as concept_mesh_router
        
        app.include_router(soliton_router, prefix="/api/soliton", tags=["soliton"])
        app.include_router(concept_mesh_router, prefix="/api/concept-mesh", tags=["concept-mesh"])
        logger.info("Main API routers included in Prajna app")
    except ImportError as e:
        logger.warning(f"Could not import individual routers: {e}")

# Prajna-specific endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "prajna",
        "routers_available": routers_available,
        "main_api_mounted": routers_available
    }

@app.post("/api/answer")
async def answer_question(question: str, user_id: str = "default"):
    """Process a question through Prajna"""
    try:
        # Basic response
        answer = "This is a placeholder answer from Prajna."
        
        # If we have access to concept mesh, record the concept
        if routers_available:
            try:
                from kha.api.concept_mesh import _concepts
                concept_id = f"prajna_q_{len(_concepts)}"
                _concepts[concept_id] = {
                    "id": concept_id,
                    "name": f"Question: {question[:50]}",
                    "embedding": [0.1] * 768,  # Placeholder
                    "strength": 0.8,
                    "metadata": {"source": "prajna", "user_id": user_id}
                }
                logger.info(f"Recorded concept from question: {concept_id}")
            except Exception as e:
                logger.debug(f"Could not record concept: {e}")
        
        return {
            "question": question,
            "answer": answer,
            "concepts_extracted": [],
            "user_id": user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    endpoints = {
        "answer": "/api/answer",
        "health": "/health",
        "docs": "/docs"
    }
    
    if routers_available:
        endpoints.update({
            "soliton": "/api/soliton",
            "concept_mesh": "/api/concept-mesh",
            "main_api": "/main"  # If mounted
        })
    
    return {
        "message": "Prajna API",
        "integrated_mode": routers_available,
        "endpoints": endpoints
    }

if __name__ == "__main__":
    import uvicorn
    
    # Log startup info
    logger.info("Starting Prajna API server")
    if routers_available:
        logger.info("✅ Main API integration enabled")
    else:
        logger.warning("⚠️ Running in standalone mode (no main API)")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
