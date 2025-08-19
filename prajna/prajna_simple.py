#!/usr/bin/env python3
"""
Simple Prajna Launcher
======================

Minimal Prajna API that imports from kha.api.
"""

from fastapi import FastAPI, HTTPException
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Prajna app
app = FastAPI(title="Prajna Simple", version="1.0.0")

# Try to import and include main API routes
try:
    import kha.api.main as api_main
    from kha.api.soliton import router as soliton_router
    from kha.api.concept_mesh import router as concept_mesh_router
    
    # Include the routers
    app.include_router(soliton_router, prefix="/api/soliton")
    app.include_router(concept_mesh_router, prefix="/api/concept-mesh")
    
    logger.info("✅ Successfully integrated kha.api routes")
    integrated = True
    
except ImportError as e:
    logger.warning(f"⚠️ Could not import kha.api: {e}")
    integrated = False

# Prajna endpoints
@app.get("/")
async def root():
    return {
        "service": "prajna",
        "integrated": integrated,
        "endpoints": ["/api/answer", "/api/soliton", "/api/concept-mesh"] if integrated else ["/api/answer"]
    }

@app.post("/api/answer")
async def answer(question: str):
    return {"question": question, "answer": "Prajna response"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
