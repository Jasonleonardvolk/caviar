"""
Minimal Main API
================

Simplified main.py to ensure clean import chain.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create app first
app = FastAPI(
    title="TORI Main API",
    description="Main API for TORI system",
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

# Then import routers (avoids circular imports)
try:
    from .soliton import router as soliton_router
    app.include_router(soliton_router, prefix="/api/soliton", tags=["soliton"])
except ImportError as e:
    print(f"Warning: Could not import soliton router: {e}")

try:
    from .concept_mesh import router as concept_mesh_router
    app.include_router(concept_mesh_router, prefix="/api/concept-mesh", tags=["concept-mesh"])
except ImportError as e:
    print(f"Warning: Could not import concept_mesh router: {e}")

# Basic endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "main-api"}

@app.get("/")
async def root():
    return {
        "message": "TORI Main API",
        "endpoints": {
            "soliton": "/api/soliton",
            "concept_mesh": "/api/concept-mesh",
            "health": "/health",
            "docs": "/docs"
        }
    }
