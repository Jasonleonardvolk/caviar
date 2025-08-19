#!/usr/bin/env python3
"""
Quick API server for TORI - With REAL Soliton Routes
Enhanced version that imports actual implementations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import time
import json
from pathlib import Path
import sys

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import real soliton routes
try:
    from api.routes.soliton_production import router as soliton_router
    SOLITON_ROUTES_AVAILABLE = True
    print("✅ Real soliton routes imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import real soliton routes: {e}")
    soliton_router = None
    SOLITON_ROUTES_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="TORI Quick API - With Real Soliton",
    description="Lightweight API server with real soliton memory implementation",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic models
class AnswerRequest(BaseModel):
    user_query: str
    persona: Optional[Dict[str, Any]] = None
    context: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: float
    components: Dict[str, bool]

# Global state for component tracking
components_ready = {
    "api": True,
    "prajna": False,  # Will be False until we fix imports
    "concept_mesh": False,
    "soliton": SOLITON_ROUTES_AVAILABLE  # True if real routes loaded
}

# Health check endpoint
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="tori-quick-api",
        timestamp=time.time(),
        components=components_ready
    )

# System ready endpoint
@app.get("/api/system/ready")
async def system_ready():
    # Check if all critical components are ready
    critical = ["api"]  # Only API is critical for now
    all_ready = all(components_ready.get(comp, False) for comp in critical)
    
    if all_ready:
        return {"ready": True, "components": components_ready}
    else:
        # Return 503 if not ready
        raise HTTPException(status_code=503, detail="System not ready", 
                          headers={"Retry-After": "5"})

# Component status endpoint
@app.get("/api/system/components")
async def get_components():
    return components_ready

# Register component as ready
@app.post("/api/system/components/{component}/ready")
async def register_component_ready(component: str, details: Optional[Dict] = None):
    components_ready[component] = True
    return {"status": "ok", "component": component, "ready": True}

# Mock answer endpoint (until prajna is fixed)
@app.post("/api/answer")
async def answer_query(request: AnswerRequest):
    """Mock answer endpoint that returns a simple response"""
    return {
        "answer": f"I received your query: '{request.user_query}'. However, the Prajna language model is still initializing. This is a placeholder response.",
        "query": request.user_query,
        "persona": request.persona,
        "status": "mock_mode",
        "model": "placeholder"
    }

# Prajna stats endpoint (mock)
@app.get("/api/prajna/stats")
async def prajna_stats():
    return {
        "status": "initializing",
        "model_type": "saigon_lstm",
        "queries_processed": 0,
        "average_response_time": 0,
        "cache_hits": 0,
        "errors": 0,
        "message": "Prajna is still initializing. Heavy imports are being loaded."
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "TORI Quick API",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "answer": "/api/answer",
            "stats": "/api/prajna/stats",
            "docs": "/docs"
        }
    }

# Include real soliton routes if available
if SOLITON_ROUTES_AVAILABLE and soliton_router:
    app.include_router(soliton_router)
    print("✅ Real soliton routes included!")
    print("📍 Available soliton endpoints:")
    print("   - POST /api/soliton/init")
    print("   - POST /api/soliton/initialize")
    print("   - POST /api/soliton/store")
    print("   - GET /api/soliton/stats/{user_id}")
    print("   - POST /api/soliton/embed")
    print("   - GET /api/soliton/health")
    print("   - GET /api/soliton/diagnostic")
else:
    # Fallback: Mock soliton endpoints
    @app.post("/api/soliton/store")
    async def store_soliton(data: Dict[str, Any]):
        return {"status": "stored", "id": "mock_id", "message": "Soliton storage not yet implemented"}

    @app.get("/api/soliton/query")
    async def query_soliton(query: str):
        return {"results": [], "message": "Soliton query not yet implemented"}
    
    print("⚠️ Using mock soliton endpoints")

# Auth endpoints (mock)
@app.post("/api/auth/login")
async def login(username: str, password: str):
    if username in ["admin", "user", "test"]:
        return {
            "success": True,
            "token": "mock_token",
            "user": {"username": username, "role": "user"}
        }
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/api/auth/status")
async def auth_status():
    return {"authenticated": False, "user": None}

def main():
    """Start the quick API server"""
    print("🚀 Starting TORI Quick API Server with Real Soliton Routes...")
    print("📝 This version includes real memory system implementation")
    print("🔧 Soliton routes available:", SOLITON_ROUTES_AVAILABLE)
    
    # Get port from environment or use default
    import os
    port = int(os.environ.get('API_PORT', 8002))
    
    print(f"\n🌐 API Server starting on port {port}")
    print(f"📚 Docs will be available at: http://localhost:{port}/docs")
    print(f"❤️ Health check: http://localhost:{port}/api/health")
    
    # Save port info
    port_file = Path("api_port.json")
    port_file.write_text(json.dumps({
        "api_port": port,
        "api_url": f"http://localhost:{port}",
        "timestamp": time.time(),
        "mode": "quick_start_with_soliton"
    }))
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    main()
