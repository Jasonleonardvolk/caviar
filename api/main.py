from fastapi import FastAPI
from pydantic import BaseModel
import os
import logging
import numpy as np
import scipy.sparse as sp
from pathlib import Path
# Ricci demo router import will be added conditionally below

app = FastAPI(title="TORI API", version="1.0.0")
logger = logging.getLogger(__name__)

# Initialize FSM on app startup (optional - not required for demo routes)
try:
    # Skip FSM initialization if torch has issues
    import torch
    torch_available = True
except Exception:
    torch_available = False
    logger.warning("PyTorch not available, skipping FSM initialization")

if torch_available:
    try:
        from python.core.fractal_soliton_memory import FractalSolitonMemory
        from python.core.graph_ops import load_laplacian
        
        # Bootstrap FSM with random initial state
        fsm = FractalSolitonMemory.from_random(n=4096, seed=42)
        
        # Try to load Laplacian with safe path resolution
        try:
            # Resolve path relative to project root
            api_path = Path(__file__).resolve()
            project_root = api_path.parents[1]  # Go up from api/main.py to project root
            laplacian_path = project_root / "data" / "concept_mesh" / "L_norm.npz"
            
            L = load_laplacian(str(laplacian_path))
            fsm.set_laplacian(L, version="L_norm@1")
            logger.info(f"Loaded Laplacian from {laplacian_path}")
        except Exception as e:
            logger.warning(f"Could not load Laplacian: {e}")
            # Use identity matrix as fallback
            L = sp.identity(4096, format="csr")
            fsm.set_laplacian(L, version="identity@fallback")
        
        app.state.fsm = fsm
        app.state.fsm_lambda = 0.0  # Adjust if you want quartic term active
        logger.info("FSM initialized and attached to app.state")
    except ImportError as e:
        logger.error(f"Required modules not found: {e}")
        logger.error("Please ensure python.core.fractal_soliton_memory and python.core.graph_ops exist")
        app.state.fsm = None
    except Exception as e:
        logger.error(f"Could not initialize FSM: {e}")
        app.state.fsm = None
else:
    app.state.fsm = None
    app.state.fsm_lambda = 0.0

# Import and register memory routes
try:
    from api.routes.memory_vault_routes import router as memory_router
    app.include_router(memory_router)
    logger.info("Memory vault routes registered")
except ImportError as e:
    logger.warning(f"Could not import memory vault routes: {e}")

# Include Ricci demo router if available
try:
    from api.demo import router as ricci_demo_router
    app.include_router(ricci_demo_router, prefix="/admin/ricci", tags=["ricci-admin"])
    logger.info("Ricci demo routes registered")
except ImportError as e:
    logger.warning(f"Could not import ricci demo routes: {e}")

@app.get("/health")
def health():
    fsm_status = "initialized" if hasattr(app.state, "fsm") and app.state.fsm is not None else "not_initialized"
    return {
        "status": "ok",
        "api_port": int(os.environ.get("API_PORT", 0)),
        "fsm_status": fsm_status,
        "routes_loaded": "memory_vault" in [r.name for r in app.routes if hasattr(r, "name")]
    }

class Echo(BaseModel):
    text: str

@app.post("/api/echo")
def api_echo(payload: Echo):
    return {"echo": payload.text}

if __name__ == "__main__":
    # For ad-hoc local runs (not used by launcher)
    import uvicorn
    port = int(os.environ.get("API_PORT", "8002"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
