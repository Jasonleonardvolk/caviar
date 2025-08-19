"""
API endpoints for the ELFIN visualization dashboard.

Provides data streaming and querying for dashboard components.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, Request, Response, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import ELFIN components
from alan_backend.elfin.stability.core.phase_engine import PhaseEngine
from alan_backend.elfin.stability.core.vectorized_phase_engine import VectorizedPhaseEngine
from alan_backend.elfin.koopman.spectral_analyzer import SpectralAnalyzer
from alan_backend.elfin.koopman.snapshot_buffer import SnapshotBuffer
from alan_backend.elfin.visualization.barrier_stream import generate_barrier_events

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="ELFIN Visualization API", 
             description="API for the ELFIN visualization dashboard",
             version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global registry for active systems
active_systems: Dict[str, Dict[str, Any]] = {}


# Pydantic models for request/response validation
class SystemInfo(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    type: str
    config: Dict[str, Any]


class HealthData(BaseModel):
    sync_ratio: float
    stability_index: float
    max_delta_v: float
    unstable_modes: int
    dominant_modes: List[str]
    timestamp: int
    concept_count: int
    connection_count: int
    performance: Dict[str, float]


# Helper for SSE responses
async def event_generator(
    data_func: Callable[[], Dict[str, Any]], 
    interval: float = 1.0
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events from a data function.
    
    Args:
        data_func: Function that returns data
        interval: Update interval in seconds
        
    Yields:
        SSE formatted data
    """
    while True:
        try:
            # Get data from function
            data = data_func()
            
            # Format as SSE
            yield f"data: {json.dumps(data)}\n\n"
            
            # Wait for next update
            await asyncio.sleep(interval)
        except Exception as e:
            logger.error(f"Error in event generator: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(interval)


# API Routes

@app.get("/")
async def root():
    """API root endpoint."""
    return {"message": "ELFIN Visualization API"}


@app.get("/api/v1/systems")
async def list_systems():
    """List active systems."""
    systems = []
    for sys_id, system in active_systems.items():
        systems.append({
            "id": sys_id,
            "name": system.get("name", sys_id),
            "type": system.get("type", "unknown"),
            "description": system.get("description", ""),
            "created_at": system.get("created_at", datetime.now().isoformat()),
            "status": "active"
        })
    return {"systems": systems}


@app.get("/api/v1/systems/{system_id}")
async def get_system_info(system_id: str):
    """Get system information."""
    if system_id not in active_systems:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")
    
    system = active_systems[system_id]
    return {
        "id": system_id,
        "name": system.get("name", system_id),
        "type": system.get("type", "unknown"),
        "description": system.get("description", ""),
        "created_at": system.get("created_at", datetime.now().isoformat()),
        "status": "active",
        "metrics": {
            "concept_count": len(system.get("engine", {}).get("phases", {})),
            "connection_count": system.get("engine", {}).get("weights_matrix", {}).get("nnz", 0)
        }
    }


@app.get("/api/v1/stream/health")
async def stream_health_data(sys: str = Query(...), freq: float = Query(1.0)):
    """
    Stream system health data.
    
    Args:
        sys: System ID
        freq: Update frequency in seconds
    
    Returns:
        Server-Sent Events stream of health data
    """
    if sys not in active_systems:
        raise HTTPException(status_code=404, detail=f"System {sys} not found")
    
    system = active_systems[sys]
    engine = system.get("engine")
    analyzer = system.get("analyzer")
    
    if not engine:
        raise HTTPException(status_code=400, detail="System has no engine")
    
    def get_health_data():
        """Get current health data from engine and analyzer."""
        start_time = time.time()
        
        # Get sync ratio from engine
        engine_state = engine.export_state()
        sync_ratio = engine_state.get("sync_ratio", 0.0)
        
        # Get stability information from analyzer
        stability_index = 0.0
        max_delta_v = 0.0
        unstable_modes = 0
        dominant_modes = []
        spectral_time = 0.0
        
        if analyzer:
            spectral_start = time.time()
            stability_index = analyzer.calculate_stability_index()
            
            # Get dominant and unstable modes
            if hasattr(analyzer, "dominant_modes"):
                dominant_modes = analyzer.dominant_modes
            
            if hasattr(analyzer, "unstable_modes"):
                unstable_modes = len(analyzer.unstable_modes)
            
            # Calculate max Lyapunov change if we have the data
            if hasattr(analyzer, "last_result") and analyzer.last_result:
                if hasattr(analyzer.last_result, "growth_rates"):
                    max_delta_v = max(analyzer.last_result.growth_rates) if analyzer.last_result.growth_rates.size > 0 else 0.0
            
            spectral_time = time.time() - spectral_start
        
        # Get counts from engine state
        if isinstance(engine, VectorizedPhaseEngine):
            concept_count = len(engine.node_ids)
            if hasattr(engine, "weights_matrix"):
                connection_count = engine.weights_matrix.nnz
            else:
                connection_count = 0
        else:
            # Regular PhaseEngine
            concept_count = len(engine_state.get("phases", {}))
            connection_count = len(engine_state.get("graph", {}).get("edges", []))
        
        # Performance metrics
        step_time = engine_state.get("performance", {}).get("last_step_time_ms", 0.0)
        if not step_time and hasattr(engine, "last_step_time"):
            step_time = engine.last_step_time * 1000  # Convert to ms
        
        return {
            "sync_ratio": sync_ratio,
            "stability_index": stability_index,
            "max_delta_v": max_delta_v,
            "unstable_modes": unstable_modes,
            "dominant_modes": dominant_modes,
            "timestamp": int(time.time() * 1000),  # Unix timestamp in ms
            "concept_count": concept_count,
            "connection_count": connection_count,
            "performance": {
                "step_time_ms": step_time,
                "spectral_time_ms": spectral_time * 1000
            }
        }
    
    # Create SSE response
    return StreamingResponse(
        event_generator(get_health_data, interval=freq),
        media_type="text/event-stream"
    )


@app.get("/api/v1/stream/barrier")
async def stream_barrier_data(sys: str = Query(...), freq: float = Query(0.05)):
    """
    Stream barrier function values and thresholds.
    
    This endpoint provides a real-time stream of barrier function values and 
    thresholds, which are used for safety monitoring in the dashboard.
    
    Args:
        sys: System ID
        freq: Update frequency in seconds (default: 50ms = 20Hz)
    
    Returns:
        Server-Sent Events stream of barrier data
    """
    # Check if system exists in registry, but allow simulated data even if not
    system_type = None
    if sys in active_systems:
        system = active_systems[sys]
        system_type = system.get("type")
    
    # Create SSE response using barrier stream generator
    return StreamingResponse(
        generate_barrier_events(sys, freq),
        media_type="text/event-stream"
    )


@app.get("/api/v1/barrier/isosurface")
async def get_barrier_isosurface(sys: str, lvl: float = 0.0, type: str = 'barrier'):
    """
    Get isosurface at a specific level.
    
    Args:
        sys: System ID
        lvl: Isosurface level
        type: Function type ('barrier' or 'lyapunov')
    
    Returns:
        Binary GLB file containing the isosurface mesh
    """
    from alan_backend.elfin.visualization.isosurface_generator import get_or_generate_isosurface
    
    # Allow missing systems for demo purposes
    if sys not in active_systems:
        # For demo, we'll generate one anyway
        logger.info(f"System {sys} not found, generating demo isosurface")
    
    # Get or generate isosurface
    isosurface_path = get_or_generate_isosurface(sys, type, lvl)
    
    if isosurface_path and isosurface_path.exists():
        return FileResponse(isosurface_path, media_type="model/gltf-binary")
    
    # If generation failed
    raise HTTPException(status_code=404, detail="Failed to generate isosurface")


@app.get("/api/v1/field")
async def get_field(type: str, grid: int = 100, sys: str = Query(...)):
    """
    Get field data (Lyapunov or barrier function values).
    
    Args:
        type: Field type ('lyap' or 'barrier')
        grid: Grid resolution
        sys: System ID
        
    Returns:
        2D or 3D array of field values
    """
    if sys not in active_systems:
        raise HTTPException(status_code=404, detail=f"System {sys} not found")
    
    if type not in ['lyap', 'barrier']:
        raise HTTPException(status_code=400, detail="Type must be 'lyap' or 'barrier'")
    
    # This would normally compute the field values on-demand
    # For now, we'll just return mock data
    
    # Create a simple 2D grid with a point attractor at the origin
    x = np.linspace(-5, 5, grid)
    y = np.linspace(-5, 5, grid)
    X, Y = np.meshgrid(x, y)
    
    if type == 'lyap':
        # Simple quadratic Lyapunov function V(x,y) = x^2 + y^2
        Z = X**2 + Y**2
    else:
        # Simple barrier function B(x,y) = 1 - (x^2 + y^2)/25
        Z = 1.0 - (X**2 + Y**2) / 25.0
    
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "z": Z.tolist(),
        "type": type
    }


@app.post("/api/v1/koopman/params")
async def update_koopman_params(
    params: Dict[str, Any],
    sys: str = Query(...)
):
    """
    Update Koopman parameters.
    
    Args:
        params: Parameter updates
        sys: System ID
    
    Returns:
        Updated parameter values
    """
    if sys not in active_systems:
        raise HTTPException(status_code=404, detail=f"System {sys} not found")
    
    system = active_systems[sys]
    analyzer = system.get("analyzer")
    
    if not analyzer:
        raise HTTPException(status_code=400, detail="System has no analyzer")
    
    updated = {}
    
    # Handle specific parameters
    if 'lambda_cut' in params:
        lambda_cut = float(params['lambda_cut'])
        
        # This would normally update a parameter in the analyzer
        # For now, we'll just echo it back
        updated['lambda_cut'] = lambda_cut
    
    return {"status": "ok", "updated": updated}


@app.post("/api/v1/systems")
async def create_system(system: SystemInfo):
    """
    Create a new system.
    
    Args:
        system: System information
        
    Returns:
        Created system information
    """
    # Check if system already exists
    if system.id in active_systems:
        raise HTTPException(status_code=400, detail=f"System {system.id} already exists")
    
    # Create engine based on type
    if system.type == 'phase_engine':
        engine = PhaseEngine(
            coupling_strength=system.config.get('coupling_strength', 0.1)
        )
    elif system.type == 'vectorized_phase_engine':
        engine = VectorizedPhaseEngine(
            coupling_strength=system.config.get('coupling_strength', 0.1),
            use_gpu=system.config.get('use_gpu', False)
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown system type: {system.type}")
    
    # Create analyzer if requested
    analyzer = None
    if system.config.get('create_analyzer', True):
        buffer = SnapshotBuffer(capacity=100)
        analyzer = SpectralAnalyzer(buffer)
    
    # Store in active systems
    active_systems[system.id] = {
        "name": system.name,
        "description": system.description,
        "type": system.type,
        "created_at": datetime.now().isoformat(),
        "engine": engine,
        "analyzer": analyzer,
        "config": system.config
    }
    
    return {"status": "created", "id": system.id}


@app.delete("/api/v1/systems/{system_id}")
async def delete_system(system_id: str):
    """
    Delete a system.
    
    Args:
        system_id: System ID
        
    Returns:
        Deletion status
    """
    if system_id not in active_systems:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")
    
    # Remove from active systems
    del active_systems[system_id]
    
    return {"status": "deleted", "id": system_id}


# Support function to create a demo system
def create_demo_system(sys_id: str, num_concepts: int = 10, num_edges: int = 20, coupling_strength: float = 0.2):
    """
    Create a demo system with random concepts and connections.
    
    Args:
        sys_id: System ID
        num_concepts: Number of concepts
        num_edges: Number of edges
        coupling_strength: Coupling strength
        
    Returns:
        Created system information
    """
    import random
    
    # Create engine
    engine = PhaseEngine(coupling_strength=coupling_strength)
    
    # Add random concepts
    for i in range(num_concepts):
        concept_id = f"concept_{i}"
        initial_phase = random.random() * 2 * np.pi
        natural_frequency = random.random() * 0.1  # Small natural frequencies
        
        engine.add_concept(concept_id, initial_phase, natural_frequency)
    
    # Add random edges
    concept_ids = list(engine.phases.keys())
    for _ in range(num_edges):
        source = random.choice(concept_ids)
        target = random.choice(concept_ids)
        
        if source != target:
            weight = random.random() * 0.5 + 0.5  # [0.5, 1.0]
            phase_offset = random.random() * np.pi / 4  # Small offsets
            
            engine.add_edge(source, target, weight, phase_offset)
    
    # Create analyzer
    buffer = SnapshotBuffer(capacity=100)
    analyzer = SpectralAnalyzer(buffer)
    
    # Run a few steps to initialize
    for _ in range(10):
        phases = engine.step(dt=0.1)
        buffer.add_snapshot(phases)
    
    # Run analysis
    if len(buffer.buffer) > 1:
        analyzer.edmd_decompose()
    
    # Store in active systems
    active_systems[sys_id] = {
        "name": f"Demo System {sys_id}",
        "description": "Automatically generated demo system",
        "type": "phase_engine",
        "created_at": datetime.now().isoformat(),
        "engine": engine,
        "analyzer": analyzer,
        "config": {
            "coupling_strength": coupling_strength,
            "num_concepts": num_concepts,
            "num_edges": num_edges
        }
    }
    
    return {"status": "created", "id": sys_id}


# Create demo system on startup
@app.on_event("startup")
async def startup_event():
    """Create a demo system on startup."""
    try:
        import numpy as np
        create_demo_system("demo_system", num_concepts=20, num_edges=50)
        logger.info("Created demo system 'demo_system'")
    except Exception as e:
        logger.error(f"Failed to create demo system: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
