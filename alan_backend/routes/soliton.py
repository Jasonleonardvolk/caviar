"""
alan_backend/routes/soliton.py - Soliton Memory API routes

This module handles all Soliton Memory API requests from the frontend,
forwarding them to the Soliton engine or concept mesh.
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Response, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import logging
import time
import json
from datetime import datetime

# Import Soliton engine connector
try:
    from core.soliton_engine import SolitonEngine, ConceptMeshConnector
    from core.soliton_types import MemoryEntry, MemoryStats, PhaseChangeEvent
    SOLITON_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ Soliton engine imports failed. Using direct concept mesh connection.")
    try:
        from python.core import ConceptMeshConnector
        from core.soliton_types import MemoryEntry, MemoryStats, PhaseChangeEvent
        SOLITON_AVAILABLE = False
    except ImportError:
        logging.error("❌ CRITICAL: Both Soliton engine and concept mesh imports failed!")
        raise RuntimeError("Soliton Memory system unavailable - cannot start backend")

# Setup router
router = APIRouter(prefix="/api/soliton", tags=["soliton"])

# Setup logger
logger = logging.getLogger("soliton_api")

# Initialize the engine connection
if SOLITON_AVAILABLE:
    engine = SolitonEngine()
    mesh = engine.get_mesh_connector()
else:
    # Direct connection to concept mesh
    mesh = ConceptMeshConnector()

# Track initialization status
is_initialized = True

# API Models
class InitializeRequest(BaseModel):
    userId: str
    options: Optional[Dict[str, Any]] = None

class StoreMemoryRequest(BaseModel):
    userId: str
    memory: MemoryEntry

class RecallRequest(BaseModel):
    userId: str
    query: str
    limit: Optional[int] = 5
    minStrength: Optional[float] = 0.3
    tags: Optional[List[str]] = None

class EmbeddingRequest(BaseModel):
    text: str
    dimension: Optional[int] = 128

class PhaseChangeRequest(BaseModel):
    userId: str
    phase: str
    amplitude: float
    frequency: float
    coherence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class ApiResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

# Endpoints

@router.get("/health")
async def health_check():
    """Check if the Soliton memory system is operational"""
    if not is_initialized:
        return {
            "status": "initializing",
            "engine": "soliton" if SOLITON_AVAILABLE else "concept_mesh_direct",
            "message": "Soliton Memory system is initializing"
        }
    
    try:
        # Test the engine connection
        status = "operational"
        message = "Soliton Memory system is operational"
        engine_type = "soliton" if SOLITON_AVAILABLE else "concept_mesh_direct"
        
        # Quick ping test to verify connection
        start_time = time.time()
        mesh.ping()
        response_time = time.time() - start_time
        
        return {
            "status": status,
            "engine": engine_type,
            "message": message,
            "response_time_ms": round(response_time * 1000, 2)
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return {
            "status": "error",
            "engine": "soliton" if SOLITON_AVAILABLE else "concept_mesh_direct",
            "message": f"Soliton Memory system error: {str(e)}"
        }

@router.post("/initialize")
async def initialize_memory(request: InitializeRequest) -> ApiResponse:
    """Initialize the Soliton memory for a user"""
    try:
        user_id = request.userId
        options = request.options or {}
        
        logger.info(f"🌊 Initializing Soliton Memory for user: {user_id}")
        
        # Initialize user memory space in the concept mesh
        result = mesh.initialize_user(user_id, **options)
        
        if not result:
            raise Exception("Failed to initialize user memory space")
        
        return ApiResponse(
            success=True,
            message=f"Memory initialized for user {user_id}"
        )
    except Exception as e:
        logger.error(f"❌ Memory initialization failed: {str(e)}")
        # Do NOT fall back - expose the error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory initialization failed: {str(e)}"
        )

@router.post("/store")
async def store_memory(request: StoreMemoryRequest) -> ApiResponse:
    """Store a memory in the Soliton system"""
    try:
        user_id = request.userId
        memory = request.memory
        
        # Add metadata about storage source
        memory.metadata = memory.metadata or {}
        memory.metadata["storage_time"] = datetime.now().isoformat()
        memory.metadata["storage_source"] = "api"
        
        # Store in concept mesh
        memory_id = mesh.store_memory(user_id, memory)
        
        if not memory_id:
            raise Exception("Failed to store memory")
        
        # Log success with memory info
        logger.info(f"🌊 Memory stored for user {user_id}: {memory.id} (length: {len(memory.content)})")
        
        return ApiResponse(
            success=True,
            message=f"Memory stored with ID {memory_id}"
        )
    except Exception as e:
        logger.error(f"❌ Memory storage failed: {str(e)}")
        # Do NOT fall back - expose the error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory storage failed: {str(e)}"
        )

@router.post("/recall")
async def recall_memories(request: RecallRequest) -> Dict[str, Any]:
    """Find memories related to a query"""
    try:
        user_id = request.userId
        query = request.query
        limit = request.limit
        min_strength = request.minStrength
        tags = request.tags
        
        # Find related memories
        memories = mesh.find_related_memories(
            user_id, 
            query, 
            limit=limit,
            min_strength=min_strength,
            tags=tags
        )
        
        # Log recall stats
        memory_count = len(memories) if memories else 0
        logger.info(f"🌊 Recalled {memory_count} memories for user {user_id} with query: {query[:30]}...")
        
        return {
            "success": True,
            "memories": memories,
            "count": memory_count
        }
    except Exception as e:
        logger.error(f"❌ Memory recall failed: {str(e)}")
        # Do NOT fall back - expose the error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory recall failed: {str(e)}"
        )

@router.get("/stats")
async def get_memory_stats(userId: str) -> Dict[str, Any]:
    """Get statistics about a user's memory"""
    try:
        # Get stats from concept mesh
        stats = mesh.get_user_stats(userId)
        
        if not stats:
            raise Exception(f"Failed to get stats for user {userId}")
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"❌ Memory stats retrieval failed: {str(e)}")
        # Do NOT fall back - expose the error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory stats retrieval failed: {str(e)}"
        )

@router.post("/embed")
async def create_embedding(request: EmbeddingRequest) -> Dict[str, Any]:
    """Generate an embedding vector for text"""
    try:
        text = request.text
        dimension = request.dimension
        
        # Generate embedding
        embedding = mesh.generate_embedding(text, dimension)
        
        if not embedding:
            raise Exception("Failed to generate embedding")
        
        return {
            "success": True,
            "embedding": embedding,
            "dimension": len(embedding)
        }
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {str(e)}")
        # Do NOT fall back - expose the error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )

@router.post("/phase-change")
async def record_phase_change(request: PhaseChangeRequest) -> ApiResponse:
    """Record a phase change event in the Ghost AI"""
    try:
        user_id = request.userId
        phase_event = PhaseChangeEvent(
            userId=user_id,
            phase=request.phase,
            amplitude=request.amplitude,
            frequency=request.frequency,
            coherence=request.coherence,
            timestamp=datetime.now(),
            metadata=request.metadata or {}
        )
        
        # Record the phase change
        event_id = mesh.record_phase_change(user_id, phase_event)
        
        if not event_id:
            raise Exception("Failed to record phase change")
        
        logger.info(f"👻 Ghost phase change for user {user_id}: {request.phase} " 
                    f"(amplitude: {request.amplitude:.2f}, frequency: {request.frequency:.2f})")
        
        return ApiResponse(
            success=True,
            message=f"Phase change recorded with ID {event_id}"
        )
    except Exception as e:
        logger.error(f"❌ Phase change recording failed: {str(e)}")
        # Do NOT fall back - expose the error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Phase change recording failed: {str(e)}"
        )

@router.delete("/memories/{memoryId}")
async def delete_memory(memoryId: str, userId: str) -> ApiResponse:
    """Delete a specific memory"""
    try:
        # Delete from concept mesh
        success = mesh.delete_memory(userId, memoryId)
        
        if not success:
            raise Exception(f"Failed to delete memory {memoryId}")
        
        logger.info(f"🗑️ Deleted memory {memoryId} for user {userId}")
        
        return ApiResponse(
            success=True,
            message=f"Memory {memoryId} deleted"
        )
    except Exception as e:
        logger.error(f"❌ Memory deletion failed: {str(e)}")
        # Do NOT fall back - expose the error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory deletion failed: {str(e)}"
        )

@router.post("/clear")
async def clear_memories(userId: str) -> ApiResponse:
    """Clear all memories for a user (dangerous operation)"""
    try:
        # Clear all memories
        success = mesh.clear_user_memories(userId)
        
        if not success:
            raise Exception(f"Failed to clear memories for user {userId}")
        
        logger.warning(f"🧹 Cleared all memories for user {userId}")
        
        return ApiResponse(
            success=True,
            message=f"All memories cleared for user {userId}"
        )
    except Exception as e:
        logger.error(f"❌ Memory clearing failed: {str(e)}")
        # Do NOT fall back - expose the error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory clearing failed: {str(e)}"
        )

@router.get("/debug/status")
async def debug_status() -> Dict[str, Any]:
    """Get detailed status information (development/debugging only)"""
    if not is_initialized:
        return {
            "initialized": False,
            "message": "Soliton Memory system is initializing"
        }
    
    try:
        # Get detailed status from the mesh
        mesh_status = mesh.get_status()
        
        return {
            "initialized": True,
            "engine_type": "soliton" if SOLITON_AVAILABLE else "concept_mesh_direct",
            "concept_mesh_status": mesh_status,
            "api_version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"❌ Debug status check failed: {str(e)}")
        return {
            "initialized": True,
            "engine_type": "soliton" if SOLITON_AVAILABLE else "concept_mesh_direct",
            "error": str(e),
            "api_version": "1.0.0"
        }

# Startup event to ensure the engine is ready
@router.on_event("startup")
async def startup_event():
    global is_initialized
    
    try:
        # Initialize the engine connection
        if SOLITON_AVAILABLE:
            engine.initialize()
        else:
            mesh.initialize()
        
        is_initialized = True
        logger.info("🌊 Soliton Memory API routes initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Soliton Memory API: {str(e)}")
        is_initialized = False
        # Critical error - we won't fall back
        raise RuntimeError(f"Soliton Memory API initialization failed: {str(e)}")

# Shutdown event to clean up resources
@router.on_event("shutdown")
async def shutdown_event():
    try:
        if SOLITON_AVAILABLE:
            engine.shutdown()
        else:
            mesh.shutdown()
        
        logger.info("🌊 Soliton Memory API shutdown completed")
    except Exception as e:
        logger.error(f"❌ Error during Soliton Memory API shutdown: {str(e)}")
