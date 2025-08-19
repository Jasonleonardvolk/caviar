"""
🧠 TORI Cognitive Engine FastAPI Bridge
Python FastAPI service that bridges to the Node.js cognitive microservice
Provides seamless Python integration with the full TORI cognitive system
"""

import asyncio
import httpx
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
COGNITIVE_MICROSERVICE_URL = "http://localhost:4321"
API_TIMEOUT = 30.0

# ===== PYDANTIC MODELS =====

class CognitiveRequest(BaseModel):
    """Request model for cognitive processing"""
    message: str = Field(..., description="The message/prompt to process")
    glyphs: List[str] = Field(..., description="Array of symbolic glyphs to execute")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")

class BatchCognitiveRequest(BaseModel):
    """Request model for batch cognitive processing"""
    requests: List[CognitiveRequest] = Field(..., description="Array of cognitive requests")

class ConceptRequest(BaseModel):
    """Request model for creating concepts"""
    essence: str = Field(..., description="The essence/name of the concept")
    activationLevel: float = Field(0.5, ge=0.0, le=1.0, description="Initial activation level")

class GhostQueryRequest(BaseModel):
    """Request model for ghost collective queries"""
    query: str = Field(..., description="Query to select appropriate persona")

class CognitiveTrace(BaseModel):
    """Model for cognitive processing trace"""
    loopId: str
    prompt: str
    glyphPath: List[str]
    closed: bool
    scarFlag: bool
    processingTime: int
    coherenceTrace: List[float]
    contradictionTrace: List[float]
    phaseTrace: List[float]
    metadata: Optional[Dict[str, Any]] = None

class CognitiveResponse(BaseModel):
    """Response model for cognitive processing"""
    success: bool
    answer: str
    trace: CognitiveTrace
    fullLoop: Optional[Dict[str, Any]] = None
    timestamp: str
    cognitive: Optional[Dict[str, Any]] = None

# ===== GLOBAL HTTP CLIENT =====

# Global HTTP client for connection pooling
http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with proper HTTP client handling"""
    global http_client
    
    # Startup
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(API_TIMEOUT),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )
    
    # Check cognitive microservice health
    try:
        response = await http_client.get(f"{COGNITIVE_MICROSERVICE_URL}/api/health")
        if response.status_code == 200:
            logger.info("✅ Connected to TORI Cognitive Microservice")
        else:
            logger.warning("⚠️ Cognitive microservice health check failed")
    except Exception as e:
        logger.error(f"❌ Failed to connect to cognitive microservice: {e}")
    
    yield
    
    # Shutdown
    if http_client:
        await http_client.aclose()

# ===== FASTAPI APP SETUP =====

app = FastAPI(
    title="TORI Cognitive Engine FastAPI Bridge",
    description="Python FastAPI bridge to the Node.js TORI Cognitive Engine microservice",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== UTILITY FUNCTIONS =====

async def call_cognitive_service(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
    """Call the cognitive microservice with proper error handling"""
    if not http_client:
        raise HTTPException(status_code=503, detail="HTTP client not initialized")
    
    url = f"{COGNITIVE_MICROSERVICE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = await http_client.get(url)
        elif method.upper() == "POST":
            response = await http_client.post(url, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Cognitive service error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Cognitive service error: {response.text}"
            )
            
    except httpx.RequestError as e:
        logger.error(f"Request to cognitive service failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Cognitive microservice unavailable"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from cognitive service: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Cognitive service error: {e.response.text}"
        )

def generate_smart_glyph_sequence(message: str, complexity: str = "standard") -> List[str]:
    """Generate intelligent glyph sequences based on message content and complexity"""
    
    # Base sequence
    glyphs = ["anchor"]
    
    # Analyze message content
    message_lower = message.lower()
    
    # Content-based glyph selection
    if any(word in message_lower for word in ["analyze", "compare", "evaluate", "assess"]):
        glyphs.extend(["concept-synthesizer", "paradox-analyzer"])
    
    if any(word in message_lower for word in ["problem", "issue", "error", "bug"]):
        glyphs.extend(["scar-repair", "contradiction-resolver"])
    
    if any(word in message_lower for word in ["creative", "innovative", "new", "novel"]):
        glyphs.extend(["novelty-detector", "intent-bifurcation"])
    
    if any(word in message_lower for word in ["memory", "remember", "recall", "past"]):
        glyphs.extend(["memory-anchor", "braid-weaver"])
    
    if any(word in message_lower for word in ["explain", "understand", "clarify"]):
        glyphs.extend(["meta-echo:reflect", "coherence-boost"])
    
    # Complexity-based additions
    if complexity == "simple":
        pass  # Keep minimal
    elif complexity == "complex":
        glyphs.extend(["phase-drift", "holographic-projection"])
    elif complexity == "research":
        glyphs.extend(["deep-analysis", "concept-clustering", "emergence-detection"])
    
    # Always end with return
    glyphs.append("return")
    
    # Remove duplicates while preserving order
    unique_glyphs = []
    for glyph in glyphs:
        if glyph not in unique_glyphs:
            unique_glyphs.append(glyph)
    
    return unique_glyphs

# ===== MAIN COGNITIVE ENDPOINTS =====

@app.post("/api/chat", response_model=CognitiveResponse)
async def chat_endpoint(request: CognitiveRequest):
    """
    Main chat endpoint that processes messages through the cognitive engine
    This is the primary endpoint for FastAPI -> Node.js cognitive processing
    """
    logger.info(f"🧠 Processing chat request: {request.message[:100]}...")
    
    # Use provided glyphs or generate smart sequence
    glyphs = request.glyphs
    if not glyphs or len(glyphs) == 0:
        glyphs = generate_smart_glyph_sequence(request.message)
        logger.info(f"🎯 Generated glyph sequence: {glyphs}")
    
    # Prepare data for cognitive microservice
    data = {
        "message": request.message,
        "glyphs": glyphs,
        "metadata": request.metadata or {}
    }
    
    # Call cognitive microservice
    result = await call_cognitive_service("/api/engine", "POST", data)
    
    # Return structured response
    return CognitiveResponse(**result)

@app.post("/api/cognitive/process")
async def process_cognitive(request: CognitiveRequest):
    """Direct cognitive processing endpoint with full control"""
    logger.info(f"🧠 Direct cognitive processing: {request.message}")
    
    data = {
        "message": request.message,
        "glyphs": request.glyphs,
        "metadata": request.metadata or {}
    }
    
    result = await call_cognitive_service("/api/engine", "POST", data)
    return result

@app.post("/api/cognitive/batch")
async def batch_cognitive_processing(request: BatchCognitiveRequest):
    """Batch cognitive processing for multiple requests"""
    logger.info(f"🧠 Batch processing {len(request.requests)} requests")
    
    # Convert to format expected by microservice
    requests_data = []
    for req in request.requests:
        requests_data.append({
            "message": req.message,
            "glyphs": req.glyphs,
            "metadata": req.metadata or {}
        })
    
    data = {"requests": requests_data}
    result = await call_cognitive_service("/api/engine/batch", "POST", data)
    return result

@app.post("/api/cognitive/stream")
async def stream_cognitive_processing(request: CognitiveRequest):
    """Stream cognitive processing with real-time updates"""
    
    async def generate_stream():
        data = {
            "message": request.message,
            "glyphs": request.glyphs,
            "metadata": request.metadata or {}
        }
        
        url = f"{COGNITIVE_MICROSERVICE_URL}/api/engine/stream"
        
        async with http_client.stream("POST", url, json=data) as response:
            async for chunk in response.aiter_text():
                if chunk.strip():
                    yield f"data: {chunk}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

# ===== SMART HELPER ENDPOINTS =====

@app.post("/api/smart/ask")
async def smart_ask(request: dict):
    """
    Smart ask endpoint that automatically selects the best processing approach
    """
    message = request.get("message", "")
    complexity = request.get("complexity", "standard")  # simple, standard, complex, research
    
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    # Generate optimal glyph sequence
    glyphs = generate_smart_glyph_sequence(message, complexity)
    
    # Add metadata about the smart processing
    metadata = {
        "smartProcessing": True,
        "autoGeneratedGlyphs": True,
        "complexity": complexity,
        "processedBy": "smart_ask_endpoint"
    }
    
    # Process through cognitive engine
    data = {
        "message": message,
        "glyphs": glyphs,
        "metadata": metadata
    }
    
    result = await call_cognitive_service("/api/engine", "POST", data)
    return result

@app.post("/api/smart/research")
async def smart_research(request: dict):
    """
    Advanced research endpoint with comprehensive cognitive processing
    """
    query = request.get("query", "")
    depth = request.get("depth", "standard")  # shallow, standard, deep
    
    if not query:
        raise HTTPException(status_code=400, detail="Research query is required")
    
    # Research-specific glyph sequences
    if depth == "shallow":
        glyphs = ["anchor", "concept-synthesizer", "return"]
    elif depth == "deep":
        glyphs = [
            "anchor", "concept-synthesizer", "deep-analysis", 
            "paradox-analyzer", "memory-anchor", "braid-weaver",
            "holographic-projection", "emergence-detection", 
            "meta-echo:reflect", "return"
        ]
    else:  # standard
        glyphs = [
            "anchor", "concept-synthesizer", "paradox-analyzer",
            "memory-anchor", "meta-echo:reflect", "return"
        ]
    
    metadata = {
        "researchMode": True,
        "depth": depth,
        "processedBy": "smart_research_endpoint"
    }
    
    data = {
        "message": f"Research Query: {query}",
        "glyphs": glyphs,
        "metadata": metadata
    }
    
    result = await call_cognitive_service("/api/engine", "POST", data)
    return result

# ===== MEMORY AND CONCEPT ENDPOINTS =====

@app.get("/api/memory/query")
async def query_memory(digest: Optional[str] = None, limit: int = 10):
    """Query braid memory system"""
    params = f"?digest={digest}&limit={limit}" if digest else f"?limit={limit}"
    result = await call_cognitive_service(f"/api/memory/query{params}")
    return result

@app.get("/api/memory/holographic")
async def get_holographic_memory():
    """Get holographic memory visualization data"""
    result = await call_cognitive_service("/api/memory/holographic")
    return result

@app.post("/api/memory/concept")
async def create_concept(request: ConceptRequest):
    """Create a new concept in holographic memory"""
    data = {
        "essence": request.essence,
        "activationLevel": request.activationLevel
    }
    result = await call_cognitive_service("/api/memory/concept", "POST", data)
    return result

# ===== GHOST COLLECTIVE ENDPOINTS =====

@app.post("/api/ghosts/query")
async def query_ghost_collective(request: GhostQueryRequest):
    """Query ghost collective for persona selection"""
    data = {"query": request.query}
    result = await call_cognitive_service("/api/ghosts/query", "POST", data)
    return result

@app.get("/api/ghosts/personas")
async def get_personas():
    """Get all available personas from ghost collective"""
    result = await call_cognitive_service("/api/ghosts/personas")
    return result

# ===== STATUS AND DIAGNOSTICS =====

@app.get("/api/status")
async def get_status():
    """Get comprehensive system status"""
    try:
        # Get status from cognitive microservice
        cognitive_status = await call_cognitive_service("/api/status")
        
        # Add FastAPI bridge status
        bridge_status = {
            "bridge": {
                "service": "TORI Cognitive FastAPI Bridge",
                "version": "1.0.0",
                "status": "online",
                "timestamp": datetime.now().isoformat(),
                "httpClientStatus": "connected" if http_client else "disconnected"
            },
            "cognitive": cognitive_status
        }
        
        return bridge_status
        
    except Exception as e:
        return {
            "bridge": {
                "service": "TORI Cognitive FastAPI Bridge",
                "status": "online",
                "cognitiveServiceStatus": "disconnected",
                "error": str(e)
            }
        }

@app.get("/api/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "TORI Cognitive FastAPI Bridge"
    }

@app.get("/api/metrics")
async def get_metrics():
    """Get comprehensive system metrics"""
    try:
        cognitive_metrics = await call_cognitive_service("/api/metrics")
        return {
            "bridge": {
                "timestamp": datetime.now().isoformat(),
                "httpClientConnected": http_client is not None
            },
            "cognitive": cognitive_metrics
        }
    except Exception as e:
        return {
            "bridge": {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        }

# ===== UTILITY ENDPOINTS =====

@app.get("/api/glyph-suggestions")
async def get_glyph_suggestions(message: str, complexity: str = "standard"):
    """Get suggested glyph sequences for a message"""
    glyphs = generate_smart_glyph_sequence(message, complexity)
    
    return {
        "message": message,
        "complexity": complexity,
        "suggestedGlyphs": glyphs,
        "description": "Auto-generated glyph sequence based on message analysis",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "TORI Cognitive Engine FastAPI Bridge",
        "version": "1.0.0",
        "description": "Python FastAPI bridge to Node.js TORI Cognitive Engine",
        "endpoints": {
            "main": "/api/chat",
            "smart": "/api/smart/ask",
            "research": "/api/smart/research",
            "status": "/api/status",
            "health": "/api/health",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "cognitive_bridge:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
