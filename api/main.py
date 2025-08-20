"""
TORI API Main Module
FastAPI server with health checks and core endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="TORI API",
    version="3.0.0",
    description="Enhanced TORI Backend API with Bulletproof Architecture"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get ports from environment
API_PORT = int(os.getenv("API_PORT", "8002"))
MCP_PORT = int(os.getenv("MCP_PORT", "6660"))
BRIDGE_AUDIO_PORT = int(os.getenv("BRIDGE_AUDIO_PORT", "8501"))
BRIDGE_CONCEPT_MESH_PORT = int(os.getenv("BRIDGE_CONCEPT_MESH_PORT", "8502"))


# Models
class HealthStatus(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]
    
class SolitonStats(BaseModel):
    admin_user: str
    active_connections: int
    total_requests: int
    uptime_seconds: float
    memory_usage_mb: float
    cpu_percent: float
    
class InferenceRequest(BaseModel):
    text: str
    model: Optional[str] = "saigon-v5"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    
class InferenceResponse(BaseModel):
    text: str
    model: str
    tokens_used: int
    processing_time_ms: float


# Global state
START_TIME = datetime.now()
REQUEST_COUNT = 0


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "TORI API v3.0 - Enhanced Bulletproof Edition",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "stats": "/api/soliton/stats/adminuser",
            "inference": "/api/inference"
        }
    }


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint for service monitoring"""
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    
    # Check service availability
    services = {
        "api": "healthy",
        "database": "healthy",  # Placeholder
        "cache": "healthy",     # Placeholder
        "mcp": "unknown",       # Would check MCP server
        "bridges": "unknown"    # Would check bridge services
    }
    
    # Try to check if MCP is responsive (simplified)
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', MCP_PORT))
        sock.close()
        services["mcp"] = "healthy" if result == 0 else "unreachable"
    except:
        services["mcp"] = "error"
    
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="3.0.0",
        services=services
    )


@app.get("/api/soliton/stats/adminuser", response_model=SolitonStats)
async def soliton_stats():
    """Soliton admin stats endpoint (for compatibility)"""
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    
    uptime = (datetime.now() - START_TIME).total_seconds()
    
    # Get system stats if psutil available
    memory_mb = 0.0
    cpu_percent = 0.0
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
    except:
        pass
    
    return SolitonStats(
        admin_user="tori_admin",
        active_connections=1,  # Placeholder
        total_requests=REQUEST_COUNT,
        uptime_seconds=uptime,
        memory_usage_mb=memory_mb,
        cpu_percent=cpu_percent
    )


@app.post("/api/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Main inference endpoint"""
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    
    import time
    start_time = time.time()
    
    # Placeholder inference logic
    # In production, this would call the actual Saigon/TORI inference engine
    
    try:
        # Simulate processing
        time.sleep(0.1)  # Simulate processing time
        
        # Generate response (placeholder)
        response_text = f"Processed: {request.text[:100]}..."
        
        processing_time = (time.time() - start_time) * 1000
        
        return InferenceResponse(
            text=response_text,
            model=request.model,
            tokens_used=len(request.text.split()),
            processing_time_ms=processing_time
        )
    except Exception as e:
        LOG.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "api_port": API_PORT,
        "mcp_port": MCP_PORT,
        "bridge_audio_port": BRIDGE_AUDIO_PORT,
        "bridge_concept_mesh_port": BRIDGE_CONCEPT_MESH_PORT,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "debug": os.getenv("DEBUG", "false").lower() == "true"
    }


@app.post("/api/shutdown")
async def shutdown():
    """Graceful shutdown endpoint (protected in production)"""
    # In production, this should be protected with authentication
    LOG.warning("Shutdown requested via API")
    
    # Trigger graceful shutdown
    import signal
    import threading
    
    def trigger_shutdown():
        import time
        time.sleep(1)  # Give time for response
        os.kill(os.getpid(), signal.SIGTERM)
    
    threading.Thread(target=trigger_shutdown).start()
    
    return {"message": "Shutting down..."}


# WebSocket support (optional)
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back or process
            await websocket.send_text(f"Echo: {data}")
            # Broadcast to all clients
            await manager.broadcast(f"Client said: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    LOG.info("TORI API starting up...")
    LOG.info(f"API Port: {API_PORT}")
    LOG.info(f"MCP Port: {MCP_PORT}")
    LOG.info(f"Bridge Audio Port: {BRIDGE_AUDIO_PORT}")
    LOG.info(f"Bridge Concept Mesh Port: {BRIDGE_CONCEPT_MESH_PORT}")
    
    # Initialize services here
    # - Database connections
    # - Cache connections
    # - Load models
    # - etc.
    
    LOG.info("TORI API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    LOG.info("TORI API shutting down...")
    
    # Cleanup here
    # - Close database connections
    # - Save state
    # - Close cache connections
    # - etc.
    
    LOG.info("TORI API shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or command line
    port = int(sys.argv[1]) if len(sys.argv) > 1 else API_PORT
    
    LOG.info(f"Starting TORI API on port {port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
