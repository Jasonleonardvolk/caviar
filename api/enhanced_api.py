#!/usr/bin/env python3
"""
TORI/KHA Enhanced API Layer
Adds /multiply, /intent, WebSocket support, and integrates GraphQL
Designed to work with the existing prajna_api.py
"""

import asyncio
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, AsyncIterator
from pathlib import Path
import sys

from fastapi import WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import strawberry
from api.routes.soliton import router as soliton_router
from api.routes.concept_mesh import router as concept_mesh_router

# Import log rotation configuration
try:
    from config.log_rotation import setup_log_rotation
    setup_log_rotation()
except ImportError:
    logging.warning("Log rotation not available")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import TORI components
try:
    from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply
    from python.core.intent_driven_reasoning import IntentDrivenReasoning
    from python.core.eigenvalue_monitor import EigenvalueMonitor
    from python.core.chaos_control_layer import ChaosControlLayer
    from python.core.CognitiveEngine import CognitiveEngine
    CORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core components not available: {e}")
    CORE_AVAILABLE = False

# Import existing Prajna API app to extend it
try:
    from prajna.api.prajna_api import app
    PRAJNA_APP_AVAILABLE = True
except ImportError:
    logging.warning("Prajna API app not available, creating new FastAPI instance")
    from fastapi import FastAPI
    app = FastAPI(
        title="TORI Enhanced API",
        description="Extended API with /multiply, /intent, and WebSocket support",
        version="1.0.0"
    )
    PRAJNA_APP_AVAILABLE = False


# Include routers
app.include_router(soliton_router)
app.include_router(concept_mesh_router)
logger = logging.getLogger(__name__)

# ========== Request/Response Models ==========

class MatrixMultiplyRequest(BaseModel):
    """Request for hyperbolic matrix multiplication"""
    matrix_a: List[List[float]] = Field(..., description="First matrix")
    matrix_b: List[List[float]] = Field(..., description="Second matrix")
    curvature: float = Field(default=-1.0, description="Hyperbolic curvature")
    precision: int = Field(default=100, description="Precision for computation")

class MatrixMultiplyResponse(BaseModel):
    """Response from matrix multiplication"""
    result: List[List[float]]
    computation_time: float
    curvature_used: float
    eigenvalues: Optional[List[complex]] = None
    is_stable: bool = True

class IntentRequest(BaseModel):
    """Request for intent-driven reasoning"""
    query: str = Field(..., description="User query")
    context: Optional[Dict[str, Any]] = None
    max_reasoning_depth: int = Field(default=3, description="Maximum reasoning depth")
    enable_chaos: bool = Field(default=False, description="Enable chaos-enhanced reasoning")

class IntentResponse(BaseModel):
    """Response from intent reasoning"""
    intent: str
    confidence: float
    reasoning_path: List[str]
    response: str
    processing_time: float
    chaos_used: bool = False

class SystemHealthResponse(BaseModel):
    """System health status"""
    status: str
    components: Dict[str, bool]
    metrics: Dict[str, float]
    timestamp: str

# ========== WebSocket Manager ==========

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.stability_subscribers: List[WebSocket] = []
        self.chaos_subscribers: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket, channel: str = "general"):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if channel == "stability":
            self.stability_subscribers.append(websocket)
        elif channel == "chaos":
            self.chaos_subscribers.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.stability_subscribers:
            self.stability_subscribers.remove(websocket)
        if websocket in self.chaos_subscribers:
            self.chaos_subscribers.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str, channel: str = "general"):
        if channel == "stability":
            connections = self.stability_subscribers
        elif channel == "chaos":
            connections = self.chaos_subscribers
        else:
            connections = self.active_connections
        
        for connection in connections:
            try:
                await connection.send_text(message)
            except:
                # Connection might be closed
                pass

manager = ConnectionManager()

# ========== Global Components ==========

# Initialize components if available
eigenvalue_monitor = None
intent_reasoner = None
chaos_controller = None

if CORE_AVAILABLE:
    try:
        eigenvalue_monitor = EigenvalueMonitor()
        intent_reasoner = IntentDrivenReasoning()
        logger.info("✅ Core components initialized")
    except Exception as e:
        logger.error(f"Failed to initialize core components: {e}")

# ========== API Endpoints ==========

@app.post("/multiply", response_model=MatrixMultiplyResponse)
async def multiply_matrices(request: MatrixMultiplyRequest):
    """
    Perform hyperbolic matrix multiplication
    
    This endpoint multiplies two matrices in hyperbolic space with the specified curvature.
    """
    start_time = time.time()
    
    try:
        # Convert to numpy arrays
        A = np.array(request.matrix_a)
        B = np.array(request.matrix_b)
        
        # Validate matrix dimensions
        if A.shape[1] != B.shape[0]:
            raise HTTPException(
                status_code=400,
                detail=f"Matrix dimensions incompatible: {A.shape} and {B.shape}"
            )
        
        # Perform hyperbolic multiplication
        if CORE_AVAILABLE and hyperbolic_matrix_multiply:
            result = hyperbolic_matrix_multiply(A, B, curvature=request.curvature)
        else:
            # Fallback to standard multiplication
            result = A @ B
            logger.warning("Using standard matrix multiplication (hyperbolic not available)")
        
        # Compute eigenvalues for stability analysis
        eigenvalues = None
        is_stable = True
        
        if eigenvalue_monitor:
            try:
                analysis = await eigenvalue_monitor.analyze_matrix(result)
                eigenvalues = analysis.eigenvalues.tolist()
                is_stable = analysis.is_stable
            except:
                pass
        
        computation_time = time.time() - start_time
        
        # Broadcast stability update if unstable
        if not is_stable:
            await manager.broadcast(
                json.dumps({
                    "type": "stability_warning",
                    "max_eigenvalue": max(abs(e) for e in eigenvalues) if eigenvalues else 0,
                    "timestamp": datetime.now().isoformat()
                }),
                channel="stability"
            )
        
        return MatrixMultiplyResponse(
            result=result.tolist(),
            computation_time=computation_time,
            curvature_used=request.curvature,
            eigenvalues=eigenvalues,
            is_stable=is_stable
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Matrix multiplication failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intent", response_model=IntentResponse)
async def process_intent(request: IntentRequest):
    """
    Process query through intent-driven reasoning
    
    This endpoint analyzes the user's intent and provides a reasoned response.
    """
    start_time = time.time()
    
    try:
        if not CORE_AVAILABLE or not intent_reasoner:
            # Fallback response
            return IntentResponse(
                intent="unknown",
                confidence=0.5,
                reasoning_path=["Intent reasoning not available"],
                response="Intent-driven reasoning is not available. Please use the standard /api/answer endpoint.",
                processing_time=time.time() - start_time,
                chaos_used=False
            )
        
        # Process through intent reasoning
        result = await intent_reasoner.process(
            query=request.query,
            context=request.context or {},
            max_depth=request.max_reasoning_depth,
            enable_chaos=request.enable_chaos
        )
        
        processing_time = time.time() - start_time
        
        # Broadcast intent event
        await manager.broadcast(
            json.dumps({
                "type": "intent_processed",
                "intent": result.intent,
                "confidence": result.confidence,
                "timestamp": datetime.now().isoformat()
            }),
            channel="general"
        )
        
        return IntentResponse(
            intent=result.intent,
            confidence=result.confidence,
            reasoning_path=result.reasoning_path,
            response=result.response,
            processing_time=processing_time,
            chaos_used=request.enable_chaos
        )
        
    except Exception as e:
        logger.error(f"Intent processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/extended", response_model=SystemHealthResponse)
async def extended_health_check():
    """
    Extended health check with component status
    """
    components = {
        "core_available": CORE_AVAILABLE,
        "eigenvalue_monitor": eigenvalue_monitor is not None,
        "intent_reasoner": intent_reasoner is not None,
        "chaos_controller": chaos_controller is not None,
        "websocket_manager": True,
        "prajna_integrated": PRAJNA_APP_AVAILABLE
    }
    
    metrics = {}
    
    # Get eigenvalue metrics if available
    if eigenvalue_monitor:
        try:
            status = eigenvalue_monitor.get_stability_metrics()
            if status.get('has_data'):
                metrics['max_eigenvalue'] = status['current_analysis']['max_eigenvalue']
                metrics['stability_score'] = status['current_analysis']['stability_score']
        except:
            pass
    
    # Get active connections
    metrics['active_websockets'] = len(manager.active_connections)
    metrics['stability_subscribers'] = len(manager.stability_subscribers)
    metrics['chaos_subscribers'] = len(manager.chaos_subscribers)
    
    return SystemHealthResponse(
        status="healthy" if any(components.values()) else "degraded",
        components=components,
        metrics=metrics,
        timestamp=datetime.now().isoformat()
    )

# ========== WebSocket Endpoints ==========

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    General WebSocket endpoint for real-time updates
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now
            await manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/stability")
async def stability_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time stability updates
    """
    await manager.connect(websocket, channel="stability")
    
    try:
        # Send initial status
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "channel": "stability",
            "timestamp": datetime.now().isoformat()
        }))
        
        # If eigenvalue monitor available, start monitoring
        if eigenvalue_monitor:
            async def send_stability_updates():
                while True:
                    try:
                        status = eigenvalue_monitor.get_stability_metrics()
                        if status.get('has_data'):
                            await websocket.send_text(json.dumps({
                                "type": "stability_update",
                                "max_eigenvalue": status['current_analysis']['max_eigenvalue'],
                                "is_stable": status['current_analysis']['is_stable'],
                                "stability_score": status['current_analysis']['stability_score'],
                                "timestamp": datetime.now().isoformat()
                            }))
                    except:
                        break
                    await asyncio.sleep(1.0)  # Update every second
            
            # Start monitoring task
            monitor_task = asyncio.create_task(send_stability_updates())
            
            # Keep connection alive
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        else:
            # Just keep connection alive without monitoring
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        if 'monitor_task' in locals():
            monitor_task.cancel()

@app.websocket("/ws/chaos")
async def chaos_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for chaos events
    """
    await manager.connect(websocket, channel="chaos")
    
    try:
        # Send initial status
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "channel": "chaos",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Keep connection alive and handle chaos events
        while True:
            data = await websocket.receive_text()
            # Could trigger chaos events based on client requests
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ========== HTML Test Page ==========

@app.get("/ws/test")
async def websocket_test_page():
    """
    Simple HTML page to test WebSocket connections
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TORI WebSocket Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .channel { border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
            .log { height: 200px; overflow-y: auto; border: 1px solid #eee; padding: 5px; }
            button { margin: 5px; }
        </style>
    </head>
    <body>
        <h1>TORI WebSocket Test</h1>
        
        <div class="channel">
            <h2>Stability Channel</h2>
            <button onclick="connectStability()">Connect</button>
            <button onclick="disconnectStability()">Disconnect</button>
            <div id="stability-log" class="log"></div>
        </div>
        
        <div class="channel">
            <h2>Chaos Channel</h2>
            <button onclick="connectChaos()">Connect</button>
            <button onclick="disconnectChaos()">Disconnect</button>
            <div id="chaos-log" class="log"></div>
        </div>
        
        <script>
            let stabilityWs = null;
            let chaosWs = null;
            
            function log(channel, message) {
                const logDiv = document.getElementById(channel + '-log');
                const entry = document.createElement('div');
                entry.textContent = new Date().toLocaleTimeString() + ': ' + message;
                logDiv.appendChild(entry);
                logDiv.scrollTop = logDiv.scrollHeight;
            }
            
            function connectStability() {
                if (stabilityWs) return;
                
                stabilityWs = new WebSocket('ws://localhost:8002/ws/stability');
                
                stabilityWs.onopen = () => log('stability', 'Connected');
                stabilityWs.onmessage = (e) => log('stability', e.data);
                stabilityWs.onerror = (e) => log('stability', 'Error: ' + e);
                stabilityWs.onclose = () => {
                    log('stability', 'Disconnected');
                    stabilityWs = null;
                };
                
                // Send ping every 5 seconds
                setInterval(() => {
                    if (stabilityWs && stabilityWs.readyState === WebSocket.OPEN) {
                        stabilityWs.send('ping');
                    }
                }, 5000);
            }
            
            function disconnectStability() {
                if (stabilityWs) {
                    stabilityWs.close();
                }
            }
            
            function connectChaos() {
                if (chaosWs) return;
                
                chaosWs = new WebSocket('ws://localhost:8002/ws/chaos');
                
                chaosWs.onopen = () => log('chaos', 'Connected');
                chaosWs.onmessage = (e) => log('chaos', e.data);
                chaosWs.onerror = (e) => log('chaos', 'Error: ' + e);
                chaosWs.onclose = () => {
                    log('chaos', 'Disconnected');
                    chaosWs = null;
                };
            }
            
            function disconnectChaos() {
                if (chaosWs) {
                    chaosWs.close();
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# ========== GraphQL Integration ==========

@app.get("/graphql/status")
async def graphql_status():
    """
    Check GraphQL availability
    """
    try:
        from python.core.graphql_api import GRAPHQL_AVAILABLE
        if GRAPHQL_AVAILABLE:
            return {
                "graphql_available": True,
                "endpoint": "/graphql",
                "message": "GraphQL is available. Use a separate process to run the GraphQL server."
            }
    except:
        pass
    
    return {
        "graphql_available": False,
        "message": "GraphQL not available. Install strawberry-graphql: pip install strawberry-graphql"
    }

# ========== Startup Event ==========

@app.on_event("startup")
async def startup_enhanced_api():
    """
    Initialize enhanced API components
    """
    logger.info("🚀 Starting TORI Enhanced API Layer")
    logger.info("✅ Added endpoints: /multiply, /intent")
    logger.info("✅ WebSocket channels: /ws/stability, /ws/chaos")
    logger.info("📊 WebSocket test page: /ws/test")
    logger.info("🔗 GraphQL status: /graphql/status")
    
    # Start background tasks if components available
    if eigenvalue_monitor:
        logger.info("✅ Eigenvalue monitoring active")
    
    if intent_reasoner:
        logger.info("✅ Intent reasoning active")
    
    logger.info("🎯 Enhanced API ready!")

# ========== Module Info ==========

if __name__ == "__main__":
    print("""
    TORI Enhanced API Layer
    ======================
    
    This module enhances the existing Prajna API with:
    
    1. /multiply - Hyperbolic matrix multiplication
    2. /intent - Intent-driven reasoning
    3. WebSocket support for real-time updates
    4. Integration with GraphQL (separate server)
    
    To use:
    1. This module is automatically loaded when using enhanced_launcher.py
    2. Access endpoints at http://localhost:8002/multiply, /intent, etc.
    3. Test WebSockets at http://localhost:8002/ws/test
    4. For GraphQL, run: python python/core/graphql_api.py
    
    The enhanced API integrates seamlessly with the existing Prajna API.
    """)
