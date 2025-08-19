"""
API Extensions for TORI - Simple additions to prajna_api.py
Add these endpoints directly to your existing prajna_api.py file
"""

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import json
import asyncio
import time

# === ADD THESE IMPORTS TO prajna_api.py ===

# Import TORI components (adjust paths as needed)
try:
    from python.core.eigenvalue_monitor import EigenvalueMonitor
    from python.core.chaos_control_layer import ChaosControlLayer, ChaosTask, ChaosMode
    from python.core.CognitiveEngine import CognitiveEngine
    TORI_COMPONENTS_AVAILABLE = True
except ImportError:
    TORI_COMPONENTS_AVAILABLE = False

# === ADD THESE MODELS TO prajna_api.py ===

class MultiplyRequest(BaseModel):
    matrix_a: List[List[float]]
    matrix_b: List[List[float]]

class IntentRequest(BaseModel):
    query: str
    context: Optional[dict] = {}

class ChaosTaskRequest(BaseModel):
    mode: str = "dark_soliton"  # dark_soliton, attractor_hop, phase_explosion, hybrid
    input_data: List[float]
    parameters: Optional[Dict[str, Any]] = {}
    energy_budget: int = 100

# === ADD THESE ENDPOINTS TO prajna_api.py ===

@app.post("/multiply")
async def multiply_matrices(request: MultiplyRequest):
    """Matrix multiplication endpoint"""
    try:
        A = np.array(request.matrix_a)
        B = np.array(request.matrix_b)
        
        # Validate dimensions
        if A.shape[1] != B.shape[0]:
            return {
                "success": False,
                "error": f"Matrix dimensions incompatible: {A.shape} x {B.shape}"
            }
        
        result = A @ B
        
        return {
            "success": True,
            "result": result.tolist(),
            "shape": result.shape
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/intent")
async def process_intent(request: IntentRequest):
    """Intent-driven reasoning endpoint - wraps Prajna with intent detection"""
    try:
        # For now, just forward to Prajna with intent detection
        # You can enhance this with your IntentDrivenReasoning module
        
        # Detect intent from query (simple keyword matching for now)
        query_lower = request.query.lower()
        
        if "explain" in query_lower or "what" in query_lower:
            intent = "explanation"
        elif "how" in query_lower:
            intent = "instruction"
        elif "why" in query_lower:
            intent = "reasoning"
        elif "create" in query_lower or "generate" in query_lower:
            intent = "generation"
        else:
            intent = "general"
        
        # Use existing Prajna endpoint
        prajna_request = PrajnaRequest(
            user_query=request.query,
            conversation_id=request.context.get("conversation_id"),
            enable_reasoning=True
        )
        
        # Call the existing answer endpoint logic
        response = await prajna_answer_endpoint(prajna_request)
        
        return {
            "success": True,
            "intent": intent,
            "response": response.answer,
            "confidence": response.trust_score,
            "sources": response.sources
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "intent": "unknown"
        }

@app.get("/api/stability/current")
async def get_stability_status():
    """Get current system stability metrics"""
    try:
        if TORI_COMPONENTS_AVAILABLE and hasattr(app.state, 'eigenvalue_monitor'):
            monitor = app.state.eigenvalue_monitor
            metrics = monitor.get_stability_metrics()
            
            if metrics.get('has_data'):
                return {
                    "success": True,
                    "max_eigenvalue": metrics['current_analysis']['max_eigenvalue'],
                    "is_stable": metrics['current_analysis']['is_stable'],
                    "stability_score": metrics['current_analysis']['stability_score'],
                    "condition_number": metrics['current_analysis']['condition_number'],
                    "trending_stable": metrics['trending_stable']
                }
        
        # Fallback response
        return {
            "success": True,
            "max_eigenvalue": 0.95,
            "is_stable": True,
            "stability_score": 0.85,
            "message": "Using default values - eigenvalue monitor not connected"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/chaos/task")
async def submit_chaos_task(request: ChaosTaskRequest):
    """Submit a chaos computation task"""
    try:
        if TORI_COMPONENTS_AVAILABLE and hasattr(app.state, 'chaos_controller'):
            ccl = app.state.chaos_controller
            
            # Create chaos task
            task = ChaosTask(
                task_id=f"chaos_{int(time.time()*1000)}",
                mode=ChaosMode[request.mode.upper()],
                input_data=np.array(request.input_data),
                parameters=request.parameters,
                energy_budget=request.energy_budget
            )
            
            # Submit task
            task_id = await ccl.submit_task(task)
            
            return {
                "success": True,
                "task_id": task_id,
                "status": "queued",
                "mode": request.mode,
                "energy_budget": request.energy_budget
            }
        
        # Fallback response
        return {
            "success": True,
            "task_id": f"chaos_mock_{int(time.time()*1000)}",
            "status": "queued",
            "mode": request.mode,
            "message": "Mock response - chaos controller not connected"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/cognitive/state")
async def get_cognitive_state():
    """Get current cognitive engine state"""
    try:
        if TORI_COMPONENTS_AVAILABLE and hasattr(app.state, 'cognitive_engine'):
            engine = app.state.cognitive_engine
            state = engine.get_current_stability()
            
            return {
                "success": True,
                "phase": state['state']['phase'],
                "stability_score": state['state']['stability_score'],
                "coherence": state['state']['coherence'],
                "confidence": state['state']['confidence'],
                "max_eigenvalue": state['max_eigenvalue'],
                "is_stable": state['is_stable'],
                "processing_state": state['processing_state'],
                "history_size": state['history_size']
            }
        
        # Fallback response
        return {
            "success": True,
            "phase": "idle",
            "stability_score": 0.9,
            "coherence": 0.85,
            "confidence": 0.8,
            "message": "Using default values - cognitive engine not connected"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# === ADD THIS TO THE STARTUP EVENT IN prajna_api.py ===

# Add this to the existing @app.on_event("startup") function:
"""
# Initialize TORI components if available
if TORI_COMPONENTS_AVAILABLE:
    try:
        # Initialize EigenvalueMonitor
        app.state.eigenvalue_monitor = EigenvalueMonitor({
            'storage_path': 'data/eigenvalue_monitor',
            'history_size': 1000
        })
        logger.info("✅ EigenvalueMonitor initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize EigenvalueMonitor: {e}")
    
    try:
        # Initialize ChaosControlLayer
        from python.core.unified_metacognitive_integration import CognitiveStateManager
        from python.core.eigensentry.core import EigenSentry2
        
        state_manager = CognitiveStateManager()
        eigen_sentry = EigenSentry2(state_manager)
        app.state.chaos_controller = ChaosControlLayer(eigen_sentry, state_manager)
        
        # Start chaos task processing
        asyncio.create_task(app.state.chaos_controller.process_tasks())
        logger.info("✅ ChaosControlLayer initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize ChaosControlLayer: {e}")
    
    try:
        # Initialize CognitiveEngine
        app.state.cognitive_engine = CognitiveEngine({
            'storage_path': 'data/cognitive',
            'vector_dim': 512
        })
        logger.info("✅ CognitiveEngine initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize CognitiveEngine: {e}")
"""

# === SIMPLE WEBSOCKET FOR REAL-TIME UPDATES (OPTIONAL) ===

# WebSocket manager
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

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """Simple WebSocket for real-time events"""
    await manager.connect(websocket)
    try:
        # Send stability updates every second
        while True:
            # Get current stability
            stability = await get_stability_status()
            
            # Broadcast to all connected clients
            await manager.broadcast(json.dumps({
                "type": "stability_update",
                "data": stability,
                "timestamp": time.time()
            }))
            
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
