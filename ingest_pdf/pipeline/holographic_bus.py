"""
pipeline/holographic_bus.py

Event bus integration between ingestion pipeline and holographic display.
Bridges backend processing with frontend visualization.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger("holographic_bus")

@dataclass
class HolographicEvent:
    """Event structure for holographic display updates"""
    type: str  # "progress", "waveform", "concept", "complete", "error"
    data: Dict[str, Any]
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def to_json(self) -> str:
        """Convert to JSON for WebSocket transmission"""
        return json.dumps({
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp
        })

class HolographicEventBus:
    """
    Event bus for real-time holographic visualization updates.
    
    This connects the ingestion pipeline to the holographic display,
    allowing real-time visualization of the extraction process.
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[HolographicEvent] = []
        self.max_history = 1000
        self._lock = asyncio.Lock()
        
        # Waveform state
        self.current_waveform = {
            "amplitude": 0.0,
            "frequency": 1.0,
            "phase": 0.0,
            "coherence": 0.8,
            "interference_pattern": []
        }
    
    async def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to events of a specific type"""
        async with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to {event_type} events")
    
    async def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from events"""
        async with self._lock:
            if event_type in self.subscribers:
                self.subscribers[event_type].remove(callback)
    
    async def publish(self, event: Dict[str, Any]) -> None:
        """Publish an event to all subscribers"""
        event_obj = HolographicEvent(
            type=event.get("type", "unknown"),
            data=event.get("data", {})
        )
        
        # Store in history
        async with self._lock:
            self.event_history.append(event_obj)
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)
        
        # Update waveform based on event type
        await self._update_waveform(event_obj)
        
        # Notify subscribers
        callbacks = self.subscribers.get(event_obj.type, []) + self.subscribers.get("*", [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_obj)
                else:
                    callback(event_obj)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    async def _update_waveform(self, event: HolographicEvent) -> None:
        """Update holographic waveform based on event"""
        if event.type == "progress":
            # Map progress to waveform parameters
            percent = event.data.get("percent", 0)
            stage = event.data.get("stage", "")
            
            # Update amplitude based on progress
            self.current_waveform["amplitude"] = percent / 100.0
            
            # Map stage to frequency
            stage_frequencies = {
                "init": 0.5,
                "ocr": 1.0,
                "chunks": 1.5,
                "concepts": 2.0,
                "analysis": 2.5,
                "pruning": 3.0,
                "storage": 3.5,
                "complete": 4.0
            }
            self.current_waveform["frequency"] = stage_frequencies.get(stage, 1.0)
            
            # Add slight phase shift for movement
            self.current_waveform["phase"] += 0.1
            
        elif event.type == "concept":
            # Concept extraction creates interference ripples
            score = event.data.get("score", 0.5)
            
            # Create interference pattern
            ripple = {
                "center": [np.random.random(), np.random.random()],
                "amplitude": score,
                "wavelength": 1.0 / (score + 0.1),
                "decay": 0.95
            }
            
            self.current_waveform["interference_pattern"].append(ripple)
            
            # Limit pattern size
            if len(self.current_waveform["interference_pattern"]) > 50:
                self.current_waveform["interference_pattern"].pop(0)
        
        elif event.type == "ingest_complete":
            # Success creates coherent wave burst
            self.current_waveform["coherence"] = 1.0
            self.current_waveform["amplitude"] = 1.0
            
            # Create success pattern
            concept_count = event.data.get("concept_count", 0)
            for i in range(min(concept_count, 10)):
                angle = (i / 10) * 2 * np.pi
                self.current_waveform["interference_pattern"].append({
                    "center": [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)],
                    "amplitude": 0.8,
                    "wavelength": 0.1,
                    "decay": 0.98
                })
        
        elif event.type == "error":
            # Error creates chaotic interference
            self.current_waveform["coherence"] = 0.2
            self.current_waveform["amplitude"] = 0.5
            
            # Random noise pattern
            for _ in range(5):
                self.current_waveform["interference_pattern"].append({
                    "center": [np.random.random(), np.random.random()],
                    "amplitude": np.random.random() * 0.5,
                    "wavelength": np.random.random() * 0.5,
                    "decay": 0.9
                })
        
        # Publish waveform update (debounced)
        asyncio.create_task(self.publish({
            "type": "waveform",
            "data": self.current_waveform.copy()
        }))
    
    def get_current_waveform(self) -> Dict[str, Any]:
        """Get current waveform state"""
        return self.current_waveform.copy()
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get recent event history"""
        history = self.event_history
        
        if event_type:
            history = [e for e in history if e.type == event_type]
        
        return [
            {
                "type": e.type,
                "data": e.data,
                "timestamp": e.timestamp
            }
            for e in history[-limit:]
        ]

# === WebSocket Bridge ===
class WebSocketBridge:
    """Bridge between event bus and WebSocket clients"""
    
    def __init__(self, event_bus: HolographicEventBus):
        self.event_bus = event_bus
        self.clients: List[Any] = []  # WebSocket connections
        self._running = False
    
    async def start(self):
        """Start the bridge"""
        self._running = True
        
        # Subscribe to all events
        await self.event_bus.subscribe("*", self._forward_to_clients)
        
        logger.info("WebSocket bridge started")
    
    async def stop(self):
        """Stop the bridge"""
        self._running = False
        await self.event_bus.unsubscribe("*", self._forward_to_clients)
    
    async def add_client(self, websocket):
        """Add a WebSocket client"""
        self.clients.append(websocket)
        
        # Send current waveform state
        await websocket.send(json.dumps({
            "type": "waveform",
            "data": self.event_bus.get_current_waveform()
        }))
        
        logger.info(f"WebSocket client connected. Total clients: {len(self.clients)}")
    
    async def remove_client(self, websocket):
        """Remove a WebSocket client"""
        if websocket in self.clients:
            self.clients.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total clients: {len(self.clients)}")
    
    async def _forward_to_clients(self, event: HolographicEvent):
        """Forward event to all connected clients"""
        if not self.clients:
            return
        
        message = event.to_json()
        
        # Send to all clients
        disconnected = []
        for client in self.clients:
            try:
                await client.send(message)
            except Exception as e:
                logger.debug(f"Failed to send to client: {e}")
                disconnected.append(client)
        
        # Remove disconnected clients
        for client in disconnected:
            await self.remove_client(client)

# === Holographic Display API ===
class HolographicDisplayAPI:
    """
    High-level API for holographic display updates.
    Simplifies integration with the ingestion pipeline.
    """
    
    def __init__(self, event_bus: HolographicEventBus):
        self.event_bus = event_bus
    
    async def update_progress(self, stage: str, percent: int, message: str):
        """Update progress visualization"""
        await self.event_bus.publish({
            "type": "progress",
            "data": {
                "stage": stage,
                "percent": percent,
                "message": message
            }
        })
    
    async def add_concept(self, concept: Dict[str, Any]):
        """Add a concept to the visualization"""
        await self.event_bus.publish({
            "type": "concept",
            "data": {
                "name": concept.get("name", ""),
                "score": concept.get("score", 0.5),
                "metadata": concept.get("metadata", {})
            }
        })
    
    async def show_spectral_analysis(self, spectral_features: Dict[str, Any]):
        """Show audio/video spectral analysis"""
        await self.event_bus.publish({
            "type": "spectral",
            "data": spectral_features
        })
    
    async def complete(self, summary: Dict[str, Any]):
        """Show completion animation"""
        await self.event_bus.publish({
            "type": "ingest_complete",
            "data": summary
        })
    
    async def error(self, error_message: str):
        """Show error state"""
        await self.event_bus.publish({
            "type": "error",
            "data": {"message": error_message}
        })

# === Global Instance ===
_global_bus = None
_global_bridge = None

def get_event_bus() -> HolographicEventBus:
    """Get global event bus instance"""
    global _global_bus
    if _global_bus is None:
        _global_bus = HolographicEventBus()
    return _global_bus

def get_display_api() -> HolographicDisplayAPI:
    """Get global display API"""
    return HolographicDisplayAPI(get_event_bus())

async def start_websocket_bridge():
    """Start the WebSocket bridge"""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = WebSocketBridge(get_event_bus())
        await _global_bridge.start()
    return _global_bridge

# === Integration Example ===
"""
# In your FastAPI app:

from fastapi import WebSocket
from .holographic_bus import get_event_bus, start_websocket_bridge

@app.on_event("startup")
async def startup():
    await start_websocket_bridge()

@app.websocket("/ws/hologram")
async def hologram_websocket(websocket: WebSocket):
    await websocket.accept()
    bridge = await start_websocket_bridge()
    
    try:
        await bridge.add_client(websocket)
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        await bridge.remove_client(websocket)

# In your ingestion handlers:

from .holographic_bus import get_display_api

async def ingest_with_hologram(file_path):
    display = get_display_api()
    
    await display.update_progress("init", 0, "Starting...")
    
    # Process file...
    
    for concept in extracted_concepts:
        await display.add_concept(concept)
    
    await display.complete({"concept_count": len(concepts)})
"""

logger.info("Holographic event bus initialized")

