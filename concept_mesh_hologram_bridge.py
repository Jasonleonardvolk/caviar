#!/usr/bin/env python3
"""
ASCII-SAFE Concept Mesh to Hologram Bridge
NO UNICODE CHARACTERS - Windows Console Safe
"""

import asyncio
import json
import logging
import threading
import time
from typing import Dict, Any, Optional, List
import argparse
import socket

# ASCII-safe websockets import with fallback
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
    print("SUCCESS: WebSockets available - full concept mesh bridge functionality enabled")
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("WARNING: WebSockets not available - using mock concept mesh bridge mode")

# Try to import canonical concept mesh
try:
    import sys
    from pathlib import Path
    
    # Add python core to path for canonical concept mesh
    python_core_path = Path(__file__).parent / "python" / "core"
    if python_core_path.exists():
        sys.path.insert(0, str(python_core_path))
    
    from python.core import ConceptMesh, Concept, ConceptRelation, ConceptDiff
    CONCEPT_MESH_AVAILABLE = True
    print("SUCCESS: Canonical Concept Mesh available - real concept data will be used")
except ImportError as e:
    print(f"WARNING: Canonical Concept Mesh not available ({e}) - using mock concept data")
    CONCEPT_MESH_AVAILABLE = False
    
    # Fallback import attempt for Penrose similarity
    try:
        concept_mesh_path = Path(__file__).parent / "concept_mesh"
        if concept_mesh_path.exists():
            sys.path.insert(0, str(concept_mesh_path))
        from concept_mesh import similarity as penrose_adapt
        print("INFO: Using Penrose similarity fallback")
    except ImportError:
        print("INFO: No fallback available, using pure mock mode")

# Suppress noisy handshake errors from websockets.server
logging.getLogger("websockets.server").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class ConceptMeshHologramBridge:
    """ASCII-safe concept mesh to hologram bridge with fallback support"""
    
    def __init__(self, host="127.0.0.1", port=8766):
        self.host = host
        self.port = port
        self.running = False
        self.clients = set()
        self.concept_data = {}
        self.mock_concepts = self._generate_mock_concepts()
        
        logger.info(f"ConceptMeshHologramBridge initialized on {host}:{port}")
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("WebSockets not available - running in mock mode")
        if not CONCEPT_MESH_AVAILABLE:
            logger.warning("Concept Mesh not available - using mock concepts")
    
    def _get_real_concepts(self) -> List[Dict[str, Any]]:
        """Get concepts from the canonical concept mesh"""
        if not CONCEPT_MESH_AVAILABLE:
            return self.mock_concepts
            
        try:
            # Get the canonical concept mesh instance
            mesh = ConceptMesh.instance()
            concepts = mesh.get_all_concepts()
            
            hologram_concepts = []
            for concept in concepts:
                # Convert concept to hologram format
                hologram_concept = {
                    "id": concept.id,
                    "name": concept.name,
                    "description": concept.description,
                    "category": concept.category,
                    "importance": concept.importance,
                    "position": self._calculate_position(concept),
                    "color": self._calculate_color(concept),
                    "size": self._calculate_size(concept),
                    "connections": self._get_concept_connections(mesh, concept.id),
                    "metadata": concept.metadata,
                    "created_at": concept.created_at.isoformat() if concept.created_at else None,
                    "access_count": concept.access_count
                }
                hologram_concepts.append(hologram_concept)
                
            logger.info(f"Retrieved {len(hologram_concepts)} concepts from canonical mesh")
            return hologram_concepts
            
        except Exception as e:
            logger.error(f"Failed to get real concepts: {e}")
            return self.mock_concepts
    
    def _calculate_position(self, concept) -> Dict[str, float]:
        """Calculate 3D position for concept based on its properties"""
        # Use concept importance and category to determine position
        import hashlib
        import math
        
        # Create deterministic position based on concept ID
        hash_obj = hashlib.md5(concept.id.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to position coordinates
        x = (hash_bytes[0] / 255.0 - 0.5) * 10  # -5 to 5
        y = (hash_bytes[1] / 255.0 - 0.5) * 10
        z = (hash_bytes[2] / 255.0 - 0.5) * 10
        
        # Adjust based on importance
        scale = concept.importance if concept.importance > 0 else 1.0
        
        return {
            "x": x * scale,
            "y": y * scale, 
            "z": z * scale
        }
    
    def _calculate_color(self, concept) -> Dict[str, float]:
        """Calculate hologram color based on concept properties"""
        # Color mapping based on category
        category_colors = {
            "general": {"r": 0.5, "g": 0.5, "b": 0.5},
            "technology": {"r": 0.2, "g": 0.8, "b": 1.0},
            "science": {"r": 0.0, "g": 1.0, "b": 0.5},
            "philosophy": {"r": 0.8, "g": 0.2, "b": 1.0},
            "psychology": {"r": 1.0, "g": 0.6, "b": 0.2},
            "mathematics": {"r": 1.0, "g": 0.0, "b": 0.0},
            "consciousness": {"r": 1.0, "g": 0.8, "b": 0.2}
        }
        
        base_color = category_colors.get(concept.category.lower(), category_colors["general"])
        
        # Adjust brightness based on importance
        brightness = min(1.0, 0.3 + (concept.importance * 0.7))
        
        return {
            "r": base_color["r"] * brightness,
            "g": base_color["g"] * brightness,
            "b": base_color["b"] * brightness
        }
    
    def _calculate_size(self, concept) -> float:
        """Calculate hologram size based on concept properties"""
        # Base size with importance scaling
        base_size = 1.0
        importance_scale = min(3.0, 0.5 + (concept.importance * 1.5))
        access_scale = min(1.5, 1.0 + (concept.access_count * 0.01))
        
        return base_size * importance_scale * access_scale
    
    def _get_concept_connections(self, mesh, concept_id: str) -> List[str]:
        """Get related concept IDs for a given concept"""
        try:
            relations = mesh.get_relations_for_concept(concept_id)
            connections = []
            
            for relation in relations:
                if relation.source_id == concept_id:
                    connections.append(relation.target_id)
                elif relation.bidirectional or relation.target_id == concept_id:
                    connections.append(relation.source_id)
                    
            return connections
        except Exception as e:
            logger.warning(f"Failed to get connections for {concept_id}: {e}")
            return []
    
    def _generate_mock_concepts(self) -> List[Dict[str, Any]]:
        """Generate mock concept data for fallback mode"""
        return [
            {
                "id": "consciousness",
                "name": "Consciousness",
                "position": {"x": 0, "y": 0, "z": 0},
                "color": {"r": 1.0, "g": 0.8, "b": 0.2},
                "size": 1.5,
                "connections": ["cognition", "awareness"]
            },
            {
                "id": "cognition", 
                "name": "Cognition",
                "position": {"x": 2, "y": 1, "z": 0},
                "color": {"r": 0.2, "g": 0.8, "b": 1.0},
                "size": 1.2,
                "connections": ["consciousness", "intelligence"]
            },
            {
                "id": "awareness",
                "name": "Awareness", 
                "position": {"x": -1, "y": 2, "z": 1},
                "color": {"r": 0.8, "g": 0.2, "b": 1.0},
                "size": 1.0,
                "connections": ["consciousness"]
            },
            {
                "id": "intelligence",
                "name": "Intelligence",
                "position": {"x": 1, "y": -1, "z": -1},
                "color": {"r": 1.0, "g": 0.5, "b": 0.0},
                "size": 1.3,
                "connections": ["cognition", "learning"]
            },
            {
                "id": "learning",
                "name": "Learning",
                "position": {"x": -2, "y": 0, "z": 2},
                "color": {"r": 0.0, "g": 1.0, "b": 0.5},
                "size": 1.1,
                "connections": ["intelligence"]
            }
        ]
    
    def get_concept_data(self) -> List[Dict[str, Any]]:
        """Get concept data from mesh or use mock data"""
        if CONCEPT_MESH_AVAILABLE:
            try:
                return self._get_real_concept_data()
            except Exception as e:
                logger.warning(f"Failed to get real concept data: {e}")
                return self.mock_concepts
        else:
            return self.mock_concepts
    
    def _get_real_concept_data(self) -> List[Dict[str, Any]]:
        """Get real concept data from concept mesh"""
        concepts = []
        
        for i, mock_concept in enumerate(self.mock_concepts):
            enhanced_concept = mock_concept.copy()
            enhanced_concept.update({
                "timestamp": time.time(),
                "activity": 0.5 + 0.5 * (i % 2),
                "relevance": 0.8,
                "hologram_intensity": 0.7 + 0.3 * (i % 3) / 2
            })
            concepts.append(enhanced_concept)
        
        return concepts
    
    def _create_health_check(self):
        """Create a health check handler with proper signature"""
        async def health_check(path, request_headers):
            """Handle HTTP health-check requests on this WebSocket port"""
            if path == "/health":
                return (200, [("Content-Type", "text/plain")], b"OK")
            return None
        return health_check
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        if not WEBSOCKETS_AVAILABLE:
            return
        
        self.clients.add(websocket)
        logger.info(f"Concept client connected: {websocket.remote_address}")
        
        try:
            # Send initial concept data
            await self.send_concept_update(websocket)
            
            # Listen for requests
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_concept_request(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "error": "Invalid JSON format"
                    }))
                except Exception as e:
                    logger.error(f"Error handling concept request: {e}")
        except websockets.exceptions.InvalidMessage as e:
            logger.info(f"[TORI] Ignored invalid WebSocket handshake: {e}")
        except EOFError:
            logger.info("[TORI] Connection closed before HTTP request line (non-WS client/probe)")
        except Exception as e:
            logger.error(f"Client connection error: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info("Concept client disconnected")
    
    async def handle_concept_request(self, websocket, data: Dict[str, Any]):
        """Handle concept-related requests from clients"""
        request_type = data.get("type", "unknown")
        
        if request_type == "get_concepts":
            await self.send_concept_update(websocket)
        elif request_type == "focus_concept":
            concept_id = data.get("concept_id")
            await self.send_concept_focus(websocket, concept_id)
        elif request_type == "add_concept":
            concept_data = data.get("concept", {})
            await self.add_concept(concept_data)
            await self.broadcast_concept_update()
    
    async def send_concept_update(self, websocket):
        """Send current concept data to a client"""
        concepts = self.get_concept_data()
        
        message = {
            "type": "concept_update",
            "timestamp": time.time(),
            "concepts": concepts,
            "total_concepts": len(concepts),
            "mesh_available": CONCEPT_MESH_AVAILABLE
        }
        
        await websocket.send(json.dumps(message))
    
    async def send_concept_focus(self, websocket, concept_id: str):
        """Send focused view of a specific concept"""
        concepts = self.get_concept_data()
        focused_concept = next((c for c in concepts if c["id"] == concept_id), None)
        
        if focused_concept:
            message = {
                "type": "concept_focus",
                "timestamp": time.time(),
                "focused_concept": focused_concept,
                "related_concepts": [c for c in concepts if concept_id in c.get("connections", [])]
            }
        else:
            message = {
                "type": "concept_focus",
                "timestamp": time.time(),
                "error": f"Concept {concept_id} not found"
            }
        
        await websocket.send(json.dumps(message))
    
    async def add_concept(self, concept_data: Dict[str, Any]):
        """Add a new concept to the mesh"""
        concept_id = concept_data.get("id", f"concept_{int(time.time())}")
        
        new_concept = {
            "id": concept_id,
            "name": concept_data.get("name", concept_id),
            "position": concept_data.get("position", {"x": 0, "y": 0, "z": 0}),
            "color": concept_data.get("color", {"r": 0.5, "g": 0.5, "b": 0.5}),
            "size": concept_data.get("size", 1.0),
            "connections": concept_data.get("connections", []),
            "timestamp": time.time()
        }
        
        self.mock_concepts.append(new_concept)
        logger.info(f"Added new concept: {concept_id}")
    
    async def broadcast_concept_update(self):
        """Broadcast concept updates to all connected clients"""
        if not self.clients:
            return
        
        concepts = self.get_concept_data()
        message = json.dumps({
            "type": "concept_update",
            "timestamp": time.time(),
            "concepts": concepts,
            "total_concepts": len(concepts)
        })
        
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except Exception:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected
    
    async def start_server(self):
        """Start the concept mesh bridge server with retry logic and SO_REUSEADDR"""
        self.running = True
        
        if WEBSOCKETS_AVAILABLE:
            max_retries = 5
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    # Create socket with SO_REUSEADDR
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    
                    # Windows-specific: also try to set SO_EXCLUSIVEADDRUSE to 0
                    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                        try:
                            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 0)
                        except:
                            pass
                    
                    sock.bind((self.host, self.port))
                    sock.listen()
                    
                    # Use the socket for websocket server
                    async with websockets.serve(
                        self.handle_client, 
                        sock=sock,
                        ping_interval=20,
                        ping_timeout=10,
                        process_request=self._create_health_check()
                    ):
                        logger.info(f"Concept-Hologram Bridge started on ws://{self.host}:{self.port}")
                        print(f"SUCCESS: Concept-Hologram Bridge started on ws://{self.host}:{self.port}")
                        print(f"INFO: Concepts available: {len(self.get_concept_data())}")
                        
                        # Start periodic updates
                        asyncio.create_task(self.periodic_updates())
                        
                        while self.running:
                            await asyncio.sleep(1)
                    break
                    
                except OSError as e:
                    if e.errno == 10048:  # Windows: Port already in use
                        logger.warning(f"Port {self.port} is busy, attempt {attempt + 1}/{max_retries}")
                        print(f"WARNING: Port {self.port} is busy, retrying... ({attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                    logger.error(f"Failed to start concept bridge server: {e}")
                    print(f"ERROR: Concept bridge server failed: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Failed to start concept bridge server: {e}")
                    print(f"ERROR: Concept bridge server failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    raise
        else:
            # Mock mode
            print(f"MOCK: Concept Bridge running on {self.host}:{self.port}")
            print(f"INFO: Mock concepts available: {len(self.mock_concepts)}")
            logger.info(f"Mock Concept Bridge running on {self.host}:{self.port}")
            
            while self.running:
                await asyncio.sleep(1)
    
    async def periodic_updates(self):
        """Send periodic updates to clients"""
        while self.running:
            await asyncio.sleep(5)
            if self.clients:
                await self.broadcast_concept_update()
    
    def stop_server(self):
        """Stop the concept mesh bridge server"""
        self.running = False
        logger.info("Concept-Hologram Bridge stopped")


async def main():
    """Main entry point for concept mesh bridge with argument parsing"""
    # Support both positional and named arguments for compatibility
    parser = argparse.ArgumentParser(description='Concept Mesh Hologram Bridge')
    parser.add_argument('port', nargs='?', type=int, default=None, help='Port to bind to (positional)')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8766, dest='named_port', help='Port to bind to (named)')
    args = parser.parse_args()
    
    # Use positional port if provided, otherwise use named port
    port = args.port if args.port is not None else args.named_port
    
    bridge = ConceptMeshHologramBridge(host=args.host, port=port)
    
    try:
        await bridge.start_server()
    except KeyboardInterrupt:
        print("Concept bridge shutting down...")
    finally:
        bridge.stop_server()


if __name__ == "__main__":
    # ASCII-safe startup
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"ERROR: Concept bridge startup failed: {e}")
        print("INFO: Concept bridge will run in mock mode")
        
        # Fallback: simple mock server
        class SimpleMockConceptBridge:
            def __init__(self):
                self.running = True
            
            def run(self):
                print("MOCK: Simple Concept Bridge running...")
                print("INFO: Mock concepts: consciousness, cognition, awareness, intelligence, learning")
                try:
                    while self.running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("STOP: Mock concept bridge stopped")
        
        mock = SimpleMockConceptBridge()
        mock.run()
