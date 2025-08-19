#!/usr/bin/env python3
"""
EigenSentry Metrics WebSocket Service
Exposes live metrics to UI via WebSocket at /ws/eigensentry
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime, timezone
from typing import Set, Dict, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alan_backend.eigensentry_guard import get_guard

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# WebSocket configuration
WS_HOST = "localhost"
WS_PORT = 8765
WS_PATH = "/ws/eigensentry"

# Update interval (seconds)
UPDATE_INTERVAL = 0.1  # 100ms for real-time feel

class MetricsWebSocketServer:
    """
    WebSocket server for EigenSentry metrics
    Broadcasts real-time guard status to connected clients
    """
    
    def __init__(self):
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.guard = get_guard()
        self.running = False
        
    async def register(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client"""
        self.clients.add(websocket)
        self.guard.register_websocket(websocket)
        logger.info(f"Client connected from {websocket.remote_address}. Total: {len(self.clients)}")
        
        # Send initial state
        await self.send_initial_state(websocket)
        
    async def unregister(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client"""
        self.clients.discard(websocket)
        self.guard.unregister_websocket(websocket)
        logger.info(f"Client disconnected. Total: {len(self.clients)}")
        
    async def send_initial_state(self, websocket: websockets.WebSocketServerProtocol):
        """Send initial metrics state to new client"""
        message = {
            'type': 'initial_state',
            'data': {
                'metrics': self.guard.metrics,
                'config': {
                    'base_threshold': self.guard.base_threshold,
                    'curvature_sensitivity': self.guard.curvature_sensitivity,
                    'update_interval': UPDATE_INTERVAL
                }
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        await websocket.send(json.dumps(message))
        
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle a WebSocket client connection"""
        if path != WS_PATH:
            await websocket.close(code=404, reason="Not found")
            return
            
        await self.register(websocket)
        
        try:
            async for message in websocket:
                # Handle client messages
                try:
                    data = json.loads(message)
                    await self.handle_message(websocket, data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from client: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
            
    async def handle_message(self, websocket: websockets.WebSocketServerProtocol, 
                           data: Dict[str, Any]):
        """Handle incoming client messages"""
        msg_type = data.get('type')
        
        if msg_type == 'ping':
            # Respond to ping
            await websocket.send(json.dumps({
                'type': 'pong',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }))
            
        elif msg_type == 'request_history':
            # Send eigenvalue history
            history = list(self.guard.eigenvalue_history)[-100:]  # Last 100 points
            await websocket.send(json.dumps({
                'type': 'eigenvalue_history',
                'data': history,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }))
            
        elif msg_type == 'inject_test':
            # Inject test blow-up (for debugging)
            state, eigenvalues = self.guard.inject_synthetic_blowup()
            action = self.guard.check_eigenvalues(eigenvalues, state)
            
            await websocket.send(json.dumps({
                'type': 'test_injection',
                'data': {
                    'action': action,
                    'max_eigenvalue': float(np.max(np.real(eigenvalues)))
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }))
            
    async def broadcast_loop(self):
        """Periodic broadcast of metrics"""
        while self.running:
            if self.clients:
                # Prepare broadcast message
                message = {
                    'type': 'metrics_update',
                    'data': self.guard.metrics,
                    'bdg_stability': {
                        'lambda_max': self.guard.metrics.get('lambda_max', 0.0),
                        'unstable_modes': self.guard.metrics.get('unstable_modes', 0),
                        'adaptive_dt': self.guard.metrics.get('adaptive_dt', 0.01)
                    },
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # Check for alerts
                if self.guard.metrics['lyapunov_exponent'] > 0.2:
                    message['alert'] = {
                        'type': 'high_lyapunov',
                        'value': self.guard.metrics['lyapunov_exponent'],
                        'message': 'Lyapunov exponent exceeds 0.2'
                    }
                    
                if self.guard.metrics['damping_active']:
                    message['alert'] = {
                        'type': 'damping_active',
                        'message': 'Damping is currently active'
                    }
                
                # Broadcast to all clients
                disconnected = set()
                for client in self.clients:
                    try:
                        await client.send(json.dumps(message))
                    except websockets.exceptions.ConnectionClosed:
                        disconnected.add(client)
                        
                # Clean up disconnected clients
                for client in disconnected:
                    await self.unregister(client)
                    
            await asyncio.sleep(UPDATE_INTERVAL)
            
    async def start(self):
        """Start the WebSocket server"""
        self.running = True
        
        # Start broadcast loop
        broadcast_task = asyncio.create_task(self.broadcast_loop())
        
        # Start WebSocket server
        logger.info(f"Starting EigenSentry metrics WebSocket server on ws://{WS_HOST}:{WS_PORT}{WS_PATH}")
        
        async with websockets.serve(self.handle_client, WS_HOST, WS_PORT):
            try:
                await asyncio.Future()  # Run forever
            except asyncio.CancelledError:
                pass
            finally:
                self.running = False
                broadcast_task.cancel()
                
# Convenience function for external use
async def start_metrics_websocket():
    """Start the metrics WebSocket server"""
    server = MetricsWebSocketServer()
    await server.start()
    
# Standalone execution
if __name__ == "__main__":
    import numpy as np  # Import for test injection
    
    print("EigenSentry Metrics WebSocket Server")
    print(f"Listening on ws://{WS_HOST}:{WS_PORT}{WS_PATH}")
    print("Press Ctrl+C to stop")
    
    try:
        asyncio.run(start_metrics_websocket())
    except KeyboardInterrupt:
        print("\nShutting down...")
