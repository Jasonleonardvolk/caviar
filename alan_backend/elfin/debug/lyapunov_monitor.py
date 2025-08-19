"""
Lyapunov Monitor for ELFIN

This module provides Lyapunov function monitoring capabilities for ELFIN controllers,
sending real-time stability information to the debugging UI.
"""

import json
import time
import asyncio
import logging
import websockets
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Set, Callable, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("elfin.debug.lyapunov_monitor")

class LyapunovMonitor:
    """
    Monitor for Lyapunov stability in ELFIN controller systems.
    
    This class tracks Lyapunov function values and derivatives in real-time,
    providing stability feedback and generating events when stability conditions
    are violated.
    """
    
    def __init__(self, port: int = 8642):
        """
        Initialize a Lyapunov monitor.
        
        Args:
            port: WebSocket server port for UI connections
        """
        self.port = port
        self.running = False
        self.clients = set()
        self.server = None
        self.server_task = None
        self.seq_counter = 0
        
        # Barrier functions being monitored
        self.barrier_functions = {}
        
        # Breakpoints - conditions that pause execution
        self.breakpoints = []
        
        # State variables
        self.state_vars = {}
        
        # Events that have occurred
        self.events = []
        
        # Lyapunov function and its derivative
        self.V = 0.0
        self.Vdot = 0.0
        
        # System metadata
        self.metadata = {
            "mode": "default",
            "status": "initializing",
            "units": {}
        }
        
        # Last send time to limit update frequency
        self.last_send_time = 0
        self.min_update_interval = 0.01  # 100Hz maximum update rate
        
        # Handshake packet sent at connection
        self.handshake_packet = {
            "type": "handshake",
            "schema": {
                "vars": {},
                "barriers": []
            },
            "dt_nominal": 0.01
        }
    
    async def start_server(self):
        """Start the WebSocket server."""
        if self.server is not None:
            logger.warning("Server already running")
            return
        
        self.running = True
        self.server = await websockets.serve(
            self.client_handler, 
            "localhost", 
            self.port
        )
        logger.info(f"Lyapunov monitor server started on port {self.port}")
    
    def start(self):
        """Start the monitor in a background thread."""
        def run_server_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.server_task = loop.create_task(self.start_server())
            loop.run_forever()
        
        # Start in a separate thread
        server_thread = threading.Thread(target=run_server_thread, daemon=True)
        server_thread.start()
        logger.info("Lyapunov monitor started in background thread")
        
        # Update handshake with registered barriers
        self.handshake_packet["schema"]["barriers"] = list(self.barrier_functions.keys())
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        if self.server is None:
            return
        
        self.running = False
        self.server.close()
        await self.server.wait_closed()
        self.server = None
        
        # Close all client connections
        close_coroutines = [client.close() for client in self.clients]
        if close_coroutines:
            await asyncio.gather(*close_coroutines, return_exceptions=True)
        
        self.clients.clear()
        logger.info("Lyapunov monitor server stopped")
    
    def stop(self):
        """Stop the monitor."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(self.stop_server())
        else:
            loop.run_until_complete(self.stop_server())
        logger.info("Lyapunov monitor stopped")
    
    async def client_handler(self, websocket, path):
        """
        Handle WebSocket client connections.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        try:
            # Add to clients set
            self.clients.add(websocket)
            logger.info(f"Client connected: {websocket.remote_address}")
            
            # Send handshake packet
            await websocket.send(json.dumps(self.handshake_packet))
            
            # Keep connection open and handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    # Handle client message (future: command processing)
                    await self.handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON message from client")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        finally:
            # Remove from clients set
            self.clients.remove(websocket)
    
    async def handle_client_message(self, websocket, data):
        """
        Handle messages from clients.
        
        Args:
            websocket: Client WebSocket
            data: Message data
        """
        # Future: implement command handling
        pass
    
    async def broadcast(self, message):
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast
        """
        if not self.clients:
            return
        
        # Convert message to JSON
        if isinstance(message, dict):
            message = json.dumps(message)
        
        # Send to all clients
        websockets_to_remove = set()
        
        for websocket in self.clients:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                websockets_to_remove.add(websocket)
        
        # Remove closed connections
        for websocket in websockets_to_remove:
            self.clients.remove(websocket)
    
    def register_barrier_function(self, name: str, func: Callable, threshold: float, description: str = ""):
        """
        Register a barrier function to monitor.
        
        Args:
            name: Barrier function name
            func: Barrier function (callable)
            threshold: Safety threshold (barrier is safe when func >= threshold)
            description: Human-readable description
        """
        self.barrier_functions[name] = {
            "func": func,
            "threshold": threshold,
            "description": description,
            "value": None
        }
        
        # Update handshake packet
        self.handshake_packet["schema"]["barriers"] = list(self.barrier_functions.keys())
        
        logger.info(f"Registered barrier function: {name}")
    
    def register_state_var(self, name: str, unit: Optional[str] = None):
        """
        Register a state variable with optional unit information.
        
        Args:
            name: Variable name
            unit: Optional unit specification (e.g., 'm', 'rad/s')
        """
        self.state_vars[name] = None
        
        # Update handshake packet with unit information
        if unit:
            if "vars" not in self.handshake_packet["schema"]:
                self.handshake_packet["schema"]["vars"] = {}
            self.handshake_packet["schema"]["vars"][name] = unit
            
            # Also store in metadata
            self.metadata["units"][name] = unit
        
        logger.info(f"Registered state variable: {name} ({unit or 'no unit'})")
    
    def set_lyapunov_function(self, V_func: Callable, Vdot_func: Callable):
        """
        Set the Lyapunov function and its derivative for stability monitoring.
        
        Args:
            V_func: Lyapunov function V(x)
            Vdot_func: Lyapunov derivative function dV/dt(x)
        """
        self.V_func = V_func
        self.Vdot_func = Vdot_func
        logger.info("Lyapunov functions set")
    
    def add_breakpoint(self, condition: str, expression: str):
        """
        Add a breakpoint that pauses execution when a condition is met.
        
        Args:
            condition: Condition string (e.g., "V > 0.2")
            expression: Python expression to evaluate
        """
        self.breakpoints.append({
            "condition": condition,
            "expression": expression,
            "enabled": True
        })
        logger.info(f"Added breakpoint: {condition}")
    
    def update(self, **kwargs):
        """
        Update state variables and check stability conditions.
        
        Args:
            **kwargs: State variable values to update
        """
        # Update state variables
        for name, value in kwargs.items():
            self.state_vars[name] = value
        
        # Update Lyapunov function and derivative
        try:
            if hasattr(self, 'V_func'):
                self.V = self.V_func(**self.state_vars)
            if hasattr(self, 'Vdot_func'):
                self.Vdot = self.Vdot_func(**self.state_vars)
        except Exception as e:
            logger.error(f"Error computing Lyapunov functions: {e}")
        
        # Update barrier functions
        for name, barrier in self.barrier_functions.items():
            try:
                barrier["value"] = barrier["func"](**self.state_vars)
            except Exception as e:
                logger.error(f"Error computing barrier function {name}: {e}")
        
        # Check breakpoints
        event = None
        for bp in self.breakpoints:
            if not bp["enabled"]:
                continue
                
            try:
                # Create evaluation context with state variables
                eval_context = {**self.state_vars}
                eval_context['V'] = self.V
                eval_context['Vdot'] = self.Vdot
                
                # Add barrier functions
                for b_name, barrier in self.barrier_functions.items():
                    eval_context[b_name] = barrier["value"]
                
                # Evaluate the breakpoint condition
                if eval(bp["expression"], {"__builtins__": {}}, eval_context):
                    event = f"break:{bp['condition']}"
                    self.events.append({
                        "type": "breakpoint",
                        "condition": bp["condition"],
                        "time": time.time()
                    })
                    break
            except Exception as e:
                logger.error(f"Error evaluating breakpoint {bp['condition']}: {e}")
        
        # Check barrier violations
        if event is None:
            for name, barrier in self.barrier_functions.items():
                if barrier["value"] is not None and barrier["value"] < barrier["threshold"]:
                    event = f"warn:barrier:{name}"
                    self.events.append({
                        "type": "barrier_violation",
                        "barrier": name,
                        "value": barrier["value"],
                        "threshold": barrier["threshold"],
                        "time": time.time()
                    })
                    break
        
        # Check Lyapunov instability
        if event is None and self.Vdot > 0:
            event = f"warn:unstable"
            self.events.append({
                "type": "instability",
                "V": self.V,
                "Vdot": self.Vdot,
                "time": time.time()
            })
        
        # Send update to connected clients
        self.send_update(event)
    
    def send_update(self, event: Optional[str] = None):
        """
        Send a state update to connected clients if enough time has passed.
        
        Args:
            event: Optional event string
        """
        # Check if enough time has passed since last update
        now = time.time()
        if now - self.last_send_time < self.min_update_interval and event is None:
            return
        
        self.last_send_time = now
        
        # Prepare state packet
        packet = {
            "type": "state",
            "seq": self.seq_counter,
            "t": now,
            "vars": {**self.state_vars},
            "V": self.V,
            "Vdot": self.Vdot,
            "barriers": {
                name: barrier["value"] 
                for name, barrier in self.barrier_functions.items()
                if barrier["value"] is not None
            },
            "event": event,
            "meta": self.metadata
        }
        
        # Increment sequence counter
        self.seq_counter += 1
        
        # Send asynchronously
        asyncio.create_task(self.broadcast(packet))


# Create a singleton instance
lyapunov_monitor = LyapunovMonitor()
