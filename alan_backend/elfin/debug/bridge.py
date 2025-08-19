"""
ELFIN Debug Bridge for TORI IDE Integration

This module provides a bridge between ELFIN's debug infrastructure and the TORI IDE,
enabling seamless integration of ELFIN's debugging capabilities with TORI's visualization.
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import websockets
from typing import Dict, Any, Set, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("elfin.debug.bridge")

class ElfinDebugBridge:
    """
    Bridge between ELFIN Debug Server and TORI IDE.
    
    This class relays messages between ELFIN's debugging infrastructure and
    the TORI IDE, translating between their respective protocols as needed.
    """
    
    def __init__(self, 
                 elfin_port: int = 8642, 
                 tori_port: int = 8643, 
                 dap_port: Optional[int] = None,
                 host: str = "localhost"):
        """
        Initialize the debug bridge.
        
        Args:
            elfin_port: Port for the ELFIN debug server
            tori_port: Port for the TORI IDE WebSocket connection
            dap_port: Optional port for VS Code Debug Adapter Protocol connection
            host: Host address to bind to
        """
        self.elfin_port = elfin_port
        self.tori_port = tori_port
        self.dap_port = dap_port
        self.host = host
        
        # Clients connected to the bridge
        self.tori_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.dap_clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Connection to ELFIN debug server
        self.elfin_socket: Optional[websockets.WebSocketClientProtocol] = None
        
        # Last received handshake packet
        self.last_handshake = None
        
        # Running state
        self.running = False
    
    async def start(self):
        """Start the debug bridge."""
        self.running = True
        
        # Start tasks
        await asyncio.gather(
            self.run_tori_server(),
            self.run_elfin_client(),
            *(
                [self.run_dap_server()] 
                if self.dap_port is not None else []
            )
        )
    
    async def run_tori_server(self):
        """Run the WebSocket server for TORI IDE connections."""
        async with websockets.serve(
            self.handle_tori_client, 
            self.host, 
            self.tori_port
        ):
            logger.info(f"TORI IDE server started on {self.host}:{self.tori_port}")
            
            # Keep the server running
            while self.running:
                await asyncio.sleep(1)
    
    async def run_dap_server(self):
        """Run the WebSocket server for VS Code DAP connections."""
        if self.dap_port is None:
            return
        
        async with websockets.serve(
            self.handle_dap_client, 
            self.host, 
            self.dap_port
        ):
            logger.info(f"VS Code DAP server started on {self.host}:{self.dap_port}")
            
            # Keep the server running
            while self.running:
                await asyncio.sleep(1)
    
    async def run_elfin_client(self):
        """Connect to the ELFIN debug server and relay messages."""
        while self.running:
            try:
                uri = f"ws://{self.host}:{self.elfin_port}/state"
                async with websockets.connect(uri) as websocket:
                    self.elfin_socket = websocket
                    logger.info(f"Connected to ELFIN debug server at {uri}")
                    
                    async for message in websocket:
                        await self.handle_elfin_message(message)
                    
                    self.elfin_socket = None
            
            except (websockets.ConnectionClosed, ConnectionRefusedError) as e:
                logger.warning(f"ELFIN debug server connection lost: {e}")
                self.elfin_socket = None
                await asyncio.sleep(3)  # Reconnection delay
    
    async def handle_tori_client(self, websocket, path):
        """
        Handle a TORI IDE client connection.
        
        Args:
            websocket: WebSocket connection
            path: WebSocket path
        """
        try:
            # Add to clients set
            self.tori_clients.add(websocket)
            logger.info(f"TORI IDE client connected: {websocket.remote_address}")
            
            # Send handshake if available
            if self.last_handshake:
                await websocket.send(json.dumps(self.last_handshake))
            
            # Handle client messages
            async for message in websocket:
                # Forward commands to ELFIN
                if self.elfin_socket:
                    await self.elfin_socket.send(message)
        
        except websockets.ConnectionClosed:
            logger.info(f"TORI IDE client disconnected: {websocket.remote_address}")
        finally:
            self.tori_clients.remove(websocket)
    
    async def handle_dap_client(self, websocket, path):
        """
        Handle a VS Code DAP client connection.
        
        Args:
            websocket: WebSocket connection
            path: WebSocket path
        """
        try:
            # Add to clients set
            self.dap_clients.add(websocket)
            logger.info(f"VS Code DAP client connected: {websocket.remote_address}")
            
            # Handle client messages (DAP messages from VS Code)
            async for message in websocket:
                # Parse and process DAP message
                try:
                    data = json.loads(message)
                    await self.handle_dap_message(data, websocket)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON from DAP client")
        
        except websockets.ConnectionClosed:
            logger.info(f"VS Code DAP client disconnected: {websocket.remote_address}")
        finally:
            self.dap_clients.remove(websocket)
    
    async def handle_elfin_message(self, message):
        """
        Handle a message from the ELFIN debug server.
        
        Args:
            message: Message from ELFIN
        """
        try:
            data = json.loads(message)
            
            # Store handshake packet for new clients
            if data.get("type") == "handshake":
                self.last_handshake = data
            
            # Process event notifications
            event = data.get("event")
            if event:
                await self.process_event(event, data)
            
            # Forward to all TORI clients
            await self.broadcast_to_tori(message)
        
        except json.JSONDecodeError:
            logger.error("Invalid JSON from ELFIN server")
    
    async def handle_dap_message(self, data, client):
        """
        Handle a DAP message from VS Code.
        
        Args:
            data: DAP message data
            client: Client WebSocket
        """
        # In a real implementation, we would handle various DAP commands here
        # For example, continue, pause, set breakpoints, etc.
        logger.debug(f"Received DAP message: {data.get('command', 'unknown')}")
        
        # Example: Convert breakpoint commands to ELFIN format
        if data.get("command") == "setBreakpoints":
            # Extract breakpoint info
            source = data.get("arguments", {}).get("source", {})
            breakpoints = data.get("arguments", {}).get("breakpoints", [])
            
            # Convert to ELFIN format and send to ELFIN server
            for bp in breakpoints:
                # In a real implementation, we would translate line-based breakpoints
                # to condition-based breakpoints in ELFIN where possible
                logger.info(f"Setting breakpoint at {source.get('path')}:{bp.get('line')}")
    
    async def process_event(self, event, data):
        """
        Process an event from ELFIN.
        
        Args:
            event: Event string
            data: Full data packet
        """
        # Forward breakpoint events to DAP
        if event.startswith("break:"):
            await self.send_dap_stopped_event(event, data)
    
    async def send_dap_stopped_event(self, event, data):
        """
        Send a stopped event to DAP clients.
        
        Args:
            event: ELFIN event string
            data: Full data packet
        """
        # Skip if no DAP clients
        if not self.dap_clients:
            return
        
        # Extract reason from event
        reason = "breakpoint"
        description = event
        
        if event.startswith("break:"):
            description = event[6:]  # Remove "break:" prefix
        elif event.startswith("warn:"):
            reason = "exception"
            description = event[5:]  # Remove "warn:" prefix
        
        # Create DAP stopped event
        dap_event = {
            "type": "event",
            "event": "stopped",
            "body": {
                "reason": reason,
                "description": description,
                "threadId": 1,  # Main thread
                "preserveFocusHint": False,
                "text": description,
                "allThreadsStopped": True
            }
        }
        
        # Add variables if available
        if "vars" in data:
            dap_event["body"]["variables"] = data["vars"]
        
        # Send to all DAP clients
        dap_message = json.dumps(dap_event)
        await asyncio.gather(*[
            client.send(dap_message) for client in self.dap_clients
        ])
    
    async def broadcast_to_tori(self, message):
        """
        Broadcast a message to all TORI clients.
        
        Args:
            message: Message to broadcast
        """
        if not self.tori_clients:
            return
        
        # Remove closed connections
        closed_clients = set()
        
        # Send to all clients
        for client in self.tori_clients:
            try:
                await client.send(message)
            except websockets.ConnectionClosed:
                closed_clients.add(client)
        
        # Remove closed clients
        for client in closed_clients:
            self.tori_clients.remove(client)


async def run_bridge(args):
    """
    Run the debug bridge with the specified arguments.
    
    Args:
        args: Command-line arguments
    """
    # Create and start the bridge
    bridge = ElfinDebugBridge(
        elfin_port=args.elfin_port,
        tori_port=args.tori_port,
        dap_port=args.dap_port,
        host=args.host
    )
    
    await bridge.start()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ELFIN Debug Bridge")
    parser.add_argument(
        "--elfin-port", 
        type=int, 
        default=8642, 
        help="Port for ELFIN debug server"
    )
    parser.add_argument(
        "--tori-port", 
        type=int, 
        default=8643, 
        help="Port for TORI IDE WebSocket connection"
    )
    parser.add_argument(
        "--dap-port", 
        type=int, 
        default=None, 
        help="Port for VS Code Debug Adapter Protocol connection"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost", 
        help="Host address to bind to"
    )
    
    args = parser.parse_args()
    
    # Run the bridge
    try:
        asyncio.run(run_bridge(args))
    except KeyboardInterrupt:
        logger.info("Bridge stopped by user")


if __name__ == "__main__":
    main()
