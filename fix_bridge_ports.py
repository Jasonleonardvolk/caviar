#!/usr/bin/env python3
"""
Fix for Audio and Concept Bridge Port Binding Issues
Adds SO_REUSEADDR and retry logic to prevent port binding errors
"""

import asyncio
import socket
import websockets
from websockets.server import serve
from functools import partial


def create_reusable_server_socket(host, port):
    """Create a socket with SO_REUSEADDR enabled"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Windows-specific: also set SO_EXCLUSIVEADDRUSE to 0
    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 0)
        except:
            pass
    
    sock.bind((host, port))
    return sock


async def start_websocket_server_with_retry(handler, host, port, max_retries=5, retry_delay=1):
    """Start websocket server with retry logic and SO_REUSEADDR"""
    for attempt in range(max_retries):
        try:
            # Create a reusable socket
            sock = create_reusable_server_socket(host, port)
            
            # Use the socket for websocket server
            server = await serve(
                handler,
                sock=sock,
                # Additional options for stability
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            print(f"✅ Server started successfully on {host}:{port}")
            return server
            
        except OSError as e:
            if e.errno == 10048:  # Windows: Port already in use
                print(f"⚠️ Port {port} is busy, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
            raise
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            raise
    
    raise Exception(f"Failed to bind to port {port} after {max_retries} attempts")


# Example usage for audio bridge
async def audio_bridge_with_retry(handler, host='127.0.0.1', port=8765):
    """Start audio bridge with retry and SO_REUSEADDR"""
    return await start_websocket_server_with_retry(handler, host, port)


# Example usage for concept bridge  
async def concept_bridge_with_retry(handler, host='127.0.0.1', port=8766):
    """Start concept bridge with retry and SO_REUSEADDR"""
    return await start_websocket_server_with_retry(handler, host, port)


if __name__ == "__main__":
    print("""
🔧 Bridge Port Fix Helper
========================

This module provides functions to fix port binding issues:
- create_reusable_server_socket() - Creates socket with SO_REUSEADDR
- start_websocket_server_with_retry() - Starts server with retry logic

To use in your bridges:

1. Import this module:
   from fix_bridge_ports import start_websocket_server_with_retry

2. Replace websockets.serve() with:
   server = await start_websocket_server_with_retry(self.handle_client, self.host, self.port)

3. The server will retry up to 5 times with SO_REUSEADDR enabled
""")
