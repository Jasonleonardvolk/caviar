#!/usr/bin/env python3
"""Hologram Bridge - WebSocket server for holographic updates"""

import asyncio
import websockets
import json

async def handle_client(websocket, path):
    print(f"Client connected: {websocket.remote_address}")
    
    try:
        # Send initial state
        await websocket.send(json.dumps({
            "type": "hologram_ready",
            "capabilities": ["webgpu", "fft", "multiview"]
        }))
        
        # Handle messages
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data['type']}")
            
            # Echo back for now
            await websocket.send(json.dumps({
                "type": "ack",
                "original": data['type']
            }))
            
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    print("🔮 Hologram Bridge starting on port 8767...")
    async with websockets.serve(handle_client, "localhost", 8767):
        print("✅ Hologram Bridge ready!")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
