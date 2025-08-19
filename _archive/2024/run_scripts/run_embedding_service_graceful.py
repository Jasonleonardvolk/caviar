# run_embedding_service_graceful.py - Embedding service with graceful shutdown
import sys
import os
import signal
import asyncio
import uvicorn
from contextlib import asynccontextmanager

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Apply the av.logging fix
import importlib, types
try:
    av = importlib.import_module("av")
    if not hasattr(av, "logging"):
        av.logging = types.SimpleNamespace(
            ERROR=0,
            WARNING=1,
            INFO=2,
            DEBUG=3,
            set_level=lambda *_, **__: None,
        )
except ModuleNotFoundError:
    pass

# Global shutdown event
shutdown_event = asyncio.Event()

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    print("\n🛑 Shutdown signal received. Closing gracefully...")
    shutdown_event.set()

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if sys.platform == "win32":
    signal.signal(signal.SIGBREAK, signal_handler)

# Import after fixing av.logging
from serve_embeddings_noauth import app

async def run_server():
    """Run the server with graceful shutdown"""
    port = int(os.getenv("EMBED_PORT", 8080))
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    
    # Run server in background task
    server_task = asyncio.create_task(server.serve())
    
    # Wait for shutdown signal
    await shutdown_event.wait()
    
    # Gracefully shutdown
    print("🔄 Shutting down server...")
    await server.shutdown()
    
    # Wait for server to finish
    await server_task
    print("✅ Server shutdown complete")

if __name__ == "__main__":
    print("🚀 Starting TORI Embedding Service with graceful shutdown support")
    print("📍 Press Ctrl+C to shutdown gracefully")
    
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\n✅ Shutdown complete")
    except Exception as e:
        print(f"\n❌ Error during shutdown: {e}")
