# run_embedding_service.py - Launcher that ensures av.logging fix is applied
import sys
import os

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

# Now import and run the service
from serve_embeddings_noauth import app
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("EMBED_PORT", 8080))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        reload=False,
        access_log=True
    )
