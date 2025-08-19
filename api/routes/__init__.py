# API Routes Package
"""
Central export point for all API route modules.
Import and register these routers in the main FastAPI application.
"""

# Import all routers with error handling
routers_available = {}

# Memory routes
try:
    from .memory_routes import router as memory_router
    routers_available['memory'] = memory_router
except ImportError as e:
    print(f"Warning: Could not import memory_routes: {e}")
    memory_router = None

# Chat routes
try:
    from .chat_routes import router as chat_router
    routers_available['chat'] = chat_router
except ImportError as e:
    print(f"Warning: Could not import chat_routes: {e}")
    chat_router = None

# PDF routes
try:
    from .pdf_routes import router as pdf_router
    routers_available['pdf'] = pdf_router
except ImportError as e:
    print(f"Warning: Could not import pdf_routes: {e}")
    pdf_router = None

# Soliton routes
try:
    from .soliton import router as soliton_router
    routers_available['soliton'] = soliton_router
except ImportError as e:
    print(f"Warning: Could not import soliton routes: {e}")
    soliton_router = None

# Concept mesh routes
try:
    from .concept_mesh import router as concept_mesh_router
    routers_available['concept_mesh'] = concept_mesh_router
except ImportError as e:
    print(f"Warning: Could not import concept_mesh routes: {e}")
    concept_mesh_router = None

# Phase visualization routes
try:
    from .phase_visualization import router as phase_router
    routers_available['phase'] = phase_router
except ImportError as e:
    print(f"Warning: Could not import phase_visualization routes: {e}")
    phase_router = None

# Hologram routes
try:
    from .hologram import router as hologram_router
    routers_available['hologram'] = hologram_router
except ImportError as e:
    print(f"Warning: Could not import hologram routes: {e}")
    hologram_router = None

# V1 aggregated routes
try:
    from .v1 import api_v1_router
    routers_available['v1'] = api_v1_router
except ImportError as e:
    print(f"Warning: Could not import v1 routes: {e}")
    api_v1_router = None

# Export all available routers
__all__ = [
    'memory_router',
    'chat_router', 
    'pdf_router',
    'soliton_router',
    'concept_mesh_router',
    'phase_router',
    'hologram_router',
    'api_v1_router',
    'routers_available'
]

def get_available_routers():
    """Return a list of all successfully imported routers"""
    return {name: router for name, router in routers_available.items() if router is not None}

def register_all_routes(app):
    """
    Helper function to register all available routes with a FastAPI app.
    Usage: register_all_routes(app)
    """
    for name, router in get_available_routers().items():
        try:
            app.include_router(router)
            print(f"✅ Registered {name} router")
        except Exception as e:
            print(f"❌ Failed to register {name} router: {e}")
