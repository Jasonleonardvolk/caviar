"""
Add Soliton routes to Prajna API
This patch adds the missing soliton API endpoints
"""

# Add this after the other imports in prajna_api.py

# Import soliton routes
try:
    # Fix import path for soliton routes
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from api.routes.soliton import router as soliton_router
    SOLITON_ROUTES_AVAILABLE = True
    print("[Prajna] Soliton routes loaded successfully!")
except ImportError as e:
    print(f"[Prajna] Soliton routes not available: {e}")
    soliton_router = None
    SOLITON_ROUTES_AVAILABLE = False

# Then add this after app creation (after app = FastAPI(...))

# Include soliton routes if available
if SOLITON_ROUTES_AVAILABLE and soliton_router:
    app.include_router(soliton_router)
    print("[Prajna] Soliton routes registered at /api/soliton/*")
else:
    print("[Prajna] WARNING: Soliton routes not included - import failed")

# That's it! The soliton routes should now be available
