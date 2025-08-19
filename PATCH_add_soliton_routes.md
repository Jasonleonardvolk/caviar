# Add Soliton Routes to Prajna API

Add this after line 33 (after the TONKA import block):

```python
# Import soliton routes
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from api.routes.soliton import router as soliton_router
    SOLITON_ROUTES_AVAILABLE = True
    print("[Prajna] Soliton routes loaded successfully!")
except ImportError as e:
    print(f"[Prajna] Soliton routes not available: {e}")
    soliton_router = None
    SOLITON_ROUTES_AVAILABLE = False
```

Then add this after line 473 (after the TONKA router is included):

```python
    # Add soliton router if available
    if SOLITON_ROUTES_AVAILABLE and soliton_router:
        app.include_router(soliton_router)
        logger.info("[STARTUP] Soliton memory endpoints added at /api/soliton/*")
    else:
        logger.warning("[STARTUP] Soliton routes not available - memory features disabled")
```

This will register all the soliton endpoints under /api/soliton/*
