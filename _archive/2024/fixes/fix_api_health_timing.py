#!/usr/bin/env python3
"""
Fix the /api/health endpoint timing issue
Apply this patch to prajna/api/prajna_api.py
"""

# Add this at the TOP of prajna_api.py, right after FastAPI import
# BEFORE any heavy imports or initialization

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create app IMMEDIATELY
app = FastAPI(title="Prajna API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CRITICAL: Define health endpoint FIRST, before ANY other imports
@app.get("/api/health")
async def health_check():
    """Ultra-lightweight health check - available immediately"""
    return {"status": "ok", "message": "API running"}

# NOW do the heavy imports after the health endpoint is registered
# Move all the existing imports here...

print("[API] Health endpoint registered early - will respond immediately")
