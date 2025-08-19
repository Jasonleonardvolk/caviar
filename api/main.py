"""
⚠️ DEPRECATED - DO NOT USE THIS FILE ⚠️
=====================================

This file is no longer used. The actual API is served through:
- prajna/api/prajna_api.py (main API with integrated routes)
- api/routes/soliton.py (actual soliton routes)

This file was importing from api/soliton.py which is also deprecated.

Kept for reference only. Will be removed in future cleanup.
"""

"""
Main API Application
====================

Central FastAPI application that includes all routes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import routers
from .soliton import router as soliton_router
from .concept_mesh import router as concept_mesh_router


# === TORI AUTH ENDPOINTS (QUICK FIX) ===
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional

auth_router = APIRouter(prefix="/api/auth", tags=["authentication"])

class LoginRequest(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    username: str
    email: Optional[str] = None
    role: str = "user"

@auth_router.post("/login")
async def login(credentials: LoginRequest, response: Response):
    """Simple login endpoint for testing"""
    # TODO: Replace with real authentication
    if credentials.username in ["admin", "user", "test"] and credentials.password == credentials.username:
        # Set a simple cookie for now
        response.set_cookie(
            key="tori_session",
            value=f"session_{credentials.username}_123",
            httponly=True,
            samesite="lax"
        )
        return {
            "success": True,
            "user": UserResponse(username=credentials.username, email=f"{credentials.username}@tori.local"),
            "message": "Login successful"
        }
    raise HTTPException(status_code=401, detail="Invalid username or password")

@auth_router.post("/logout")
async def logout(response: Response):
    """Logout endpoint"""
    response.delete_cookie("tori_session")
    return {"success": True, "message": "Logged out successfully"}

@auth_router.get("/status")
async def auth_status():
    """Check authentication status"""
    # TODO: Check real session
    return {
        "authenticated": False,
        "user": None
    }

@auth_router.get("/me")
async def get_current_user():
    """Get current user info"""
    # TODO: Get from real session
    return {
        "authenticated": False,
        "user": None
    }

# === END AUTH ENDPOINTS ===



logger = logging.getLogger(__name__)

# Create main FastAPI app
app = FastAPI(
    title="TORI Main API",
    description="Main API for TORI system including Soliton Memory and Concept Mesh",
    version="1.0.0"
)

# Register auth router
app.include_router(auth_router)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(soliton_router, prefix="/api/soliton", tags=["soliton"])
app.include_router(concept_mesh_router, prefix="/api/concept-mesh", tags=["concept-mesh"])

# Import and include avatar WebSocket router
from api.routes.avatar_ws import router as avatar_ws_router
app.include_router(avatar_ws_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "main-api"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "TORI Main API",
        "endpoints": {
            "soliton": "/api/soliton",
            "concept_mesh": "/api/concept-mesh",
            "health": "/health",
            "docs": "/docs"
        }
    }

logger.info("Main API application initialized")
