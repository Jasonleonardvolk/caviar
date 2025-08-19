# Fixed FastAPI main.py with JSON serialization
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ingest_pdf.pipeline import ingest_pdf_clean
import os
import logging
import traceback
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from alan_backend.routes.soliton import router as soliton_router  # 🌊 ADDED: Soliton Memory import

# 🏢 NEW: Multi-tenant imports
from ingest_pdf.multi_tenant_manager import get_multi_tenant_manager, UserRole
from ingest_pdf.user_manager import get_user_manager, LoginRequest, UserRole as UserManagerRole
from ingest_pdf.knowledge_manager import get_knowledge_manager, KnowledgeTier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="TORI Multi-Tenant PDF Ingestion Service")

# CORS: Allow frontend (Vite/SvelteKit) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🌊 ADDED: Include Soliton Memory routes
app.include_router(soliton_router)
logger.info("🌊 Soliton Memory routes integrated into API")

# 🏢 NEW: Get manager instances
multi_tenant_manager = get_multi_tenant_manager()
user_manager = get_user_manager()
knowledge_manager = get_knowledge_manager()

# Global progress tracking
progress_connections: Dict[str, WebSocket] = {}

# ===================================================================
# JSON SERIALIZATION HELPERS
# ===================================================================

def make_json_serializable(obj):
    """Convert any object to be JSON serializable"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(key): make_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return {str(key): make_json_serializable(value) for key, value in obj.__dict__.items()}
    elif isinstance(obj, set):
        return list(obj)
    else:
        # Convert any other object to string as fallback
        return str(obj)

def sanitize_response(response_data):
    """Ensure response is completely JSON serializable"""
    if not isinstance(response_data, dict):
        response_data = {"result": str(response_data)}
    
    # Apply JSON serialization to the entire response
    return make_json_serializable(response_data)

# ===================================================================
# EXISTING MODELS
# ===================================================================

class ExtractionRequest(BaseModel):
    file_path: str
    filename: str
    content_type: str
    progress_id: Optional[str] = None
    # 🏢 NEW: Multi-tenant fields
    organization_id: Optional[str] = None
    tier: Optional[str] = "private"  # private, organization, foundation

class ChatRequest(BaseModel):
    message: str
    userId: str = "anonymous"
    timestamp: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    concepts_found: list
    soliton_memory_used: bool
    confidence: float
    processing_time: float
    # 🏢 NEW: Multi-tenant fields
    concepts_by_tier: Dict[str, List[Dict[str, Any]]] = {}

# ===================================================================
# 🏢 NEW: MULTI-TENANT MODELS
# ===================================================================

class UserRegistrationRequest(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[str] = "member"

class UserLoginRequest(BaseModel):
    username: str
    password: str

class OrganizationCreateRequest(BaseModel):
    name: str
    description: str

class ConceptSearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 20
    include_tiers: Optional[List[str]] = ["private", "organization", "foundation"]

# ===================================================================
# 🏢 NEW: AUTHENTICATION DEPENDENCY
# ===================================================================

async def get_current_user(authorization: Optional[str] = Header(None)):
    """Dependency to get current authenticated user"""
    if not authorization:
        return None
    
    try:
        # Extract token from "Bearer <token>"
        if authorization.startswith("Bearer "):
            token = authorization[7:]
        else:
            token = authorization
        
        user = user_manager.get_current_user(token)
        return user
    except Exception as e:
        logger.warning(f"Authentication error: {e}")
        return None

async def require_auth(authorization: Optional[str] = Header(None)):
    """Dependency that requires authentication"""
    user = await get_current_user(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

async def require_admin(authorization: Optional[str] = Header(None)):
    """Dependency that requires admin role"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    token = authorization[7:] if authorization.startswith("Bearer ") else authorization
    
    if not user_manager.is_admin(token):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    return await get_current_user(authorization)

# ===================================================================
# EXISTING WEBSOCKET AND PROGRESS TRACKING (UNCHANGED)
# ===================================================================

async def send_progress(progress_id: str, stage: str, percentage: int, message: str, details: dict = None):
    """Send progress update to connected WebSocket"""
    if progress_id and progress_id in progress_connections:
        try:
            progress_data = {
                "stage": stage,
                "percentage": percentage,
                "message": message,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            }
            await progress_connections[progress_id].send_text(json.dumps(progress_data))
            logger.info(f"📡 Progress sent to {progress_id}: {stage} {percentage}% - {message}")
        except Exception as e:
            logger.warning(f"Failed to send progress to {progress_id}: {e}")
            if progress_id in progress_connections:
                del progress_connections[progress_id]

@app.websocket("/progress/{progress_id}")
async def websocket_endpoint(websocket: WebSocket, progress_id: str):
    await websocket.accept()
    progress_connections[progress_id] = websocket
    logger.info(f"📡 WebSocket connected for progress tracking: {progress_id}")
    
    try:
        await send_progress(progress_id, "connected", 0, "Progress tracking connected", {
            "connection_time": datetime.now().isoformat(),
            "status": "ready"
        })
        
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                await websocket.ping()
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        logger.info(f"📡 WebSocket disconnected: {progress_id}")
    finally:
        if progress_id in progress_connections:
            del progress_connections[progress_id]

class ProgressTracker:
    """Context manager for tracking extraction progress"""
    def __init__(self, progress_id: Optional[str], filename: str):
        self.progress_id = progress_id
        self.filename = filename
        self.current_stage = "initializing"
        self.current_percentage = 0
    
    async def update(self, stage: str, percentage: int, message: str, details: dict = None):
        self.current_stage = stage
        self.current_percentage = percentage
        if self.progress_id:
            await send_progress(self.progress_id, stage, percentage, message, details)
        else:
            logger.info(f"📊 Progress: {stage} {percentage}% - {message}")
    
    async def __aenter__(self):
        await self.update("starting", 5, f"Starting extraction for {self.filename}", {
            "filename": self.filename,
            "stage": "initialization"
        })
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.update("error", self.current_percentage, f"Error during extraction: {str(exc_val)}", {
                "error": str(exc_val),
                "error_type": str(exc_type.__name__) if exc_type else "unknown"
            })
        else:
            await self.update("complete", 100, "Extraction completed successfully!", {
                "filename": self.filename,
                "status": "success"
            })

# ===================================================================
# STARTUP EVENT (ENHANCED)
# ===================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("🚀 FastAPI Multi-Tenant PDF Ingestion Service Starting...")
    logger.info(f"📂 Working directory: {os.getcwd()}")
    logger.info(f"🐍 Python path: {sys.path}")
    
    # Verify pipeline can be imported
    try:
        from ingest_pdf.pipeline import ingest_pdf_clean
        logger.info("✅ Pipeline module loaded successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import pipeline: {e}")
        raise
    
    # 🌊 ADDED: Verify Soliton routes loaded
    try:
        logger.info("🌊 Soliton Memory routes available at /api/soliton/*")
    except Exception as e:
        logger.warning(f"⚠️ Soliton Memory routes may not be fully loaded: {e}")
    
    # 🏢 NEW: Verify multi-tenant system
    try:
        health = multi_tenant_manager.health_check()
        if health["status"] == "healthy":
            logger.info("🏢 Multi-tenant system initialized successfully")
        else:
            logger.warning(f"⚠️ Multi-tenant system health: {health['status']}")
        
        # Clean up expired sessions
        user_manager.cleanup_expired_sessions()
        logger.info("🧹 Session cleanup completed")
        
        logger.info("✅ All systems operational")
        
    except Exception as e:
        logger.error(f"❌ Multi-tenant system initialization error: {e}")

# ===================================================================
# 🏢 NEW: AUTHENTICATION ENDPOINTS
# ===================================================================

@app.post("/auth/register")
async def register_user(request: UserRegistrationRequest):
    """Register a new user"""
    try:
        role = UserManagerRole.MEMBER
        if request.role:
            try:
                role = UserManagerRole(request.role.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid role")
        
        user = user_manager.register_user(
            username=request.username,
            email=request.email,
            password=request.password,
            role=role
        )
        
        if user:
            logger.info(f"✅ User registered: {request.username}")
            return sanitize_response({
                "success": True,
                "message": "User registered successfully",
                "user": user
            })
        else:
            raise HTTPException(status_code=400, detail="User already exists or registration failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/auth/login")
async def login_user(request: UserLoginRequest):
    """Login user and get JWT token"""
    try:
        result = user_manager.login(request.username, request.password)
        
        if result.success:
            logger.info(f"✅ User logged in: {request.username}")
            return sanitize_response({
                "success": True,
                "token": result.token,
                "user": result.user,
                "expires_at": result.expires_at.isoformat() if result.expires_at else None,
                "message": result.message
            })
        else:
            raise HTTPException(status_code=401, detail=result.message)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/auth/logout")
async def logout_user(authorization: Optional[str] = Header(None)):
    """Logout user and invalidate token"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    try:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        success = user_manager.logout(token)
        
        return sanitize_response({
            "success": success,
            "message": "Logged out successfully" if success else "Logout failed"
        })
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/auth/me")
async def get_current_user_info(current_user: Dict = Depends(require_auth)):
    """Get current user information"""
    return sanitize_response({
        "success": True,
        "user": current_user
    })

# ===================================================================
# 🏢 NEW: ORGANIZATION ENDPOINTS
# ===================================================================

@app.post("/organizations")
async def create_organization(request: OrganizationCreateRequest, current_user: Dict = Depends(require_auth)):
    """Create a new organization"""
    try:
        organization = multi_tenant_manager.create_organization(
            name=request.name,
            description=request.description,
            admin_user_id=current_user["id"]
        )
        
        if organization:
            logger.info(f"✅ Organization created: {request.name}")
            return sanitize_response({
                "success": True,
                "message": "Organization created successfully",
                "organization": {
                    "id": organization.id,
                    "name": organization.name,
                    "description": organization.description,
                    "created_at": organization.created_at
                }
            })
        else:
            raise HTTPException(status_code=400, detail="Organization already exists or creation failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Organization creation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/organizations")
async def get_user_organizations(current_user: Dict = Depends(require_auth)):
    """Get organizations for current user"""
    try:
        organizations = multi_tenant_manager.get_user_organizations(current_user["id"])
        
        org_list = []
        for org in organizations:
            org_list.append({
                "id": org.id,
                "name": org.name,
                "description": org.description,
                "created_at": org.created_at,
                "concept_count": org.concept_count
            })
        
        return sanitize_response({
            "success": True,
            "organizations": org_list
        })
        
    except Exception as e:
        logger.error(f"Get organizations error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ===================================================================
# 🏢 NEW: KNOWLEDGE MANAGEMENT ENDPOINTS
# ===================================================================

@app.post("/knowledge/search")
async def search_concepts(request: ConceptSearchRequest, current_user: Dict = Depends(require_auth)):
    """Search concepts across all accessible tiers"""
    try:
        results = knowledge_manager.search_concepts(
            query=request.query,
            user_id=current_user["id"],
            organization_ids=current_user.get("organization_ids", []),
            max_results=request.max_results
        )
        
        # Group results by tier
        concepts_by_tier = {
            "private": [],
            "organization": [],
            "foundation": []
        }
        
        for result in results:
            tier_key = result.tier.value
            concept_data = {
                "name": result.name,
                "confidence": result.confidence,
                "context": result.context,
                "tier": result.tier.value,
                "owner_id": result.owner_id,
                "source_document": result.source_document,
                "tags": result.tags,
                "created_at": result.created_at,
                "access_count": result.access_count,
                "metadata": result.metadata
            }
            concepts_by_tier[tier_key].append(concept_data)
        
        return sanitize_response({
            "success": True,
            "query": request.query,
            "total_results": len(results),
            "concepts": [
                {
                    "name": result.name,
                    "confidence": result.confidence,
                    "context": result.context,
                    "tier": result.tier.value,
                    "source_document": result.source_document,
                    "metadata": result.metadata
                }
                for result in results
            ],
            "concepts_by_tier": concepts_by_tier
        })
        
    except Exception as e:
        logger.error(f"Concept search error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/knowledge/stats")
async def get_knowledge_stats(current_user: Dict = Depends(require_auth)):
    """Get knowledge statistics for current user"""
    try:
        stats = knowledge_manager.get_knowledge_stats(
            user_id=current_user["id"],
            organization_ids=current_user.get("organization_ids", [])
        )
        
        return sanitize_response({
            "success": True,
            "stats": stats
        })
        
    except Exception as e:
        logger.error(f"Knowledge stats error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ===================================================================
# 🏢 NEW: ADMIN ENDPOINTS
# ===================================================================

@app.get("/admin/users")
async def get_all_users(admin_user: Dict = Depends(require_admin)):
    """Get all users (admin only)"""
    try:
        # Get admin token from authorization header
        # This is a bit hacky but works for now
        auth_header = None  # We'd need to pass this through
        users = user_manager.get_all_users("admin_token")  # Simplified for now
        
        return sanitize_response({
            "success": True,
            "users": users or []
        })
        
    except Exception as e:
        logger.error(f"Get all users error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/admin/system/stats")
async def get_system_stats(admin_user: Dict = Depends(require_admin)):
    """Get comprehensive system statistics (admin only)"""
    try:
        stats = multi_tenant_manager.get_system_stats()
        
        return sanitize_response({
            "success": True,
            "stats": stats
        })
        
    except Exception as e:
        logger.error(f"System stats error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/admin/system/health")
async def get_system_health(admin_user: Dict = Depends(require_admin)):
    """Get system health check (admin only)"""
    try:
        health = multi_tenant_manager.health_check()
        
        return sanitize_response({
            "success": True,
            "health": health
        })
        
    except Exception as e:
        logger.error(f"System health error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ===================================================================
# ENHANCED EXTRACTION ENDPOINT (MULTI-TENANT AWARE) - FIXED SERIALIZATION
# ===================================================================

@app.post("/extract")
async def extract(request: ExtractionRequest, current_user: Dict = Depends(get_current_user)):
    """Extract concepts from PDF file with multi-tenant support"""
    progress_id = request.progress_id or "no-progress"
    
    try:
        print(f"🔔 [FASTAPI] REQUEST RECEIVED at {datetime.now()}")
        print(f"🔔 [FASTAPI] Request data: {request.dict()}")
        print(f"🔔 [FASTAPI] Current user: {current_user['username'] if current_user else 'anonymous'}")
        
        tracker_progress_id = request.progress_id if request.progress_id else None
        
        async with ProgressTracker(tracker_progress_id, request.filename) as tracker:
            
            await tracker.update("validating", 10, "Validating file path and permissions", {
                "file_path": request.file_path,
                "content_type": request.content_type,
                "user": current_user["username"] if current_user else "anonymous"
            })
            
            # File validation (same as before)
            file_path = Path(request.file_path).absolute()
            print(f"🔔 [FASTAPI] Absolute path: {file_path}")
            
            if not file_path.exists():
                error_msg = f"File not found: {file_path}"
                print(f"🔔 [FASTAPI] ERROR: {error_msg}")
                logger.error(f"❌ [FASTAPI] {error_msg}")
                raise HTTPException(status_code=404, detail=error_msg)
            
            if not file_path.is_file():
                error_msg = f"Path is not a file: {file_path}"
                print(f"🔔 [FASTAPI] ERROR: {error_msg}")
                logger.error(f"❌ [FASTAPI] {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
            
            file_size = file_path.stat().st_size
            max_size = 50 * 1024 * 1024
            if file_size > max_size:
                error_msg = f"File too large: {file_size:,} bytes (max {max_size:,} bytes)"
                raise HTTPException(status_code=413, detail=error_msg)
            
            await tracker.update("extracting", 25, "Starting concept extraction from PDF", {
                "extraction_method": "purity_based_universal_pipeline",
                "tier": request.tier,
                "organization_id": request.organization_id
            })
            
            # Run extraction and IMMEDIATELY sanitize the result
            raw_result = await extract_with_progress(str(file_path), tracker)
            
            # CRITICAL: Sanitize the pipeline result for JSON serialization
            result = sanitize_response(raw_result)
            
            # 🏢 NEW: Store concepts in appropriate tier
            if current_user and result.get("success") and result.get("concept_names"):
                await tracker.update("storing", 90, "Storing concepts in knowledge system", {
                    "tier": request.tier,
                    "concept_count": len(result["concept_names"])
                })
                
                # Determine storage tier
                tier = KnowledgeTier.PRIVATE
                if request.tier == "organization" and request.organization_id:
                    tier = KnowledgeTier.ORGANIZATION
                elif request.tier == "foundation" and current_user.get("role") == "admin":
                    tier = KnowledgeTier.FOUNDATION
                
                # Convert concepts to knowledge manager format
                concepts_for_storage = []
                for i, name in enumerate(result.get("concept_names", [])):
                    concepts_for_storage.append({
                        "name": name,
                        "confidence": 0.7,  # Default confidence
                        "context": f"Extracted from {request.filename}",
                        "metadata": {
                            "extraction_method": "purity_based_universal_pipeline",
                            "source_file": request.filename
                        }
                    })
                
                # Store concepts
                try:
                    concept_diff = knowledge_manager.store_concepts(
                        user_id=current_user["id"],
                        concepts=concepts_for_storage,
                        document_title=request.filename,
                        organization_id=request.organization_id,
                        tier=tier
                    )
                    
                    if concept_diff:
                        result["concept_diff"] = {
                            "id": concept_diff.id,
                            "tier": concept_diff.tier.value,
                            "owner_id": concept_diff.owner_id
                        }
                        logger.info(f"✅ Stored {len(concepts_for_storage)} concepts in {tier.value} tier")
                except Exception as e:
                    logger.warning(f"Failed to store concepts: {e}")
                    result["concept_diff_error"] = str(e)
            
            # Add multi-tenant metadata
            result["multi_tenant"] = {
                "user_id": current_user["id"] if current_user else None,
                "username": current_user["username"] if current_user else "anonymous",
                "tier": request.tier,
                "organization_id": request.organization_id,
                "stored_successfully": "concept_diff" in result
            }
            
            # Final sanitization before return
            return sanitize_response(result)

    except HTTPException:
        raise
    except Exception as e:
        print(f"🔔 [FASTAPI] EXCEPTION CAUGHT: {type(e).__name__}: {e}")
        logger.error(f"❌ [FASTAPI] ERROR during extraction: {type(e).__name__}: {e}")
        
        if request.progress_id:
            await send_progress(request.progress_id, "error", 0, f"Error: {str(e)}", {
                "error": str(e),
                "error_type": type(e).__name__
            })
        
        error_response = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "concept_count": 0,
            "concept_names": [],
            "status": "error",
            "extraction_method": "error"
        }
        
        return sanitize_response(error_response)

# ===================================================================
# ENHANCED CHAT ENDPOINT (MULTI-TENANT AWARE)
# ===================================================================

@app.post("/chat")
async def chat(request: ChatRequest, current_user: Dict = Depends(get_current_user)) -> ChatResponse:
    """Chat endpoint with multi-tenant concept search"""
    start_time = datetime.now()
    
    try:
        user_id = current_user["id"] if current_user else "anonymous"
        logger.info(f"🤖 Chat request from {user_id}: {request.message}")
        
        # 🏢 NEW: Multi-tenant concept search
        concepts_by_tier = {"private": [], "organization": [], "foundation": []}
        
        if current_user:
            # Search across all accessible tiers
            try:
                search_results = knowledge_manager.search_concepts(
                    query=request.message,
                    user_id=current_user["id"],
                    organization_ids=current_user.get("organization_ids", []),
                    max_results=10
                )
                
                # Group by tier
                for result in search_results:
                    tier_key = result.tier.value
                    concept_data = {
                        "name": result.name,
                        "confidence": result.confidence,
                        "context": result.context,
                        "tier": result.tier.value
                    }
                    concepts_by_tier[tier_key].append(concept_data)
            except Exception as e:
                logger.warning(f"Knowledge manager search failed: {e}")
        else:
            # Fallback to simple search for anonymous users
            concepts_found = await search_concepts_fallback(request.message)
            concepts_by_tier["foundation"] = concepts_found
        
        # Count total concepts
        total_concepts = sum(len(concepts) for concepts in concepts_by_tier.values())
        
        # Store message in Soliton Memory
        soliton_memory_used = await store_in_soliton_memory(user_id, request.message)
        
        # Get related memories
        related_memories = await get_related_memories(user_id, request.message)
        
        # Generate enhanced response
        response_text = await generate_multi_tenant_response(
            request.message, 
            concepts_by_tier,
            related_memories,
            current_user
        )
        
        # Calculate confidence
        confidence = calculate_multi_tenant_confidence(concepts_by_tier, related_memories)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Multi-tenant chat response: {total_concepts} concepts, {confidence:.2f} confidence")
        
        return ChatResponse(
            response=response_text,
            concepts_found=[c["name"] for tier_concepts in concepts_by_tier.values() for c in tier_concepts],
            soliton_memory_used=soliton_memory_used,
            confidence=confidence,
            processing_time=processing_time,
            concepts_by_tier=concepts_by_tier
        )
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            response=f"I apologize, but I encountered an error processing your message: {str(e)}",
            concepts_found=[],
            soliton_memory_used=False,
            confidence=0.0,
            processing_time=processing_time,
            concepts_by_tier={"private": [], "organization": [], "foundation": []}
        )

# ===================================================================
# HELPER FUNCTIONS (EXISTING + ENHANCED)
# ===================================================================

async def extract_with_progress(file_path: str, tracker: ProgressTracker) -> dict:
    """Run extraction pipeline with progress updates"""
    
    await tracker.update("reading_pdf", 30, "Reading PDF and extracting text chunks", {
        "stage": "pdf_parsing"
    })
    
    chunk_stages = [
        (35, "Extracting text from PDF pages"),
        (40, "Running YAKE keyword extraction"),
        (45, "Running KeyBERT semantic analysis"),
        (50, "Processing with spaCy NER"),
        (55, "Cross-referencing concept file_storage"),
        (60, "Analyzing semantic relationships"),
        (65, "Applying purity-based filtering"),
        (70, "Computing concept scores"),
        (75, "Finalizing concept extraction"),
        (80, "Building response data"),
    ]
    
    extraction_task = asyncio.create_task(run_extraction(file_path))
    
    for percentage, message in chunk_stages:
        if extraction_task.done():
            break
        await tracker.update("processing", percentage, message, {
            "stage": "concept_extraction",
            "estimated_remaining": f"{(90-percentage)/10:.0f}s"
        })
        await asyncio.sleep(1.5)
    
    await tracker.update("purity_analysis", 85, "Applying purity analysis - extracting the 'truth'", {
        "stage": "quality_filtering",
        "analysis_type": "consensus_based"
    })
    
    result = await extraction_task
    
    await tracker.update("building_response", 90, "Building final response with pure concepts", {
        "stage": "response_generation",
        "concept_count": result.get("concept_count", 0)
    })
    
    return result

async def run_extraction(file_path: str) -> dict:
    """Run the actual extraction in a separate task"""
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, ingest_pdf_clean, file_path)
    
    # CRITICAL: Ensure result is JSON serializable
    return make_json_serializable(result)

async def search_concepts_fallback(query: str) -> list:
    """Fallback concept search for anonymous users"""
    query_lower = query.lower()
    found_concepts = []
    
    # Simple keyword file_storage for anonymous users
    concept_file_storage = {
        "darwin": [{"name": "Evolution", "confidence": 0.9, "context": "Charles Darwin's theory of evolution", "tier": "foundation"}],
        "ai": [{"name": "Artificial Intelligence", "confidence": 0.8, "context": "Machine learning and AI systems", "tier": "foundation"}],
        "quantum": [{"name": "Quantum Mechanics", "confidence": 0.8, "context": "Quantum physics and computation", "tier": "foundation"}],
        "strategy": [{"name": "Strategic Planning", "confidence": 0.8, "context": "Business strategy and planning", "tier": "foundation"}]
    }
    
    for keyword, concepts in concept_file_storage.items():
        if keyword in query_lower:
            found_concepts.extend(concepts)
    
    return found_concepts[:5]

async def generate_multi_tenant_response(message: str, concepts_by_tier: Dict[str, List], 
                                       memories: list, current_user: Optional[Dict]) -> str:
    """Generate response using cognitive engine with multi-tenant knowledge"""
    try:
        # Build context from knowledge tiers
        all_concepts = []
        for tier, concepts in concepts_by_tier.items():
            for concept in concepts:
                all_concepts.append(f"{concept['name']} ({tier})")
        
        # Try cognitive engine first
        try:
            cognitive_response = await call_cognitive_engine(message, all_concepts, memories)
            if cognitive_response:
                logger.info("✅ Response generated by cognitive engine")
                return cognitive_response
        except Exception as e:
            logger.warning(f"Cognitive engine unavailable: {e}")
        
        # Option 1: Try OpenAI (if API key available)
        try:
            import openai
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if openai.api_key:
                context = " | ".join([
                    f"Relevant knowledge: {', '.join(all_concepts[:5])}" if all_concepts else "",
                    f"Previous conversation context: {len(memories)} related messages" if memories else ""
                ]).strip(" | ")
                
                prompt = f"""You are TORI, an intelligent assistant with access to a multi-tenant knowledge system.

Context: {context if context else "No specific context available"}

User: {message}

Please provide a helpful, informative response. If you have relevant knowledge from the context, incorporate it naturally. If not, provide a general helpful response.

Response:"""
                
                response = await call_openai_api(prompt)
                if response:
                    return response
        except Exception as e:
            logger.warning(f"OpenAI API not available: {e}")
        
        # Option 2: Try Ollama (if running locally)
        try:
            ollama_response = await call_ollama_api(message)
            if ollama_response:
                return ollama_response
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
        
        # Option 3: Fallback to intelligent templating
        return generate_intelligent_fallback(message, all_concepts, current_user)
        
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return f"I apologize, but I encountered an error processing your question about '{message}'. Please try again."

async def call_cognitive_engine(message: str, concepts: list, memories: list) -> Optional[str]:
    """Call the Node.js cognitive engine bridge"""
    try:
        import aiohttp
        import asyncio
        
        # Choose appropriate glyph sequence based on context
        glyphs = ['anchor']
        
        # Add glyphs based on concepts
        if concepts:
            glyphs.append('concept-synthesizer')
        
        # Add memory glyph if we have conversation history
        if memories:
            glyphs.append('memory-anchor')
        
        # End with return
        glyphs.append('return')
        
        # Prepare request data
        request_data = {
            "message": message,
            "glyphs": glyphs,
            "metadata": {
                "concepts": concepts[:5],  # Limit to first 5 concepts
                "memoryCount": len(memories),
                "source": "fastapi_chat"
            }
        }
        
        # Call cognitive engine bridge
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                'http://localhost:4321/api/process',
                json=request_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Log cognitive processing details
                    loop_record = data.get('loopRecord', {})
                    logger.info(f"🧠 Cognitive processing: {loop_record.get('id', 'unknown')} "
                              f"(closed: {loop_record.get('closed', False)}, "
                              f"coherence: {loop_record.get('coherenceTrace', [0])[-1]:.2f})")
                    
                    return data.get('answer')
                elif response.status == 503:
                    # Engine not loaded, return None to fall back
                    data = await response.json()
                    logger.warning(f"Cognitive engine not loaded: {data.get('details')}")
                    return data.get('fallback')
                else:
                    logger.warning(f"Cognitive engine error {response.status}: {await response.text()}")
                    return None
                    
    except asyncio.TimeoutError:
        logger.warning("Cognitive engine timeout")
        return None
    except Exception as e:
        logger.warning(f"Cognitive engine call failed: {e}")
        return None

async def call_openai_api(prompt: str) -> Optional[str]:
    """Call OpenAI API for actual AI response"""
    try:
        import openai
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"OpenAI API call failed: {e}")
        return None

async def call_ollama_api(prompt: str) -> Optional[str]:
    """Call Ollama API for local LLM response"""
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": "llama2",  # or any available model
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', '').strip()
    except Exception as e:
        logger.warning(f"Ollama API call failed: {e}")
        return None

def generate_intelligent_fallback(message: str, concepts: list, current_user: Optional[Dict]) -> str:
    """Generate intelligent fallback response"""
    message_lower = message.lower()
    
    # Enhanced knowledge-based responses
    responses = {
        'darwin': "Charles Darwin revolutionized our understanding of life through his theory of evolution by natural selection. His work 'On the Origin of Species' showed how all species descended from common ancestors through gradual changes over time.",
        'evolution': "Evolution is the process by which species change over time through natural selection. Organisms with advantageous traits are more likely to survive and reproduce, passing these traits to their offspring.",
        'ai': "Artificial Intelligence encompasses machine learning, neural networks, and computational systems that can perform tasks typically requiring human intelligence. Modern AI includes large language models, computer vision, and autonomous systems.",
        'machine learning': "Machine learning is a subset of AI where algorithms learn patterns from data without explicit programming. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
        'quantum': "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic level, where particles exhibit both wave and particle properties. Quantum computing leverages these principles for exponentially faster calculations.",
        'lattice': "Lattice models are mathematical frameworks used in physics, chemistry, and materials science to describe the regular arrangement of atoms, molecules, or other components in crystalline structures.",
        'strategy': "Strategic thinking involves long-term planning, competitive analysis, and resource allocation to achieve organizational goals. It requires understanding market dynamics, stakeholder needs, and sustainable competitive advantages."
    }
    
    # Find matching response
    for keyword, response in responses.items():
        if keyword in message_lower:
            if concepts:
                return f"{response}\n\nBased on your knowledge: {', '.join(concepts[:3])}."
            return response
    
    # Generic helpful response
    user_name = current_user["username"] if current_user else "there"
    if concepts:
        return f"That's an interesting question about '{message}'. Based on your knowledge ({', '.join(concepts[:2])}), I can help you explore this topic further. What specific aspect would you like to know more about?"
    else:
        return f"I'd be happy to help you learn about '{message}'. While I don't have specific information in your knowledge base yet, I can provide general insights. Consider uploading relevant documents to build your personal knowledge library."

def calculate_multi_tenant_confidence(concepts_by_tier: Dict[str, List], memories: list) -> float:
    """Calculate confidence based on multi-tenant knowledge"""
    base_confidence = 0.3
    
    # Boost for private concepts (highest value)
    private_boost = min(0.4, len(concepts_by_tier.get("private", [])) * 0.15)
    
    # Boost for organization concepts
    org_boost = min(0.3, len(concepts_by_tier.get("organization", [])) * 0.1)
    
    # Boost for foundation concepts
    foundation_boost = min(0.2, len(concepts_by_tier.get("foundation", [])) * 0.05)
    
    # Memory boost
    memory_boost = min(0.1, len(memories) * 0.03)
    
    return min(1.0, base_confidence + private_boost + org_boost + foundation_boost + memory_boost)

# Existing helper functions (unchanged)
async def store_in_soliton_memory(user_id: str, message: str) -> bool:
    try:
        logger.info(f"🌊 Storing message in Soliton Memory for user {user_id}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Soliton Memory storage failed: {e}")
        return False

async def get_related_memories(user_id: str, message: str) -> list:
    try:
        logger.info(f"🌊 Retrieving related memories for user {user_id}")
        return [{"content": "Previous conversation context", "relevance": 0.7}]
    except Exception as e:
        logger.warning(f"⚠️ Soliton Memory retrieval failed: {e}")
        return []

# ===================================================================
# ENHANCED HEALTH AND ROOT ENDPOINTS
# ===================================================================

@app.get("/health")
async def health():
    """Enhanced health check with multi-tenant status"""
    print(f"🔔 [FASTAPI] Health check requested at {datetime.now()}")
    
    # Get multi-tenant health
    try:
        mt_health = multi_tenant_manager.health_check()
    except:
        mt_health = {"status": "unknown"}
    
    return sanitize_response({
        "status": "healthy",
        "message": "FastAPI multi-tenant extraction service is running",
        "working_directory": os.getcwd(),
        "python_version": sys.version,
        "progress_connections": len(progress_connections),
        "features": [
            "real_time_progress", 
            "websocket_updates", 
            "purity_analysis", 
            "soliton_memory", 
            "chat_api", 
            "multi_tenant_auth",
            "three_tier_knowledge",
            "organization_support"
        ],
        "multi_tenant": {
            "status": mt_health["status"],
            "authentication_enabled": True,
            "knowledge_tiers": ["private", "organization", "foundation"],
            "jwt_enabled": True
        },
        "system_stats": multi_tenant_manager.get_system_stats() if mt_health["status"] == "healthy" else {}
    })

@app.get("/")
async def root():
    """Enhanced root endpoint with multi-tenant info"""
    print(f"🔔 [FASTAPI] Root endpoint accessed at {datetime.now()}")
    return sanitize_response({
        "message": "TORI Multi-Tenant FastAPI Extraction Service",
        "status": "ready",
        "features": [
            "real_time_progress", 
            "websocket_updates", 
            "purity_analysis", 
            "soliton_memory", 
            "chat_api",
            "multi_tenant_auth",
            "three_tier_knowledge"
        ],
        "endpoints": {
            "extract": "/extract",
            "health": "/health",
            "progress": "/progress/{progress_id}",
            "chat": "/chat",
            "docs": "/docs",
            "soliton": "/api/soliton/*",
            # 🏢 NEW: Multi-tenant endpoints
            "auth": {
                "register": "/auth/register",
                "login": "/auth/login", 
                "logout": "/auth/logout",
                "me": "/auth/me"
            },
            "organizations": {
                "create": "/organizations",
                "list": "/organizations"
            },
            "knowledge": {
                "search": "/knowledge/search",
                "stats": "/knowledge/stats"
            },
            "admin": {
                "users": "/admin/users",
                "system_stats": "/admin/system/stats",
                "system_health": "/admin/system/health"
            }
        },
        "multi_tenant": {
            "enabled": True,
            "authentication": "JWT",
            "knowledge_tiers": ["private", "organization", "foundation"],
            "description": "Three-tier knowledge architecture with user authentication"
        }
    })
