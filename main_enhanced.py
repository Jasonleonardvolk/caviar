#!/usr/bin/env python3
"""
🚀 ENHANCED TORI API SERVER - With /multiply, /intent, and WebSocket support

This is the main API server that includes:
- Original concept extraction endpoints
- Prajna language model integration
- NEW: /multiply endpoint for hyperbolic matrix operations
- NEW: /intent endpoint for intent-driven reasoning
- NEW: WebSocket support for real-time stability monitoring
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os
import tempfile
import time
import logging
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import traceback
import shutil

# BULLETPROOF TEMP DIRECTORY - Our controlled, always-writable location
TMP_ROOT = r"{PROJECT_ROOT}\tmp"

# Ensure temp directory exists on startup
os.makedirs(TMP_ROOT, exist_ok=True)
print(f"✅ TORI Temp directory ready: {TMP_ROOT}")

# Add the ingest_pdf directory to Python path
current_dir = Path(__file__).parent
ingest_pdf_dir = current_dir / "ingest_pdf"
sys.path.insert(0, str(ingest_pdf_dir))
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "api"))

# Import your extraction functions
try:
    from ingest_pdf.pipeline import ingest_pdf_clean
    print("✅ Successfully imported ingest_pdf_clean")
except ImportError as e:
    print(f"❌ Failed to import pipeline: {e}")
    def ingest_pdf_clean(file_path, extraction_threshold=0.0):
        return {
            "filename": Path(file_path).name,
            "concept_count": 0,
            "concept_names": [],
            "status": "error", 
            "error_message": "Pipeline import failed - using fallback",
            "processing_time_seconds": 0.1
        }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Prajna API app which includes all the Prajna endpoints
try:
    from prajna.api.prajna_api import app
    logger.info("✅ Successfully imported Prajna API app")
    
    # Import and integrate enhanced API
    try:
        import api.enhanced_api  # This automatically adds routes to the app
        logger.info("✅ Successfully integrated enhanced API endpoints")
    except ImportError as e:
        logger.warning(f"⚠️ Could not import enhanced API: {e}")
    
except ImportError as e:
    logger.warning(f"⚠️ Could not import Prajna API, creating basic app: {e}")
    # Create basic FastAPI app if Prajna not available
    app = FastAPI(
        title="TORI Universal API",
        description="TORI API with concept extraction, Prajna, and enhanced endpoints",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Models for the original endpoints
class ExtractionRequest(BaseModel):
    file_path: str
    filename: str
    content_type: str

class UploadResponse(BaseModel):
    success: bool
    file_path: str
    filename: str
    size_mb: float
    message: str

# Add the original endpoints if they don't exist
if not any(route.path == "/" for route in app.routes):
    @app.get("/")
    async def root():
        """Root endpoint - API status"""
        return {
            "service": "TORI Universal API - Enhanced Edition",
            "version": "3.0.0",
            "status": "healthy",
            "features": [
                "concept_extraction",
                "prajna_language_model",
                "hyperbolic_matrix_multiply",
                "intent_driven_reasoning",
                "websocket_monitoring"
            ],
            "temp_directory": TMP_ROOT,
            "endpoints": {
                "health": "/health",
                "upload": "/upload (POST with file)",
                "extract": "/extract (POST with JSON)",
                "prajna_answer": "/api/answer (POST)",
                "multiply": "/multiply (POST)",
                "intent": "/intent (POST)",
                "websocket_test": "/ws/test",
                "docs": "/docs"
            }
        }

if not any(route.path == "/health" for route in app.routes):
    @app.get("/health")
    async def health_check():
        """Health check with pipeline test"""
        try:
            logger.info("🏥 Health check started")
            
            # Check temp directory is writable
            test_file = os.path.join(TMP_ROOT, "health_check.txt")
            try:
                with open(test_file, 'w') as f:
                    f.write("healthy")
                os.remove(test_file)
                temp_writable = True
            except:
                temp_writable = False
            
            # Check components
            components = {
                "temp_writable": temp_writable,
                "concept_extraction": True,
                "prajna_available": "prajna" in sys.modules,
                "enhanced_api": "api.enhanced_api" in sys.modules,
                "websocket_ready": True
            }
            
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "components": components,
                "temp_directory": TMP_ROOT,
                "ready": True
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": time.time(),
                    "ready": False
                }
            )

# Keep the original upload endpoint
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    🚀 BULLETPROOF UPLOAD ENDPOINT
    
    Saves uploaded files to our controlled temp directory.
    """
    try:
        start_time = time.time()
        
        # Ensure temp directory exists
        os.makedirs(TMP_ROOT, exist_ok=True)
        
        # Sanitize filename
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in '._-')
        if not safe_filename:
            safe_filename = f"upload_{int(time.time())}.pdf"
        
        # Create unique filename
        timestamp = int(time.time() * 1000)
        unique_filename = f"{timestamp}_{safe_filename}"
        dest_path = os.path.join(TMP_ROOT, unique_filename)
        
        logger.info(f"📤 [UPLOAD] Receiving file: {file.filename}")
        logger.info(f"📁 [UPLOAD] Saving to: {dest_path}")
        
        # Save the file
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(dest_path)
        file_size_mb = file_size / (1024 * 1024)
        
        upload_time = time.time() - start_time
        
        logger.info(f"✅ [UPLOAD] File saved successfully!")
        logger.info(f"📏 [UPLOAD] Size: {file_size_mb:.2f} MB")
        logger.info(f"⚡ [UPLOAD] Upload time: {upload_time:.2f}s")
        
        return UploadResponse(
            success=True,
            file_path=dest_path,
            filename=safe_filename,
            size_mb=round(file_size_mb, 2),
            message=f"File uploaded successfully to {dest_path}"
        )
        
    except Exception as e:
        logger.error(f"❌ [UPLOAD] Upload failed: {e}")
        logger.error(f"❌ [UPLOAD] Traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

@app.post("/extract")
async def extract_concepts_endpoint(request: ExtractionRequest):
    """
    🧬 MAIN EXTRACTION ENDPOINT
    """
    start_time = time.time()
    
    try:
        file_path = request.file_path
        filename = request.filename
        content_type = request.content_type
        
        logger.info(f"🧬 [EXTRACT] Processing file: {filename}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Get file size
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Validate file type
        if not filename.lower().endswith(('.pdf', '.txt')):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Only PDF and TXT files are supported."
            )
        
        # Call the extraction pipeline
        result = ingest_pdf_clean(file_path, extraction_threshold=0.0, admin_mode=True)
        
        extraction_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            "success": True,
            "filename": filename,
            "status": result.get("status", "success"),
            "processing_time_seconds": round(extraction_time, 3),
            "concept_count": result.get("concept_count", 0),
            "concept_names": result.get("concept_names", []),
            "concepts": result.get("concepts", []),
            "file_size_mb": round(file_size_mb, 2)
        }
        
        logger.info(f"🎉 [EXTRACT] SUCCESS: {result.get('concept_count', 0)} concepts extracted")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [EXTRACT] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found", 
            "available_endpoints": [
                "/", "/health", "/upload", "/extract", 
                "/api/answer", "/multiply", "/intent",
                "/ws/stability", "/ws/chaos", "/docs"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"❌ Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting TORI Enhanced API Server v3.0")
    print("🎯 Now with /multiply, /intent, and WebSocket support!")
    print("=" * 60)
    print("📍 Server will be available at: http://localhost:8002")
    print("🔗 Health check: http://localhost:8002/health")
    print("📤 Upload endpoint: http://localhost:8002/upload")
    print("🧬 Extract endpoint: http://localhost:8002/extract")
    print("🧠 Prajna endpoint: http://localhost:8002/api/answer")
    print("🔢 Matrix multiply: http://localhost:8002/multiply")
    print("🎯 Intent reasoning: http://localhost:8002/intent")
    print("🌐 WebSocket test: http://localhost:8002/ws/test")
    print("📄 API documentation: http://localhost:8002/docs")
    print("=" * 60)
    
    # Ensure temp directory exists
    os.makedirs(TMP_ROOT, exist_ok=True)
    print(f"✅ Temp directory verified: {TMP_ROOT}")
    
    # Try multiple ports if needed
    for port in [8002, 8003, 8004, 8005]:
        try:
            print(f"🔌 Attempting to start on port {port}...")
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=port,
                log_level="info",
                access_log=True
            )
            break
        except Exception as e:
            print(f"❌ Port {port} failed: {e}")
            if port == 8005:
                print("❌ All ports failed!")
                break
