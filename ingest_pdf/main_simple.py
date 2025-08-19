# SIMPLE FASTAPI FIX - JSON SERIALIZATION ONLY
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ingest_pdf.pipeline import ingest_pdf_clean
import os
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="TORI PDF Ingestion Service")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================================================
# ATOMIC JSON SERIALIZATION FIX
# ===================================================================

def clean_for_json(obj):
    """Convert any object to JSON-serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(key): clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, 'dict'):  # Pydantic model
        return obj.dict()
    elif hasattr(obj, '__dict__'):  # Regular class instance
        return {str(key): clean_for_json(value) for key, value in obj.__dict__.items()}
    else:
        return str(obj)  # Fallback to string

# ===================================================================
# MODELS
# ===================================================================

class ExtractionRequest(BaseModel):
    file_path: str
    filename: str
    content_type: str
    progress_id: Optional[str] = None

# ===================================================================
# MAIN EXTRACTION ENDPOINT - FIXED
# ===================================================================

@app.post("/extract")
async def extract(request: ExtractionRequest):
    """Extract concepts from PDF file - JSON serialization fixed"""
    
    try:
        print(f"🔔 [FASTAPI] REQUEST RECEIVED at {datetime.now()}")
        print(f"🔔 [FASTAPI] Request data: {request.dict()}")
        
        # File validation
        file_path = Path(request.file_path).absolute()
        print(f"🔔 [FASTAPI] Absolute path: {file_path}")
        
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            raise HTTPException(status_code=404, detail=error_msg)
        
        if not file_path.is_file():
            error_msg = f"Path is not a file: {file_path}"
            raise HTTPException(status_code=400, detail=error_msg)
        
        file_size = file_path.stat().st_size
        max_size = 50 * 1024 * 1024
        if file_size > max_size:
            error_msg = f"File too large: {file_size:,} bytes (max {max_size:,} bytes)"
            raise HTTPException(status_code=413, detail=error_msg)
        
        # Run extraction
        print(f"🔔 [FASTAPI] Starting extraction...")
        raw_result = ingest_pdf_clean(str(file_path))
        print(f"🔔 [FASTAPI] Raw result type: {type(raw_result)}")
        
        # CRITICAL: Convert to JSON-serializable format
        clean_result = clean_for_json(raw_result)
        print(f"🔔 [FASTAPI] Clean result type: {type(clean_result)}")
        
        # Ensure it's a dict for FastAPI
        if not isinstance(clean_result, dict):
            clean_result = {"result": clean_result}
        
        # Add success flag
        clean_result["success"] = True
        clean_result["extraction_method"] = "purity_based_universal_pipeline"
        clean_result["timestamp"] = datetime.now().isoformat()
        
        print(f"🔔 [FASTAPI] Returning clean result")
        return clean_result

    except HTTPException:
        raise
    except Exception as e:
        print(f"🔔 [FASTAPI] EXCEPTION: {type(e).__name__}: {e}")
        logger.error(f"❌ [FASTAPI] ERROR: {type(e).__name__}: {e}")
        
        # Return JSON-serializable error
        error_response = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "concept_count": 0,
            "concept_names": [],
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }
        
        return error_response

# ===================================================================
# SIMPLE ENDPOINTS
# ===================================================================

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "message": "FastAPI extraction service is running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "TORI FastAPI Extraction Service",
        "status": "ready",
        "endpoints": {
            "extract": "/extract",
            "health": "/health",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }

# ===================================================================
# STARTUP
# ===================================================================

@app.on_event("startup")
async def startup_event():
    """Startup"""
    logger.info("🚀 FastAPI Simple PDF Ingestion Service Starting...")
    
    try:
        from ingest_pdf.pipeline import ingest_pdf_clean
        logger.info("✅ Pipeline module loaded successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import pipeline: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
