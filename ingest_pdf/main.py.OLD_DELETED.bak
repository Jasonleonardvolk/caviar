# BULLETPROOF FASTAPI - HANDLES ANY RETURN TYPE
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ingest_pdf.pipeline import ingest_pdf_clean
import os
import logging
import sys
import json
import numpy as np
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
# BULLETPROOF JSON SERIALIZATION
# ===================================================================

def ensure_serializable(obj):
    """
    Convert ANY object to JSON-serializable format - NumPy 2.0+ compatible
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.hex()
    # Robust NumPy handling for 2.0+
    elif "numpy" in str(type(obj)):
        # Try converting to Python scalar
        try:
            return obj.item()
        except Exception:
            # Try array to list if scalar fails
            try:
                return obj.tolist()
            except Exception:
                return str(obj)
    elif isinstance(obj, dict):
        return {str(k): ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [ensure_serializable(item) for item in obj]
    elif hasattr(obj, 'dict'):
        try:
            return obj.dict()
        except:
            pass
    elif hasattr(obj, '__dict__'):
        return {str(k): ensure_serializable(v) for k, v in obj.__dict__.items()}
    else:
        return str(obj)



# ===================================================================
# MODELS
# ===================================================================

class ExtractionRequest(BaseModel):
    file_path: str
    filename: str
    content_type: str
    progress_id: Optional[str] = None

# ===================================================================
# BULLETPROOF EXTRACTION ENDPOINT
# ===================================================================

@app.post("/extract")
async def extract(request: ExtractionRequest):
    """Extract concepts from PDF file - 100% bulletproof serialization"""
    
    try:
        print(f"🔔 [FASTAPI] REQUEST RECEIVED at {datetime.now()}")
        print(f"🔔 [FASTAPI] Request: {request.filename}")
        print(f"🔔 [FASTAPI] File path: {request.file_path}")
        
        # File validation
        file_path = Path(request.file_path).absolute()
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail=f"Path is not a file: {file_path}")
        
        file_size = file_path.stat().st_size
        max_size = 50 * 1024 * 1024
        if file_size > max_size:
            raise HTTPException(status_code=413, detail=f"File too large: {file_size:,} bytes")
        
        # 🔧 CRITICAL FIX: Call pipeline with correct parameters
        print(f"🔔 [FASTAPI] Starting extraction with admin mode...")
        raw_result = ingest_pdf_clean(
            pdf_path=str(file_path),
            doc_id=None,  # Let it auto-generate from filename
            extraction_threshold=0.0,  # No threshold filtering
            admin_mode=True  # Enable full concept extraction
        )
        print(f"🔔 [FASTAPI] Extraction complete. Result type: {type(raw_result)}")
        
        # Log key details about the result
        if isinstance(raw_result, dict):
            concept_count = raw_result.get('concept_count', 0)
            status = raw_result.get('status', 'unknown')
            print(f"🔔 [FASTAPI] Extraction status: {status}, concept_count: {concept_count}")
            
            if concept_count > 0:
                concept_names = raw_result.get('concept_names', [])
                print(f"🔔 [FASTAPI] Top 3 concepts: {concept_names[:3]}")
            else:
                # Debug why no concepts were found
                error_msg = raw_result.get('error_message', 'No error message')
                print(f"🔔 [FASTAPI] No concepts found. Error: {error_msg}")
                
                # Check if purity analysis has details
                purity_analysis = raw_result.get('purity_analysis', {})
                if purity_analysis:
                    raw_concepts = purity_analysis.get('raw_concepts', 0)
                    pure_concepts = purity_analysis.get('pure_concepts', 0)
                    print(f"🔔 [FASTAPI] Purity analysis: {raw_concepts} raw -> {pure_concepts} pure")
        
        # CRITICAL: Convert to JSON-serializable format
        clean_result = ensure_serializable(raw_result)
        
        # Ensure it's a dict
        if not isinstance(clean_result, dict):
            clean_result = {"result": clean_result}
        
        # Add metadata
        clean_result["success"] = True
        clean_result["timestamp"] = datetime.now().isoformat()
        clean_result["filename"] = request.filename
        
        # Log final response details
        final_concept_count = clean_result.get('concept_count', 0)
        print(f"🔔 [FASTAPI] Final response - success: True, concepts: {final_concept_count}")
        print(f"🔔 [FASTAPI] Response size: ~{len(str(clean_result))} characters")
        print(f"🔔 [FASTAPI] Returning clean result")
        
        return clean_result

    except HTTPException:
        raise
    except Exception as e:
        print(f"🔔 [FASTAPI] EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        print(f"🔔 [FASTAPI] TRACEBACK: {traceback.format_exc()}")
        
        # Bulletproof error response
        error_response = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "concept_count": 0,
            "concept_names": [],
            "concepts": [],
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "filename": getattr(request, 'filename', 'unknown')
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
        "message": "FastAPI bulletproof extraction service is running",
        "timestamp": datetime.now().isoformat(),
        "pipeline_ready": True
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return {
        "message": "API is working!",
        "timestamp": datetime.now().isoformat(),
        "test_data": {
            "success": True,
            "concept_count": 5,
            "concept_names": ["test concept 1", "test concept 2"],
            "status": "success"
        }
    }

@app.get("/debug/pipeline")
async def debug_pipeline():
    """Debug endpoint to test pipeline directly"""
    try:
        # Test the pipeline import and basic functionality
        from ingest_pdf.pipeline import ingest_pdf_clean
        
        # Try to call with a non-existent file to test error handling
        test_result = ingest_pdf_clean(
            pdf_path="nonexistent_test_file.pdf",
            doc_id="test",
            extraction_threshold=0.0,
            admin_mode=True
        )
        
        return {
            "pipeline_import": "success",
            "test_call_result": ensure_serializable(test_result),
            "expected_error": "File should not exist, so should get error response"
        }
    except Exception as e:
        return {
            "pipeline_import": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/debug/last-upload")
async def debug_last_upload():
    """Debug endpoint to check last upload files"""
    try:
        tmp_dir = Path("C:\\Users\\jason\\Desktop\\tori\\kha\\tmp")
        if tmp_dir.exists():
            files = list(tmp_dir.glob("*"))
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            recent_files = []
            for f in files[:5]:  # Last 5 files
                recent_files.append({
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / (1024*1024), 2),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    "exists": f.exists()
                })
            
            return {
                "tmp_directory": str(tmp_dir),
                "recent_files": recent_files,
                "total_files": len(files)
            }
        else:
            return {"error": "Tmp directory not found"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "TORI Bulletproof FastAPI Extraction Service",
        "status": "ready",
        "features": ["bulletproof_serialization", "concept_extraction", "purity_analysis", "admin_mode"],
        "endpoints": {
            "extract": "/extract",
            "health": "/health",
            "test": "/test",
            "debug_pipeline": "/debug/pipeline",
            "debug_uploads": "/debug/last-upload",
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
    logger.info("🚀 FastAPI Bulletproof PDF Ingestion Service Starting...")
    
    try:
        from ingest_pdf.pipeline import ingest_pdf_clean
        logger.info("✅ Pipeline module loaded successfully")
        
        # Test pipeline with dummy call
        try:
            test_result = ingest_pdf_clean("test_nonexistent.pdf", admin_mode=True)
            if test_result.get('status') == 'error':
                logger.info("✅ Pipeline responds correctly to test call")
            else:
                logger.warning(f"⚠️ Unexpected test result: {test_result}")
        except Exception as e:
            logger.error(f"❌ Pipeline test call failed: {e}")
            
    except ImportError as e:
        logger.error(f"❌ Failed to import pipeline: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
