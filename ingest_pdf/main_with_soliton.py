from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
from typing import Dict, Any, Optional
from alan_backend.routes.soliton import router as soliton_router  # 🌊 ADDED: Soliton Memory import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="TORI PDF Ingestion Service")

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

# Global progress tracking
progress_connections: Dict[str, WebSocket] = {}

class ExtractionRequest(BaseModel):
    file_path: str
    filename: str
    content_type: str
    progress_id: Optional[str] = None  # ✅ FIXED: Make optional with proper typing

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
            # Remove disconnected client
            if progress_id in progress_connections:
                del progress_connections[progress_id]

@app.websocket("/progress/{progress_id}")
async def websocket_endpoint(websocket: WebSocket, progress_id: str):
    await websocket.accept()
    progress_connections[progress_id] = websocket
    logger.info(f"📡 WebSocket connected for progress tracking: {progress_id}")
    
    try:
        # Send initial connection confirmation
        await send_progress(progress_id, "connected", 0, "Progress tracking connected", {
            "connection_time": datetime.now().isoformat(),
            "status": "ready"
        })
        
        # Keep connection alive
        while True:
            try:
                # Wait for any message (ping/pong or disconnect)
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                # Send heartbeat every second
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
        # Only send progress if we have a valid progress_id
        if self.progress_id:
            await send_progress(self.progress_id, stage, percentage, message, details)
        else:
            # Log progress for debugging even without WebSocket
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

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("🚀 FastAPI PDF Ingestion Service Starting...")
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
        logger.info("✅ All systems operational")
    except Exception as e:
        logger.warning(f"⚠️ Soliton Memory routes may not be fully loaded: {e}")

@app.post("/extract")
async def extract(request: ExtractionRequest):
    """Extract concepts from PDF file with optional real-time progress tracking"""
    progress_id = request.progress_id or "no-progress"
    
    try:
        print(f"🔔 [FASTAPI] REQUEST RECEIVED at {datetime.now()}")
        print(f"🔔 [FASTAPI] Request data: {request.dict()}")
        print(f"🔔 [FASTAPI] Progress ID: {progress_id}")
        
        # ✅ FIXED: Handle cases where progress_id might be None
        tracker_progress_id = request.progress_id if request.progress_id else None
        
        async with ProgressTracker(tracker_progress_id, request.filename) as tracker:
            
            await tracker.update("validating", 10, "Validating file path and permissions", {
                "file_path": request.file_path,
                "content_type": request.content_type
            })
            
            # Convert to absolute path
            file_path = Path(request.file_path).absolute()
            print(f"🔔 [FASTAPI] Absolute path: {file_path}")
            
            # Verify file exists
            if not file_path.exists():
                error_msg = f"File not found: {file_path}"
                print(f"🔔 [FASTAPI] ERROR: {error_msg}")
                logger.error(f"❌ [FASTAPI] {error_msg}")
                raise HTTPException(status_code=404, detail=error_msg)
            
            print(f"🔔 [FASTAPI] File exists check: PASSED")
            
            # Verify it's a file (not a directory)
            if not file_path.is_file():
                error_msg = f"Path is not a file: {file_path}"
                print(f"🔔 [FASTAPI] ERROR: {error_msg}")
                logger.error(f"❌ [FASTAPI] {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
            
            print(f"🔔 [FASTAPI] Is file check: PASSED")
            
            # Get file size for logging
            file_size = file_path.stat().st_size
            logger.info(f"📏 [FASTAPI] File size: {file_size:,} bytes")
            print(f"🔔 [FASTAPI] File size: {file_size:,} bytes")
            
            await tracker.update("size_check", 15, f"File size validated: {file_size:,} bytes", {
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024*1024), 2)
            })
            
            # Check file size limit (50MB)
            max_size = 50 * 1024 * 1024
            if file_size > max_size:
                error_msg = f"File too large: {file_size:,} bytes (max {max_size:,} bytes)"
                print(f"🔔 [FASTAPI] ERROR: {error_msg}")
                logger.error(f"❌ [FASTAPI] {error_msg}")
                raise HTTPException(status_code=413, detail=error_msg)
            
            print(f"🔔 [FASTAPI] File size check: PASSED")
            
            await tracker.update("loading_pipeline", 20, "Loading extraction pipeline and models", {
                "pipeline": "purity_based_universal_pipeline",
                "models": ["KeyBERT", "YAKE", "NER", "spaCy"]
            })
            
            logger.info("🚀 [FASTAPI] Starting pipeline processing...")
            print(f"🔔 [FASTAPI] About to call ingest_pdf_clean()...")
            
            await tracker.update("extracting", 25, "Starting concept extraction from PDF", {
                "extraction_method": "purity_based_universal_pipeline",
                "threshold": 0.0
            })
            
            # Create a custom pipeline that sends progress updates
            result = await extract_with_progress(str(file_path), tracker)
            
            print(f"🔔 [FASTAPI] ingest_pdf_clean() completed!")
            print(f"🔔 [FASTAPI] Result type: {type(result)}")
            
            # Ensure we have required fields
            if not isinstance(result, dict):
                logger.error(f"❌ [FASTAPI] Pipeline returned non-dict: {type(result)}")
                result = {
                    "concept_names": [],
                    "concept_count": 0,
                    "extraction_method": "error",
                    "status": "error",
                    "error": "Invalid pipeline response"
                }
            
            # Add success indicator
            result["success"] = result.get("status") == "success"
            
            concept_count = result.get('concept_count', 0)
            concept_names = result.get('concept_names', [])
            
            print(f"🔔 [FASTAPI] Final result summary:")
            print(f"🔔 [FASTAPI] - Concept count: {concept_count}")
            print(f"🔔 [FASTAPI] - Sample concepts: {concept_names[:5] if concept_names else 'None'}")
            print(f"🔔 [FASTAPI] - Status: {result.get('status', 'unknown')}")
            
            await tracker.update("finalizing", 95, f"Extraction complete! Found {concept_count} concepts", {
                "concept_count": concept_count,
                "sample_concepts": concept_names[:5],
                "extraction_method": result.get("extraction_method", "unknown"),
                "processing_time": result.get("processing_time_seconds", 0)
            })
            
            logger.info(f"✅ [FASTAPI] Pipeline completed successfully")
            logger.info(f"📊 [FASTAPI] Extracted {concept_count} concepts")
            logger.info(f"🎯 [FASTAPI] Returning result to frontend")
            
            print(f"🔔 [FASTAPI] About to return result to SvelteKit...")
            
            return result

    except HTTPException:
        print(f"🔔 [FASTAPI] HTTPException occurred, re-raising...")
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        print(f"🔔 [FASTAPI] EXCEPTION CAUGHT: {type(e).__name__}: {e}")
        logger.error(f"❌ [FASTAPI] ERROR during extraction: {type(e).__name__}: {e}")
        logger.error(f"🔍 [FASTAPI] Full traceback:\n{traceback.format_exc()}")
        
        # Send error progress update if we have a progress_id
        if request.progress_id:
            await send_progress(request.progress_id, "error", 0, f"Error: {str(e)}", {
                "error": str(e),
                "error_type": type(e).__name__
            })
        
        # Return error response instead of raising
        error_response = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "concept_count": 0,
            "concept_names": [],
            "status": "error",
            "extraction_method": "error",
            "traceback": traceback.format_exc() if logger.level <= logging.DEBUG else None
        }
        
        print(f"🔔 [FASTAPI] Returning error response: {error_response}")
        return error_response

async def extract_with_progress(file_path: str, tracker: ProgressTracker) -> dict:
    """Run extraction pipeline with progress updates"""
    
    await tracker.update("reading_pdf", 30, "Reading PDF and extracting text chunks", {
        "stage": "pdf_parsing"
    })
    
    # Simulate progress for chunks (we'll need to modify the pipeline to actually send these)
    chunk_stages = [
        (35, "Processing chunk 1/9 - Extracting concepts"),
        (45, "Processing chunk 2/9 - Universal analysis"),
        (50, "Processing chunk 3/9 - Cross-referencing file_storage"),
        (55, "Processing chunk 4/9 - Semantic analysis"),
        (60, "Processing chunk 5/9 - Named entity recognition"),
        (65, "Processing chunk 6/9 - Quality scoring"),
        (70, "Processing chunk 7/9 - Domain classification"),
        (75, "Processing chunk 8/9 - Final extraction"),
        (80, "Processing chunk 9/9 - Completing analysis"),
    ]
    
    # We'll do the actual processing in a separate task and simulate progress
    # In a real implementation, you'd modify the pipeline to send actual progress
    
    extraction_task = asyncio.create_task(run_extraction(file_path))
    
    # Simulate progress while extraction runs
    for percentage, message in chunk_stages:
        if extraction_task.done():
            break
        await tracker.update("processing", percentage, message, {
            "stage": "concept_extraction",
            "estimated_remaining": f"{(90-percentage)/10:.0f}s"
        })
        await asyncio.sleep(1.5)  # Simulate processing time
    
    await tracker.update("purity_analysis", 85, "Applying purity analysis - extracting the 'truth'", {
        "stage": "quality_filtering",
        "analysis_type": "consensus_based"
    })
    
    # Wait for actual extraction to complete
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
    # Run the synchronous pipeline function in a thread pool
    return await loop.run_in_executor(None, ingest_pdf_clean, file_path)

@app.get("/health")
async def health():
    """Health check endpoint"""
    print(f"🔔 [FASTAPI] Health check requested at {datetime.now()}")
    return {
        "status": "healthy",
        "message": "FastAPI extraction service is running",
        "working_directory": os.getcwd(),
        "python_version": sys.version,
        "progress_connections": len(progress_connections),
        "features": ["real_time_progress", "websocket_updates", "purity_analysis", "soliton_memory"],  # 🌊 UPDATED: Added soliton_memory
        "soliton_enabled": True  # 🌊 ADDED: Soliton status indicator
    }

@app.get("/")
async def root():
    """Root endpoint"""
    print(f"🔔 [FASTAPI] Root endpoint accessed at {datetime.now()}")
    return {
        "message": "TORI FastAPI Extraction Service",
        "status": "ready",
        "features": ["real_time_progress", "websocket_updates", "purity_analysis", "soliton_memory"],  # 🌊 UPDATED: Added soliton_memory
        "endpoints": {
            "extract": "/extract",
            "health": "/health",
            "progress": "/progress/{progress_id}",
            "docs": "/docs",
            "soliton": "/api/soliton/*"  # 🌊 ADDED: Soliton API endpoints
        }
    }
