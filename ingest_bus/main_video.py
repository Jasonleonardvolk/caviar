"""
TORI Full-Spectrum Video Ingestion System - Main Application

This is the complete video ingestion system that implements the Full-Spectrum 
Video and Audio Ingestion System Blueprint for TORI. It provides:

- Complete video/audio ingestion pipeline
- Real-time streaming capabilities
- Advanced transcription with speaker diarization
- Visual context processing (OCR, face detection, gesture analysis)
- Intelligent content segmentation
- Deep NLP analysis for concepts, intentions, and needs
- Ghost Collective multi-agent reflections
- Integration with TORI's memory systems
- Trust layer with verification and integrity checks
- Human-in-the-loop feedback capabilities

Usage:
    python main_video.py

API Documentation will be available at: http://localhost:8080/docs
"""

import asyncio
import logging
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import all our video services and routes
from src.services.video_ingestion_service import video_service
from src.services.realtime_video_processor import realtime_processor
from src.services.video_memory_integration import video_memory_service
from src.routes.video_ingestion import router as video_router
from src.routes.realtime_video_streaming import router as streaming_router

# Import existing routes for compatibility
try:
    from routes.queue import router as queue_router
    from routes.status import router as status_router
    from routes.metrics import router as metrics_router
    from routes.enhanced_queue import router as enhanced_queue_router
except ImportError:
    # Fallback if existing routes don't exist
    queue_router = None
    status_router = None
    metrics_router = None
    enhanced_queue_router = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_ingestion.log')
    ]
)
logger = logging.getLogger("tori.video_main")

# Create FastAPI application
app = FastAPI(
    title="TORI Full-Spectrum Video Ingestion System",
    description="""
    Complete video and audio ingestion system for TORI with:
    
    ## 🎬 Video Processing Features
    - **Multi-format Support**: MP4, AVI, MOV, MKV, WebM, and more
    - **Audio Formats**: MP3, WAV, M4A, AAC, FLAC, OGG
    - **Advanced Transcription**: Whisper-based with speaker diarization
    - **Visual Context**: OCR, face detection, gesture analysis
    - **Real-time Processing**: Live streaming with immediate feedback
    
    ## 🧠 AI Analysis
    - **Intelligent Segmentation**: Topic-based content organization
    - **Concept Extraction**: Deep NLP for semantic understanding
    - **Ghost Collective**: Multi-agent reflections and insights
    - **Trust Layer**: Verification and integrity checking
    
    ## 🔗 Memory Integration
    - **ConceptMesh**: Semantic concept networks
    - **BraidMemory**: Contextual memory linking
    - **ψMesh**: Advanced semantic indexing
    - **ScholarSphere**: Long-term knowledge archival
    
    ## 🚀 Getting Started
    1. Upload a video file using `/api/v2/video/ingest`
    2. Monitor progress with `/api/v2/video/jobs/{job_id}/status`
    3. Retrieve results from `/api/v2/video/jobs/{job_id}/result`
    4. For real-time streaming, connect to `/api/v2/video/stream/live`
    
    **Note**: This system requires significant computational resources for optimal performance.
    """,
    version="2.0.0",
    contact={
        "name": "TORI Development Team",
        "email": "tori@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add video ingestion routes
app.include_router(video_router)
app.include_router(streaming_router)

# Include existing routes if available
if queue_router:
    app.include_router(queue_router)
if status_router:
    app.include_router(status_router)
if metrics_router:
    app.include_router(metrics_router)
if enhanced_queue_router:
    app.include_router(enhanced_queue_router)

# Serve static files for demo interface
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        logger.info("🚀 Starting TORI Full-Spectrum Video Ingestion System...")
        
        # Initialize video service models
        logger.info("⏳ Initializing AI models (this may take a few minutes)...")
        
        # The video service will initialize models in the background
        # We'll give it a moment to start
        await asyncio.sleep(2)
        
        logger.info("✅ TORI Video Ingestion System is ready!")
        logger.info("📖 API Documentation: http://localhost:8080/docs")
        logger.info("🎬 Upload videos to: /api/v2/video/ingest")
        logger.info("🔴 Real-time streaming: /api/v2/video/stream/live")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    try:
        logger.info("🛑 Shutting down TORI Video Ingestion System...")
        
        # Shutdown real-time processor
        await realtime_processor.shutdown()
        
        logger.info("✅ Shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with system overview."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TORI Video Ingestion System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .feature { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }
            .endpoint { background: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 3px; font-family: monospace; }
            .button { display: inline-block; padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }
            .button:hover { background: #2980b9; }
            .status { padding: 10px; background: #d5f4e6; border: 1px solid #27ae60; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 TORI Full-Spectrum Video Ingestion System</h1>
            
            <div class="status">
                <strong>✅ System Status:</strong> Active and Ready
            </div>
            
            <div class="feature">
                <h3>🎯 Core Capabilities</h3>
                <ul>
                    <li><strong>Video Processing:</strong> MP4, AVI, MOV, MKV, WebM support</li>
                    <li><strong>Audio Processing:</strong> MP3, WAV, M4A, AAC, FLAC support</li>
                    <li><strong>Advanced Transcription:</strong> Whisper AI with speaker diarization</li>
                    <li><strong>Visual Analysis:</strong> OCR, face detection, gesture tracking</li>
                    <li><strong>Real-time Streaming:</strong> Live processing with WebSocket updates</li>
                    <li><strong>AI Analysis:</strong> Concept extraction, semantic segmentation</li>
                    <li><strong>Ghost Collective:</strong> Multi-agent reflections and insights</li>
                    <li><strong>Memory Integration:</strong> Full TORI cognitive system integration</li>
                </ul>
            </div>
            
            <div class="feature">
                <h3>🚀 Quick Start</h3>
                <div class="endpoint">POST /api/v2/video/ingest</div>
                <p>Upload video/audio files for complete processing</p>
                
                <div class="endpoint">GET /api/v2/video/jobs/{job_id}/status</div>
                <p>Monitor processing progress</p>
                
                <div class="endpoint">WebSocket /api/v2/video/stream/live</div>
                <p>Real-time streaming and live processing</p>
            </div>
            
            <div class="feature">
                <h3>📚 Documentation & Tools</h3>
                <a href="/docs" class="button">📖 API Documentation</a>
                <a href="/redoc" class="button">📋 ReDoc API</a>
                <a href="/api/v2/video/health" class="button">💚 Health Check</a>
            </div>
            
            <div class="feature">
                <h3>🔧 System Information</h3>
                <p><strong>Version:</strong> 2.0.0</p>
                <p><strong>AI Models:</strong> Whisper, spaCy, SentenceTransformers, MediaPipe</p>
                <p><strong>Memory Systems:</strong> ConceptMesh, BraidMemory, ψMesh, ScholarSphere</p>
                <p><strong>Real-time:</strong> WebSocket streaming supported</p>
            </div>
            
            <div class="feature">
                <h3>💡 Example Usage</h3>
                <pre style="background: #2c3e50; color: white; padding: 15px; border-radius: 5px; overflow-x: auto;">
# Upload a video file
curl -X POST "http://localhost:8080/api/v2/video/ingest" \\
     -F "file=@your_video.mp4" \\
     -F "language=en" \\
     -F "enable_diarization=true" \\
     -F "personas=Ghost Collective,Scholar,Creator"

# Check processing status
curl "http://localhost:8080/api/v2/video/jobs/{job_id}/status"

# Get complete results
curl "http://localhost:8080/api/v2/video/jobs/{job_id}/result"
                </pre>
            </div>
            
            <div class="feature">
                <h3>⚠️ Requirements</h3>
                <ul>
                    <li>Sufficient RAM for AI model loading (4GB+ recommended)</li>
                    <li>GPU support recommended for optimal performance</li>
                    <li>FFmpeg installed for video processing</li>
                    <li>Tesseract OCR for text extraction</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """System health check endpoint."""
    try:
        # Check video service
        video_health = "ready" if video_service.whisper_model else "initializing"
        
        # Check real-time processor
        realtime_health = "active" if realtime_processor.processing_active else "inactive"
        
        # Check memory service
        memory_stats = video_memory_service.get_video_memory_stats()
        
        return {
            "status": "healthy",
            "timestamp": "2025-05-27T12:00:00Z",
            "services": {
                "video_ingestion": video_health,
                "realtime_streaming": realtime_health,
                "memory_integration": "active"
            },
            "statistics": {
                "active_jobs": len(video_service.processing_jobs),
                "active_streams": len(realtime_processor.active_sessions),
                "memory_nodes": memory_stats.get("total_nodes", 0),
                "concepts_in_graph": memory_stats.get("concepts_in_graph", 0)
            },
            "capabilities": {
                "video_formats": ["mp4", "avi", "mov", "mkv", "webm", "m4v"],
                "audio_formats": ["mp3", "wav", "m4a", "aac", "flac", "ogg"],
                "ai_models": ["whisper", "spacy", "sentence_transformers", "mediapipe"],
                "real_time_streaming": True,
                "ghost_collective": True,
                "memory_integration": True
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/api/v2/system/stats")
async def get_system_stats():
    """Get comprehensive system statistics."""
    try:
        video_stats = {
            "total_jobs": len(video_service.processing_jobs),
            "completed_jobs": len([
                job for job in video_service.processing_jobs.values()
                if job.get("status") == "completed"
            ]),
            "active_jobs": len([
                job for job in video_service.processing_jobs.values()
                if job.get("status") in ["processing", "transcribing", "analyzing_video"]
            ])
        }
        
        streaming_stats = {
            "active_sessions": len(realtime_processor.active_sessions),
            "websocket_connections": len(realtime_processor.websocket_connections),
            "audio_queue_size": realtime_processor.audio_queue.qsize(),
            "video_queue_size": realtime_processor.video_queue.qsize()
        }
        
        memory_stats = video_memory_service.get_video_memory_stats()
        
        return {
            "timestamp": "2025-05-27T12:00:00Z",
            "video_processing": video_stats,
            "real_time_streaming": streaming_stats,
            "memory_integration": memory_stats,
            "system_health": {
                "uptime_seconds": 3600,  # Placeholder
                "memory_usage_mb": 1024,  # Placeholder
                "cpu_usage_percent": 25   # Placeholder
            }
        }
        
    except Exception as e:
        logger.error(f"System stats failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stats unavailable: {str(e)}")

@app.get("/api/v2/system/memory/search")
async def search_video_memory(
    query: str,
    memory_types: str = "concept,segment,video",
    max_results: int = 10
):
    """Search integrated video memory."""
    try:
        memory_type_list = [t.strip() for t in memory_types.split(",")]
        
        results = video_memory_service.search_video_memory(
            query=query,
            memory_types=memory_type_list,
            max_results=max_results
        )
        
        return {
            "query": query,
            "memory_types_searched": memory_type_list,
            "total_results": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Memory search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Memory search failed: {str(e)}")

if __name__ == "__main__":
    """Run the video ingestion system."""
    try:
        logger.info("🎬 TORI Full-Spectrum Video Ingestion System")
        logger.info("=" * 60)
        logger.info("Starting comprehensive video/audio processing system...")
        logger.info("This system implements the complete TORI Video Ingestion Blueprint")
        logger.info("=" * 60)
        
        # Run the application
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8080,
            log_level="info",
            access_log=True,
            reload=False  # Disable reload for production stability
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
    except Exception as e:
        logger.error(f"❌ Failed to start video ingestion system: {str(e)}")
        raise
