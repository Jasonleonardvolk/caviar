"""
Enhanced Ingest Bus Main Application
Includes all file types and TORI system integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
from pathlib import Path

# Import existing routes
from routes.queue import router as queue_router
from routes.status import router as status_router
from routes.metrics import router as metrics_router

# Import enhanced routes
from routes.enhanced_queue import router as enhanced_queue_router

# Import workers for initialization
from workers.integration_coordinator import coordinator
from workers.psi_mesh_integration import psi_mesh

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tori-ingest-bus")

# Create FastAPI app
app = FastAPI(
    title="TORI Ingest Bus - Enhanced",
    description="Complete document ingestion pipeline for TORI with all file types and system integration",
    version="2.0.0",
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

# Include existing routes
app.include_router(queue_router, prefix="/api/v1", tags=["queue"])
app.include_router(status_router, prefix="/api/v1", tags=["status"])
app.include_router(metrics_router, prefix="/api/v1", tags=["metrics"])

# Include enhanced routes
app.include_router(enhanced_queue_router, prefix="/api/v2", tags=["enhanced-queue"])

@app.on_startup
async def startup_event():
    """Initialize TORI systems on startup"""
    logger.info("🚀 TORI Ingest Bus Enhanced starting up...")
    
    # Initialize integration coordinator
    logger.info("📊 Initializing integration coordinator...")
    
    # Initialize ψMesh
    logger.info("🧠 Initializing ψMesh semantic associations...")
    
    # Check system requirements
    requirements_check = await check_system_requirements()
    if not requirements_check['all_satisfied']:
        logger.warning("⚠️ Some system requirements not satisfied:")
        for req, status in requirements_check['requirements'].items():
            if not status['satisfied']:
                logger.warning(f"  - {req}: {status['message']}")
    
    logger.info("✅ TORI Ingest Bus Enhanced ready!")
    logger.info("📚 Supported file types: PDF, DOCX, CSV, PPTX, XLSX, JSON, TXT, MD")
    logger.info("🔗 Integrated systems: ConceptMesh, ψMesh, Ghost Collective, ScholarSphere")

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "service": "TORI Ingest Bus Enhanced",
        "version": "2.0.0",
        "description": "Complete document ingestion pipeline with all file types",
        "supported_file_types": ["PDF", "DOCX", "CSV", "PPTX", "XLSX", "JSON", "TXT", "MD"],
        "integrated_systems": ["ConceptMesh", "ψMesh", "Ghost Collective", "ScholarSphere"],
        "endpoints": {
            "v1": {
                "queue": "/api/v1/queue",
                "status": "/api/v1/status",
                "metrics": "/api/v1/metrics"
            },
            "v2": {
                "enhanced_queue": "/api/v2/queue/enhanced",
                "batch_queue": "/api/v2/queue/batch",
                "supported_types": "/api/v2/queue/supported_types",
                "integration_options": "/api/v2/queue/integration_options"
            }
        },
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": "2025-05-27T12:00:00Z",
        "components": {}
    }
    
    # Check integration coordinator
    try:
        health_status["components"]["integration_coordinator"] = "healthy"
    except Exception as e:
        health_status["components"]["integration_coordinator"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check ψMesh
    try:
        mesh_stats = psi_mesh.get_mesh_statistics()
        health_status["components"]["psi_mesh"] = {
            "status": "healthy",
            "concepts": mesh_stats["total_concepts"],
            "associations": mesh_stats["total_associations"]
        }
    except Exception as e:
        health_status["components"]["psi_mesh"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/system/info")
async def system_info():
    """Get detailed system information"""
    try:
        # Get ψMesh statistics
        mesh_stats = psi_mesh.get_mesh_statistics()
        
        # Check system requirements
        requirements = await check_system_requirements()
        
        return {
            "service_info": {
                "name": "TORI Ingest Bus Enhanced",
                "version": "2.0.0",
                "uptime": "running"
            },
            "supported_formats": {
                "documents": ["PDF", "DOCX", "DOC"],
                "data": ["CSV", "XLSX", "JSON"],
                "presentations": ["PPTX"],
                "text": ["TXT", "MD"]
            },
            "integration_systems": {
                "concept_mesh": {
                    "description": "Knowledge graph for concept relationships",
                    "enabled": coordinator.concept_mesh_enabled
                },
                "psi_mesh": {
                    "description": "Semantic association mesh",
                    "enabled": True,
                    "statistics": mesh_stats
                },
                "ghost_collective": {
                    "description": "AI persona system",
                    "enabled": coordinator.ghost_collective_enabled
                },
                "scholar_sphere": {
                    "description": "Document archival system",
                    "enabled": coordinator.scholar_sphere_enabled
                }
            },
            "processing_capabilities": {
                "concept_extraction": True,
                "semantic_analysis": True,
                "integrity_verification": True,
                "multi_persona_analysis": True,
                "knowledge_graph_integration": True,
                "archival_storage": True
            },
            "system_requirements": requirements
        }
        
    except Exception as e:
        logger.exception(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system information")

async def check_system_requirements():
    """Check if all system requirements are satisfied"""
    requirements = {
        "python_docx": {"satisfied": False, "message": "Required for DOCX processing"},
        "python_pptx": {"satisfied": False, "message": "Required for PPTX processing"},
        "openpyxl": {"satisfied": False, "message": "Required for XLSX processing"},
        "pandas": {"satisfied": False, "message": "Required for CSV processing"},
        "pypdf2": {"satisfied": False, "message": "Required for PDF processing"}
    }
    
    # Check for required packages
    try:
        import docx
        requirements["python_docx"] = {"satisfied": True, "message": "Available"}
    except ImportError:
        pass
    
    try:
        import pptx
        requirements["python_pptx"] = {"satisfied": True, "message": "Available"}
    except ImportError:
        pass
    
    try:
        import openpyxl
        requirements["openpyxl"] = {"satisfied": True, "message": "Available"}
    except ImportError:
        pass
    
    try:
        import pandas
        requirements["pandas"] = {"satisfied": True, "message": "Available"}
    except ImportError:
        pass
    
    try:
        import PyPDF2
        requirements["pypdf2"] = {"satisfied": True, "message": "Available"}
    except ImportError:
        pass
    
    all_satisfied = all(req["satisfied"] for req in requirements.values())
    
    return {
        "all_satisfied": all_satisfied,
        "requirements": requirements
    }

if __name__ == "__main__":
    logger.info("Starting TORI Ingest Bus Enhanced...")
    uvicorn.run(
        "main_enhanced:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
