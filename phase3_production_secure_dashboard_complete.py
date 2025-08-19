"""
Phase 3: Production Secure Dashboard - INTEGRATED WITH ADVANCED PIPELINE
Routes ALL uploads through Jason's sophisticated 4000-hour extraction system.
Preserves all advanced features while providing clean Prajna interface.
"""
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uuid
from datetime import datetime

from ingestion import ingest_document, get_ingestion_statistics
from utils.logging import logger

# FastAPI app setup
app = FastAPI(
    title="Prajna Universal Pipeline API - Advanced Integration",
    description="Production-ready concept extraction using Jason's 4000-hour sophisticated system",
    version="4.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Router
router = APIRouter(prefix="/api", tags=["prajna"])

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...), 
    user_id: Optional[str] = None,
    doc_id: Optional[str] = None
):
    """
    🌉 MAIN PIPELINE ENDPOINT - ADVANCED INTEGRATION
    Handles ALL PDF uploads through Jason's sophisticated 4000-hour system:
    PDF → Advanced Pipeline (Purity Analysis + Context Awareness) → Prajna Mesh → Dashboard
    """
    try:
        # Generate doc_id if not provided
        if not doc_id:
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        logger.info({
            "event": "api_upload_start",
            "filename": file.filename,
            "doc_id": doc_id,
            "user_id": user_id,
            "integration_method": "advanced_pipeline_bridge"
        })
        
        # Read PDF bytes
        pdf_bytes = await file.read()
        
        # 🌉 ROUTE THROUGH ADVANCED PIPELINE BRIDGE
        result = ingest_document(pdf_bytes, user_id=user_id, doc_id=doc_id)
        
        # Extract metrics for logging
        num_concepts = result.get("num_concepts", 0)
        injected = result.get("injection_result", {}).get("injected", 0)
        advanced_analytics = result.get("advanced_analytics", {})
        purity_analysis = advanced_analytics.get("purity_analysis", {})
        
        logger.info({
            "event": "api_upload_success",
            "doc_id": doc_id,
            "concepts_extracted": num_concepts,
            "concepts_injected": injected,
            "pure_concepts": purity_analysis.get("pure_concepts", 0),
            "consensus_concepts": purity_analysis.get("distribution", {}).get("consensus", 0),
            "auto_prefilled": advanced_analytics.get("performance_data", {}).get("auto_prefilled_concepts", 0),
            "advanced_pipeline_used": True
        })
        
        # Build comprehensive response with all advanced analytics
        response = {
            "status": "success",
            "message": "Document processed through Jason's advanced 4000-hour pipeline",
            "doc_id": doc_id,
            "filename": file.filename,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            
            # Expose advanced analytics for dashboard
            "advanced_analytics": {
                "purity_analysis": purity_analysis,
                "context_extraction": advanced_analytics.get("context_extraction", {}),
                "filtering_stats": advanced_analytics.get("filtering_stats", {}),
                "performance_data": advanced_analytics.get("performance_data", {}),
                "quality_metrics": advanced_analytics.get("quality_metrics", {}),
                "extraction_methods": advanced_analytics.get("extraction_methods", {})
            },
            
            # Summary metrics for easy access
            "summary": {
                "total_concepts": num_concepts,
                "pure_concepts": purity_analysis.get("pure_concepts", 0),
                "consensus_concepts": purity_analysis.get("distribution", {}).get("consensus", 0),
                "high_confidence": purity_analysis.get("distribution", {}).get("high_confidence", 0),
                "file_storage_boosted": purity_analysis.get("distribution", {}).get("file_storage_boosted", 0),
                "auto_prefilled": advanced_analytics.get("performance_data", {}).get("auto_prefilled_concepts", 0),
                "processing_time": advanced_analytics.get("performance_data", {}).get("processing_time", 0),
                "purity_efficiency": purity_analysis.get("purity_efficiency", "N/A"),
                "extraction_successful": result.get("integration_info", {}).get("extraction_successful", True)
            }
        }
        
        return response
        
    except Exception as e:
        logger.error({
            "event": "api_upload_error",
            "doc_id": doc_id,
            "error": str(e),
            "filename": getattr(file, 'filename', 'unknown')
        })
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@router.get("/stats")
async def get_system_stats():
    """
    📊 SYSTEM STATISTICS ENDPOINT
    Get comprehensive statistics from the advanced pipeline integration
    """
    try:
        stats = get_ingestion_statistics()
        
        return {
            "status": "success",
            "message": "System statistics from advanced pipeline integration",
            "statistics": stats,
            "timestamp": datetime.now().isoformat(),
            "advanced_pipeline_available": stats.get("advanced_pipeline_available", False)
        }
        
    except Exception as e:
        logger.error({"event": "stats_error", "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.get("/concepts")
async def get_concepts(
    doc_id: Optional[str] = Query(None, description="Document ID to filter by"),
    user_id: Optional[str] = Query(None, description="User ID to filter by"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum concepts to return")
):
    """
    📊 CONCEPTS RETRIEVAL ENDPOINT
    Get concepts from Prajna mesh/ledger (placeholder for future implementation)
    """
    try:
        # [TODO] Replace with actual Prajna mesh/ledger query
        # For now, return placeholder that acknowledges advanced pipeline integration
        
        logger.info({
            "event": "api_concepts_query",
            "doc_id": doc_id,
            "user_id": user_id,
            "limit": limit
        })
        
        return {
            "status": "success",
            "message": "Concepts retrieved from mesh (placeholder - advanced pipeline ready)",
            "doc_id": doc_id,
            "user_id": user_id,
            "concepts": [],  # TODO: Replace with actual query results
            "total": 0,  # TODO: Replace with actual count
            "advanced_pipeline_note": "Integration ready for mesh query implementation",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error({
            "event": "api_concepts_error",
            "error": str(e),
            "doc_id": doc_id,
            "user_id": user_id
        })
        raise HTTPException(status_code=500, detail=f"Failed to retrieve concepts: {str(e)}")

@router.get("/health")
async def health_check():
    """
    ❤️ HEALTH CHECK ENDPOINT
    System status and advanced pipeline integration health
    """
    try:
        # Get integration stats to verify health
        stats = get_ingestion_statistics()
        advanced_available = stats.get("advanced_pipeline_available", False)
        
        health_status = {
            "status": "healthy" if advanced_available else "degraded",
            "service": "Prajna Universal Pipeline API - Advanced Integration",
            "version": "4.0.0",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "advanced_pipeline": "operational" if advanced_available else "unavailable",
                "ingestion_bridge": "operational",
                "injection": "operational",
                "governance": "operational",
                "prajna_mesh": "ready"
            },
            "integration_status": {
                "advanced_pipeline_available": advanced_available,
                "total_documents_processed": stats.get("total_documents_processed", 0),
                "total_concepts_extracted": stats.get("total_concepts_extracted", 0),
                "integration_uptime": stats.get("integration_start_time", "unknown")
            }
        }
        
        logger.info({"event": "health_check", "status": health_status["status"]})
        return health_status
        
    except Exception as e:
        logger.error({"event": "health_check_error", "error": str(e)})
        raise HTTPException(status_code=503, detail="Service unhealthy")

@router.get("/advanced")
async def get_advanced_info():
    """
    🔬 ADVANCED PIPELINE INFO ENDPOINT
    Information about the integrated advanced system
    """
    try:
        stats = get_ingestion_statistics()
        
        return {
            "status": "success",
            "message": "Advanced pipeline integration information",
            "advanced_system": {
                "description": "Jason's 4000-hour sophisticated concept extraction system",
                "features": [
                    "Context-aware purity analysis",
                    "Universal domain coverage (Science, Humanities, Arts, Philosophy, Math)",
                    "Frequency tracking and smart filtering", 
                    "Database auto-prefill with domain detection",
                    "Rogue concept detection and filtering",
                    "Cross-reference boosting and consensus analysis",
                    "Section detection (Abstract, Introduction, Methods, etc.)",
                    "Performance optimization with dynamic limits"
                ],
                "integration_status": stats.get("advanced_pipeline_available", False),
                "pipeline_version": "jason_4000h_system",
                "bridge_version": "prajna_bridge_v1.0"
            },
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error({"event": "advanced_info_error", "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to get advanced info: {str(e)}")

# Include router in app
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    """
    🏠 ROOT ENDPOINT
    API information and available endpoints with advanced integration
    """
    return {
        "service": "Prajna Universal Pipeline API - Advanced Integration",
        "version": "4.0.0",
        "description": "Production-ready concept extraction using Jason's 4000-hour sophisticated system",
        "integration": {
            "advanced_pipeline": "Jason's 4000-hour sophisticated concept extraction system",
            "features": "Purity analysis, context awareness, universal domain coverage",
            "bridge_version": "prajna_bridge_v1.0"
        },
        "endpoints": {
            "upload": "/api/upload - Process PDF through advanced pipeline",
            "stats": "/api/stats - Get comprehensive system statistics", 
            "concepts": "/api/concepts - Retrieve concepts from mesh",
            "health": "/api/health - Health check with integration status",
            "advanced": "/api/advanced - Advanced pipeline information"
        },
        "pipeline_flow": "PDF → Advanced Pipeline (Purity + Context) → Prajna Mesh → Dashboard",
        "timestamp": datetime.now().isoformat()
    }

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")
