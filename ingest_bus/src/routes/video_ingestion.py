"""
Video Ingestion API Routes for TORI

This module implements the REST API endpoints for the full-spectrum video and audio
ingestion system. It provides endpoints for uploading videos, tracking processing
status, retrieving results, and managing the ingestion pipeline.

API Endpoints:
- POST /api/v2/video/ingest - Upload and process video/audio files
- GET /api/v2/video/jobs/{job_id}/status - Get processing status
- GET /api/v2/video/jobs/{job_id}/result - Get processing results
- POST /api/v2/video/jobs/{job_id}/feedback - Submit human feedback
- GET /api/v2/video/search - Search processed video content
- WebSocket /api/v2/video/stream/{job_id} - Real-time processing updates
"""

import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi import Depends, Query, Form
from pydantic import BaseModel, Field
import aiofiles
import json

# Import our video service
from ..services.video_ingestion_service import video_service, ProcessingStatus, VideoIngestionResult

# Configure logging
logger = logging.getLogger("tori.video_api")

# Create router
router = APIRouter(prefix="/api/v2/video", tags=["video-ingestion"])

# Pydantic models for API
class VideoIngestionRequest(BaseModel):
    """Request model for video ingestion."""
    language: str = Field(default="en", description="Language code for transcription")
    enable_diarization: bool = Field(default=True, description="Enable speaker diarization")
    enable_visual_context: bool = Field(default=True, description="Enable visual processing")
    enable_real_time: bool = Field(default=False, description="Enable real-time processing")
    extract_slides: bool = Field(default=True, description="Extract slide content via OCR")
    detect_faces: bool = Field(default=True, description="Detect and track faces")
    analyze_gestures: bool = Field(default=True, description="Analyze gestures and poses")
    segment_threshold: float = Field(default=0.7, description="Semantic segmentation threshold")
    personas: List[str] = Field(
        default=["Ghost Collective", "Scholar", "Creator"],
        description="Ghost personas to activate for reflection"
    )

class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: ProcessingStatus
    progress: float
    created_at: datetime
    last_updated: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    error: Optional[str] = None
    messages: List[str] = Field(default_factory=list)
    
class ConceptFeedback(BaseModel):
    """Model for concept feedback."""
    concept: str
    action: str  # "verify", "reject", "flag"
    note: Optional[str] = None

class VideoFeedbackRequest(BaseModel):
    """Request model for human feedback."""
    concept_feedback: List[ConceptFeedback] = Field(default_factory=list)
    transcript_corrections: Dict[str, str] = Field(default_factory=dict)  # segment_id -> corrected_text
    segment_adjustments: Dict[str, Any] = Field(default_factory=dict)
    general_feedback: Optional[str] = None

class VideoSearchRequest(BaseModel):
    """Request model for video search."""
    query: str
    video_ids: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    concepts: Optional[List[str]] = None
    speakers: Optional[List[str]] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None

# Connection manager for WebSocket updates
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """Accept WebSocket connection."""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        logger.info(f"WebSocket connected for job {job_id}")
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        """Disconnect WebSocket."""
        if job_id in self.active_connections:
            try:
                self.active_connections[job_id].remove(websocket)
                if not self.active_connections[job_id]:
                    del self.active_connections[job_id]
            except ValueError:
                pass
        logger.info(f"WebSocket disconnected for job {job_id}")
    
    async def send_update(self, job_id: str, message: dict):
        """Send update to all connections for a job."""
        if job_id in self.active_connections:
            connections_to_remove = []
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except:
                    connections_to_remove.append(connection)
            
            # Remove dead connections
            for connection in connections_to_remove:
                self.disconnect(connection, job_id)

manager = ConnectionManager()

# Helper function to save uploaded file
async def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location."""
    try:
        # Create temporary file
        suffix = Path(upload_file.filename).suffix if upload_file.filename else ".tmp"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.close()
        
        # Save file content
        async with aiofiles.open(temp_file.name, 'wb') as f:
            content = await upload_file.read()
            await f.write(content)
        
        logger.info(f"Uploaded file saved to: {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

# Validate file type
def validate_video_file(filename: str) -> bool:
    """Validate if file is a supported video/audio format."""
    if not filename:
        return False
    
    supported_extensions = {
        # Video formats
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v',
        # Audio formats
        '.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.wma'
    }
    
    return Path(filename).suffix.lower() in supported_extensions

@router.post("/ingest", response_model=Dict[str, str])
async def ingest_video(
    file: UploadFile = File(...),
    language: str = Form(default="en"),
    enable_diarization: bool = Form(default=True),
    enable_visual_context: bool = Form(default=True),
    enable_real_time: bool = Form(default=False),
    extract_slides: bool = Form(default=True),
    detect_faces: bool = Form(default=True),
    analyze_gestures: bool = Form(default=True),
    segment_threshold: float = Form(default=0.7),
    personas: str = Form(default="Ghost Collective,Scholar,Creator")
):
    """
    Upload and process video/audio file.
    
    This endpoint accepts video or audio files and starts the full ingestion pipeline
    including transcription, visual analysis, concept extraction, and Ghost reflections.
    
    Args:
        file: Video or audio file to process
        language: Language code for transcription (default: "en")
        enable_diarization: Enable speaker diarization
        enable_visual_context: Enable visual processing (for video files)
        enable_real_time: Enable real-time processing mode
        extract_slides: Extract slide content via OCR
        detect_faces: Detect and track faces
        analyze_gestures: Analyze gestures and poses
        segment_threshold: Semantic segmentation threshold (0.0-1.0)
        personas: Comma-separated list of Ghost personas to activate
        
    Returns:
        Dict containing job_id for tracking progress
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not validate_video_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: mp4, avi, mov, mkv, mp3, wav, m4a, etc."
            )
        
        # Save uploaded file
        file_path = await save_uploaded_file(file)
        
        # Parse personas
        persona_list = [p.strip() for p in personas.split(",")] if personas else ["Ghost Collective"]
        
        # Prepare options
        options = {
            "language": language,
            "enable_diarization": enable_diarization,
            "enable_visual_context": enable_visual_context,
            "enable_real_time": enable_real_time,
            "extract_slides": extract_slides,
            "detect_faces": detect_faces,
            "analyze_gestures": analyze_gestures,
            "segment_threshold": segment_threshold,
            "personas": persona_list
        }
        
        # Start ingestion
        job_id = await video_service.ingest_video(file_path, options)
        
        # Start WebSocket update task
        asyncio.create_task(monitor_job_progress(job_id))
        
        logger.info(f"Started video ingestion job {job_id} for file: {file.filename}")
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": f"Video ingestion started for {file.filename}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video ingestion failed: {str(e)}")

@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get processing status for a video ingestion job.
    
    Args:
        job_id: Job identifier returned from /ingest endpoint
        
    Returns:
        Job status including progress, messages, and current stage
    """
    try:
        job_data = video_service.get_job_status(job_id)
        
        if not job_data:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return JobStatusResponse(
            job_id=job_id,
            status=job_data["status"],
            progress=job_data.get("progress", 0.0),
            created_at=job_data["created_at"],
            last_updated=job_data.get("last_updated"),
            completed_at=job_data.get("completed_at"),
            failed_at=job_data.get("failed_at"),
            error=job_data.get("error"),
            messages=job_data.get("messages", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting job status: {str(e)}")

@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Get processing results for a completed video ingestion job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Complete processing results including transcript, concepts, reflections, etc.
    """
    try:
        result = video_service.get_job_result(job_id)
        
        if not result:
            job_data = video_service.get_job_status(job_id)
            if not job_data:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            elif job_data["status"] != ProcessingStatus.COMPLETED:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Job {job_id} is not completed (status: {job_data['status']})"
                )
            else:
                raise HTTPException(status_code=500, detail="Result not available")
        
        # Convert result to dict for JSON serialization
        result_dict = {
            "video_id": result.video_id,
            "source_file": result.source_file,
            "processing_time": result.processing_time,
            "status": result.status,
            "transcript": [
                {
                    "id": seg.id,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.text,
                    "speaker_id": seg.speaker_id,
                    "speaker_name": seg.speaker_name,
                    "confidence": seg.confidence
                }
                for seg in result.transcript
            ],
            "segments": result.segments,
            "speakers": [
                {
                    "id": speaker.id,
                    "name": speaker.name,
                    "total_speaking_time": speaker.total_speaking_time,
                    "face_encodings_count": len(speaker.face_encodings or [])
                }
                for speaker in result.speakers
            ],
            "concepts": [
                {
                    "term": concept.term,
                    "concept_type": concept.concept_type,
                    "confidence": concept.confidence,
                    "source_segments": concept.source_segments,
                    "context": concept.context,
                    "timestamp_ranges": concept.timestamp_ranges
                }
                for concept in result.concepts
            ],
            "questions": result.questions,
            "intentions": result.intentions,
            "action_items": result.action_items,
            "integrity_score": result.integrity_score,
            "trust_flags": result.trust_flags,
            "source_hash": result.source_hash,
            "ghost_reflections": result.ghost_reflections,
            "created_at": result.created_at.isoformat(),
            "duration": result.duration,
            "file_size": result.file_size,
            "frames_processed": len(result.frames),
            "visual_context_available": any(frame.ocr_text or frame.faces or frame.gestures for frame in result.frames)
        }
        
        return result_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting job result: {str(e)}")

@router.post("/jobs/{job_id}/feedback")
async def submit_feedback(job_id: str, feedback: VideoFeedbackRequest):
    """
    Submit human feedback for a processed video.
    
    This endpoint allows users to provide corrections and feedback on the
    processing results, which can be used to improve accuracy and train
    the system.
    
    Args:
        job_id: Job identifier
        feedback: Feedback data including concept corrections, transcript edits, etc.
        
    Returns:
        Confirmation of feedback submission
    """
    try:
        # Verify job exists and is completed
        result = video_service.get_job_result(job_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found or not completed")
        
        # Process feedback
        feedback_summary = {
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "concept_feedback_count": len(feedback.concept_feedback),
            "transcript_corrections_count": len(feedback.transcript_corrections),
            "segment_adjustments_count": len(feedback.segment_adjustments),
            "general_feedback": feedback.general_feedback
        }
        
        # Log feedback for analysis
        logger.info(f"Received feedback for job {job_id}: {feedback_summary}")
        
        # Here you would typically:
        # 1. Update the stored results with corrections
        # 2. Flag concepts for review
        # 3. Update training data
        # 4. Trigger reprocessing if needed
        
        # For now, we'll just acknowledge the feedback
        return {
            "status": "feedback_received",
            "job_id": job_id,
            "summary": feedback_summary,
            "message": "Feedback received and will be used to improve future processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

@router.get("/search")
async def search_videos(
    query: str = Query(..., description="Search query"),
    video_ids: Optional[str] = Query(None, description="Comma-separated video IDs to search within"),
    concepts: Optional[str] = Query(None, description="Comma-separated concepts to filter by"),
    speakers: Optional[str] = Query(None, description="Comma-separated speaker IDs to filter by"),
    min_duration: Optional[float] = Query(None, description="Minimum video duration in seconds"),
    max_duration: Optional[float] = Query(None, description="Maximum video duration in seconds"),
    limit: int = Query(default=10, description="Maximum number of results to return")
):
    """
    Search processed video content.
    
    This endpoint allows searching through processed videos by content,
    concepts, speakers, and other metadata.
    
    Args:
        query: Text query to search for
        video_ids: Optional filter by specific video IDs
        concepts: Optional filter by concepts
        speakers: Optional filter by speakers
        min_duration: Optional minimum duration filter
        max_duration: Optional maximum duration filter
        limit: Maximum results to return
        
    Returns:
        Search results with matching videos and segments
    """
    try:
        # Parse filters
        video_id_list = [id.strip() for id in video_ids.split(",")] if video_ids else None
        concept_list = [c.strip() for c in concepts.split(",")] if concepts else None
        speaker_list = [s.strip() for s in speakers.split(",")] if speakers else None
        
        # This is a placeholder implementation
        # In a real system, you would:
        # 1. Query your vector file_storage for semantic similarity
        # 2. Filter by metadata
        # 3. Rank and return results
        
        search_results = {
            "query": query,
            "total_results": 0,
            "results": [],
            "filters_applied": {
                "video_ids": video_id_list,
                "concepts": concept_list,
                "speakers": speaker_list,
                "min_duration": min_duration,
                "max_duration": max_duration
            },
            "message": "Search functionality is being implemented. This is a placeholder response."
        }
        
        return search_results
        
    except Exception as e:
        logger.error(f"Error searching videos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching videos: {str(e)}")

@router.websocket("/stream/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time processing updates.
    
    This endpoint provides real-time updates about video processing progress,
    including status changes, partial results, and Ghost reflections.
    
    Args:
        websocket: WebSocket connection
        job_id: Job identifier to stream updates for
    """
    await manager.connect(websocket, job_id)
    try:
        # Send initial status
        job_data = video_service.get_job_status(job_id)
        if job_data:
            await websocket.send_json({
                "type": "status_update",
                "job_id": job_id,
                "status": job_data["status"],
                "progress": job_data.get("progress", 0.0),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for client messages (ping/pong, etc.)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle client messages if needed
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass
                    
            except asyncio.TimeoutError:
                # Send keep-alive ping
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {str(e)}")
        manager.disconnect(websocket, job_id)

# Helper function to monitor job progress and send WebSocket updates
async def monitor_job_progress(job_id: str):
    """Monitor job progress and send WebSocket updates."""
    try:
        last_status = None
        last_progress = 0.0
        
        while True:
            job_data = video_service.get_job_status(job_id)
            
            if not job_data:
                break
            
            current_status = job_data["status"]
            current_progress = job_data.get("progress", 0.0)
            
            # Send update if status or significant progress change
            if (current_status != last_status or 
                abs(current_progress - last_progress) > 0.05 or
                current_status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]):
                
                update_message = {
                    "type": "progress_update",
                    "job_id": job_id,
                    "status": current_status,
                    "progress": current_progress,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Add result data if completed
                if current_status == ProcessingStatus.COMPLETED:
                    result = video_service.get_job_result(job_id)
                    if result:
                        update_message["summary"] = {
                            "duration": result.duration,
                            "segments_count": len(result.segments),
                            "concepts_count": len(result.concepts),
                            "questions_count": len(result.questions),
                            "integrity_score": result.integrity_score
                        }
                
                await manager.send_update(job_id, update_message)
                
                last_status = current_status
                last_progress = current_progress
            
            # Stop monitoring if job is complete or failed
            if current_status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                break
            
            # Wait before next check
            await asyncio.sleep(2.0)
            
    except Exception as e:
        logger.error(f"Error monitoring job {job_id}: {str(e)}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint for video ingestion service."""
    try:
        # Check if video service is initialized
        service_status = "healthy" if video_service.whisper_model else "initializing"
        
        return {
            "status": service_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_jobs": len(video_service.processing_jobs),
            "websocket_connections": sum(len(conns) for conns in manager.active_connections.values())
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

# Export router
__all__ = ["router"]
