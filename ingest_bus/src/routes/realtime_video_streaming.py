"""
Real-Time Video Streaming API Routes for TORI

This module provides REST and WebSocket endpoints for real-time video/audio
streaming and processing. It enables live ingestion with immediate transcription,
concept extraction, and Ghost Collective reflections.

API Endpoints:
- WebSocket /api/v2/video/stream/live - Start real-time streaming session
- POST /api/v2/video/stream/{session_id}/audio - Send audio chunk
- POST /api/v2/video/stream/{session_id}/video - Send video frame
- POST /api/v2/video/stream/{session_id}/stop - Stop streaming session
- GET /api/v2/video/stream/{session_id}/status - Get session status
"""

import asyncio
import logging
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi import Form, File, UploadFile, Depends
from pydantic import BaseModel, Field
import json

# Import our real-time processor
from ..services.realtime_video_processor import realtime_processor, StreamingStatus

# Configure logging
logger = logging.getLogger("tori.realtime_api")

# Create router
router = APIRouter(prefix="/api/v2/video/stream", tags=["real-time-video"])

# Pydantic models
class StreamingOptions(BaseModel):
    """Options for real-time streaming."""
    language: str = Field(default="en", description="Language code for transcription")
    enable_diarization: bool = Field(default=True, description="Enable speaker diarization")
    enable_visual_context: bool = Field(default=True, description="Enable visual processing")
    personas: List[str] = Field(
        default=["Ghost Collective", "Scholar"],
        description="Ghost personas for real-time reflection"
    )
    immediate_reflection: bool = Field(default=True, description="Enable immediate reflections")
    chunk_size: float = Field(default=3.0, description="Audio chunk size in seconds")
    quality: str = Field(default="fast", description="Processing quality (fast/balanced/accurate)")

class AudioChunkData(BaseModel):
    """Audio chunk data for streaming."""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    timestamp: Optional[float] = Field(None, description="Timestamp for the chunk")
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    channels: int = Field(default=1, description="Number of audio channels")

class VideoFrameData(BaseModel):
    """Video frame data for streaming."""
    frame_data: str = Field(..., description="Base64 encoded frame data (JPEG)")
    timestamp: Optional[float] = Field(None, description="Timestamp for the frame")
    width: Optional[int] = Field(None, description="Frame width")
    height: Optional[int] = Field(None, description="Frame height")

@router.websocket("/live")
async def websocket_streaming_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming.
    
    This endpoint handles the full real-time streaming workflow:
    1. Client connects and sends configuration
    2. Client streams audio/video data
    3. Server provides real-time processing updates
    4. Session ends when client disconnects or sends stop signal
    
    WebSocket Message Format:
    - {"type": "start_session", "options": {...}} - Start streaming
    - {"type": "audio_chunk", "data": "base64...", "timestamp": 123456} - Audio data
    - {"type": "video_frame", "data": "base64...", "timestamp": 123456} - Video frame
    - {"type": "stop_session"} - Stop streaming
    - {"type": "ping"} - Keep-alive ping
    """
    await websocket.accept()
    session_id = None
    
    try:
        logger.info("New WebSocket connection for real-time streaming")
        
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "Real-time video streaming ready",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                message_type = message.get("type")
                
                if message_type == "start_session":
                    # Start new streaming session
                    options = message.get("options", {})
                    session_id = await realtime_processor.start_streaming_session(
                        websocket, options
                    )
                    
                    await websocket.send_json({
                        "type": "session_started",
                        "session_id": session_id,
                        "status": "active"
                    })
                    
                elif message_type == "audio_chunk" and session_id:
                    # Process audio chunk
                    audio_data_b64 = message.get("data", "")
                    timestamp = message.get("timestamp")
                    
                    if audio_data_b64:
                        try:
                            audio_bytes = base64.b64decode(audio_data_b64)
                            await realtime_processor.process_audio_chunk(
                                session_id, audio_bytes, timestamp
                            )
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Audio processing error: {str(e)}"
                            })
                
                elif message_type == "video_frame" and session_id:
                    # Process video frame
                    frame_data_b64 = message.get("data", "")
                    timestamp = message.get("timestamp")
                    
                    if frame_data_b64:
                        try:
                            frame_bytes = base64.b64decode(frame_data_b64)
                            await realtime_processor.process_video_frame(
                                session_id, frame_bytes, timestamp
                            )
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Video processing error: {str(e)}"
                            })
                
                elif message_type == "stop_session" and session_id:
                    # Stop streaming session
                    summary = await realtime_processor.stop_streaming_session(session_id)
                    
                    await websocket.send_json({
                        "type": "session_stopped",
                        "summary": summary
                    })
                    
                    session_id = None
                
                elif message_type == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    })
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON message format"
                })
            
            except Exception as e:
                logger.error(f"WebSocket message processing error: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    
    finally:
        # Cleanup session if still active
        if session_id:
            try:
                await realtime_processor.stop_streaming_session(session_id)
            except:
                pass

@router.post("/{session_id}/audio")
async def upload_audio_chunk(
    session_id: str,
    audio_chunk: AudioChunkData
):
    """
    Upload audio chunk for processing.
    
    Alternative to WebSocket for clients that prefer REST API.
    
    Args:
        session_id: Active streaming session ID
        audio_chunk: Audio data and metadata
        
    Returns:
        Processing confirmation
    """
    try:
        # Decode audio data
        audio_bytes = base64.b64decode(audio_chunk.audio_data)
        
        # Process audio chunk
        await realtime_processor.process_audio_chunk(
            session_id, 
            audio_bytes, 
            audio_chunk.timestamp
        )
        
        return {
            "status": "processed",
            "session_id": session_id,
            "chunk_size": len(audio_bytes),
            "timestamp": audio_chunk.timestamp or datetime.now(timezone.utc).timestamp()
        }
        
    except Exception as e:
        logger.error(f"Audio chunk processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

@router.post("/{session_id}/video")
async def upload_video_frame(
    session_id: str,
    video_frame: VideoFrameData
):
    """
    Upload video frame for processing.
    
    Alternative to WebSocket for clients that prefer REST API.
    
    Args:
        session_id: Active streaming session ID
        video_frame: Video frame data and metadata
        
    Returns:
        Processing confirmation
    """
    try:
        # Decode frame data
        frame_bytes = base64.b64decode(video_frame.frame_data)
        
        # Process video frame
        await realtime_processor.process_video_frame(
            session_id,
            frame_bytes,
            video_frame.timestamp
        )
        
        return {
            "status": "processed",
            "session_id": session_id,
            "frame_size": len(frame_bytes),
            "timestamp": video_frame.timestamp or datetime.now(timezone.utc).timestamp()
        }
        
    except Exception as e:
        logger.error(f"Video frame processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

@router.post("/{session_id}/stop")
async def stop_streaming_session(session_id: str):
    """
    Stop a real-time streaming session.
    
    Args:
        session_id: Session to stop
        
    Returns:
        Final session summary
    """
    try:
        summary = await realtime_processor.stop_streaming_session(session_id)
        
        return {
            "status": "stopped",
            "session_id": session_id,
            "summary": summary
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error stopping session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping session: {str(e)}")

@router.get("/{session_id}/status")
async def get_streaming_status(session_id: str):
    """
    Get current status of a streaming session.
    
    Args:
        session_id: Session to check
        
    Returns:
        Current session status and statistics
    """
    try:
        status = realtime_processor.get_session_status(session_id)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@router.get("/active")
async def get_active_sessions():
    """
    Get list of currently active streaming sessions.
    
    Returns:
        List of active sessions with basic info
    """
    try:
        active_sessions = []
        
        for session_id in realtime_processor.active_sessions:
            status = realtime_processor.get_session_status(session_id)
            if status:
                active_sessions.append(status)
        
        return {
            "active_sessions": active_sessions,
            "total_count": len(active_sessions),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting active sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting active sessions: {str(e)}")

@router.post("/start")
async def start_rest_session(options: StreamingOptions):
    """
    Start a new streaming session via REST API.
    
    This creates a session that can receive audio/video via REST endpoints
    instead of WebSocket. Updates can be retrieved by polling the status endpoint.
    
    Args:
        options: Streaming configuration options
        
    Returns:
        New session information
    """
    try:
        # Create a dummy WebSocket for the session
        # (Real-time updates won't be available via REST)
        class DummyWebSocket:
            async def send_json(self, data):
                pass  # No-op for REST sessions
        
        dummy_ws = DummyWebSocket()
        
        session_id = await realtime_processor.start_streaming_session(
            dummy_ws, options.dict()
        )
        
        return {
            "session_id": session_id,
            "status": "active",
            "message": "REST streaming session started",
            "endpoints": {
                "audio": f"/api/v2/video/stream/{session_id}/audio",
                "video": f"/api/v2/video/stream/{session_id}/video",
                "status": f"/api/v2/video/stream/{session_id}/status",
                "stop": f"/api/v2/video/stream/{session_id}/stop"
            }
        }
        
    except Exception as e:
        logger.error(f"Error starting REST session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")

@router.get("/health")
async def streaming_health_check():
    """Health check for real-time streaming service."""
    try:
        processor_status = "healthy" if realtime_processor.processing_active else "inactive"
        
        return {
            "status": processor_status,
            "active_sessions": len(realtime_processor.active_sessions),
            "websocket_connections": len(realtime_processor.websocket_connections),
            "audio_queue_size": realtime_processor.audio_queue.qsize(),
            "video_queue_size": realtime_processor.video_queue.qsize(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Streaming health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Streaming service unhealthy")

# Export router
__all__ = ["router"]
