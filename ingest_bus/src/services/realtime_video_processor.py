"""
Real-Time Video Streaming Processor for TORI

This module implements real-time video/audio ingestion capabilities,
allowing TORI to process live streams and provide immediate feedback
and reflections as content is being captured.

Key Features:
- Live audio/video stream processing
- Real-time transcription with streaming Whisper
- Immediate Ghost Collective reflections
- Progressive concept extraction
- Live WebSocket updates to clients
- Buffered analysis for context maintenance
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime, timezone
import uuid
import io

# Core processing imports
import cv2
import whisper
from pydub import AudioSegment
import tempfile
import threading
from queue import Queue, Empty
import time

# Data structures
from dataclasses import dataclass
from enum import Enum

# WebSocket for real-time updates
from fastapi import WebSocket

# Import our base video service components
from .video_ingestion_service import (
    VideoIngestionService, ProcessingStatus, TranscriptSegment,
    VideoFrame, ExtractedConcept, Speaker
)

logger = logging.getLogger("tori.realtime_video")

class StreamingStatus(str, Enum):
    """Real-time streaming status."""
    CONNECTING = "connecting"
    ACTIVE = "active"
    BUFFERING = "buffering"
    PROCESSING = "processing"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class StreamingSession:
    """Real-time streaming session data."""
    session_id: str
    status: StreamingStatus
    created_at: datetime
    duration: float = 0.0
    
    # Accumulated data
    transcript_segments: List[TranscriptSegment] = None
    concepts: List[ExtractedConcept] = None
    questions: List[str] = None
    current_speakers: List[str] = None
    
    # Real-time processing state
    audio_buffer: bytes = b""
    video_buffer: List[np.ndarray] = None
    last_reflection_time: float = 0.0
    
    def __post_init__(self):
        if self.transcript_segments is None:
            self.transcript_segments = []
        if self.concepts is None:
            self.concepts = []
        if self.questions is None:
            self.questions = []
        if self.current_speakers is None:
            self.current_speakers = []
        if self.video_buffer is None:
            self.video_buffer = []

class RealTimeVideoProcessor:
    """
    Real-time video processor that handles live streams.
    
    This processor works differently from batch processing:
    - Processes audio/video in small chunks
    - Provides immediate transcription and analysis
    - Maintains context across chunks
    - Sends live updates via WebSocket
    """
    
    def __init__(self):
        """Initialize the real-time processor."""
        self.base_service = VideoIngestionService()
        self.active_sessions: Dict[str, StreamingSession] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}
        
        # Real-time processing parameters
        self.audio_chunk_duration = 3.0  # seconds
        self.video_frame_interval = 1.0  # seconds
        self.reflection_interval = 10.0  # seconds
        
        # Processing queues
        self.audio_queue = Queue()
        self.video_queue = Queue()
        
        # Start background processors
        self.processing_active = True
        self.audio_processor_task = None
        self.video_processor_task = None
        
    async def start_streaming_session(
        self,
        websocket: WebSocket,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new real-time streaming session.
        
        Args:
            websocket: WebSocket connection for real-time updates
            options: Processing options
            
        Returns:
            Session ID for the streaming session
        """
        try:
            session_id = str(uuid.uuid4())
            
            # Default options for real-time processing
            if options is None:
                options = {}
            
            default_options = {
                "language": "en",
                "enable_diarization": True,
                "enable_visual_context": True,
                "personas": ["Ghost Collective", "Scholar"],
                "immediate_reflection": True,
                "chunk_size": 3.0,
                "quality": "fast"  # Prioritize speed over accuracy
            }
            
            processing_options = {**default_options, **options}
            
            # Create streaming session
            session = StreamingSession(
                session_id=session_id,
                status=StreamingStatus.CONNECTING,
                created_at=datetime.now(timezone.utc)
            )
            
            self.active_sessions[session_id] = session
            self.websocket_connections[session_id] = websocket
            
            # Start processing tasks if not already running
            if not self.audio_processor_task:
                self.audio_processor_task = asyncio.create_task(self._audio_processor())
            if not self.video_processor_task:
                self.video_processor_task = asyncio.create_task(self._video_processor())
            
            logger.info(f"Started real-time streaming session: {session_id}")
            
            # Send initial status
            await self._send_update(session_id, {
                "type": "session_started",
                "session_id": session_id,
                "status": StreamingStatus.CONNECTING,
                "options": processing_options
            })
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start streaming session: {str(e)}")
            raise
    
    async def process_audio_chunk(
        self,
        session_id: str,
        audio_data: bytes,
        timestamp: Optional[float] = None
    ):
        """
        Process incoming audio chunk in real-time.
        
        Args:
            session_id: Streaming session ID
            audio_data: Raw audio data
            timestamp: Optional timestamp for the chunk
        """
        try:
            if session_id not in self.active_sessions:
                logger.warning(f"Audio chunk received for unknown session: {session_id}")
                return
            
            session = self.active_sessions[session_id]
            
            # Update session status
            if session.status == StreamingStatus.CONNECTING:
                session.status = StreamingStatus.ACTIVE
                await self._send_update(session_id, {
                    "type": "status_change",
                    "status": StreamingStatus.ACTIVE
                })
            
            # Add to audio buffer
            session.audio_buffer += audio_data
            current_time = timestamp or time.time()
            
            # Process if we have enough audio data
            buffer_duration = len(session.audio_buffer) / (16000 * 2)  # Assuming 16kHz 16-bit
            
            if buffer_duration >= self.audio_chunk_duration:
                # Queue for processing
                self.audio_queue.put({
                    "session_id": session_id,
                    "audio_data": session.audio_buffer,
                    "timestamp": current_time
                })
                
                # Clear buffer
                session.audio_buffer = b""
                
                logger.debug(f"Queued audio chunk for processing: {session_id}")
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}")
            await self._send_error(session_id, f"Audio processing error: {str(e)}")
    
    async def process_video_frame(
        self,
        session_id: str,
        frame_data: bytes,
        timestamp: Optional[float] = None
    ):
        """
        Process incoming video frame in real-time.
        
        Args:
            session_id: Streaming session ID
            frame_data: Raw frame data (JPEG or similar)
            timestamp: Optional timestamp for the frame
        """
        try:
            if session_id not in self.active_sessions:
                logger.warning(f"Video frame received for unknown session: {session_id}")
                return
            
            session = self.active_sessions[session_id]
            current_time = timestamp or time.time()
            
            # Decode frame
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.warning(f"Failed to decode video frame for session: {session_id}")
                return
            
            # Add to video buffer (keep only recent frames)
            session.video_buffer.append(frame)
            if len(session.video_buffer) > 10:  # Keep last 10 frames
                session.video_buffer.pop(0)
            
            # Queue for processing at intervals
            if current_time - session.last_reflection_time > self.video_frame_interval:
                self.video_queue.put({
                    "session_id": session_id,
                    "frame": frame.copy(),
                    "timestamp": current_time
                })
                
                logger.debug(f"Queued video frame for processing: {session_id}")
            
        except Exception as e:
            logger.error(f"Error processing video frame: {str(e)}")
            await self._send_error(session_id, f"Video processing error: {str(e)}")
    
    async def _audio_processor(self):
        """Background task to process audio chunks."""
        while self.processing_active:
            try:
                # Get audio chunk from queue (non-blocking)
                try:
                    chunk_data = self.audio_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                session_id = chunk_data["session_id"]
                audio_data = chunk_data["audio_data"]
                timestamp = chunk_data["timestamp"]
                
                if session_id not in self.active_sessions:
                    continue
                
                session = self.active_sessions[session_id]
                
                # Update status
                session.status = StreamingStatus.PROCESSING
                await self._send_update(session_id, {
                    "type": "processing_status",
                    "stage": "transcribing_audio"
                })
                
                # Save audio to temporary file for Whisper
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    # Convert raw audio bytes to WAV format
                    audio_segment = AudioSegment(
                        data=audio_data,
                        sample_width=2,  # 16-bit
                        frame_rate=16000,
                        channels=1
                    )
                    audio_segment.export(temp_audio.name, format="wav")
                    
                    # Transcribe with Whisper
                    try:
                        result = self.base_service.whisper_model.transcribe(
                            temp_audio.name,
                            language="en",
                            word_timestamps=True
                        )
                        
                        # Process transcription results
                        if result["segments"]:
                            new_segments = []
                            base_time = timestamp - self.audio_chunk_duration
                            
                            for i, segment in enumerate(result["segments"]):
                                transcript_segment = TranscriptSegment(
                                    id=f"live_{session_id}_{len(session.transcript_segments) + i}",
                                    start_time=base_time + segment["start"],
                                    end_time=base_time + segment["end"],
                                    text=segment["text"].strip(),
                                    speaker_id=f"Speaker_Live_{i % 2}",  # Simple alternating
                                    confidence=segment.get("avg_logprob", 0.0)
                                )
                                new_segments.append(transcript_segment)
                                session.transcript_segments.append(transcript_segment)
                            
                            # Send transcript update
                            await self._send_update(session_id, {
                                "type": "transcript_update",
                                "segments": [
                                    {
                                        "id": seg.id,
                                        "start_time": seg.start_time,
                                        "end_time": seg.end_time,
                                        "text": seg.text,
                                        "speaker_id": seg.speaker_id,
                                        "confidence": seg.confidence
                                    }
                                    for seg in new_segments
                                ]
                            })
                            
                            # Quick concept extraction on new content
                            await self._extract_live_concepts(session_id, new_segments)
                        
                    except Exception as e:
                        logger.error(f"Transcription failed for session {session_id}: {str(e)}")
                    
                    finally:
                        # Cleanup temp file
                        try:
                            import os
                            os.unlink(temp_audio.name)
                        except:
                            pass
                
                # Update status back to active
                session.status = StreamingStatus.ACTIVE
                
                # Check if it's time for Ghost reflection
                if timestamp - session.last_reflection_time > self.reflection_interval:
                    await self._generate_live_reflection(session_id)
                    session.last_reflection_time = timestamp
                
            except Exception as e:
                logger.error(f"Audio processor error: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _video_processor(self):
        """Background task to process video frames."""
        while self.processing_active:
            try:
                # Get frame from queue (non-blocking)
                try:
                    frame_data = self.video_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                session_id = frame_data["session_id"]
                frame = frame_data["frame"]
                timestamp = frame_data["timestamp"]
                
                if session_id not in self.active_sessions:
                    continue
                
                session = self.active_sessions[session_id]
                
                # Quick OCR check for slide content
                try:
                    import pytesseract
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ocr_text = pytesseract.image_to_string(gray, config='--psm 6')
                    
                    if ocr_text.strip():
                        # Send OCR update
                        await self._send_update(session_id, {
                            "type": "visual_content",
                            "timestamp": timestamp,
                            "ocr_text": ocr_text.strip(),
                            "content_type": "slide_text"
                        })
                        
                        # Quick concept extraction from visual content
                        visual_concepts = await self._extract_visual_concepts(ocr_text)
                        if visual_concepts:
                            session.concepts.extend(visual_concepts)
                            
                            await self._send_update(session_id, {
                                "type": "concepts_update",
                                "source": "visual",
                                "concepts": [
                                    {
                                        "term": concept.term,
                                        "type": concept.concept_type,
                                        "confidence": concept.confidence
                                    }
                                    for concept in visual_concepts
                                ]
                            })
                
                except Exception as e:
                    logger.debug(f"OCR processing failed: {str(e)}")
                
                # Face detection (lightweight)
                try:
                    import face_recognition
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    
                    if face_locations:
                        await self._send_update(session_id, {
                            "type": "faces_detected",
                            "timestamp": timestamp,
                            "face_count": len(face_locations),
                            "locations": face_locations
                        })
                
                except Exception as e:
                    logger.debug(f"Face detection failed: {str(e)}")
                
            except Exception as e:
                logger.error(f"Video processor error: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _extract_live_concepts(self, session_id: str, segments: List[TranscriptSegment]):
        """Extract concepts from new transcript segments in real-time."""
        try:
            session = self.active_sessions[session_id]
            
            # Combine text from new segments
            text = " ".join([seg.text for seg in segments])
            
            if not text.strip():
                return
            
            # Quick NLP processing
            doc = self.base_service.nlp_model(text)
            
            new_concepts = []
            
            # Extract named entities
            for ent in doc.ents:
                concept = ExtractedConcept(
                    term=ent.text,
                    concept_type=ent.label_,
                    confidence=0.8,
                    source_segments=[seg.id for seg in segments],
                    context=text,
                    timestamp_ranges=[(seg.start_time, seg.end_time) for seg in segments]
                )
                new_concepts.append(concept)
            
            # Look for questions
            if "?" in text:
                sentences = text.split(".")
                for sentence in sentences:
                    if "?" in sentence:
                        session.questions.append(sentence.strip())
            
            if new_concepts:
                session.concepts.extend(new_concepts)
                
                # Send concept update
                await self._send_update(session_id, {
                    "type": "concepts_update",
                    "source": "transcript",
                    "concepts": [
                        {
                            "term": concept.term,
                            "type": concept.concept_type,
                            "confidence": concept.confidence
                        }
                        for concept in new_concepts
                    ]
                })
            
            if session.questions:
                await self._send_update(session_id, {
                    "type": "questions_detected",
                    "questions": session.questions[-5:]  # Last 5 questions
                })
            
        except Exception as e:
            logger.error(f"Live concept extraction failed: {str(e)}")
    
    async def _extract_visual_concepts(self, ocr_text: str) -> List[ExtractedConcept]:
        """Extract concepts from OCR text."""
        try:
            if not ocr_text.strip():
                return []
            
            doc = self.base_service.nlp_model(ocr_text)
            concepts = []
            
            for ent in doc.ents:
                concept = ExtractedConcept(
                    term=ent.text,
                    concept_type=f"visual_{ent.label_}",
                    confidence=0.7,
                    source_segments=["visual"],
                    context=ocr_text,
                    timestamp_ranges=[(time.time(), time.time())]
                )
                concepts.append(concept)
            
            return concepts
            
        except Exception as e:
            logger.error(f"Visual concept extraction failed: {str(e)}")
            return []
    
    async def _generate_live_reflection(self, session_id: str):
        """Generate Ghost Collective reflection on current session content."""
        try:
            session = self.active_sessions[session_id]
            
            # Prepare content summary for Ghost analysis
            recent_transcript = " ".join([
                seg.text for seg in session.transcript_segments[-10:]  # Last 10 segments
            ])
            
            recent_concepts = [c.term for c in session.concepts[-10:]]  # Last 10 concepts
            
            if not recent_transcript and not recent_concepts:
                return
            
            # Generate Ghost reflection
            reflection_data = {
                "persona": "Ghost Collective",
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content_summary": {
                    "recent_segments": len(session.transcript_segments[-10:]),
                    "recent_concepts": recent_concepts[:5],
                    "questions_count": len(session.questions),
                    "duration": session.duration
                }
            }
            
            # Generate reflection message based on content
            if recent_concepts:
                reflection_data["message"] = f"I'm noticing emerging themes around: {', '.join(recent_concepts[:3])}. The conversation is developing depth in these areas."
            elif recent_transcript:
                reflection_data["message"] = f"The discussion is actively progressing. I've detected {len(session.transcript_segments)} segments of dialogue so far."
            else:
                reflection_data["message"] = "Listening and ready to provide insights as the conversation develops."
            
            reflection_data["confidence"] = 0.9
            
            # Send reflection
            await self._send_update(session_id, {
                "type": "ghost_reflection",
                "reflection": reflection_data
            })
            
            logger.info(f"Generated live reflection for session {session_id}")
            
        except Exception as e:
            logger.error(f"Live reflection generation failed: {str(e)}")
    
    async def _send_update(self, session_id: str, message: Dict[str, Any]):
        """Send update to WebSocket client."""
        try:
            if session_id in self.websocket_connections:
                websocket = self.websocket_connections[session_id]
                
                # Add session metadata
                message.update({
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                await websocket.send_json(message)
                
        except Exception as e:
            logger.error(f"Failed to send update to session {session_id}: {str(e)}")
            # Remove dead connection
            if session_id in self.websocket_connections:
                del self.websocket_connections[session_id]
    
    async def _send_error(self, session_id: str, error_message: str):
        """Send error message to WebSocket client."""
        await self._send_update(session_id, {
            "type": "error",
            "error": error_message
        })
    
    async def stop_streaming_session(self, session_id: str) -> Dict[str, Any]:
        """
        Stop a streaming session and return final results.
        
        Args:
            session_id: Session to stop
            
        Returns:
            Final session summary
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            session.status = StreamingStatus.STOPPED
            
            # Calculate final duration
            session.duration = (datetime.now(timezone.utc) - session.created_at).total_seconds()
            
            # Create final summary
            summary = {
                "session_id": session_id,
                "duration": session.duration,
                "total_segments": len(session.transcript_segments),
                "total_concepts": len(session.concepts),
                "questions_detected": len(session.questions),
                "speakers_detected": len(set([seg.speaker_id for seg in session.transcript_segments if seg.speaker_id])),
                "final_transcript": [
                    {
                        "id": seg.id,
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "text": seg.text,
                        "speaker_id": seg.speaker_id
                    }
                    for seg in session.transcript_segments
                ],
                "concepts": [
                    {
                        "term": concept.term,
                        "type": concept.concept_type,
                        "confidence": concept.confidence
                    }
                    for concept in session.concepts
                ],
                "questions": session.questions
            }
            
            # Send final update
            await self._send_update(session_id, {
                "type": "session_ended",
                "summary": summary
            })
            
            # Cleanup
            del self.active_sessions[session_id]
            if session_id in self.websocket_connections:
                del self.websocket_connections[session_id]
            
            logger.info(f"Stopped streaming session {session_id} (duration: {session.duration:.1f}s)")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error stopping session {session_id}: {str(e)}")
            raise
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a streaming session."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "status": session.status,
            "duration": (datetime.now(timezone.utc) - session.created_at).total_seconds(),
            "segments_count": len(session.transcript_segments),
            "concepts_count": len(session.concepts),
            "questions_count": len(session.questions),
            "is_connected": session_id in self.websocket_connections
        }
    
    async def shutdown(self):
        """Shutdown the real-time processor."""
        self.processing_active = False
        
        # Stop background tasks
        if self.audio_processor_task:
            self.audio_processor_task.cancel()
        if self.video_processor_task:
            self.video_processor_task.cancel()
        
        # Close all sessions
        for session_id in list(self.active_sessions.keys()):
            try:
                await self.stop_streaming_session(session_id)
            except:
                pass
        
        logger.info("Real-time video processor shut down")

# Global instance
realtime_processor = RealTimeVideoProcessor()
