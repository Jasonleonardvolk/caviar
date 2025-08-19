"""
Audio/Visual Enhancement Module for Prajna
==========================================

Adds audio transcription, TTS, video processing, and avatar state management.
"""

import os
import asyncio
import tempfile
import hashlib
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import time
import logging

from fastapi import File, UploadFile, Form, WebSocket, HTTPException, Query
from pydantic import BaseModel

# Audio processing imports
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ Whisper not available - install with: pip install openai-whisper")

# TTS imports
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("⚠️ Edge-TTS not available - install with: pip install edge-tts")

# Video processing imports
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available - install with: pip install opencv-python")

# Image analysis imports
try:
    from PIL import Image
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    print("⚠️ BLIP not available - install transformers and torch")

logger = logging.getLogger("prajna.audio_visual")

# Avatar state management
class AvatarState:
    """Global avatar state manager"""
    def __init__(self):
        self.state = "idle"  # idle, listening, thinking, speaking, processing
        self.mood = "neutral"  # neutral, happy, confused, focused
        self.audio_level = 0.0
        self.last_update = time.time()
        self.subscribers: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def update_state(self, state: str, mood: Optional[str] = None, audio_level: Optional[float] = None):
        async with self._lock:
            self.state = state
            if mood:
                self.mood = mood
            if audio_level is not None:
                self.audio_level = audio_level
            self.last_update = time.time()
            
            # Notify subscribers
            update = {
                "state": self.state,
                "mood": self.mood,
                "audio_level": self.audio_level,
                "timestamp": self.last_update
            }
            
            disconnected = []
            for ws in self.subscribers:
                try:
                    await ws.send_json(update)
                except:
                    disconnected.append(ws)
            
            # Remove disconnected clients
            for ws in disconnected:
                self.subscribers.remove(ws)
    
    async def subscribe(self, websocket: WebSocket):
        async with self._lock:
            self.subscribers.append(websocket)
    
    async def unsubscribe(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.subscribers:
                self.subscribers.remove(websocket)
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "mood": self.mood,
            "audio_level": self.audio_level,
            "last_update": self.last_update
        }

# Global avatar state instance
avatar_state = AvatarState()

# Audio/Visual models
class AudioVisualModels:
    """Lazy-loaded models for audio/visual processing"""
    def __init__(self):
        self.whisper_model = None
        self.blip_processor = None
        self.blip_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if 'torch' in globals() else None
    
    def load_whisper(self, model_size="base"):
        """Load Whisper model for speech-to-text"""
        if not WHISPER_AVAILABLE:
            raise HTTPException(status_code=503, detail="Whisper not available")
        
        if not self.whisper_model:
            logger.info(f"Loading Whisper model ({model_size})...")
            self.whisper_model = whisper.load_model(model_size)
            logger.info("✅ Whisper model loaded")
        
        return self.whisper_model
    
    def load_blip(self):
        """Load BLIP model for image captioning"""
        if not BLIP_AVAILABLE:
            raise HTTPException(status_code=503, detail="BLIP not available")
        
        if not self.blip_model:
            logger.info("Loading BLIP model...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
            logger.info("✅ BLIP model loaded")
        
        return self.blip_processor, self.blip_model

# Global models instance
av_models = AudioVisualModels()

# Request/Response models
class AudioTranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    confidence: Optional[float] = None

class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-AriaNeural"  # Default voice
    speed: float = 1.0
    pitch: float = 1.0

class VideoAnalysisResponse(BaseModel):
    frames_analyzed: int
    duration: float
    key_frames: List[Dict[str, Any]]
    summary: str
    objects_detected: List[str]
    transcription: Optional[str] = None

# Audio processing functions
async def transcribe_audio(file_path: str) -> AudioTranscriptionResponse:
    """Transcribe audio file using Whisper"""
    try:
        # Update avatar state
        await avatar_state.update_state("processing", "focused")
        
        # Load model
        model = av_models.load_whisper()
        
        # Transcribe
        logger.info(f"Transcribing audio: {file_path}")
        result = model.transcribe(file_path)
        
        # Extract info
        text = result.get("text", "").strip()
        language = result.get("language", "unknown")
        
        # Calculate confidence (Whisper doesn't provide this directly)
        segments = result.get("segments", [])
        if segments:
            avg_prob = sum(s.get("avg_logprob", 0) for s in segments) / len(segments)
            confidence = min(1.0, max(0.0, 1.0 + avg_prob / 10))  # Rough conversion
        else:
            confidence = 0.5
        
        logger.info(f"✅ Transcription complete: {text[:50]}...")
        
        return AudioTranscriptionResponse(
            text=text,
            language=language,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        await avatar_state.update_state("idle", "confused")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        await avatar_state.update_state("idle")

async def generate_tts(text: str, voice: str = "en-US-AriaNeural", output_dir: str = "tmp") -> str:
    """Generate TTS audio using edge-tts"""
    if not EDGE_TTS_AVAILABLE:
        logger.warning("TTS not available")
        return None
    
    try:
        # Update avatar state
        await avatar_state.update_state("speaking")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        output_file = os.path.join(output_dir, f"tts_{text_hash}_{int(time.time())}.mp3")
        
        # Generate TTS
        logger.info(f"Generating TTS for: {text[:50]}...")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        
        logger.info(f"✅ TTS generated: {output_file}")
        
        # Simulate audio level changes during speech
        asyncio.create_task(_simulate_speech_levels(len(text) * 0.05))  # Rough duration estimate
        
        return output_file
        
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        await avatar_state.update_state("idle", "confused")
        return None

async def _simulate_speech_levels(duration: float):
    """Simulate audio levels during speech"""
    steps = int(duration * 10)  # 10 updates per second
    for i in range(steps):
        # Create natural speech pattern
        level = 0.3 + 0.4 * abs(np.sin(i * 0.5)) + 0.3 * np.random.random()
        await avatar_state.update_state("speaking", audio_level=min(1.0, level))
        await asyncio.sleep(0.1)
    
    await avatar_state.update_state("idle", audio_level=0.0)

# Video processing functions
async def analyze_video(file_path: str, max_frames: int = 10) -> VideoAnalysisResponse:
    """Analyze video file and extract key information"""
    if not CV2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Video processing not available")
    
    try:
        await avatar_state.update_state("processing", "focused")
        
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Sample frames evenly
        frame_interval = max(1, frame_count // max_frames)
        key_frames = []
        objects_detected = set()
        
        # Load BLIP if available for frame analysis
        if BLIP_AVAILABLE:
            processor, model = av_models.load_blip()
        
        frame_idx = 0
        while cap.isOpened() and len(key_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Analyze frame
                if BLIP_AVAILABLE:
                    inputs = processor(pil_image, return_tensors="pt").to(av_models.device)
                    out = model.generate(**inputs, max_length=50)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    
                    # Extract objects from caption
                    for word in caption.split():
                        if len(word) > 3:  # Simple object extraction
                            objects_detected.add(word.lower())
                else:
                    caption = "Frame analysis not available"
                
                key_frames.append({
                    "frame_number": frame_idx,
                    "timestamp": frame_idx / fps if fps > 0 else 0,
                    "caption": caption
                })
            
            frame_idx += 1
        
        cap.release()
        
        # Extract audio for transcription if available
        transcription = None
        if WHISPER_AVAILABLE:
            # Extract audio track
            audio_path = file_path.replace('.mp4', '_audio.wav')
            try:
                # Use ffmpeg or similar to extract audio
                # For now, skip if not available
                pass
            except:
                pass
        
        # Generate summary
        summary = f"Video with {frame_count} frames ({duration:.1f}s). "
        if key_frames:
            summary += f"Key scenes: {', '.join(kf['caption'][:30] + '...' for kf in key_frames[:3])}"
        
        return VideoAnalysisResponse(
            frames_analyzed=len(key_frames),
            duration=duration,
            key_frames=key_frames,
            summary=summary,
            objects_detected=list(objects_detected),
            transcription=transcription
        )
        
    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        await avatar_state.update_state("idle", "confused")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    finally:
        await avatar_state.update_state("idle")

# Image analysis function
async def analyze_image(file_path: str) -> Dict[str, Any]:
    """Analyze image and generate caption"""
    if not BLIP_AVAILABLE:
        return {"caption": "Image analysis not available", "objects": []}
    
    try:
        await avatar_state.update_state("processing", "focused")
        
        # Load image
        image = Image.open(file_path)
        
        # Load BLIP
        processor, model = av_models.load_blip()
        
        # Generate caption
        inputs = processor(image, return_tensors="pt").to(av_models.device)
        out = model.generate(**inputs, max_length=100)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Extract potential objects
        objects = []
        for word in caption.split():
            if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'from']:
                objects.append(word.lower())
        
        return {
            "caption": caption,
            "objects": objects,
            "size": image.size,
            "mode": image.mode
        }
        
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return {"caption": f"Analysis failed: {str(e)}", "objects": []}
    finally:
        await avatar_state.update_state("idle")

# Audio/Visual API endpoints to add to Prajna
def create_audio_visual_endpoints(app):
    """Add audio/visual endpoints to the FastAPI app"""
    
    @app.post("/api/answer/audio", response_model=Dict[str, Any])
    async def answer_audio_query(
        audio: UploadFile = File(...),
        conversation_id: Optional[str] = Form(None),
        focus_concept: Optional[str] = Form(None),
        generate_audio_response: bool = Form(True)
    ):
        """
        Process audio query through Prajna
        
        1. Transcribe audio to text
        2. Process through Prajna pipeline
        3. Optionally generate TTS response
        """
        # Save uploaded audio
        temp_dir = Path("tmp/audio")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        audio_path = temp_dir / f"audio_{timestamp}_{audio.filename}"
        
        try:
            # Save audio file
            with open(audio_path, "wb") as f:
                content = await audio.read()
                f.write(content)
            
            # Transcribe audio
            await avatar_state.update_state("listening")
            transcription = await transcribe_audio(str(audio_path))
            
            # Process through Prajna
            await avatar_state.update_state("thinking")
            
            # Import Prajna components
            from prajna.api.prajna_api import answer_query, PrajnaRequest
            
            # Create request
            request = PrajnaRequest(
                user_query=transcription.text,
                focus_concept=focus_concept,
                conversation_id=conversation_id
            )
            
            # Get answer
            response = await answer_query(request)
            
            # Generate TTS if requested
            audio_url = None
            if generate_audio_response and response.answer:
                tts_file = await generate_tts(response.answer)
                if tts_file:
                    # In production, upload to CDN and return URL
                    # For now, return local path
                    audio_url = f"/static/audio/{Path(tts_file).name}"
            
            # Add audio-specific fields
            result = response.dict() if hasattr(response, 'dict') else response
            result['transcription'] = transcription.dict()
            result['audio_url'] = audio_url
            
            return result
            
        except Exception as e:
            logger.error(f"Audio query error: {e}")
            await avatar_state.update_state("idle", "confused")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Cleanup
            try:
                audio_path.unlink()
            except:
                pass
    
    @app.post("/api/answer/video", response_model=Dict[str, Any])
    async def answer_video_query(
        video: UploadFile = File(...),
        conversation_id: Optional[str] = Form(None),
        focus_concept: Optional[str] = Form(None),
        question: Optional[str] = Form(None)
    ):
        """
        Process video query through Prajna
        
        1. Analyze video frames
        2. Extract audio and transcribe if present
        3. Combine with user question
        4. Process through Prajna
        """
        # Save uploaded video
        temp_dir = Path("tmp/video")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        video_path = temp_dir / f"video_{timestamp}_{video.filename}"
        
        try:
            # Save video file
            with open(video_path, "wb") as f:
                content = await video.read()
                f.write(content)
            
            # Analyze video
            analysis = await analyze_video(str(video_path))
            
            # Build context from video
            context = f"Video Analysis: {analysis.summary}\n"
            if analysis.objects_detected:
                context += f"Objects detected: {', '.join(analysis.objects_detected)}\n"
            if analysis.transcription:
                context += f"Audio transcription: {analysis.transcription}\n"
            
            # Combine with user question
            if question:
                full_query = f"{context}\n\nQuestion: {question}"
            else:
                full_query = f"Please analyze this video:\n{context}"
            
            # Process through Prajna
            await avatar_state.update_state("thinking")
            
            from prajna.api.prajna_api import answer_query, PrajnaRequest
            
            request = PrajnaRequest(
                user_query=full_query,
                focus_concept=focus_concept,
                conversation_id=conversation_id
            )
            
            response = await answer_query(request)
            
            # Add video-specific fields
            result = response.dict() if hasattr(response, 'dict') else response
            result['video_analysis'] = analysis.dict()
            
            return result
            
        except Exception as e:
            logger.error(f"Video query error: {e}")
            await avatar_state.update_state("idle", "confused")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Cleanup
            try:
                video_path.unlink()
            except:
                pass
    
    @app.post("/api/answer/image", response_model=Dict[str, Any])
    async def answer_image_query(
        image: UploadFile = File(...),
        conversation_id: Optional[str] = Form(None),
        focus_concept: Optional[str] = Form(None),
        question: Optional[str] = Form("What is in this image?")
    ):
        """
        Process image query through Prajna
        
        1. Analyze image
        2. Combine with user question
        3. Process through Prajna
        """
        # Save uploaded image
        temp_dir = Path("tmp/images")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        image_path = temp_dir / f"image_{timestamp}_{image.filename}"
        
        try:
            # Save image file
            with open(image_path, "wb") as f:
                content = await image.read()
                f.write(content)
            
            # Analyze image
            analysis = await analyze_image(str(image_path))
            
            # Build query
            full_query = f"Image shows: {analysis['caption']}\n\nQuestion: {question}"
            
            # Process through Prajna
            await avatar_state.update_state("thinking")
            
            from prajna.api.prajna_api import answer_query, PrajnaRequest
            
            request = PrajnaRequest(
                user_query=full_query,
                focus_concept=focus_concept,
                conversation_id=conversation_id
            )
            
            response = await answer_query(request)
            
            # Add image-specific fields
            result = response.dict() if hasattr(response, 'dict') else response
            result['image_analysis'] = analysis
            
            return result
            
        except Exception as e:
            logger.error(f"Image query error: {e}")
            await avatar_state.update_state("idle", "confused")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Cleanup
            try:
                image_path.unlink()
            except:
                pass
    
    @app.get("/api/avatar/state")
    async def get_avatar_state():
        """Get current avatar state"""
        return avatar_state.get_state()
    
    @app.websocket("/api/avatar/updates")
    async def avatar_updates(websocket: WebSocket):
        """WebSocket for real-time avatar state updates"""
        await websocket.accept()
        await avatar_state.subscribe(websocket)
        
        try:
            # Send initial state
            await websocket.send_json(avatar_state.get_state())
            
            # Keep connection alive
            while True:
                # Wait for any message (heartbeat)
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
                    
        except WebSocketDisconnect:
            logger.info("Avatar WebSocket disconnected")
        finally:
            await avatar_state.unsubscribe(websocket)
    
    @app.post("/api/tts/generate")
    async def generate_tts_endpoint(request: TTSRequest):
        """Generate TTS audio from text"""
        if not EDGE_TTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="TTS not available")
        
        try:
            # Generate TTS
            output_file = await generate_tts(
                text=request.text,
                voice=request.voice
            )
            
            if not output_file:
                raise HTTPException(status_code=500, detail="TTS generation failed")
            
            # Return file info
            return {
                "audio_file": output_file,
                "audio_url": f"/static/audio/{Path(output_file).name}",
                "voice": request.voice,
                "text_length": len(request.text)
            }
            
        except Exception as e:
            logger.error(f"TTS endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    logger.info("✅ Audio/Visual endpoints added to Prajna")
    return app

# Export for use in prajna_api.py
__all__ = ['create_audio_visual_endpoints', 'avatar_state', 'AudioTranscriptionResponse', 'TTSRequest', 'VideoAnalysisResponse']
