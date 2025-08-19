import asyncio
import numpy as np
import whisper
import torch
from typing import AsyncGenerator, Dict, Any, Optional, List
from collections import deque
import io
import wave
import logging
import time

from .emotion import compute_spectral_features, detect_emotion_enhanced
from .spectral_oscillator import BanksyOscillator, create_oscillator_for_context

logger = logging.getLogger(__name__)

# Global model instance
_whisper_model = None

def get_whisper_model():
    """Lazy load Whisper model"""
    global _whisper_model
    if _whisper_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _whisper_model = whisper.load_model("base.en", device=device)
        logger.info(f"Loaded Whisper model on {device}")
    return _whisper_model

class AudioStreamProcessor:
    """Process streaming audio with buffering and incremental transcription"""
    
    def __init__(self, session_id: str = "stream", sample_rate: int = 16000):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.buffer = bytearray()  # Use bytearray for efficiency
        self.sample_count = 0  # Track samples directly
        self.chunk_duration = 1.0
        self.chunk_samples = int(sample_rate * self.chunk_duration)
        
        # Use our new BanksyOscillator
        self.oscillator = create_oscillator_for_context("voice")  # Voice-optimized
        
        self.transcript_buffer = []
        self.last_n_tokens = []
        
        # Performance tracking
        self.chunks_processed = 0
        self.total_samples_processed = 0
        self.start_time = time.time()
        
        # Audio statistics
        self.audio_stats = {
            'min_level': float('inf'),
            'max_level': 0.0,
            'avg_level': 0.0,
            'silence_ratio': 0.0
        }
        
        # Streaming configuration
        self.overlap_samples = int(sample_rate * 0.1)  # 100ms overlap
        self.min_speech_duration = 0.3  # Minimum speech duration
        self.vad_threshold = 0.02  # Voice activity detection threshold
        
        # Whisper streaming state
        self.previous_tokens = []
        self.temperature = 0.0
        self.compression_ratio_threshold = 2.4
        self.no_speech_threshold = 0.6
        
    async def process_chunk(self, audio_bytes: bytes) -> AsyncGenerator[Dict[str, Any], None]:
        """Process an audio chunk and yield results"""
        
        # Add to buffer
        self.buffer.extend(audio_bytes)
        self.sample_count += len(audio_bytes) // 2  # 16-bit samples
        
        # Log buffer status periodically
        if self.chunks_processed % 10 == 0:
            logger.debug(f"Buffer: {self.sample_count} samples, {len(self.buffer)} bytes")
        
        # Check if we have enough samples for processing
        if self.sample_count < self.chunk_samples:
            return
        
        # Process while we have enough samples
        while self.sample_count >= self.chunk_samples:
            # Extract chunk_samples worth of data
            bytes_needed = self.chunk_samples * 2
            chunk_data = bytes(self.buffer[:bytes_needed])
            
            # Update buffer and sample count
            self.buffer = self.buffer[bytes_needed:]
            self.sample_count -= self.chunk_samples
            
            # Keep overlap for smoother transitions
            if self.overlap_samples > 0 and len(self.buffer) >= self.overlap_samples * 2:
                overlap_bytes = self.overlap_samples * 2
                self.buffer[:0] = chunk_data[-overlap_bytes:]
                self.sample_count += self.overlap_samples
            
            # Process the chunk
            async for result in self._process_audio(chunk_data):
                yield result
            
            self.chunks_processed += 1
            self.total_samples_processed += self.chunk_samples
    
    async def _process_audio(self, audio_bytes: bytes) -> AsyncGenerator[Dict[str, Any], None]:
        """Process audio bytes and yield incremental results"""
        
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Update audio statistics
            self._update_audio_stats(audio_array)
            
            # Voice activity detection
            rms = np.sqrt(np.mean(audio_array**2))
            is_speech = rms > self.vad_threshold
            
            # Compute spectral features
            spectral_features = compute_spectral_features(audio_array, self.sample_rate)
            
            # Update BanksyOscillator with spectral features
            centroid = spectral_features.get('spectral_centroid', 440)
            emotion_confidence = 0.5  # Default if no speech
            
            # Skip transcription if audio is too quiet
            if not is_speech:
                # Still update oscillator for visualization
                self.oscillator.map_parameters(centroid, emotion_confidence, rms)
                self.oscillator.step(self.chunk_duration)
                psi_state = self.oscillator.psi_state()
                
                yield {
                    "transcript": " ".join(self.transcript_buffer[-10:]),
                    "spectral": {
                        "centroid": spectral_features.get('spectral_centroid', 0),
                        "rms": spectral_features.get('rms', 0)
                    },
                    "emotion": {
                        "label": "neutral",
                        "confidence": 0.0
                    },
                    "hologram_hint": self._generate_hologram_hint(spectral_features, psi_state),
                    "is_final": False,
                    "is_speech": False,
                    "psi_state": {
                        "phase": psi_state.get('psi_phase', 0),
                        "coherence": psi_state.get('phase_coherence', 0),
                        "magnitude": psi_state.get('psi_magnitude', 0)
                    },
                    "audio_stats": self.audio_stats.copy()
                }
                return
            
            # Get Whisper model
            model = get_whisper_model()
            
            # Prepare audio for Whisper (pad if needed)
            if len(audio_array) < self.sample_rate * 0.1:  # Minimum 100ms
                audio_array = np.pad(audio_array, (0, self.sample_rate // 10 - len(audio_array)))
            
            # Transcribe with streaming optimizations
            result = model.transcribe(
                audio_array,
                language='en',
                fp16=torch.cuda.is_available(),
                verbose=False,
                temperature=self.temperature,
                compression_ratio_threshold=self.compression_ratio_threshold,
                no_speech_threshold=self.no_speech_threshold,
                condition_on_previous_text=True,
                initial_prompt=self._get_prompt(),
                suppress_tokens=[-1],  # Don't suppress any tokens
                without_timestamps=True  # Faster for streaming
            )
            
            # Extract transcript and tokens
            text = result.get('text', '').strip()
            tokens = result.get('tokens', [])
            
            # Update transcript buffer with deduplication
            if text and text != self.transcript_buffer[-1:]:
                self.transcript_buffer.append(text)
                self.previous_tokens = tokens[-10:]  # Keep last 10 tokens for context
            
            # Enhanced emotion detection
            # First create a basic psi_state for emotion detection
            self.oscillator.map_parameters(centroid, 0.5, rms)  # Initial mapping
            self.oscillator.step(self.chunk_duration)
            initial_psi_state = self.oscillator.psi_state()
            
            emotion_result = detect_emotion_enhanced(
                spectral_features,
                initial_psi_state,
                audio_array,
                self.sample_rate
            )
            
            # Now update oscillator with emotion intensity
            emotion_intensity = emotion_result.get('confidence', 0.0)
            self.oscillator.map_parameters(centroid, emotion_intensity, rms)
            self.oscillator.step(self.chunk_duration)
            psi_state = self.oscillator.psi_state()
            
            # Generate hologram hint
            hologram_hint = self._generate_hologram_hint(spectral_features, psi_state)
            
            # Calculate processing metrics
            elapsed_time = time.time() - self.start_time
            real_time_factor = self.total_samples_processed / (self.sample_rate * elapsed_time) if elapsed_time > 0 else 0
            
            # Yield partial result
            yield {
                "transcript": " ".join(self.transcript_buffer[-10:]),  # Last 10 segments
                "spectral": {
                    "centroid": spectral_features.get('spectral_centroid', 0),
                    "rms": spectral_features.get('rms', 0),
                    "spread": spectral_features.get('spectral_spread', 0),
                    "flux": spectral_features.get('spectral_flux', 0)
                },
                "emotion": {
                    "label": emotion_result.get('label', 'neutral'),
                    "confidence": emotion_result.get('confidence', 0.0),
                    "valence": emotion_result.get('valence', 0.0),
                    "arousal": emotion_result.get('arousal', 0.0)
                },
                "hologram_hint": hologram_hint,
                "is_final": False,
                "is_speech": is_speech,
                "psi_state": {
                    "phase": psi_state.get('psi_phase', 0),
                    "coherence": psi_state.get('phase_coherence', 0),
                    "magnitude": psi_state.get('psi_magnitude', 0),
                    "oscillator_phases": psi_state.get('oscillator_phases', []),
                    "coupling_strength": psi_state.get('coupling_strength', 0.1)
                },
                "metrics": {
                    "chunks_processed": self.chunks_processed,
                    "samples_processed": self.total_samples_processed,
                    "real_time_factor": real_time_factor,
                    "buffer_samples": self.sample_count
                },
                "audio_stats": self.audio_stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}", exc_info=True)
            yield {
                "error": str(e),
                "is_final": False,
                "metrics": {
                    "chunks_processed": self.chunks_processed,
                    "buffer_samples": self.sample_count
                }
            }
    
    def _update_audio_stats(self, audio_array: np.ndarray):
        """Update running audio statistics"""
        rms = np.sqrt(np.mean(audio_array**2))
        
        self.audio_stats['min_level'] = min(self.audio_stats['min_level'], rms)
        self.audio_stats['max_level'] = max(self.audio_stats['max_level'], rms)
        
        # Running average
        alpha = 0.1  # Smoothing factor
        self.audio_stats['avg_level'] = (1 - alpha) * self.audio_stats['avg_level'] + alpha * rms
        
        # Silence ratio
        silence_samples = np.sum(np.abs(audio_array) < self.vad_threshold)
        silence_ratio = silence_samples / len(audio_array)
        self.audio_stats['silence_ratio'] = (1 - alpha) * self.audio_stats['silence_ratio'] + alpha * silence_ratio
    
    def _get_prompt(self) -> Optional[str]:
        """Generate prompt from recent transcript for better context"""
        if not self.transcript_buffer:
            return None
        
        # Use last 3 segments as context
        recent_text = " ".join(self.transcript_buffer[-3:])
        
        # Limit prompt length
        if len(recent_text) > 200:
            recent_text = recent_text[-200:]
        
        return recent_text
    
    def _generate_hologram_hint(self, spectral: Dict, psi_state: Dict) -> Dict:
        """Generate hologram visualization hint with enhanced parameters"""
        centroid = spectral.get('spectral_centroid', 440)
        rms = spectral.get('rms', 0)
        spread = spectral.get('spectral_spread', 0)
        
        # Enhanced hue mapping with spread influence
        base_hue = min(360.0, (centroid / 2000.0) * 360.0)
        spread_influence = min(60.0, (spread / 1000.0) * 60.0)
        hue = (base_hue + spread_influence) % 360.0
        
        # Dynamic intensity mapping
        intensity = min(1.0, rms * 10.0)
        
        # Add slight boost for speech
        if rms > self.vad_threshold:
            intensity = min(1.0, intensity * 1.2)
        
        # Get phase coherence as psi
        psi = psi_state.get('phase_coherence', 0.5)
        
        return {
            "hue": float(hue),
            "intensity": float(intensity),
            "psi": float(psi)
        }
    
    def _get_emotion_label(self, emotion: Dict) -> str:
        """Get dominant emotion label"""
        if not emotion:
            return "neutral"
        
        # Find dominant emotion
        max_emotion = max(emotion.items(), key=lambda x: x[1])
        
        # Map to simple labels
        emotion_map = {
            'excitement': 'excited',
            'calmness': 'calm',
            'energy': 'energetic',
            'clarity': 'focused',
            'stability': 'stable'
        }
        
        return emotion_map.get(max_emotion[0], 'neutral')
    
    async def finalize(self) -> Dict[str, Any]:
        """Generate final result with complete transcript"""
        
        # Process any remaining buffer
        if self.sample_count > 0:
            # Pad remaining samples to make a complete chunk if needed
            remaining_bytes = self.sample_count * 2
            chunk_data = bytes(self.buffer[:remaining_bytes])
            
            # Clear buffer
            self.buffer.clear()
            self.sample_count = 0
            
            # Process final chunk
            async for result in self._process_audio(chunk_data):
                result['is_final'] = True
                return result
        
        # Calculate final metrics
        elapsed_time = time.time() - self.start_time
        
        # Get final oscillator state
        final_psi_state = self.oscillator.psi_state()
        
        # Return final state
        return {
            "transcript": " ".join(self.transcript_buffer),
            "is_final": True,
            "psi_state": {
                "phase": final_psi_state.get('psi_phase', 0),
                "coherence": final_psi_state.get('phase_coherence', 0),
                "magnitude": final_psi_state.get('psi_magnitude', 0),
                "oscillator_phases": final_psi_state.get('oscillator_phases', []),
                "coupling_strength": final_psi_state.get('coupling_strength', 0.1)
            },
            "metrics": {
                "total_chunks": self.chunks_processed,
                "total_samples": self.total_samples_processed,
                "total_duration": self.total_samples_processed / self.sample_rate,
                "processing_time": elapsed_time,
                "real_time_factor": self.total_samples_processed / (self.sample_rate * elapsed_time) if elapsed_time > 0 else 0
            },
            "audio_stats": self.audio_stats
        }

# Global stream processors per session with BanksyOscillator support
_stream_processors: Dict[str, AudioStreamProcessor] = {}

async def stream_transcription(audio_bytes: bytes, session_id: str = "default") -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream transcription with incremental results using BanksyOscillator.
    
    Yields partial results as audio is processed.
    """
    
    # Get or create processor for this session
    if session_id not in _stream_processors:
        _stream_processors[session_id] = AudioStreamProcessor(session_id)
    
    processor = _stream_processors[session_id]
    
    # Process chunk
    async for result in processor.process_chunk(audio_bytes):
        yield result

async def finalize_stream(session_id: str = "default") -> Dict[str, Any]:
    """Finalize a streaming session and get complete results"""
    
    if session_id in _stream_processors:
        processor = _stream_processors[session_id]
        result = await processor.finalize()
        
        # Clean up
        del _stream_processors[session_id]
        
        return result
    
    return {"transcript": "", "is_final": True}

def cleanup_stale_sessions(max_age_seconds: int = 3600):
    """Clean up stale streaming sessions"""
    current_time = time.time()
    stale_sessions = []
    
    for session_id, processor in _stream_processors.items():
        age = current_time - processor.start_time
        if age > max_age_seconds:
            stale_sessions.append(session_id)
    
    for session_id in stale_sessions:
        logger.info(f"Cleaning up stale session: {session_id}")
        del _stream_processors[session_id]
    
    return len(stale_sessions)
