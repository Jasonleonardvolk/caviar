"""
Audio Ingest Handler
Processes audio files using speech-to-text and audio analysis
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import asyncio
import numpy as np
import json

from hott_integration.ingest_handlers.base_handler import BaseIngestHandler
from hott_integration.psi_morphon import (
    PsiMorphon, PsiStrand, HolographicMemory,
    ModalityType, StrandType
)

logger = logging.getLogger(__name__)

# Try to import audio processing libraries
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    logger.warning("Whisper not available - using mock transcription")
    WHISPER_AVAILABLE = False
    whisper = None

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    logger.warning("Librosa not available - audio analysis limited")
    LIBROSA_AVAILABLE = False
    librosa = None

class AudioIngestHandler(BaseIngestHandler):
    """
    Handler for ingesting audio files
    Uses Whisper for speech-to-text and librosa for audio analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.supported_extensions = ['.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac']
        self.modality_type = ModalityType.AUDIO
        
        # Whisper settings
        self.whisper_model_name = config.get('whisper_model', 'base') if config else 'base'
        self.whisper_model = None
        
        # Audio analysis settings
        self.segment_duration = config.get('segment_duration', 30.0) if config else 30.0  # seconds
        self.sample_rate = config.get('sample_rate', 16000) if config else 16000
        
        # Initialize models
        if WHISPER_AVAILABLE:
            self._init_whisper()
    
    def _init_whisper(self):
        """Initialize Whisper model"""
        try:
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            logger.info(f"✅ Whisper model '{self.whisper_model_name}' loaded")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            self.whisper_model = None
    
    async def extract_morphons(self, file_path: Path,
                             metadata: Optional[Dict[str, Any]]) -> List[PsiMorphon]:
        """Extract morphons from audio file"""
        morphons = []
        
        # Create main audio morphon
        audio_morphon = PsiMorphon(
            modality=ModalityType.AUDIO,
            content=str(file_path),
            metadata={
                **self.extract_metadata(file_path),
                "format": file_path.suffix[1:]  # Remove dot
            },
            salience=1.0
        )
        morphons.append(audio_morphon)
        
        # Extract audio features if librosa available
        if LIBROSA_AVAILABLE:
            audio_features = await self._extract_audio_features(file_path)
            if audio_features:
                features_morphon = PsiMorphon(
                    modality=ModalityType.TEXT,
                    content=f"Audio features: tempo={audio_features.get('tempo', 'unknown')} BPM",
                    metadata=audio_features,
                    salience=0.5
                )
                morphons.append(features_morphon)
        
        # Transcribe audio if Whisper available
        if self.whisper_model:
            transcript_segments = await self._transcribe_audio(file_path)
            
            for segment in transcript_segments:
                # Create morphon for each transcript segment
                transcript_morphon = PsiMorphon(
                    modality=ModalityType.TEXT,
                    content=segment['text'],
                    metadata={
                        "source": "whisper_transcription",
                        "start_time": segment['start'],
                        "end_time": segment['end'],
                        "confidence": segment.get('confidence', 1.0)
                    },
                    temporal_index=segment['start'],
                    salience=0.9
                )
                morphons.append(transcript_morphon)
        
        return morphons
    
    async def _extract_audio_features(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract audio features using librosa"""
        if not LIBROSA_AVAILABLE:
            return None
        
        try:
            # Load audio
            y, sr = librosa.load(str(file_path), sr=self.sample_rate)
            
            # Extract features
            features = {
                "duration": float(len(y) / sr),
                "sample_rate": sr
            }
            
            # Tempo and beat
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features["tempo"] = float(tempo)
            features["num_beats"] = len(beats)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["avg_spectral_centroid"] = float(np.mean(spectral_centroids))
            
            # Zero crossing rate (indicates speech vs music)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features["avg_zero_crossing_rate"] = float(np.mean(zcr))
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            features["avg_rms_energy"] = float(np.mean(rms))
            
            return features
            
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            return None
    
    async def _transcribe_audio(self, file_path: Path) -> List[Dict[str, Any]]:
        """Transcribe audio using Whisper"""
        if not self.whisper_model:
            # Mock transcription for testing
            return [{
                "text": f"Mock transcription of {file_path.name}",
                "start": 0.0,
                "end": 10.0,
                "confidence": 0.5
            }]
        
        try:
            # Transcribe with timestamps
            result = self.whisper_model.transcribe(
                str(file_path),
                verbose=False,
                task="transcribe"
            )
            
            # Extract segments
            segments = []
            if 'segments' in result:
                for seg in result['segments']:
                    segments.append({
                        "text": seg['text'].strip(),
                        "start": seg['start'],
                        "end": seg['end'],
                        "confidence": seg.get('avg_logprob', 0) + 1.0  # Convert to 0-1 range
                    })
            else:
                # Fallback if no segments
                segments.append({
                    "text": result.get('text', ''),
                    "start": 0.0,
                    "end": 0.0,
                    "confidence": 1.0
                })
            
            logger.info(f"Transcribed {len(segments)} segments from {file_path}")
            return segments
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return []
    
    async def _generate_embedding(self, morphon: PsiMorphon) -> Optional[np.ndarray]:
        """Generate embedding for audio morphon"""
        if morphon.modality == ModalityType.AUDIO and LIBROSA_AVAILABLE:
            try:
                # Load audio
                y, sr = librosa.load(morphon.content, sr=self.sample_rate)
                
                # Extract MFCC features as embedding
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                
                # Average over time to get fixed-size embedding
                embedding = np.mean(mfccs, axis=1)
                
                # Pad to standard size (512)
                full_embedding = np.zeros(512, dtype=np.float32)
                full_embedding[:len(embedding)] = embedding
                
                return full_embedding / (np.linalg.norm(full_embedding) + 1e-8)
                
            except Exception as e:
                logger.error(f"Failed to generate audio embedding: {e}")
        
        return await super()._generate_embedding(morphon)
    
    async def create_strands(self, memory: HolographicMemory) -> List[PsiStrand]:
        """Create strands between audio and transcript segments"""
        strands = []
        
        # Get base temporal strands
        strands.extend(await super().create_strands(memory))
        
        # Find main audio morphon
        audio_morphons = memory.get_morphon_by_modality(ModalityType.AUDIO)
        if not audio_morphons:
            return strands
        
        main_audio = audio_morphons[0]
        
        # Connect audio to all transcript segments
        for morphon in memory.morphons:
            if (morphon.modality == ModalityType.TEXT and
                morphon.metadata.get('source') == 'whisper_transcription'):
                
                strand = PsiStrand(
                    source_morphon_id=main_audio.id,
                    target_morphon_id=morphon.id,
                    strand_type=StrandType.AUDIO_INSTANCE,
                    strength=morphon.metadata.get('confidence', 0.8),
                    evidence=f"Transcribed from audio at {morphon.metadata.get('start_time', 0):.1f}s",
                    confidence=morphon.metadata.get('confidence', 0.8)
                )
                strands.append(strand)
        
        # Connect sequential transcript segments with semantic strands
        transcript_morphons = sorted(
            [m for m in memory.morphons 
             if m.metadata.get('source') == 'whisper_transcription'],
            key=lambda m: m.temporal_index or 0
        )
        
        for i in range(len(transcript_morphons) - 1):
            strand = PsiStrand(
                source_morphon_id=transcript_morphons[i].id,
                target_morphon_id=transcript_morphons[i + 1].id,
                strand_type=StrandType.SEMANTIC,
                strength=0.7,
                evidence="Sequential speech segments"
            )
            strands.append(strand)
        
        return strands
