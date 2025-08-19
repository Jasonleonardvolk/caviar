"""
ingest_audio.py - Voice input pipeline for TORI's conceptual graph
Converts speech to concepts with spectral/emotional metadata
"""

import asyncio
import json
import numpy as np
import librosa
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging

# Mock imports for production dependencies
try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("Whisper not available - using mock transcription")

@dataclass
class AudioSpectralMetadata:
    """Spectral analysis metadata for holographic integration"""
    dominant_frequency: float
    mean_pitch: float
    pitch_variance: float
    spectral_centroid: float
    spectral_rolloff: float
    zero_crossing_rate: float
    mfcc_coefficients: List[float]
    emotional_indicators: Dict[str, float]

@dataclass
class AudioConceptDiff:
    """Extracted concepts from audio with metadata"""
    concept_id: str
    confidence: float
    text_source: str
    spectral_signature: AudioSpectralMetadata
    temporal_position: Tuple[float, float]  # start, end in seconds
    emotional_context: Dict[str, float]

@dataclass
class AudioIngestionResult:
    """Complete result of audio processing"""
    session_id: str
    concepts: List[AudioConceptDiff]
    transcript: str
    duration: float
    spectral_summary: AudioSpectralMetadata
    emotional_arc: List[Dict[str, float]]  # Emotional state over time
    phase_signature: str
    holographic_hints: Dict[str, Any]

class AudioEmotionDetector:
    """Analyzes vocal characteristics for emotional state"""
    
    def __init__(self):
        self.emotion_ranges = {
            'calm': {'pitch': (80, 200), 'variance': (0, 0.3)},
            'excited': {'pitch': (200, 400), 'variance': (0.3, 0.8)},
            'distressed': {'pitch': (150, 350), 'variance': (0.4, 1.0)},
            'focused': {'pitch': (100, 220), 'variance': (0, 0.2)},
            'uncertain': {'pitch': (120, 280), 'variance': (0.3, 0.6)}
        }
    
    def analyze_emotional_state(self, audio_features: AudioSpectralMetadata) -> Dict[str, float]:
        """Classify emotional state from spectral features"""
        emotions = {}
        
        for emotion, ranges in self.emotion_ranges.items():
            pitch_match = self._in_range(audio_features.mean_pitch, ranges['pitch'])
            variance_match = self._in_range(audio_features.pitch_variance, ranges['variance'])
            
            # Combine pitch and variance scores
            emotions[emotion] = (pitch_match + variance_match) / 2
        
        # Normalize scores
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v / total for k, v in emotions.items()}
        
        return emotions
    
    def _in_range(self, value: float, range_tuple: Tuple[float, float]) -> float:
        """Calculate how well a value fits within a range (0-1 score)"""
        min_val, max_val = range_tuple
        if min_val <= value <= max_val:
            # Perfect fit
            center = (min_val + max_val) / 2
            return 1.0 - abs(value - center) / (max_val - min_val)
        else:
            # Outside range - calculate distance
            if value < min_val:
                distance = min_val - value
            else:
                distance = value - max_val
            
            # Convert distance to score (closer = higher score)
            return max(0, 1.0 - distance / max_val)

class AudioSpectralAnalyzer:
    """Extracts spectral features for holographic mapping"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def analyze_spectrum(self, audio_data: np.ndarray) -> AudioSpectralMetadata:
        """Extract comprehensive spectral features"""
        
        # Basic spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
        
        # MFCC coefficients
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1).tolist()
        
        # Pitch analysis
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate)
        
        # Extract dominant pitch
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            dominant_frequency = max(set(pitch_values), key=pitch_values.count)
            mean_pitch = np.mean(pitch_values)
            pitch_variance = np.var(pitch_values)
        else:
            dominant_frequency = 0
            mean_pitch = 0
            pitch_variance = 0
        
        # Emotional indicators from spectral features
        emotional_indicators = self._extract_emotional_indicators(
            spectral_centroid, spectral_rolloff, zero_crossing_rate, pitch_variance
        )
        
        return AudioSpectralMetadata(
            dominant_frequency=float(dominant_frequency),
            mean_pitch=float(mean_pitch),
            pitch_variance=float(pitch_variance),
            spectral_centroid=float(np.mean(spectral_centroid)),
            spectral_rolloff=float(np.mean(spectral_rolloff)),
            zero_crossing_rate=float(np.mean(zero_crossing_rate)),
            mfcc_coefficients=mfcc_means,
            emotional_indicators=emotional_indicators
        )
    
    def _extract_emotional_indicators(self, centroid, rolloff, zcr, pitch_var) -> Dict[str, float]:
        """Extract emotional indicators from spectral features"""
        return {
            'brightness': float(np.mean(centroid) / 8000),  # Normalized brightness
            'harshness': float(np.mean(rolloff) / 11000),   # Spectral harshness
            'agitation': float(np.mean(zcr) * 10),          # Voice agitation
            'instability': float(pitch_var / 1000)          # Pitch instability
        }

class AudioConceptExtractor:
    """Extracts concepts from transcribed audio"""
    
    def __init__(self):
        # In production, this would use sophisticated NLP
        self.concept_keywords = {
            'programming': ['code', 'function', 'variable', 'debug', 'error', 'bug'],
            'learning': ['understand', 'concept', 'idea', 'knowledge', 'study'],
            'problem-solving': ['solve', 'issue', 'challenge', 'solution', 'fix'],
            'creativity': ['create', 'design', 'innovative', 'artistic', 'original'],
            'collaboration': ['team', 'together', 'help', 'discuss', 'share'],
            'planning': ['plan', 'strategy', 'goal', 'roadmap', 'timeline'],
            'emotion': ['feel', 'emotion', 'mood', 'happy', 'sad', 'frustrated', 'excited']
        }
    
    def extract_concepts(self, transcript: str, spectral_metadata: AudioSpectralMetadata) -> List[str]:
        """Extract concept IDs from transcript with spectral context"""
        concepts = []
        transcript_lower = transcript.lower()
        
        # Keyword-based extraction
        for concept, keywords in self.concept_keywords.items():
            if any(keyword in transcript_lower for keyword in keywords):
                concepts.append(concept)
        
        # Spectral-based concept enhancement
        emotional_state = self._infer_emotional_context(spectral_metadata)
        if emotional_state:
            concepts.extend(emotional_state)
        
        return list(set(concepts))  # Remove duplicates
    
    def _infer_emotional_context(self, spectral: AudioSpectralMetadata) -> List[str]:
        """Infer emotional concepts from spectral analysis"""
        concepts = []
        
        if spectral.emotional_indicators['agitation'] > 0.7:
            concepts.append('stress')
        elif spectral.emotional_indicators['brightness'] > 0.6:
            concepts.append('enthusiasm')
        
        if spectral.emotional_indicators['instability'] > 0.5:
            concepts.append('uncertainty')
        
        return concepts

class AudioProcessor:
    """Main audio ingestion processor"""
    
    def __init__(self):
        self.transcriber = self._init_transcriber()
        self.spectral_analyzer = AudioSpectralAnalyzer()
        self.emotion_detector = AudioEmotionDetector()
        self.concept_extractor = AudioConceptExtractor()
    
    def _init_transcriber(self):
        """Initialize speech-to-text transcriber"""
        if WHISPER_AVAILABLE:
            try:
                return whisper.load_model("base")
            except Exception as e:
                logging.warning(f"Failed to load Whisper model: {e}")
                return None
        return None
    
    async def process_audio_file(self, file_path: str) -> AudioIngestionResult:
        """Process audio file and extract concepts"""
        session_id = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self)}"
        
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(file_path, sr=22050)
            duration = len(audio_data) / sample_rate
            
            # Transcribe audio
            transcript = await self._transcribe_audio(file_path)
            
            # Analyze spectrum
            spectral_summary = self.spectral_analyzer.analyze_spectrum(audio_data)
            
            # Detect emotions
            emotional_state = self.emotion_detector.analyze_emotional_state(spectral_summary)
            
            # Extract concepts
            concept_ids = self.concept_extractor.extract_concepts(transcript, spectral_summary)
            
            # Create concept diffs
            concepts = []
            for i, concept_id in enumerate(concept_ids):
                concept_diff = AudioConceptDiff(
                    concept_id=concept_id,
                    confidence=0.8,  # Base confidence
                    text_source=transcript,
                    spectral_signature=spectral_summary,
                    temporal_position=(0, duration),  # Full audio for now
                    emotional_context=emotional_state
                )
                concepts.append(concept_diff)
            
            # Determine phase signature
            phase_signature = self._determine_phase_signature(spectral_summary, emotional_state)
            
            # Generate holographic hints
            holographic_hints = self._generate_holographic_hints(spectral_summary, emotional_state)
            
            return AudioIngestionResult(
                session_id=session_id,
                concepts=concepts,
                transcript=transcript,
                duration=duration,
                spectral_summary=spectral_summary,
                emotional_arc=[emotional_state],  # Single point for now
                phase_signature=phase_signature,
                holographic_hints=holographic_hints
            )
            
        except Exception as e:
            logging.error(f"Audio processing failed: {e}")
            # Return minimal result
            return AudioIngestionResult(
                session_id=session_id,
                concepts=[],
                transcript="",
                duration=0,
                spectral_summary=AudioSpectralMetadata(0, 0, 0, 0, 0, 0, [], {}),
                emotional_arc=[],
                phase_signature="error",
                holographic_hints={}
            )
    
    async def _transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio to text"""
        if self.transcriber and WHISPER_AVAILABLE:
            try:
                result = self.transcriber.transcribe(file_path)
                return result["text"]
            except Exception as e:
                logging.error(f"Transcription failed: {e}")
                return "Transcription failed"
        else:
            # Mock transcription for development
            return "This is a mock transcription of the audio file."
    
    def _determine_phase_signature(self, spectral: AudioSpectralMetadata, emotions: Dict[str, float]) -> str:
        """Determine phase signature from audio characteristics"""
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'neutral'
        
        phase_map = {
            'calm': 'coherence',
            'excited': 'resonance',
            'distressed': 'entropy',
            'focused': 'coherence',
            'uncertain': 'drift'
        }
        
        return phase_map.get(dominant_emotion, 'neutral')
    
    def _generate_holographic_hints(self, spectral: AudioSpectralMetadata, emotions: Dict[str, float]) -> Dict[str, Any]:
        """Generate hints for holographic visualization"""
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'neutral'
        
        # Map emotional state to wavelength
        wavelength_map = {
            'calm': 520,      # Green - peaceful
            'excited': 580,   # Yellow - energetic
            'distressed': 650,  # Red - alert
            'focused': 470,   # Blue - clarity
            'uncertain': 400  # Violet - mystery
        }
        
        return {
            'wavelength': wavelength_map.get(dominant_emotion, 550),
            'intensity': min(1.0, spectral.emotional_indicators.get('brightness', 0.5) * 2),
            'pulse_rate': spectral.emotional_indicators.get('agitation', 0.1) * 10,
            'glow_radius': 20 + spectral.emotional_indicators.get('instability', 0) * 30,
            'color_hint': dominant_emotion,
            'spectral_signature': {
                'frequency': spectral.dominant_frequency,
                'centroid': spectral.spectral_centroid,
                'brightness': spectral.emotional_indicators.get('brightness', 0)
            }
        }

# Main API function
async def ingest_audio(file_path: str) -> Dict[str, Any]:
    """
    Main entry point for audio ingestion
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Dictionary containing concept diffs and metadata
    """
    processor = AudioProcessor()
    result = await processor.process_audio_file(file_path)
    
    # Convert to dictionary for API response
    return {
        'session_id': result.session_id,
        'concepts': [asdict(concept) for concept in result.concepts],
        'transcript': result.transcript,
        'duration': result.duration,
        'spectral_summary': asdict(result.spectral_summary),
        'emotional_arc': result.emotional_arc,
        'phase_signature': result.phase_signature,
        'holographic_hints': result.holographic_hints,
        'timestamp': datetime.now().isoformat()
    }

# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    async def main():
        if len(sys.argv) != 2:
            print("Usage: python ingest_audio.py <audio_file_path>")
            return
        
        file_path = sys.argv[1]
        result = await ingest_audio(file_path)
        
        print(json.dumps(result, indent=2))
    
    asyncio.run(main())