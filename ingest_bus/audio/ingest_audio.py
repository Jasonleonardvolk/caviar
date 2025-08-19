import whisper
import numpy as np
import wave
import os
import asyncio
import logging
from pydub import AudioSegment
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import torch
import torchaudio
from dataclasses import dataclass
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# Load Whisper model once at import time with optimization
_whisper_model = None
_whisper_device = None

def get_whisper_model():
    """Lazy load Whisper model with device optimization"""
    global _whisper_model, _whisper_device
    
    if _whisper_model is None:
        # Detect best device
        if torch.cuda.is_available():
            _whisper_device = "cuda"
            logger.info("Using CUDA for Whisper inference")
        elif torch.backends.mps.is_available():
            _whisper_device = "mps"
            logger.info("Using MPS (Apple Silicon) for Whisper inference")
        else:
            _whisper_device = "cpu"
            logger.info("Using CPU for Whisper inference")
        
        # Load model with appropriate size based on available memory
        try:
            if _whisper_device == "cuda" and torch.cuda.get_device_properties(0).total_memory > 8e9:
                _whisper_model = whisper.load_model("medium.en", device=_whisper_device)
                logger.info("Loaded Whisper medium model")
            else:
                _whisper_model = whisper.load_model("base.en", device=_whisper_device)
                logger.info("Loaded Whisper base model")
        except Exception as e:
            logger.warning(f"Failed to load optimized model: {e}, falling back to base CPU model")
            _whisper_model = whisper.load_model("base.en", device="cpu")
            _whisper_device = "cpu"
    
    return _whisper_model, _whisper_device

@dataclass
class AudioMetadata:
    """Enhanced metadata for audio processing"""
    sample_rate: int
    duration: float
    channels: int
    bit_depth: int
    format: str
    file_size: int
    timestamp: str

class SpectralKoopmanOscillator:
    """Phase-coupled oscillator network for ψ-based audio processing"""
    
    def __init__(self, n_oscillators: int = 8, sample_rate: int = 44100):
        self.n_oscillators = n_oscillators
        self.sample_rate = sample_rate
        self.phases = np.random.uniform(0, 2*np.pi, n_oscillators)
        self.frequencies = np.logspace(np.log10(80), np.log10(8000), n_oscillators)  # Voice range
        self.amplitudes = np.ones(n_oscillators)
        self.coupling_matrix = self._initialize_coupling_matrix()
        self.coupling_strength = 0.1
        self.psi = 0.0  # Collective phase
        self.psi_history = []  # Track phase evolution
        self.coherence_history = []
        self.last_update_time = 0
        
    def _initialize_coupling_matrix(self) -> np.ndarray:
        """Initialize coupling matrix with frequency-based weights"""
        K = np.zeros((self.n_oscillators, self.n_oscillators))
        
        for i in range(self.n_oscillators):
            for j in range(self.n_oscillators):
                if i != j:
                    # Couple nearby frequencies more strongly
                    freq_ratio = self.frequencies[j] / self.frequencies[i]
                    if 0.5 < freq_ratio < 2.0:  # Within an octave
                        K[i, j] = np.exp(-abs(np.log(freq_ratio)))
                    else:
                        K[i, j] = 0.1  # Weak coupling for distant frequencies
        
        return K / np.max(K)  # Normalize
        
    def update_from_spectral(self, spectral_features: Dict, time_delta: float = 0.01) -> Dict:
        """Update oscillator network from spectral features and compute ψ-state"""
        centroid = spectral_features.get('spectral_centroid', 1000)
        rms = spectral_features.get('rms', 0.1)
        spread = spectral_features.get('spectral_spread', 100)
        flux = spectral_features.get('spectral_flux', 0)
        
        # Map spectral features to oscillator parameters
        coupling_modifier = np.tanh((centroid - 1000) / 500)  # Normalize around 1kHz
        energy_modifier = np.tanh(rms * 10)
        
        # Update amplitudes based on spectral spread
        for i in range(self.n_oscillators):
            freq_distance = abs(self.frequencies[i] - centroid)
            self.amplitudes[i] = np.exp(-freq_distance / (spread + 1e-6)) * energy_modifier
        
        # Kuramoto model with adaptive coupling
        phase_derivatives = np.zeros(self.n_oscillators)
        
        for i in range(self.n_oscillators):
            # Intrinsic frequency contribution
            phase_derivatives[i] = 2 * np.pi * self.frequencies[i] / self.sample_rate
            
            # Coupling term with amplitude weighting
            coupling_sum = 0
            for j in range(self.n_oscillators):
                if i != j:
                    coupling_sum += (self.coupling_matrix[i, j] * 
                                   self.amplitudes[j] * 
                                   np.sin(self.phases[j] - self.phases[i]))
            
            phase_derivatives[i] += self.coupling_strength * coupling_modifier * coupling_sum
            
            # Add spectral flux as phase perturbation
            phase_derivatives[i] += flux * 0.1 * np.random.randn()
        
        # Update phases with time integration
        self.phases += phase_derivatives * time_delta
        self.phases = self.phases % (2 * np.pi)  # Wrap to [0, 2π]
        
        # Compute collective ψ (Kuramoto order parameter)
        complex_phases = self.amplitudes * np.exp(1j * self.phases)
        psi_complex = np.mean(complex_phases)
        self.psi = np.angle(psi_complex)
        coherence = np.abs(psi_complex)
        
        # Track history for temporal analysis
        self.psi_history.append(self.psi)
        self.coherence_history.append(coherence)
        
        # Limit history size
        if len(self.psi_history) > 1000:
            self.psi_history.pop(0)
            self.coherence_history.pop(0)
        
        # Compute phase velocity and acceleration
        phase_velocity = 0
        phase_acceleration = 0
        if len(self.psi_history) > 1:
            phase_velocity = (self.psi - self.psi_history[-2]) / time_delta
            if len(self.psi_history) > 2:
                prev_velocity = (self.psi_history[-2] - self.psi_history[-3]) / time_delta
                phase_acceleration = (phase_velocity - prev_velocity) / time_delta
        
        # Generate comprehensive hologram hints
        return {
            'psi_phase': float(self.psi),
            'psi_magnitude': float(np.abs(psi_complex)),
            'oscillator_phases': self.phases.tolist(),
            'oscillator_amplitudes': self.amplitudes.tolist(),
            'oscillator_frequencies': self.frequencies.tolist(),
            'dominant_frequency': float(self.frequencies[np.argmax(self.amplitudes)]),
            'phase_coherence': float(coherence),
            'phase_velocity': float(phase_velocity),
            'phase_acceleration': float(phase_acceleration),
            'emotional_resonance': self._compute_emotional_resonance(centroid, rms, spread, flux),
            'coupling_state': {
                'matrix_norm': float(np.linalg.norm(self.coupling_matrix)),
                'effective_coupling': float(coupling_modifier * self.coupling_strength),
                'synchronization_index': float(self._compute_synchronization_index())
            }
        }
    
    def _compute_emotional_resonance(self, centroid: float, rms: float, spread: float, flux: float) -> Dict:
        """Enhanced emotional mapping based on spectral characteristics"""
        # Normalize inputs
        norm_centroid = np.tanh((centroid - 1000) / 1000)
        norm_rms = np.tanh(rms * 10)
        norm_spread = np.tanh(spread / 500)
        norm_flux = np.tanh(flux * 5)
        
        # Multi-dimensional emotional mapping
        excitement = np.clip(
            norm_centroid * 0.4 + norm_rms * 0.3 + norm_flux * 0.3,
            0, 1
        )
        
        calmness = np.clip(
            (1 - norm_flux) * 0.4 + (1 - norm_spread) * 0.3 + np.exp(-norm_rms * 2) * 0.3,
            0, 1
        )
        
        energy = np.clip(
            norm_rms * 0.5 + abs(norm_flux) * 0.3 + norm_spread * 0.2,
            0, 1
        )
        
        clarity = np.clip(
            (1 - norm_spread) * 0.4 + self._compute_harmonic_clarity() * 0.6,
            0, 1
        )
        
        # Compute emotional stability
        stability = 1.0
        if len(self.coherence_history) > 10:
            stability = 1.0 - np.std(self.coherence_history[-10:])
        
        return {
            'excitement': float(excitement),
            'calmness': float(calmness),
            'energy': float(energy),
            'clarity': float(clarity),
            'stability': float(np.clip(stability, 0, 1)),
            'valence': float(excitement - calmness),  # Positive/negative emotion
            'arousal': float(energy)  # High/low activation
        }
    
    def _compute_harmonic_clarity(self) -> float:
        """Compute harmonic clarity based on amplitude distribution"""
        if np.sum(self.amplitudes) == 0:
            return 0.0
        
        # Normalized amplitude distribution
        norm_amps = self.amplitudes / np.sum(self.amplitudes)
        
        # Entropy of amplitude distribution (lower = clearer)
        entropy = -np.sum(norm_amps * np.log(norm_amps + 1e-10))
        max_entropy = np.log(self.n_oscillators)
        
        # Convert to clarity (0-1)
        clarity = 1.0 - (entropy / max_entropy)
        
        return clarity
    
    def _compute_synchronization_index(self) -> float:
        """Compute pairwise phase synchronization index"""
        sync_index = 0
        pair_count = 0
        
        for i in range(self.n_oscillators):
            for j in range(i + 1, self.n_oscillators):
                # Phase difference
                phase_diff = abs(self.phases[i] - self.phases[j])
                phase_diff = min(phase_diff, 2 * np.pi - phase_diff)  # Wrap around
                
                # Synchronization strength (1 when in phase, 0 when π out of phase)
                sync_strength = np.cos(phase_diff) * 0.5 + 0.5
                
                # Weight by amplitudes
                weight = self.amplitudes[i] * self.amplitudes[j]
                sync_index += sync_strength * weight
                pair_count += weight
        
        return sync_index / (pair_count + 1e-10)
    
    def reset(self):
        """Reset oscillator network to initial state"""
        self.phases = np.random.uniform(0, 2*np.pi, self.n_oscillators)
        self.amplitudes = np.ones(self.n_oscillators)
        self.psi = 0.0
        self.psi_history.clear()
        self.coherence_history.clear()

# Global oscillator instance pool for different sessions
_oscillator_pool: Dict[str, SpectralKoopmanOscillator] = {}

def get_or_create_oscillator(session_id: str = "default", n_oscillators: int = 8) -> SpectralKoopmanOscillator:
    """Get or create an oscillator for a specific session"""
    if session_id not in _oscillator_pool:
        _oscillator_pool[session_id] = SpectralKoopmanOscillator(n_oscillators)
    return _oscillator_pool[session_id]

async def transcribe_audio_async(audio_path: str, session_id: str = "default", 
                                enhance_quality: bool = True) -> dict:
    """Async wrapper for audio transcription with enhanced processing"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, 
        transcribe_audio, 
        audio_path, 
        session_id,
        enhance_quality
    )

def transcribe_audio(audio_path: str, session_id: str = "default", 
                    enhance_quality: bool = True) -> dict:
    """
    Transcribe audio and generate ψ-based hologram hints using phase-coupled oscillators.
    Returns comprehensive audio analysis with holographic metadata.
    """
    try:
        # Validate input file
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        file_size = os.path.getsize(audio_path)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError(f"Audio file too large: {file_size} bytes")
        
        # Get file extension
        ext = audio_path.rsplit('.', 1)[-1].lower()
        
        # Convert to WAV if needed
        wav_path = audio_path
        if ext in ('mp3', 'webm', 'ogg', 'flac', 'm4a', 'aac'):
            wav_path = convert_to_wav(audio_path, enhance_quality)
        
        # Extract metadata
        metadata = extract_audio_metadata(wav_path)
        
        # 1) Speech-to-text with Whisper
        logger.info(f"Transcribing audio: {audio_path}")
        whisper_model, device = get_whisper_model()
        
        # Transcribe with options
        result = whisper_model.transcribe(
            wav_path,
            language='en',
            task='transcribe',
            verbose=False,
            temperature=0.0,  # More deterministic
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True,
            initial_prompt=None
        )
        
        transcript = result.get('text', '').strip()
        segments = result.get('segments', [])
        
        # 2) Enhanced spectral analysis
        spectral_features, audio_data, sample_rate = analyze_audio_spectral(wav_path)
        
        # 3) ψ-oscillator processing for hologram hints
        oscillator = get_or_create_oscillator(session_id)
        psi_state = oscillator.update_from_spectral(spectral_features)
        
        # 4) Generate comprehensive hologram metadata
        hologram_hints = generate_hologram_hints(
            spectral_features, 
            psi_state, 
            transcript, 
            metadata.duration,
            segments
        )
        
        # 5) Perform voice activity detection
        vad_segments = detect_voice_activity(audio_data, sample_rate)
        
        # 6) Extract prosodic features
        prosodic_features = extract_prosodic_features(audio_data, sample_rate)
        
        # Clean up temporary file
        if wav_path != audio_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass
        
        return {
            'transcript': transcript,
            'segments': segments,
            'spectral_centroid': spectral_features['spectral_centroid'],
            'duration': metadata.duration,
            'metadata': metadata.__dict__,
            'psi_state': psi_state,
            'hologram_hints': hologram_hints,
            'spectral_features': spectral_features,
            'vad_segments': vad_segments,
            'prosodic_features': prosodic_features,
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'processing_info': {
                'whisper_model': whisper_model.dims.__dict__ if hasattr(whisper_model, 'dims') else {},
                'device': device,
                'enhanced': enhance_quality
            }
        }
        
    except Exception as e:
        logger.error(f"Audio transcription error: {str(e)}")
        raise

def convert_to_wav(audio_path: str, enhance: bool = True) -> str:
    """Convert audio to WAV with optional enhancement"""
    try:
        audio = AudioSegment.from_file(audio_path)
        
        # Enhancement options
        if enhance:
            # Normalize audio
            audio = audio.normalize()
            
            # Remove silence from beginning and end
            audio = trim_silence(audio)
            
            # Apply subtle compression
            audio = compress_dynamic_range(audio)
        
        # Convert to mono if multi-channel
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Ensure 16kHz sample rate for Whisper
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        
        # Export as WAV
        wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
        audio.export(wav_path, format='wav', parameters=["-ar", "16000", "-ac", "1"])
        
        return wav_path
        
    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}")
        raise

def trim_silence(audio: AudioSegment, silence_threshold: float = -40.0) -> AudioSegment:
    """Trim silence from beginning and end of audio"""
    # Find non-silent chunks
    nonsilent_chunks = []
    chunk_length = 10  # ms
    
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i + chunk_length]
        if chunk.dBFS > silence_threshold:
            nonsilent_chunks.append(i)
    
    if not nonsilent_chunks:
        return audio
    
    # Trim to non-silent range
    start = max(0, nonsilent_chunks[0] - 100)  # Keep 100ms padding
    end = min(len(audio), nonsilent_chunks[-1] + 100)
    
    return audio[start:end]

def compress_dynamic_range(audio: AudioSegment, threshold: float = -20.0, 
                          ratio: float = 4.0) -> AudioSegment:
    """Apply dynamic range compression"""
    # Simple compression implementation
    return audio.compress_dynamic_range(
        threshold=threshold,
        ratio=ratio,
        attack=5.0,
        release=50.0
    )

def extract_audio_metadata(wav_path: str) -> AudioMetadata:
    """Extract comprehensive audio metadata"""
    with wave.open(wav_path, 'rb') as wf:
        return AudioMetadata(
            sample_rate=wf.getframerate(),
            duration=wf.getnframes() / wf.getframerate(),
            channels=wf.getnchannels(),
            bit_depth=wf.getsampwidth() * 8,
            format='wav',
            file_size=os.path.getsize(wav_path),
            timestamp=datetime.utcnow().isoformat()
        )

def analyze_audio_spectral(wav_path: str) -> Tuple[Dict, np.ndarray, int]:
    """Perform comprehensive spectral analysis"""
    # Load audio data
    with wave.open(wav_path, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
    
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Import enhanced spectral analysis
    from ingest_bus.audio.emotion import compute_spectral_features
    spectral_features = compute_spectral_features(data, rate)
    
    # Add additional spectral metrics
    spectral_features.update({
        'spectral_kurtosis': compute_spectral_kurtosis(data, rate),
        'spectral_skewness': compute_spectral_skewness(data, rate),
        'spectral_slope': compute_spectral_slope(data, rate),
        'mfcc_mean': compute_mfcc_features(data, rate)
    })
    
    return spectral_features, data, rate

def compute_spectral_kurtosis(data: np.ndarray, rate: int) -> float:
    """Compute spectral kurtosis"""
    fft = np.fft.rfft(data)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(data), 1.0 / rate)
    
    # Compute moments
    mean = np.average(freqs, weights=magnitude)
    variance = np.average((freqs - mean)**2, weights=magnitude)
    
    if variance == 0:
        return 0.0
    
    kurtosis = np.average((freqs - mean)**4, weights=magnitude) / (variance**2)
    
    return float(kurtosis - 3)  # Excess kurtosis

def compute_spectral_skewness(data: np.ndarray, rate: int) -> float:
    """Compute spectral skewness"""
    fft = np.fft.rfft(data)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(data), 1.0 / rate)
    
    # Compute moments
    mean = np.average(freqs, weights=magnitude)
    variance = np.average((freqs - mean)**2, weights=magnitude)
    
    if variance == 0:
        return 0.0
    
    skewness = np.average((freqs - mean)**3, weights=magnitude) / (variance**1.5)
    
    return float(skewness)

def compute_spectral_slope(data: np.ndarray, rate: int) -> float:
    """Compute spectral slope using linear regression"""
    fft = np.fft.rfft(data)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(data), 1.0 / rate)
    
    # Log scale
    log_freqs = np.log(freqs[1:] + 1e-10)
    log_magnitude = np.log(magnitude[1:] + 1e-10)
    
    # Linear regression
    slope, _ = np.polyfit(log_freqs, log_magnitude, 1)
    
    return float(slope)

def compute_mfcc_features(data: np.ndarray, rate: int, n_mfcc: int = 13) -> List[float]:
    """Compute MFCC features using torchaudio"""
    try:
        # Convert to tensor
        waveform = torch.from_numpy(data).float().unsqueeze(0)
        
        # Compute MFCCs
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=rate,
            n_mfcc=n_mfcc,
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23}
        )
        
        mfccs = mfcc_transform(waveform)
        
        # Return mean of each coefficient
        return mfccs.mean(dim=2).squeeze().tolist()
        
    except Exception as e:
        logger.warning(f"MFCC computation failed: {e}")
        return [0.0] * n_mfcc

def detect_voice_activity(data: np.ndarray, rate: int, 
                         frame_duration: float = 0.03) -> List[Dict]:
    """Detect voice activity segments"""
    frame_length = int(rate * frame_duration)
    num_frames = len(data) // frame_length
    
    vad_segments = []
    in_speech = False
    speech_start = 0
    
    for i in range(num_frames):
        frame = data[i * frame_length:(i + 1) * frame_length]
        
        # Simple energy-based VAD
        energy = np.sum(frame**2) / len(frame)
        is_speech = energy > 0.001  # Threshold
        
        if is_speech and not in_speech:
            speech_start = i * frame_duration
            in_speech = True
        elif not is_speech and in_speech:
            vad_segments.append({
                'start': speech_start,
                'end': i * frame_duration,
                'duration': i * frame_duration - speech_start
            })
            in_speech = False
    
    # Handle final segment
    if in_speech:
        vad_segments.append({
            'start': speech_start,
            'end': num_frames * frame_duration,
            'duration': num_frames * frame_duration - speech_start
        })
    
    return vad_segments

def extract_prosodic_features(data: np.ndarray, rate: int) -> Dict:
    """Extract prosodic features (pitch, timing, intensity)"""
    try:
        # Convert to tensor
        waveform = torch.from_numpy(data).float().unsqueeze(0)
        
        # Pitch detection using torchaudio
        pitch = torchaudio.functional.detect_pitch_frequency(waveform, rate)
        
        # Remove unvoiced frames
        voiced_pitch = pitch[pitch > 0]
        
        if len(voiced_pitch) > 0:
            pitch_mean = float(voiced_pitch.mean())
            pitch_std = float(voiced_pitch.std())
            pitch_range = float(voiced_pitch.max() - voiced_pitch.min())
        else:
            pitch_mean = pitch_std = pitch_range = 0.0
        
        # Speaking rate (syllables per second estimate)
        # Simple estimate based on energy peaks
        energy = np.convolve(data**2, np.ones(int(rate * 0.01)), mode='same')
        peaks = find_peaks(energy, int(rate * 0.1))
        speaking_rate = len(peaks) / (len(data) / rate)
        
        return {
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'pitch_range': pitch_range,
            'speaking_rate': float(speaking_rate),
            'intensity_mean': float(np.mean(np.abs(data))),
            'intensity_std': float(np.std(np.abs(data)))
        }
        
    except Exception as e:
        logger.warning(f"Prosodic feature extraction failed: {e}")
        return {
            'pitch_mean': 0.0,
            'pitch_std': 0.0,
            'pitch_range': 0.0,
            'speaking_rate': 0.0,
            'intensity_mean': float(np.mean(np.abs(data))),
            'intensity_std': float(np.std(np.abs(data)))
        }

def find_peaks(signal: np.ndarray, min_distance: int) -> List[int]:
    """Simple peak detection"""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            if not peaks or i - peaks[-1] >= min_distance:
                peaks.append(i)
    return peaks

def generate_hologram_hints(spectral_features: Dict, psi_state: Dict, 
                          transcript: str, duration: float,
                          segments: List[Dict]) -> Dict:
    """Generate comprehensive hints for holographic visualization"""
    
    # Map ψ-phase to 3D coordinates and colors
    psi_phase = psi_state['psi_phase']
    coherence = psi_state['phase_coherence']
    
    # Enhanced 3D position mapping with emotional influence
    emotion = psi_state['emotional_resonance']
    
    # Spherical coordinates with emotional modulation
    theta = psi_phase  # Azimuth from ψ
    phi = np.arccos(coherence * 2 - 1)  # Elevation from coherence
    
    # Radius influenced by energy and clarity
    radius = (spectral_features['rms'] * 5 + emotion['energy'] * 5) * (0.5 + emotion['clarity'] * 0.5)
    
    # Convert to Cartesian with emotional offsets
    x = radius * np.sin(phi) * np.cos(theta) + emotion['excitement'] * 0.2
    y = radius * np.sin(phi) * np.sin(theta) + emotion['calmness'] * 0.2
    z = radius * np.cos(phi) + emotion['energy'] * 0.1
    
    # Enhanced color mapping from emotional resonance
    hue = (psi_phase + np.pi) / (2 * np.pi) * 360
    
    # Saturation based on emotional intensity
    emotional_intensity = np.sqrt(emotion['excitement']**2 + emotion['energy']**2)
    saturation = np.clip(coherence * emotional_intensity * 100, 20, 100)
    
    # Lightness from clarity and stability
    lightness = np.clip((emotion['clarity'] * 0.5 + emotion['stability'] * 0.5) * 80 + 20, 20, 90)
    
    # Generate enhanced volumetric density
    density_field = generate_enhanced_density_field(spectral_features, psi_state)
    
    # Temporal alignment for segments
    temporal_markers = []
    for segment in segments:
        temporal_markers.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'],
            'psi_phase': (segment['start'] / duration) * 2 * np.pi,
            'confidence': segment.get('no_speech_prob', 0)
        })
    
    return {
        'position_3d': {'x': float(x), 'y': float(y), 'z': float(z)},
        'color_hsl': {'h': float(hue), 's': float(saturation), 'l': float(lightness)},
        'volumetric_density': density_field,
        'oscillator_visualization': {
            'phases': psi_state['oscillator_phases'],
            'amplitudes': psi_state.get('oscillator_amplitudes', []),
            'frequencies': psi_state.get('oscillator_frequencies', []),
            'coupling_strength': psi_state.get('coupling_state', {}).get('effective_coupling', 0.1)
        },
        'temporal_coherence': {
            'beat_frequency': spectral_features.get('beat_frequency', 0),
            'phase_stability': float(coherence),
            'phase_velocity': psi_state.get('phase_velocity', 0),
            'phase_acceleration': psi_state.get('phase_acceleration', 0),
            'emotional_flow': emotion
        },
        'semantic_anchors': extract_semantic_anchors(transcript, temporal_markers),
        'recommended_views': generate_view_recommendations(psi_state, emotion),
        'particle_hints': generate_particle_hints(psi_state, spectral_features),
        'audio_quality_metrics': {
            'spectral_complexity': compute_spectral_complexity(spectral_features),
            'harmonic_clarity': psi_state.get('coupling_state', {}).get('synchronization_index', 0),
            'dynamic_range': float(np.log10(spectral_features.get('rms', 0.001) + 1) * 20)
        }
    }

def generate_enhanced_density_field(spectral_features: Dict, psi_state: Dict) -> List[List[List[float]]]:
    """Generate enhanced 3D density field for volumetric rendering"""
    grid_size = 16  # Higher resolution
    density = np.zeros((grid_size, grid_size, grid_size))
    
    # Get oscillator data
    phases = np.array(psi_state['oscillator_phases'])
    amplitudes = np.array(psi_state.get('oscillator_amplitudes', np.ones(len(phases))))
    coherence = psi_state['phase_coherence']
    
    # Create multi-scale interference patterns
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                # Spatial coordinates normalized to [-1, 1]
                x = 2 * i/(grid_size-1) - 1
                y = 2 * j/(grid_size-1) - 1
                z = 2 * k/(grid_size-1) - 1
                
                # Distance from center
                r = np.sqrt(x*x + y*y + z*z)
                
                # Multi-oscillator interference pattern
                interference = 0
                for idx, (phase, amp) in enumerate(zip(phases, amplitudes)):
                    # Each oscillator contributes a wave
                    spatial_freq = (idx + 1) * np.pi
                    
                    # 3D wave with radial and angular components
                    wave = amp * np.cos(spatial_freq * r + phase)
                    
                    # Add angular modulation
                    if r > 0:
                        theta = np.arctan2(y, x)
                        phi = np.arccos(z / r)
                        wave *= np.sin(idx * theta + phase) * np.sin(idx * phi)
                    
                    interference += wave
                
                # Normalize and apply coherence scaling
                base_density = (interference / len(phases) + 1) * 0.5 * coherence
                
                # Add Gaussian envelope
                envelope = np.exp(-r*r / 0.5)
                
                # Add spectral energy contribution
                energy_contribution = spectral_features.get('rms', 0) * np.exp(-r*r / 0.3)
                
                density[i, j, k] = np.clip(base_density * envelope + energy_contribution, 0, 1)
    
    # Apply 3D Gaussian smoothing for continuity
    from scipy.ndimage import gaussian_filter
    density = gaussian_filter(density, sigma=0.5)
    
    return density.tolist()

def extract_semantic_anchors(transcript: str, temporal_markers: List[Dict]) -> List[Dict]:
    """Extract semantic concepts with temporal alignment"""
    if not transcript:
        return []
    
    # Enhanced keyword categories
    semantic_categories = {
        'joy': {
            'keywords': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'love'],
            'weight': 1.2,
            'color': {'h': 45, 's': 80, 'l': 60}
        },
        'calm': {
            'keywords': ['calm', 'peaceful', 'quiet', 'still', 'gentle', 'soft', 'serene', 'tranquil'],
            'weight': 1.0,
            'color': {'h': 200, 's': 50, 'l': 50}
        },
        'energy': {
            'keywords': ['energy', 'power', 'strong', 'intense', 'dynamic', 'vibrant', 'active', 'vigorous'],
            'weight': 1.3,
            'color': {'h': 0, 's': 90, 'l': 50}
        },
        'focus': {
            'keywords': ['focus', 'concentrate', 'attention', 'clear', 'sharp', 'precise', 'think', 'understand'],
            'weight': 1.1,
            'color': {'h': 270, 's': 60, 'l': 55}
        },
        'uncertainty': {
            'keywords': ['maybe', 'perhaps', 'might', 'could', 'possibly', 'unsure', 'doubt'],
            'weight': 0.8,
            'color': {'h': 180, 's': 30, 'l': 40}
        }
    }
    
    anchors = []
    words = transcript.lower().split()
    
    # Process each temporal segment
    for marker in temporal_markers:
        segment_words = marker['text'].lower().split()
        segment_start = marker['start']
        segment_duration = marker['end'] - marker['start']
        
        for word_idx, word in enumerate(segment_words):
            # Check each semantic category
            for category, info in semantic_categories.items():
                if word in info['keywords']:
                    # Calculate temporal position within segment
                    word_position = word_idx / max(len(segment_words), 1)
                    absolute_time = segment_start + word_position * segment_duration
                    
                    anchors.append({
                        'concept': category,
                        'keyword': word,
                        'temporal_position': absolute_time / duration if duration > 0 else 0,
                        'absolute_time': absolute_time,
                        'semantic_weight': info['weight'] * (1 - marker.get('confidence', 0)),
                        'color_hint': info['color'],
                        'segment_index': temporal_markers.index(marker)
                    })
    
    # Sort by temporal position
    anchors.sort(key=lambda x: x['temporal_position'])
    
    return anchors

def generate_view_recommendations(psi_state: Dict, emotion: Dict) -> List[Dict]:
    """Generate enhanced recommended holographic viewing angles"""
    recommendations = []
    
    # Base viewing parameters
    base_angle = psi_state['psi_phase'] * 180 / np.pi
    coherence = psi_state['phase_coherence']
    velocity = psi_state.get('phase_velocity', 0)
    
    # Primary view - aligned with ψ-phase
    recommendations.append({
        'name': 'primary',
        'azimuth': float(base_angle),
        'elevation': 0.0,
        'distance': 1.0,
        'weight': 1.0,
        'focus': 'center'
    })
    
    # Emotional state-based views
    if emotion['excitement'] > 0.7:
        # High excitement - dynamic orbiting views
        for i in range(3):
            angle_offset = (i + 1) * 120  # 120° apart
            recommendations.append({
                'name': f'excitement_{i}',
                'azimuth': float((base_angle + angle_offset + velocity * 10) % 360),
                'elevation': float(15 * (i + 1) * emotion['excitement']),
                'distance': 0.8 - emotion['energy'] * 0.2,
                'weight': emotion['excitement'] * 0.8,
                'focus': 'dynamic'
            })
    
    elif emotion['calmness'] > 0.7:
        # High calmness - stable, wide views
        recommendations.append({
            'name': 'calm_wide',
            'azimuth': 0.0,
            'elevation': 20.0,
            'distance': 1.5 + emotion['calmness'] * 0.3,
            'weight': emotion['calmness'],
            'focus': 'panoramic'
        })
        
        # Zen view - top down
        recommendations.append({
            'name': 'calm_zen',
            'azimuth': float(base_angle),
            'elevation': 85.0,
            'distance': 2.0,
            'weight': emotion['calmness'] * 0.6,
            'focus': 'meditative'
        })
    
    # Clarity-based detail view
    if emotion['clarity'] > 0.6:
        recommendations.append({
            'name': 'detail',
            'azimuth': float(base_angle + 45),
            'elevation': -10.0,
            'distance': 0.6,
            'weight': emotion['clarity'],
            'focus': 'detail'
        })
    
    # Energy flow view
    if emotion['energy'] > 0.5:
        # Follow the energy flow
        flow_angle = base_angle + np.sign(velocity) * 90
        recommendations.append({
            'name': 'energy_flow',
            'azimuth': float(flow_angle % 360),
            'elevation': float(emotion['energy'] * 30),
            'distance': 1.0 - emotion['energy'] * 0.3,
            'weight': emotion['energy'] * 0.7,
            'focus': 'flow'
        })
    
    return recommendations

def generate_particle_hints(psi_state: Dict, spectral_features: Dict) -> Dict:
    """Generate hints for particle system visualization"""
    coherence = psi_state['phase_coherence']
    emotion = psi_state['emotional_resonance']
    
    # Base particle count on coherence and energy
    base_count = int(coherence * 500 + emotion['energy'] * 500)
    
    # Particle behavior based on emotional state
    particle_config = {
        'count': base_count,
        'emission_rate': float(spectral_features.get('beat_frequency', 1) * 10),
        'lifetime': float(2.0 + emotion['stability'] * 3.0),
        'speed': float(0.5 + emotion['excitement'] * 2.0),
        'spread': float(np.pi * (1 - emotion['clarity'])),
        'size': float(0.05 + emotion['energy'] * 0.1),
        'color_variation': float(1 - emotion['stability']),
        'gravity': float(-emotion['calmness'] * 0.5),
        'turbulence': float(emotion['excitement'] * 0.3),
        'attractor_strength': float(coherence * 0.5)
    }
    
    return particle_config

def compute_spectral_complexity(features: Dict) -> float:
    """Compute overall spectral complexity metric"""
    # Combine multiple features for complexity score
    centroid_norm = np.tanh(features.get('spectral_centroid', 1000) / 2000)
    spread_norm = np.tanh(features.get('spectral_spread', 100) / 500)
    flatness = features.get('spectral_flatness', 0.5)
    flux_norm = np.tanh(features.get('spectral_flux', 0) * 5)
    
    # Weight different aspects
    complexity = (
        centroid_norm * 0.2 +
        spread_norm * 0.3 +
        (1 - flatness) * 0.3 +  # Less flat = more complex
        flux_norm * 0.2
    )
    
    return float(np.clip(complexity, 0, 1))

# Batch processing support
async def batch_transcribe_audio(audio_paths: List[str], 
                                session_id: str = "batch",
                                max_concurrent: int = 4) -> List[Dict]:
    """Process multiple audio files concurrently"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(path: str, idx: int) -> Dict:
        async with semaphore:
            logger.info(f"Processing {idx + 1}/{len(audio_paths)}: {path}")
            try:
                result = await transcribe_audio_async(path, f"{session_id}_{idx}")
                result['batch_index'] = idx
                return result
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                return {
                    'error': str(e),
                    'path': path,
                    'batch_index': idx
                }
    
    tasks = [process_with_semaphore(path, idx) for idx, path in enumerate(audio_paths)]
    results = await asyncio.gather(*tasks)
    
    return results

# Cleanup function
def cleanup_oscillator_pool(session_pattern: str = None):
    """Clean up oscillator instances"""
    global _oscillator_pool
    
    if session_pattern:
        # Remove matching sessions
        keys_to_remove = [k for k in _oscillator_pool.keys() if session_pattern in k]
        for key in keys_to_remove:
            del _oscillator_pool[key]
    else:
        # Clear all
        _oscillator_pool.clear()
    
    logger.info(f"Cleaned up {len(_oscillator_pool)} oscillator sessions")

# Export main functions
__all__ = [
    'transcribe_audio',
    'transcribe_audio_async',
    'batch_transcribe_audio',
    'get_or_create_oscillator',
    'cleanup_oscillator_pool',
    'SpectralKoopmanOscillator',
    'handle'  # Add handle to exports
]

# ---------------------------------------------------------------------------
# Router-style entry-point
# ---------------------------------------------------------------------------

async def handle(file_path: str, mime_type: str = "", **kwargs) -> dict:
    """
    Thin wrapper so pipeline/router can treat audio like any other modality.
    Re-uses the heavy-duty transcribe_audio_async already present.
    """
    from pathlib import Path
    
    # Extract kwargs
    doc_id = kwargs.get("doc_id", "router")
    
    # Call the existing transcribe function
    result_dict = await transcribe_audio_async(file_path, session_id=doc_id)
    
    # Import the result class
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))  # Go up to project root
    from ingest_pdf.pipeline.ingest_common.result import IngestResult
    
    # Extract concepts from the result
    concepts = result_dict.get("psi_state", {}).get("emotional_resonance", {})
    concept_list = []
    
    # Convert emotional resonance to concepts
    for emotion, value in concepts.items():
        if value > 0.5:  # Only include significant emotions
            concept_list.append({
                "name": emotion,
                "confidence": float(value),
                "type": "emotion",
                "source": "audio_analysis"
            })
    
    # Add spectral concepts if significant
    spectral = result_dict.get("spectral_features", {})
    if spectral.get("spectral_centroid", 0) > 2000:
        concept_list.append({
            "name": "high_frequency_content",
            "confidence": 0.8,
            "type": "spectral",
            "source": "audio_analysis"
        })
    
    # Create IngestResult
    return IngestResult(
        filename=Path(file_path).name,
        file_path=file_path,
        media_type="audio",
        transcript=result_dict.get("transcript", ""),
        concepts=concept_list,
        concept_count=len(concept_list),
        concept_names=[c.get("name", "") for c in concept_list],
        psi_state=result_dict.get("psi_state", {}),
        spectral_features=result_dict.get("spectral_features", {}),
        duration_seconds=result_dict.get("metadata", {}).get("duration"),
        sample_rate=result_dict.get("metadata", {}).get("sample_rate"),
        channels=result_dict.get("metadata", {}).get("channels"),
        file_size_bytes=result_dict.get("metadata", {}).get("file_size", 0),
        file_size_mb=result_dict.get("metadata", {}).get("file_size", 0) / (1024 * 1024),
        sha256=result_dict.get("metadata", {}).get("sha256", "unknown")
    ).to_dict()