import numpy as np
from scipy import signal
from typing import Dict, List, Tuple

def compute_spectral_features(data: np.ndarray, rate: int) -> Dict:
    """
    Comprehensive spectral analysis for ψ-oscillator input.
    Enhanced from basic spectral centroid to full feature set.
    """
    # FFT analysis
    fft = np.fft.rfft(data)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(data), 1.0 / rate)
    
    # Avoid division by zero
    total_energy = magnitude.sum()
    if total_energy == 0:
        return _empty_features()
    
    # Core spectral features
    spectral_centroid = (freqs * magnitude).sum() / total_energy
    spectral_spread = np.sqrt(((freqs - spectral_centroid)**2 * magnitude).sum() / total_energy)
    spectral_rolloff = compute_spectral_rolloff(freqs, magnitude, 0.85)
    spectral_flatness = compute_spectral_flatness(magnitude)
    
    # Temporal features
    rms = np.sqrt(np.mean(data**2))
    zero_crossing_rate = compute_zcr(data)
    
    # Enhanced features for ψ-processing
    spectral_flux = compute_spectral_flux(magnitude)
    beat_frequency = estimate_beat_frequency(data, rate)
    harmonic_ratio = compute_harmonic_ratio(freqs, magnitude)
    
    return {
        'spectral_centroid': float(spectral_centroid),
        'spectral_spread': float(spectral_spread),
        'spectral_rolloff': float(spectral_rolloff),
        'spectral_flatness': float(spectral_flatness),
        'rms': float(rms),
        'zero_crossing_rate': float(zero_crossing_rate),
        'spectral_flux': float(spectral_flux),
        'beat_frequency': float(beat_frequency),
        'harmonic_ratio': float(harmonic_ratio),
        'fundamental_frequency': estimate_f0(data, rate)
    }

def compute_spectral_centroid(data: np.ndarray, rate: int) -> float:
    """Legacy function - maintained for backward compatibility"""
    magnitude = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1.0 / rate)
    if magnitude.sum() == 0:
        return 0.0
    return float((freqs * magnitude).sum() / magnitude.sum())

def compute_rms(data: np.ndarray) -> float:
    """Compute root-mean-square of the signal."""
    return float(np.sqrt(np.mean(data.astype(float)**2)))

def compute_spectral_rolloff(freqs: np.ndarray, magnitude: np.ndarray, rolloff_point: float = 0.85) -> float:
    """Frequency below which rolloff_point of energy is contained"""
    cumsum = np.cumsum(magnitude)
    total = cumsum[-1]
    if total == 0:
        return 0.0
    
    rolloff_idx = np.where(cumsum >= rolloff_point * total)[0]
    if len(rolloff_idx) == 0:
        return freqs[-1]
    return freqs[rolloff_idx[0]]

def compute_spectral_flatness(magnitude: np.ndarray) -> float:
    """Spectral flatness (Wiener entropy) - measure of noisiness"""
    # Avoid log(0) by adding small epsilon
    magnitude = magnitude + 1e-10
    geometric_mean = np.exp(np.mean(np.log(magnitude)))
    arithmetic_mean = np.mean(magnitude)
    
    if arithmetic_mean == 0:
        return 0.0
    return geometric_mean / arithmetic_mean

def compute_zcr(data: np.ndarray) -> float:
    """Zero crossing rate - related to noisiness and pitch"""
    signs = np.sign(data)
    sign_changes = np.diff(signs)
    return len(np.where(sign_changes != 0)[0]) / len(data)

def compute_spectral_flux(magnitude: np.ndarray) -> float:
    """Rate of change in spectral content - measures onset strength"""
    # For single frame, return 0 (flux needs temporal comparison)
    # In streaming context, this would compare with previous frame
    return 0.0  # Placeholder - implement with frame buffer in streaming

def estimate_beat_frequency(data: np.ndarray, rate: int) -> float:
    """Estimate beat/tempo frequency using onset detection"""
    # Simple onset detection via amplitude envelope
    window_size = int(rate * 0.1)  # 100ms windows
    hop_size = window_size // 4
    
    onset_strength = []
    for i in range(0, len(data) - window_size, hop_size):
        window = data[i:i + window_size]
        strength = np.sum(np.abs(np.diff(window)))
        onset_strength.append(strength)
    
    if len(onset_strength) < 2:
        return 0.0
    
    # Autocorrelation to find periodicity
    onset_strength = np.array(onset_strength)
    autocorr = np.correlate(onset_strength, onset_strength, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # Find peak (excluding zero lag)
    if len(autocorr) < 2:
        return 0.0
    
    peak_idx = np.argmax(autocorr[1:]) + 1
    beat_period_frames = peak_idx
    beat_period_seconds = beat_period_frames * hop_size / rate
    
    if beat_period_seconds == 0:
        return 0.0
    
    return 1.0 / beat_period_seconds  # Convert to frequency

def compute_harmonic_ratio(freqs: np.ndarray, magnitude: np.ndarray) -> float:
    """Ratio of harmonic to non-harmonic content"""
    # Find fundamental frequency (strongest low-frequency component)
    low_freq_mask = freqs < 800  # Focus on fundamental range
    if not np.any(low_freq_mask):
        return 0.0
    
    f0_idx = np.argmax(magnitude[low_freq_mask])
    f0 = freqs[low_freq_mask][f0_idx]
    
    if f0 == 0:
        return 0.0
    
    # Sum energy at harmonic frequencies
    harmonic_energy = 0
    total_energy = np.sum(magnitude)
    
    for harmonic in range(1, 6):  # Check first 5 harmonics
        harmonic_freq = f0 * harmonic
        # Find closest frequency bin
        freq_diff = np.abs(freqs - harmonic_freq)
        closest_idx = np.argmin(freq_diff)
        
        # Only count if we're close enough (within 10% of harmonic frequency)
        if freq_diff[closest_idx] < harmonic_freq * 0.1:
            harmonic_energy += magnitude[closest_idx]
    
    if total_energy == 0:
        return 0.0
    
    return harmonic_energy / total_energy

def estimate_f0(data: np.ndarray, rate: int) -> float:
    """Estimate fundamental frequency using autocorrelation"""
    # Autocorrelation method for pitch detection
    autocorr = np.correlate(data, data, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # Find peaks in autocorrelation (excluding zero lag)
    min_period = int(rate / 800)  # 800 Hz max (high voice)
    max_period = int(rate / 80)   # 80 Hz min (low voice)
    
    if max_period >= len(autocorr):
        return 0.0
    
    search_range = autocorr[min_period:max_period]
    if len(search_range) == 0:
        return 0.0
    
    peak_idx = np.argmax(search_range) + min_period
    period_samples = peak_idx
    
    if period_samples == 0:
        return 0.0
    
    return rate / period_samples

def _empty_features() -> Dict:
    """Return empty feature dict for silent audio"""
    return {
        'spectral_centroid': 0.0,
        'spectral_spread': 0.0,
        'spectral_rolloff': 0.0,
        'spectral_flatness': 0.0,
        'rms': 0.0,
        'zero_crossing_rate': 0.0,
        'spectral_flux': 0.0,
        'beat_frequency': 0.0,
        'harmonic_ratio': 0.0,
        'fundamental_frequency': 0.0
    }

def detect_emotion(audio_path: str) -> dict:
    """
    Enhanced emotion detection with ψ-resonance mapping.
    """
    import wave
    with wave.open(audio_path, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    
    features = compute_spectral_features(data, rate)
    
    # Enhanced emotion mapping using multiple features
    centroid = features['spectral_centroid']
    spread = features['spectral_spread']
    rms = features['rms']
    harmonic_ratio = features['harmonic_ratio']
    
    # Normalize spectral flux for emotion calculation
    spectral_flux_normalized = features['spectral_flux'] / (rms + 1e-6)
    
    # Multi-dimensional emotion space
    excitement = np.tanh((centroid - 300) / 200) * np.tanh(rms * 10) * (1 + spread / 1000)
    calmness = np.exp(-(centroid - 200)**2 / 50000) * (1 - np.tanh(rms * 5)) * harmonic_ratio
    energy = np.tanh(rms * 8) * (1 + spectral_flux_normalized)
    clarity = harmonic_ratio * np.tanh(centroid / 2000)
    
    # Map to ψ-compatible emotion space
    eps = 1e-6
    if centroid >= 300.0 - eps:
        base_emotion = 'excited'
    elif centroid <= 200.0 + eps:
        base_emotion = 'calm'
    else:
        base_emotion = 'neutral'

    return {
        'spectral_centroid': centroid,
        'rms': rms,
        'emotion': base_emotion,
        'emotional_resonance': {
            'excitement': float(np.clip(excitement, 0, 1)),
            'calmness': float(np.clip(calmness, 0, 1)),
            'energy': float(np.clip(energy, 0, 1)),
            'clarity': float(np.clip(clarity, 0, 1))
        },
        'holographic_readiness': float(np.clip(energy * clarity, 0, 1))
    }

# Additional helper functions for advanced emotion analysis

def compute_emotional_trajectory(features_sequence: List[Dict]) -> Dict:
    """
    Compute emotional trajectory over time for ψ-memory integration
    """
    if not features_sequence:
        return {'trajectory': [], 'trend': 'stable', 'volatility': 0.0}
    
    # Extract emotional dimensions over time
    excitement_series = [f['emotional_resonance']['excitement'] for f in features_sequence]
    calmness_series = [f['emotional_resonance']['calmness'] for f in features_sequence]
    energy_series = [f['emotional_resonance']['energy'] for f in features_sequence]
    
    # Calculate trends
    excitement_trend = np.polyfit(range(len(excitement_series)), excitement_series, 1)[0]
    calmness_trend = np.polyfit(range(len(calmness_series)), calmness_series, 1)[0]
    energy_trend = np.polyfit(range(len(energy_series)), energy_series, 1)[0]
    
    # Calculate volatility (standard deviation of first differences)
    excitement_volatility = np.std(np.diff(excitement_series)) if len(excitement_series) > 1 else 0
    calmness_volatility = np.std(np.diff(calmness_series)) if len(calmness_series) > 1 else 0
    energy_volatility = np.std(np.diff(energy_series)) if len(energy_series) > 1 else 0
    
    overall_volatility = (excitement_volatility + calmness_volatility + energy_volatility) / 3
    
    # Determine dominant trend
    if abs(excitement_trend) > abs(calmness_trend) and excitement_trend > 0.01:
        trend = 'escalating'
    elif abs(calmness_trend) > abs(excitement_trend) and calmness_trend > 0.01:
        trend = 'settling'
    elif energy_trend < -0.01:
        trend = 'declining'
    else:
        trend = 'stable'
    
    return {
        'trajectory': {
            'excitement': excitement_series,
            'calmness': calmness_series,
            'energy': energy_series
        },
        'trends': {
            'excitement': float(excitement_trend),
            'calmness': float(calmness_trend),
            'energy': float(energy_trend)
        },
        'trend': trend,
        'volatility': float(overall_volatility)
    }

def map_emotion_to_psi_phase(emotional_resonance: Dict) -> float:
    """
    Map emotional state to ψ-phase for holographic coordination
    """
    # Create a 2D emotional space and map to phase
    x = emotional_resonance['excitement'] - emotional_resonance['calmness']
    y = emotional_resonance['energy'] - (1 - emotional_resonance['clarity'])
    
    # Convert to phase angle
    phase = np.arctan2(y, x)
    
    # Normalize to [0, 2π]
    if phase < 0:
        phase += 2 * np.pi
    
    return float(phase)

def compute_spectral_entropy(magnitude: np.ndarray) -> float:
    """
    Compute spectral entropy for complexity measurement
    """
    # Normalize magnitude spectrum to probability distribution
    magnitude = magnitude + 1e-10  # Avoid log(0)
    prob = magnitude / np.sum(magnitude)
    
    # Compute Shannon entropy
    entropy = -np.sum(prob * np.log2(prob))
    
    # Normalize by maximum possible entropy
    max_entropy = np.log2(len(magnitude))
    
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0

def detect_voice_quality(features: Dict) -> Dict:
    """
    Detect voice quality characteristics for avatar expression mapping
    """
    # Breathiness detection (high spectral flatness, low harmonic ratio)
    breathiness = features['spectral_flatness'] * (1 - features['harmonic_ratio'])
    
    # Roughness detection (high spectral flux, moderate harmonic ratio)
    roughness = features['spectral_flux'] * features['harmonic_ratio'] if features['spectral_flux'] > 0 else 0
    
    # Strain detection (high spectral centroid, high energy)
    strain = np.tanh((features['spectral_centroid'] - 2000) / 1000) * features['rms']
    
    # Clarity (high harmonic ratio, low spectral spread)
    clarity = features['harmonic_ratio'] * (1 - np.tanh(features['spectral_spread'] / 1000))
    
    return {
        'breathiness': float(np.clip(breathiness, 0, 1)),
        'roughness': float(np.clip(roughness, 0, 1)),
        'strain': float(np.clip(strain, 0, 1)),
        'clarity': float(np.clip(clarity, 0, 1))
    }

# Main enhanced emotion detection function that uses all features
def detect_emotion_enhanced(audio_path: str, include_trajectory: bool = False) -> dict:
    """
    Enhanced emotion detection with all holographic features
    """
    base_result = detect_emotion(audio_path)
    
    # Add voice quality analysis
    import wave
    with wave.open(audio_path, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    
    features = compute_spectral_features(data, rate)
    
    # Add voice quality
    base_result['voice_quality'] = detect_voice_quality(features)
    
    # Add spectral entropy
    fft = np.fft.rfft(data)
    magnitude = np.abs(fft)
    base_result['spectral_entropy'] = compute_spectral_entropy(magnitude)
    
    # Add ψ-phase mapping
    base_result['psi_phase_suggestion'] = map_emotion_to_psi_phase(base_result['emotional_resonance'])
    
    # Add all spectral features for advanced processing
    base_result['spectral_features'] = features
    
    return base_result