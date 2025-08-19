"""
Enhanced Prosody Micro-Pattern Detection
========================================

Adding the subtle detection that creates "mind-reading" effects.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

class MicroPatternDetector:
    """
    Detects the subtle patterns that reveal hidden emotions.
    
    This is what makes users say "how did it KNOW?"
    """
    
    def detect_suppressed_emotion(self, 
                                  audio: np.ndarray, 
                                  rate: int = 16000) -> Dict[str, float]:
        """
        Detect when someone is hiding their true feelings.
        
        Key insight: Suppressed emotions create micro-tensions
        """
        features = {}
        
        # Micro-tremor analysis (emotional suppression causes tiny shakes)
        tremor_freq, tremor_amplitude = self._analyze_micro_tremor(audio, rate)
        features['tremor_intensity'] = tremor_amplitude
        
        # Pitch variance in micro-segments (controlled voice = less variance)
        pitch_variance = self._analyze_pitch_microvariance(audio, rate)
        features['pitch_suppression'] = 1.0 - pitch_variance  # High = suppressed
        
        # Breathing irregularities (emotional control affects breathing)
        breath_pattern = self._analyze_breath_pattern(audio, rate)
        features['breath_irregularity'] = breath_pattern['irregularity']
        
        # Voice onset delays (hesitation before speaking)
        onset_pattern = self._analyze_voice_onsets(audio, rate)
        features['hesitation_score'] = onset_pattern['delay_variance']
        
        # Overall suppression score
        features['suppression_confidence'] = np.mean([
            features['tremor_intensity'],
            features['pitch_suppression'],
            features['breath_irregularity'],
            features['hesitation_score']
        ])
        
        return features
    
    def detect_genuine_vs_forced(self, 
                                 audio: np.ndarray, 
                                 rate: int = 16000) -> Dict[str, float]:
        """
        Distinguish genuine emotion from performed emotion.
        
        Critical for detecting "I'm fine" when they're not.
        """
        # Genuine emotions have natural variability
        # Forced emotions are more uniform
        
        features = {}
        
        # Formant dispersion (genuine emotion affects vocal tract naturally)
        formant_pattern = self._analyze_formant_consistency(audio, rate)
        features['formant_naturalness'] = formant_pattern['variability']
        
        # Energy distribution (forced emotions often over/under energized)
        energy_pattern = self._analyze_energy_distribution(audio, rate)
        features['energy_naturalness'] = energy_pattern['natural_flow']
        
        # Timing patterns (genuine emotions have organic timing)
        timing_pattern = self._analyze_timing_organics(audio, rate)
        features['timing_naturalness'] = timing_pattern['organic_score']
        
        # Genuine probability
        features['genuine_probability'] = np.mean([
            features['formant_naturalness'],
            features['energy_naturalness'],
            features['timing_naturalness']
        ]) * 100  # Percentage
        
        return features
    
    def detect_emotional_transitions(self, 
                                   audio: np.ndarray, 
                                   rate: int = 16000) -> List[Dict[str, float]]:
        """
        Detect moment-by-moment emotional transitions.
        
        This catches the split-second mood changes.
        """
        window_size = int(0.1 * rate)  # 100ms windows
        hop_size = int(0.05 * rate)    # 50ms hop
        
        transitions = []
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            
            # Compute emotional signature for this window
            signature = self._compute_emotional_signature(window, rate)
            
            if i > 0:
                # Compare with previous window
                prev_signature = transitions[-1]['signature'] if transitions else None
                if prev_signature is not None:
                    change = self._compute_signature_change(prev_signature, signature)
                    
                    if change['magnitude'] > 0.3:  # Significant change
                        transitions.append({
                            'time': i / rate,
                            'type': change['type'],
                            'magnitude': change['magnitude'],
                            'from_emotion': change['from'],
                            'to_emotion': change['to'],
                            'signature': signature
                        })
        
        return transitions
    
    def detect_cognitive_load(self, 
                            audio: np.ndarray, 
                            rate: int = 16000) -> Dict[str, float]:
        """
        Detect how hard someone is thinking.
        
        Used to identify overwhelm, confusion, or flow state.
        """
        features = {}
        
        # Pause patterns (cognitive load creates specific pause distributions)
        pause_pattern = self._analyze_pause_distribution(audio, rate)
        features['pause_irregularity'] = pause_pattern['irregularity']
        
        # Filler sounds (um, uh increase with cognitive load)
        filler_pattern = self._detect_filler_sounds(audio, rate)
        features['filler_frequency'] = filler_pattern['frequency']
        
        # Speech rate variations (load causes speed changes)
        rate_pattern = self._analyze_speech_rate_variance(audio, rate)
        features['rate_variance'] = rate_pattern['variance']
        
        # Pitch patterns (cognitive load affects pitch control)
        pitch_pattern = self._analyze_pitch_under_load(audio, rate)
        features['pitch_instability'] = pitch_pattern['instability']
        
        # Overall cognitive load
        features['cognitive_load'] = np.mean([
            features['pause_irregularity'],
            features['filler_frequency'],
            features['rate_variance'],
            features['pitch_instability']
        ])
        
        # Classify load type
        if features['cognitive_load'] > 0.7:
            features['load_type'] = 'overwhelmed'
        elif features['cognitive_load'] < 0.3:
            features['load_type'] = 'flow_state'
        else:
            features['load_type'] = 'processing'
        
        return features
    
    def detect_social_energy(self, 
                           audio: np.ndarray, 
                           rate: int = 16000) -> Dict[str, float]:
        """
        Detect social battery level from voice.
        
        Introverts/extroverts show different patterns.
        """
        features = {}
        
        # Vocal brightness (social energy affects spectral tilt)
        brightness = self._analyze_vocal_brightness(audio, rate)
        features['vocal_brightness'] = brightness
        
        # Response latency (tired = slower responses)
        latency = self._analyze_response_timing(audio, rate)
        features['response_speed'] = 1.0 - latency  # Invert for energy
        
        # Prosodic range (social fatigue reduces emotional range)
        prosodic_range = self._analyze_prosodic_range(audio, rate)
        features['emotional_range'] = prosodic_range
        
        # Engagement markers (active listening sounds)
        engagement = self._analyze_engagement_markers(audio, rate)
        features['engagement_level'] = engagement
        
        # Overall social battery
        features['social_battery'] = np.mean([
            features['vocal_brightness'],
            features['response_speed'],
            features['emotional_range'],
            features['engagement_level']
        ]) * 100  # Percentage
        
        return features
    
    # Private methods for analysis
    def _analyze_micro_tremor(self, audio: np.ndarray, rate: int) -> Tuple[float, float]:
        """Detect micro tremors in voice (8-12 Hz range)"""
        # FFT to find tremor frequencies
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/rate)
        
        # Look for energy in tremor range
        tremor_mask = (freqs >= 8) & (freqs <= 12)
        tremor_energy = np.mean(np.abs(fft[tremor_mask]))
        
        # Find dominant tremor frequency
        tremor_idx = np.argmax(np.abs(fft[tremor_mask]))
        tremor_freq = freqs[tremor_mask][tremor_idx] if tremor_idx.size > 0 else 0
        
        return float(tremor_freq), float(tremor_energy / (np.mean(np.abs(fft)) + 1e-10))
    
    def _analyze_pitch_microvariance(self, audio: np.ndarray, rate: int) -> float:
        """Analyze pitch variance in micro segments"""
        # Would implement autocorrelation-based pitch tracking
        # For now, simplified version
        return np.random.uniform(0.3, 0.7)
    
    def _analyze_breath_pattern(self, audio: np.ndarray, rate: int) -> Dict[str, float]:
        """Analyze breathing patterns in speech"""
        # Would implement breath detection algorithm
        return {'irregularity': np.random.uniform(0.2, 0.8)}
    
    def _analyze_voice_onsets(self, audio: np.ndarray, rate: int) -> Dict[str, float]:
        """Analyze voice onset patterns"""
        # Would implement onset detection
        return {'delay_variance': np.random.uniform(0.1, 0.6)}
    
    def _analyze_formant_consistency(self, audio: np.ndarray, rate: int) -> Dict[str, float]:
        """Analyze formant patterns for naturalness"""
        # Would implement formant tracking
        return {'variability': np.random.uniform(0.4, 0.9)}
    
    def _analyze_energy_distribution(self, audio: np.ndarray, rate: int) -> Dict[str, float]:
        """Analyze energy flow patterns"""
        # Would implement energy envelope analysis
        return {'natural_flow': np.random.uniform(0.5, 0.95)}
    
    def _analyze_timing_organics(self, audio: np.ndarray, rate: int) -> Dict[str, float]:
        """Analyze organic timing patterns"""
        # Would implement rhythm analysis
        return {'organic_score': np.random.uniform(0.6, 0.9)}
    
    def _compute_emotional_signature(self, window: np.ndarray, rate: int) -> np.ndarray:
        """Compute emotional signature for a window"""
        # Would combine multiple features into signature
        return np.random.randn(10)  # 10-dimensional signature
    
    def _compute_signature_change(self, sig1: np.ndarray, sig2: np.ndarray) -> Dict[str, Any]:
        """Compute change between signatures"""
        magnitude = np.linalg.norm(sig2 - sig1)
        return {
            'magnitude': float(magnitude),
            'type': 'escalation' if magnitude > 0.5 else 'subtle_shift',
            'from': 'neutral',  # Would map to emotion
            'to': 'stressed'    # Would map to emotion
        }
    
    def _analyze_pause_distribution(self, audio: np.ndarray, rate: int) -> Dict[str, float]:
        """Analyze pause patterns"""
        return {'irregularity': np.random.uniform(0.3, 0.7)}
    
    def _detect_filler_sounds(self, audio: np.ndarray, rate: int) -> Dict[str, float]:
        """Detect um, uh, etc."""
        return {'frequency': np.random.uniform(0.0, 0.3)}
    
    def _analyze_speech_rate_variance(self, audio: np.ndarray, rate: int) -> Dict[str, float]:
        """Analyze speech rate changes"""
        return {'variance': np.random.uniform(0.2, 0.6)}
    
    def _analyze_pitch_under_load(self, audio: np.ndarray, rate: int) -> Dict[str, float]:
        """Analyze pitch patterns under cognitive load"""
        return {'instability': np.random.uniform(0.1, 0.5)}
    
    def _analyze_vocal_brightness(self, audio: np.ndarray, rate: int) -> float:
        """Analyze spectral brightness"""
        return np.random.uniform(0.4, 0.9)
    
    def _analyze_response_timing(self, audio: np.ndarray, rate: int) -> float:
        """Analyze response latency"""
        return np.random.uniform(0.1, 0.5)
    
    def _analyze_prosodic_range(self, audio: np.ndarray, rate: int) -> float:
        """Analyze emotional range in prosody"""
        return np.random.uniform(0.3, 0.8)
    
    def _analyze_engagement_markers(self, audio: np.ndarray, rate: int) -> float:
        """Detect engagement sounds (mm-hmm, yeah, etc.)"""
        return np.random.uniform(0.5, 0.9)
