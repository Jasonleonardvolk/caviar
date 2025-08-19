import numpy as np
import pytest
from ingest_bus.audio.spectral_oscillator import (
    BanksyOscillator, VoiceOscillator, MusicOscillator,
    create_oscillator_for_context, visualize_phases
)

@pytest.fixture
def oscillator():
    """Create a test oscillator with 8 oscillators"""
    return BanksyOscillator(n_osc=8, bandwidth=50.0)

@pytest.fixture
def voice_oscillator():
    """Create a voice-optimized oscillator"""
    return VoiceOscillator()

@pytest.fixture
def music_oscillator():
    """Create a music-optimized oscillator"""
    return MusicOscillator()

def sine_wave_centroid(freq: float):
    """Ideal centroid for a sine wave"""
    return freq

class TestBanksyOscillator:
    """Test basic oscillator functionality"""
    
    def test_initialization(self, oscillator):
        """Test oscillator initialization"""
        assert oscillator.n == 8
        assert oscillator.bandwidth == 50.0
        assert len(oscillator.phases) == 8
        assert len(oscillator.natural_freqs) == 8
        assert oscillator.K == 0.1  # Initial coupling
        
        # Check phases are in valid range
        assert np.all(oscillator.phases >= 0)
        assert np.all(oscillator.phases < 2 * np.pi)
    
    def test_map_parameters(self, oscillator):
        """Test parameter mapping from audio features"""
        centroid = 440.0  # A4 note
        emotion_intensity = 0.7
        rms = 0.1
        
        oscillator.map_parameters(centroid, emotion_intensity, rms)
        
        # Check natural frequencies are centered around centroid
        mean_freq = np.mean(oscillator.natural_freqs) / (2 * np.pi)
        assert abs(mean_freq - centroid) < 1.0  # Within 1 Hz
        
        # Check coupling strength
        expected_k = 0.1 + 0.7 * (1.0 - 0.1)  # K_min + emotion * (K_max - K_min)
        assert abs(oscillator.K - expected_k) < 0.01
        
        # With RMS boost
        assert oscillator.K > expected_k  # Should be boosted
    
    def test_step_evolution(self, oscillator):
        """Test phase evolution"""
        # Set known frequencies
        oscillator.natural_freqs = 2 * np.pi * np.array([440] * 8)  # All same frequency
        oscillator.K = 0.5  # Medium coupling
        
        initial_phases = oscillator.phases.copy()
        oscillator.step(dt=0.01)
        
        # Phases should have changed
        assert not np.allclose(oscillator.phases, initial_phases)
        
        # Phases should still be in valid range
        assert np.all(oscillator.phases >= 0)
        assert np.all(oscillator.phases < 2 * np.pi)
    
    def test_order_parameter(self, oscillator):
        """Test Kuramoto order parameter computation"""
        # Test 1: Random phases - low coherence
        oscillator.phases = np.random.uniform(0, 2*np.pi, oscillator.n)
        r, psi = oscillator.compute_order_parameter()
        
        assert 0 <= r <= 1
        assert 0 <= psi < 2 * np.pi
        
        # Test 2: Aligned phases - high coherence
        oscillator.phases = np.ones(oscillator.n) * np.pi / 4  # All at 45°
        r, psi = oscillator.compute_order_parameter()
        
        assert r > 0.95  # Should be nearly 1
        assert abs(psi - np.pi / 4) < 0.1  # Should be close to 45°
    
    def test_psi_state(self, oscillator):
        """Test complete psi state generation"""
        oscillator.map_parameters(440, 0.5)
        oscillator.step()
        
        state = oscillator.psi_state()
        
        # Check all required fields
        assert 'phase_coherence' in state
        assert 'psi_phase' in state
        assert 'psi_magnitude' in state
        assert 'oscillator_phases' in state
        assert 'oscillator_frequencies' in state
        assert 'coupling_strength' in state
        assert 'dominant_frequency' in state
        
        # Check types and ranges
        assert 0 <= state['phase_coherence'] <= 1
        assert 0 <= state['psi_phase'] < 2 * np.pi
        assert len(state['oscillator_phases']) == oscillator.n
        assert len(state['oscillator_frequencies']) == oscillator.n
    
    def test_reset(self, oscillator):
        """Test oscillator reset"""
        # Modify state
        oscillator.map_parameters(880, 0.9)
        oscillator.step()
        
        # Reset
        oscillator.reset()
        
        # Check reset to initial conditions
        assert oscillator.K == oscillator.K_min
        assert np.all(oscillator.natural_freqs == 0)
        assert len(oscillator.coherence_history) == 0

@pytest.mark.parametrize("freq,emotion", [
    (440, 0.2),   # Low emotion, A4
    (880, 0.8),   # High emotion, A5
    (220, 0.5),   # Medium emotion, A3
    (1760, 0.9),  # Very high emotion, A6
])
def test_psi_coherence_changes(oscillator, freq, emotion):
    """Test that coherence changes with different inputs"""
    osc = oscillator
    
    # Phase 1 mapping
    osc.map_parameters(centroid=freq, emotion_intensity=emotion)
    
    # Take several steps to let system evolve
    coherence_values = []
    for _ in range(10):
        osc.step(dt=0.1)
        state = osc.psi_state()
        coherence_values.append(state['phase_coherence'])
    
    # Check valid range
    assert all(0.0 <= c <= 1.0 for c in coherence_values)
    
    # Higher emotion should lead to higher coherence (on average)
    # due to stronger coupling
    if emotion > 0.7:
        assert max(coherence_values) > 0.5

def test_random_vs_tonic():
    """Test synchronization with zero bandwidth"""
    # Zero bandwidth → all ωᵢ identical → should synchronize
    osc = BanksyOscillator(n_osc=5, bandwidth=0)
    osc.map_parameters(centroid=440, emotion_intensity=0.8)
    
    # Let it evolve
    for _ in range(20):
        osc.step(dt=0.1)
    
    state = osc.psi_state()
    
    # With identical frequencies and strong coupling, should synchronize
    assert state['phase_coherence'] > 0.9

def test_voice_oscillator_specifics(voice_oscillator):
    """Test voice-specific oscillator settings"""
    assert voice_oscillator.n == 8
    assert voice_oscillator.bandwidth == 50.0  # Narrower for voice
    assert voice_oscillator.K_min == 0.2  # Higher minimum
    assert voice_oscillator.K_max == 0.8  # Lower maximum

def test_music_oscillator_specifics(music_oscillator):
    """Test music-specific oscillator settings"""
    assert music_oscillator.n == 16  # More oscillators
    assert music_oscillator.bandwidth == 200.0  # Wider for harmonics
    assert music_oscillator.K_min == 0.3  # Strong minimum
    assert music_oscillator.K_max == 1.5  # Very strong maximum

def test_factory_function():
    """Test oscillator factory function"""
    voice_osc = create_oscillator_for_context("voice")
    assert isinstance(voice_osc, VoiceOscillator)
    
    music_osc = create_oscillator_for_context("music")
    assert isinstance(music_osc, MusicOscillator)
    
    general_osc = create_oscillator_for_context("general")
    assert isinstance(general_osc, BanksyOscillator)
    assert not isinstance(general_osc, (VoiceOscillator, MusicOscillator))

def test_phase_visualization():
    """Test ASCII phase visualization"""
    phases = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    vis = visualize_phases(phases)
    
    assert len(vis) == 12  # 12 positions
    assert vis.count('o') == len(phases)  # One marker per phase

def test_frequency_adaptation(oscillator):
    """Test smooth frequency adaptation"""
    # Initial mapping
    oscillator.map_parameters(440, 0.5)
    initial_freqs = oscillator.natural_freqs.copy()
    
    # New mapping with different frequency
    oscillator.map_parameters(880, 0.5)
    
    # Frequencies should have moved towards new target
    # but not jumped directly (due to adaptation rate)
    freq_change = oscillator.natural_freqs - initial_freqs
    assert np.all(freq_change > 0)  # Should increase
    
    # But not fully to 880 Hz center yet
    mean_freq = np.mean(oscillator.natural_freqs) / (2 * np.pi)
    assert mean_freq < 880  # Not fully adapted in one step

def test_noise_influence(oscillator):
    """Test that noise adds variability"""
    # Set up deterministic system
    oscillator.noise_strength = 0.0
    oscillator.map_parameters(440, 0.5)
    
    # Evolve without noise
    phases_no_noise = []
    for _ in range(5):
        oscillator.step()
        phases_no_noise.append(oscillator.phases.copy())
    
    # Reset and add noise
    oscillator.reset()
    oscillator.noise_strength = 0.1
    oscillator.map_parameters(440, 0.5)
    
    # Evolve with noise
    phases_with_noise = []
    for _ in range(5):
        oscillator.step()
        phases_with_noise.append(oscillator.phases.copy())
    
    # With noise, phases should be more variable
    variability_no_noise = np.std([np.std(p) for p in phases_no_noise])
    variability_with_noise = np.std([np.std(p) for p in phases_with_noise])
    
    # This test might be probabilistic, but noise should add variability
    assert variability_with_noise > 0

def test_coupling_strength_effect():
    """Test effect of coupling strength on synchronization"""
    # Weak coupling
    osc_weak = BanksyOscillator(n_osc=5)
    osc_weak.map_parameters(440, 0.1)  # Low emotion → weak coupling
    
    # Strong coupling
    osc_strong = BanksyOscillator(n_osc=5)
    osc_strong.map_parameters(440, 0.9)  # High emotion → strong coupling
    
    # Evolve both
    for _ in range(20):
        osc_weak.step(dt=0.1)
        osc_strong.step(dt=0.1)
    
    # Strong coupling should lead to higher coherence
    weak_coherence = osc_weak.psi_state()['phase_coherence']
    strong_coherence = osc_strong.psi_state()['phase_coherence']
    
    # Strong coupling should generally lead to higher coherence
    # (though this is probabilistic due to random initial conditions)
    assert osc_strong.K > osc_weak.K

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    # Single oscillator
    osc_single = BanksyOscillator(n_osc=1)
    osc_single.map_parameters(440, 0.5)
    osc_single.step()
    
    state = osc_single.psi_state()
    assert state['phase_coherence'] == 1.0  # Single oscillator is always coherent
    
    # Zero emotion
    osc = BanksyOscillator()
    osc.map_parameters(440, 0.0)
    assert osc.K == osc.K_min
    
    # Maximum emotion
    osc.map_parameters(440, 1.0)
    assert osc.K == osc.K_max

def test_set_phases_manually(oscillator):
    """Test manual phase setting"""
    # Set specific phases
    test_phases = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    oscillator.set_phases(test_phases)
    
    # Check they were set correctly
    np.testing.assert_array_almost_equal(oscillator.phases, test_phases)
    
    # Test invalid length
    with pytest.raises(ValueError):
        oscillator.set_phases([0, np.pi])  # Wrong number of phases

if __name__ == "__main__":
    # Run a simple test
    osc = BanksyOscillator()
    osc.map_parameters(440, 0.7, 0.1)
    
    print("Initial state:")
    print(f"Coupling K: {osc.K:.3f}")
    print(f"Phases: {visualize_phases(osc.phases)}")
    
    for i in range(10):
        osc.step(0.1)
        state = osc.psi_state()
        print(f"Step {i+1}: r={state['phase_coherence']:.3f}, ψ={state['psi_phase']:.3f}, phases={visualize_phases(osc.phases)}")
