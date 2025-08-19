"""
Interactive demo of the BanksyOscillator for audio-driven phase dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import sounddevice as sd
from ingest_bus.audio.spectral_oscillator import BanksyOscillator
from ingest_bus.audio.emotion import compute_spectral_features

class OscillatorVisualizer:
    """Real-time visualization of BanksyOscillator dynamics"""
    
    def __init__(self, n_osc=12, sample_rate=16000):
        self.oscillator = BanksyOscillator(n_osc=n_osc, bandwidth=100.0)
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * 0.1)  # 100ms chunks
        
        # Audio buffer
        self.audio_buffer = np.zeros(self.chunk_size)
        
        # History tracking
        self.coherence_history = []
        self.psi_history = []
        self.centroid_history = []
        self.max_history = 100
        
        # Set up the figure
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.suptitle('ψ-Oscillator Live Visualization', fontsize=16)
        
        # Create subplots
        # 1. Phase circle (oscillator visualization)
        self.ax_phase = plt.subplot(2, 3, 1, projection='polar')
        self.ax_phase.set_title('Oscillator Phases')
        self.ax_phase.set_ylim(0, 1.2)
        
        # 2. Coherence history
        self.ax_coherence = plt.subplot(2, 3, 2)
        self.ax_coherence.set_title('Phase Coherence (r)')
        self.ax_coherence.set_ylim(0, 1.1)
        self.ax_coherence.set_xlim(0, self.max_history)
        
        # 3. Collective phase history
        self.ax_psi = plt.subplot(2, 3, 3)
        self.ax_psi.set_title('Collective Phase (ψ)')
        self.ax_psi.set_ylim(0, 2*np.pi)
        self.ax_psi.set_xlim(0, self.max_history)
        
        # 4. Audio waveform
        self.ax_audio = plt.subplot(2, 3, 4)
        self.ax_audio.set_title('Audio Input')
        self.ax_audio.set_ylim(-1, 1)
        self.ax_audio.set_xlim(0, self.chunk_size)
        
        # 5. Spectral centroid
        self.ax_centroid = plt.subplot(2, 3, 5)
        self.ax_centroid.set_title('Spectral Centroid (Hz)')
        self.ax_centroid.set_ylim(0, 2000)
        self.ax_centroid.set_xlim(0, self.max_history)
        
        # 6. Emotion/coupling strength
        self.ax_emotion = plt.subplot(2, 3, 6)
        self.ax_emotion.set_title('Coupling Strength (K)')
        self.ax_emotion.set_ylim(0, 1.5)
        self.ax_emotion.set_xlim(0, self.max_history)
        
        plt.tight_layout()
        
        # Initialize plot elements
        self.phase_points = []
        self.phase_lines = []
        for i in range(self.oscillator.n):
            # Points for each oscillator
            point, = self.ax_phase.plot([], [], 'o', markersize=10)
            self.phase_points.append(point)
            # Lines from center
            line, = self.ax_phase.plot([], [], '-', alpha=0.3)
            self.phase_lines.append(line)
        
        # Collective phase indicator
        self.psi_arrow, = self.ax_phase.plot([], [], 'r-', linewidth=3)
        
        # History lines
        self.coherence_line, = self.ax_coherence.plot([], [], 'b-')
        self.psi_line, = self.ax_psi.plot([], [], 'r-')
        self.centroid_line, = self.ax_centroid.plot([], [], 'g-')
        self.emotion_line, = self.ax_emotion.plot([], [], 'm-')
        
        # Audio waveform
        self.audio_line, = self.ax_audio.plot([], [], 'k-', alpha=0.7)
        
        # Text displays
        self.coherence_text = self.ax_phase.text(0.02, 0.95, '', transform=self.ax_phase.transAxes)
        self.freq_text = self.ax_audio.text(0.02, 0.95, '', transform=self.ax_audio.transAxes)
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Store audio data
        self.audio_buffer = indata[:, 0].copy()
    
    def update_plot(self, frame):
        """Update all plots"""
        # Process audio
        audio_data = self.audio_buffer.copy()
        
        # Compute spectral features
        spectral = compute_spectral_features(audio_data, self.sample_rate)
        centroid = spectral.get('spectral_centroid', 440)
        rms = spectral.get('rms', 0.01)
        
        # Simple emotion estimation from spectral variance
        emotion_intensity = np.clip(spectral.get('spectral_flux', 0) * 10, 0, 1)
        
        # Update oscillator
        self.oscillator.map_parameters(centroid, emotion_intensity, rms)
        self.oscillator.step(0.1)
        
        # Get state
        state = self.oscillator.psi_state()
        
        # Update histories
        self.coherence_history.append(state['phase_coherence'])
        self.psi_history.append(state['psi_phase'])
        self.centroid_history.append(centroid)
        
        # Limit history length
        if len(self.coherence_history) > self.max_history:
            self.coherence_history.pop(0)
            self.psi_history.pop(0)
            self.centroid_history.pop(0)
        
        # Update phase circle
        phases = np.array(state['oscillator_phases'])
        
        # Color based on coherence
        coherence = state['phase_coherence']
        color = plt.cm.viridis(coherence)
        
        for i, (phase, point, line) in enumerate(zip(phases, self.phase_points, self.phase_lines)):
            # Update oscillator position
            x = [phase]
            y = [0.8]
            point.set_data(x, y)
            point.set_color(color)
            
            # Update line from center
            line.set_data([0, phase], [0, 0.8])
            line.set_color(color)
            line.set_alpha(0.3)
        
        # Update collective phase arrow
        psi = state['psi_phase']
        psi_magnitude = state['psi_magnitude']
        self.psi_arrow.set_data([0, psi], [0, psi_magnitude])
        
        # Update text
        self.coherence_text.set_text(f'r = {coherence:.3f}')
        self.freq_text.set_text(f'f₀ = {centroid:.1f} Hz')
        
        # Update history plots
        x = list(range(len(self.coherence_history)))
        
        self.coherence_line.set_data(x, self.coherence_history)
        self.psi_line.set_data(x, self.psi_history)
        self.centroid_line.set_data(x, self.centroid_history)
        
        # Update emotion/coupling
        coupling_history = [self.oscillator.K] * len(x)  # Current K value
        self.emotion_line.set_data(x, coupling_history[-len(x):])
        
        # Update audio waveform
        x_audio = np.arange(len(audio_data))
        self.audio_line.set_data(x_audio, audio_data)
        
        return (self.phase_points + self.phase_lines + 
                [self.psi_arrow, self.coherence_line, self.psi_line,
                 self.centroid_line, self.emotion_line, self.audio_line,
                 self.coherence_text, self.freq_text])
    
    def run(self):
        """Start the visualization"""
        # Set up audio stream
        stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size
        )
        
        # Set up animation
        ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=100,
            blit=True, cache_frame_data=False
        )
        
        # Start audio stream and show plot
        with stream:
            plt.show()

def test_oscillator_response():
    """Test oscillator response to synthetic signals"""
    oscillator = BanksyOscillator(n_osc=8)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('BanksyOscillator Response to Different Signals')
    
    # Test signals
    duration = 5.0  # seconds
    dt = 0.01
    t = np.arange(0, duration, dt)
    
    # 1. Steady tone (should synchronize)
    print("Testing steady tone...")
    coherence_steady = []
    for _ in t:
        oscillator.map_parameters(440, 0.8, 0.1)
        oscillator.step(dt)
        state = oscillator.psi_state()
        coherence_steady.append(state['phase_coherence'])
    
    axes[0, 0].plot(t, coherence_steady)
    axes[0, 0].set_title('Steady 440Hz Tone')
    axes[0, 0].set_ylabel('Coherence')
    axes[0, 0].set_ylim(0, 1)
    
    # 2. Frequency sweep (should vary)
    print("Testing frequency sweep...")
    oscillator.reset()
    coherence_sweep = []
    freqs = np.linspace(200, 800, len(t))
    
    for freq in freqs:
        oscillator.map_parameters(freq, 0.5, 0.1)
        oscillator.step(dt)
        state = oscillator.psi_state()
        coherence_sweep.append(state['phase_coherence'])
    
    axes[0, 1].plot(t, coherence_sweep)
    axes[0, 1].set_title('Frequency Sweep 200-800Hz')
    axes[0, 1].set_ylabel('Coherence')
    axes[0, 1].set_ylim(0, 1)
    
    # 3. Varying emotion (coupling strength)
    print("Testing emotion variation...")
    oscillator.reset()
    coherence_emotion = []
    emotions = 0.5 + 0.4 * np.sin(2 * np.pi * 0.5 * t)  # Oscillating emotion
    
    for emotion in emotions:
        oscillator.map_parameters(440, emotion, 0.1)
        oscillator.step(dt)
        state = oscillator.psi_state()
        coherence_emotion.append(state['phase_coherence'])
    
    axes[1, 0].plot(t, emotions, 'r-', label='Emotion')
    axes[1, 0].plot(t, coherence_emotion, 'b-', label='Coherence')
    axes[1, 0].set_title('Varying Emotion Intensity')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1)
    
    # 4. Noise (should have low coherence)
    print("Testing noise...")
    oscillator.reset()
    coherence_noise = []
    
    for _ in t:
        # Random frequency and emotion
        freq = np.random.uniform(200, 800)
        emotion = np.random.uniform(0, 1)
        oscillator.map_parameters(freq, emotion, 0.1)
        oscillator.step(dt)
        state = oscillator.psi_state()
        coherence_noise.append(state['phase_coherence'])
    
    axes[1, 1].plot(t, coherence_noise)
    axes[1, 1].set_title('Random Input (Noise)')
    axes[1, 1].set_ylabel('Coherence')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run synthetic tests
        print("Running oscillator response tests...")
        test_oscillator_response()
    else:
        # Run live visualization
        print("Starting live oscillator visualization...")
        print("Speak into your microphone to see the oscillators respond!")
        visualizer = OscillatorVisualizer()
        visualizer.run()
