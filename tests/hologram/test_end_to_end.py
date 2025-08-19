import pytest
import asyncio
import numpy as np
from ingest_bus.audio.spectral_oscillator_optimized import OptimizedBanksyOrchestrator
from api.routes.hologram_refactored import HologramService

@pytest.mark.asyncio
async def test_audio_to_hologram_pipeline():
    """Test complete pipeline from audio input to hologram generation"""
    # 1) Create a 1 kHz sine wave at 48 kHz for 0.1 s
    sample_rate = 48000
    duration = 0.1
    frequency = 1000
    
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    
    # 2) Run through the oscillator
    config = {
        'sample_rate': sample_rate,
        'num_oscillators': 16,
        'frequency_range': (20, 20000),
        'latency_ms': 10,
        'buffer_size': 1024
    }
    
    orchestrator = OptimizedBanksyOrchestrator(config=config)
    osc_output = await orchestrator.process_async(audio)
    
    # 3) Hand off to hologram service
    hologram_service = HologramService()
    holo = await hologram_service.generate(osc_output)
    
    # 4) Verify wavefield has plausible structure
    assert hasattr(holo, "width") and holo.width > 0, "Hologram width must be positive"
    assert hasattr(holo, "height") and holo.height > 0, "Hologram height must be positive"
    assert hasattr(holo, "data"), "Hologram must have data attribute"
    assert np.all(np.isfinite(holo.data)), "Wavefield contains NaNs or Infs"
    
    # Additional validations
    assert holo.width >= 256, f"Expected minimum width of 256, got {holo.width}"
    assert holo.height >= 256, f"Expected minimum height of 256, got {holo.height}"
    assert holo.data.shape == (holo.height, holo.width), "Data shape mismatch"
    
    # Check data range is reasonable (phase values should be in [-π, π] or normalized [0, 1])
    data_min, data_max = np.min(holo.data), np.max(holo.data)
    assert data_min >= -np.pi - 0.1, f"Data minimum {data_min} is too low"
    assert data_max <= np.pi + 0.1, f"Data maximum {data_max} is too high"
    
    print(f"✓ Hologram generated successfully: {holo.width}x{holo.height}")
    print(f"✓ Data range: [{data_min:.3f}, {data_max:.3f}]")


@pytest.mark.asyncio
async def test_audio_to_hologram_with_different_signals():
    """Test pipeline with various audio signals"""
    sample_rate = 48000
    duration = 0.05  # 50ms for faster testing
    
    test_cases = [
        ("sine_440hz", lambda t: 0.5 * np.sin(2 * np.pi * 440 * t)),
        ("white_noise", lambda t: np.random.randn(len(t)) * 0.1),
        ("square_wave", lambda t: 0.5 * np.sign(np.sin(2 * np.pi * 1000 * t))),
        ("chirp", lambda t: 0.5 * np.sin(2 * np.pi * (100 + 900 * t / duration) * t)),
    ]
    
    config = {
        'sample_rate': sample_rate,
        'num_oscillators': 16,
        'frequency_range': (20, 20000),
        'latency_ms': 10,
        'buffer_size': 512
    }
    
    orchestrator = OptimizedBanksyOrchestrator(config=config)
    hologram_service = HologramService()
    
    for signal_name, signal_func in test_cases:
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        audio = signal_func(t).astype(np.float32)
        
        # Process through pipeline
        osc_output = await orchestrator.process_async(audio)
        holo = await hologram_service.generate(osc_output)
        
        # Basic validation
        assert holo.width > 0 and holo.height > 0, f"{signal_name}: Invalid dimensions"
        assert np.all(np.isfinite(holo.data)), f"{signal_name}: Contains invalid values"
        
        # Check that different signals produce different holograms
        data_std = np.std(holo.data)
        assert data_std > 0.01, f"{signal_name}: Hologram appears to be constant"
        
        print(f"✓ {signal_name}: Generated {holo.width}x{holo.height} hologram, std={data_std:.3f}")


@pytest.mark.asyncio
async def test_oscillator_output_characteristics():
    """Test that oscillator produces expected output characteristics"""
    sample_rate = 48000
    duration = 0.1
    
    # Pure tone to get predictable oscillator response
    frequency = 2000
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    audio = (0.8 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    
    config = {
        'sample_rate': sample_rate,
        'num_oscillators': 16,
        'frequency_range': (20, 20000),
        'latency_ms': 10,
        'buffer_size': 1024
    }
    
    orchestrator = OptimizedBanksyOrchestrator(config=config)
    osc_output = await orchestrator.process_async(audio)
    
    # Verify oscillator output structure
    assert hasattr(osc_output, 'frequencies'), "Output missing frequencies"
    assert hasattr(osc_output, 'amplitudes'), "Output missing amplitudes"
    assert hasattr(osc_output, 'phases'), "Output missing phases"
    
    # Check dimensions match
    assert len(osc_output.frequencies) == config['num_oscillators']
    assert len(osc_output.amplitudes) == config['num_oscillators']
    assert len(osc_output.phases) == config['num_oscillators']
    
    # Verify frequency range
    assert np.all(osc_output.frequencies >= config['frequency_range'][0])
    assert np.all(osc_output.frequencies <= config['frequency_range'][1])
    
    # Check that the dominant frequency is captured
    max_amp_idx = np.argmax(osc_output.amplitudes)
    dominant_freq = osc_output.frequencies[max_amp_idx]
    assert abs(dominant_freq - frequency) < 100, f"Expected ~{frequency}Hz, got {dominant_freq}Hz"
    
    print(f"✓ Oscillator correctly identified dominant frequency: {dominant_freq:.1f}Hz")
    print(f"✓ Amplitude distribution: min={np.min(osc_output.amplitudes):.3f}, max={np.max(osc_output.amplitudes):.3f}")


@pytest.mark.parametrize("buffer_size", [256, 512, 1024, 2048])
@pytest.mark.asyncio
async def test_performance_with_different_buffer_sizes(buffer_size):
    """Test pipeline performance with various buffer sizes"""
    import time
    
    sample_rate = 48000
    duration = 0.1
    
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    
    config = {
        'sample_rate': sample_rate,
        'num_oscillators': 16,
        'frequency_range': (20, 20000),
        'latency_ms': 10,
        'buffer_size': buffer_size
    }
    
    orchestrator = OptimizedBanksyOrchestrator(config=config)
    hologram_service = HologramService()
    
    # Measure processing time
    start_time = time.perf_counter()
    
    osc_output = await orchestrator.process_async(audio)
    holo = await hologram_service.generate(osc_output)
    
    processing_time = time.perf_counter() - start_time
    
    # Performance assertion - should process faster than real-time
    assert processing_time < duration, f"Processing took {processing_time:.3f}s, longer than audio duration {duration}s"
    
    # Calculate processing rate
    processing_rate = duration / processing_time
    print(f"✓ Buffer size {buffer_size}: {processing_time*1000:.1f}ms ({processing_rate:.1f}x real-time)")
    
    # Verify output is still valid
    assert np.all(np.isfinite(holo.data)), f"Buffer size {buffer_size} produced invalid data"


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_audio_to_hologram_pipeline())
    asyncio.run(test_audio_to_hologram_with_different_signals())
    asyncio.run(test_oscillator_output_characteristics())
    
    # Run performance tests
    for buffer_size in [256, 512, 1024, 2048]:
        asyncio.run(test_performance_with_different_buffer_sizes(buffer_size))
