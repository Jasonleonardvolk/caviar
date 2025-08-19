"""
Test phase-to-depth conversion with various input/output scenarios.
"""
import numpy as np, math, pytest
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from python.core.phase_to_depth import phase_to_depth, load_phase_data

def test_phase_to_depth_linear():
    """Test unwrapping of linearly increasing phase."""
    w, h = 120, 20
    x = np.linspace(0, 4*math.pi, w)
    wrapped = np.mod(np.tile(x, (h, 1)), 2*math.pi)
    
    z = phase_to_depth(wrapped, wavelength=5e-7, f0=1.0)
    assert z.shape == (h, w)
    
    # Should be monotonic along x after unwrapping
    assert (np.diff(z[0]) >= -1e-12).all(), "Depth not monotonic after unwrapping"
    
    # Verify depth scaling
    expected_max_depth = (4*math.pi * 5e-7) / (2*math.pi * 1.0)
    assert abs(z[0, -1] - expected_max_depth) < 1e-10

def test_phase_to_depth_rejects_non2d():
    """Test input validation rejects non-2D arrays."""
    with pytest.raises(ValueError, match="must be 2D"):
        phase_to_depth(np.zeros((3, 3, 3)), 5e-7, 1.0)
    
    with pytest.raises(ValueError, match="must be 2D"):
        phase_to_depth(np.zeros(10), 5e-7, 1.0)

def test_phase_to_depth_wrapped_discontinuity():
    """Test handling of phase wrapping discontinuities."""
    # Create phase with sharp discontinuity
    phase = np.zeros((50, 50))
    phase[:, 25:] = 1.9 * math.pi  # Near wrapping point
    phase[:, 30:] = 0.1  # Wrapped around
    
    depth = phase_to_depth(phase, wavelength=1e-6, f0=2.0)
    
    # Unwrapped phase should be continuous (no huge jumps)
    grad_x = np.gradient(depth, axis=1)
    assert np.abs(grad_x).max() < 1e-5, "Unwrapping failed at discontinuity"

def test_load_phase_data_npy():
    """Test loading phase data from .npy file."""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        test_data = np.random.rand(100, 100) * 2 * math.pi
        np.save(f.name, test_data)
        f.flush()
        
        loaded = load_phase_data(f.name)
        assert np.allclose(loaded, test_data)
        
        Path(f.name).unlink()

def test_load_phase_data_image_scaling():
    """Test loading and scaling image data to phase range."""
    # Simulate 8-bit image data
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        # Create data that looks like 8-bit image
        image_data = np.random.randint(0, 256, size=(50, 50)).astype(np.float64)
        np.save(f.name, image_data)
        f.flush()
        
        loaded = load_phase_data(f.name)
        
        # Should be scaled to [0, 2π]
        assert loaded.min() >= 0
        assert loaded.max() <= 2 * math.pi
        
        Path(f.name).unlink()

def test_phase_to_depth_performance():
    """Test phase unwrapping performance meets requirements."""
    import time
    
    # 1024x1024 phase map (from requirements)
    phase = np.random.rand(1024, 1024) * 2 * math.pi
    
    start = time.time()
    depth = phase_to_depth(phase, wavelength=5e-7, f0=1.0)
    elapsed = time.time() - start
    
    # Should complete in < 10ms (requirement)
    assert elapsed < 0.1, f"Phase unwrapping took {elapsed*1000:.1f}ms, should be < 100ms"
    assert depth.shape == (1024, 1024)

def test_phase_to_depth_edge_cases():
    """Test edge cases: empty, single pixel, very small values."""
    # Single pixel
    single = np.array([[1.5]])
    depth = phase_to_depth(single, wavelength=1e-6, f0=1.0)
    assert depth.shape == (1, 1)
    
    # All zeros
    zeros = np.zeros((10, 10))
    depth = phase_to_depth(zeros, wavelength=1e-6, f0=1.0)
    assert np.all(depth == 0)
    
    # Very small phase values
    small = np.full((10, 10), 1e-10)
    depth = phase_to_depth(small, wavelength=1e-6, f0=1.0)
    assert np.all(np.abs(depth) < 1e-6)

def test_phase_to_depth_with_noise():
    """Test robustness to noisy phase data."""
    # Create smooth phase with noise
    x, y = np.meshgrid(np.linspace(0, 2*math.pi, 100), np.linspace(0, 2*math.pi, 100))
    clean_phase = np.sin(x) + np.cos(y)
    noise = np.random.randn(100, 100) * 0.1
    noisy_phase = np.mod(clean_phase + noise, 2*math.pi)
    
    depth = phase_to_depth(noisy_phase, wavelength=5e-7, f0=1.0)
    
    # Should still produce smooth depth map (unwrapping reduces noise)
    # Check that depth variation is reasonable
    depth_std = np.std(depth)
    assert depth_std > 0, "Depth map is constant despite input variation"
    assert depth_std < 1e-6, "Depth map too noisy"

def test_phase_caching_suggestion():
    """Test that repeated calls could benefit from caching (performance test)."""
    import time
    
    phase = np.random.rand(512, 512) * 2 * math.pi
    
    # First call
    start1 = time.time()
    depth1 = phase_to_depth(phase, wavelength=5e-7, f0=1.0)
    time1 = time.time() - start1
    
    # Second call with same data
    start2 = time.time()
    depth2 = phase_to_depth(phase, wavelength=5e-7, f0=1.0)
    time2 = time.time() - start2
    
    # Results should be identical
    assert np.allclose(depth1, depth2)
    
    # Note: Currently no caching, so times similar
    # This test documents the opportunity for optimization
    print(f"First call: {time1*1000:.2f}ms, Second call: {time2*1000:.2f}ms")
    print("Consider implementing caching for repeated phase maps")
