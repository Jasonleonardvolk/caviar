"""
Phase-to-depth mapping using Fourier-based phase unwrapping.

This module provides functionality to convert a 2D wrapped phase map φ(x,y)
into a continuous depth map z(x,y) using the relation:
    z(x,y) = φ_unwrapped(x,y) * λ / (2π * f0)
where λ is the wavelength and f0 is a reference frequency or optical focal factor.

Features:
 - Uses Fourier/phase unwrapping (via skimage's unwrap_phase if available, or numpy fallback) 
   to unwrap the input phase map.
 - Accepts real input from files (e.g., .npy or image files) and handles scaling if needed.
 - Provides a test harness for verifying the unwrapping on synthetic data.
"""
import numpy as np
import math
import logging
import os
from datetime import datetime
from pathlib import Path

def _log_to_file(message: str, level: str = "INFO"):
    """Write to persistent audit log file."""
    log_dir = Path(os.environ.get("TORI_LOG_DIR", "logs")) / "inference"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "phase_to_depth.log"
    
    timestamp = datetime.utcnow().isoformat()
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp}Z | {level} | {message}\n")
        f.flush()
        os.fsync(f.fileno())

try:
    from skimage.restoration import unwrap_phase
except ImportError:
    unwrap_phase = None
    _log_to_file("IMPORT_WARNING | skimage.restoration.unwrap_phase not available", "WARNING")

def phase_to_depth(phase_map: np.ndarray, wavelength: float, f0: float) -> np.ndarray:
    """
    Convert a wrapped phase map to a depth map using Fourier-based phase unwrapping.

    Parameters:
        phase_map (np.ndarray): 2D array of wrapped phase values (in radians).
        wavelength (float): Wavelength λ (in same units as desired depth, e.g. meters).
        f0 (float): Reference frequency or focal length factor.

    Returns:
        np.ndarray: 2D array of depth values corresponding to the unwrapped phase.
    """
    # Validate input shape
    if phase_map.ndim != 2:
        error_msg = f"phase_map must be 2D, got shape {phase_map.shape}"
        _log_to_file(f"VALIDATION_ERROR | {error_msg}", "ERROR")
        raise ValueError(error_msg)
    
    # Log processing start
    _log_to_file(f"PROCESS_START | shape={phase_map.shape} | wavelength={wavelength} | f0={f0}")
    
    # Ensure input is float
    phase_map = phase_map.astype(np.float64)
    # Unwrap phase. If skimage's unwrap_phase is available (for 2D unwrapping), use it.
    if unwrap_phase:
        logging.info("Using skimage.restoration.unwrap_phase for 2D phase unwrapping.")
        phase_unwrapped = unwrap_phase(phase_map)
        method = "skimage_2d"
    else:
        logging.info("skimage not available, using numpy.unwrap for phase unwrapping.")
        # NumPy's unwrap works along one dimension; apply it twice (rows, then cols) as a simple fallback.
        phase_unwrapped = np.unwrap(phase_map, axis=1)  # unwrap along x-direction (horizontal)
        phase_unwrapped = np.unwrap(phase_unwrapped, axis=0)  # unwrap along y-direction (vertical)
        method = "numpy_1d"
    # Convert phase to depth
    depth_map = (phase_unwrapped * wavelength) / (2 * math.pi * f0)
    
    # Log completion
    depth_range = (float(depth_map.min()), float(depth_map.max()))
    _log_to_file(
        f"PROCESS_COMPLETE | method={method} | depth_range={depth_range} | "
        f"output_shape={depth_map.shape}"
    )
    
    return depth_map

def load_phase_data(path: str) -> np.ndarray:
    """
    Load phase data from a file. Supports NumPy .npy files or image files.
    If an image file is provided, it will be converted to a phase array (assuming grayscale).
    If the image is 8-bit, values are scaled from [0,255] to [0, 2π].

    Parameters:
        path (str): Path to the file containing phase data.
    Returns:
        np.ndarray: 2D array of phase values (in radians).
    """
    _log_to_file(f"LOAD_START | path={path}")
    data = None
    if path.lower().endswith(".npy"):
        data = np.load(path)
        _log_to_file(f"LOAD_NPY | shape={data.shape} | dtype={data.dtype}")
    else:
        try:
            from PIL import Image
            img = Image.open(path)
            # Convert to float grayscale
            img = img.convert("F")  # 32-bit float grayscale
            data = np.array(img, dtype=np.float64)
            _log_to_file(f"LOAD_IMAGE | shape={data.shape}")
        except Exception as e:
            _log_to_file(f"LOAD_ERROR | path={path} | error={str(e)}", "ERROR")
            raise IOError(f"Failed to load phase data from {path}: {e}")
    # If data appears to be in 0-255 integer range, scale to 0-2π
    if np.issubdtype(data.dtype, np.integer) or data.max() > 2 * math.pi:
        logging.info("Scaling phase image from 8-bit to [0, 2π] range.")
        _log_to_file(f"SCALING | original_max={data.max()} | target_range=[0, 2π]")
        data = data.astype(np.float64)
        # Normalize 0-255 to 0-2π
        m = data.max()
        data *= (2 * math.pi / (255 if m > 0 else 1))
    
    _log_to_file(f"LOAD_COMPLETE | final_shape={data.shape} | range=[{data.min():.3f}, {data.max():.3f}]")
    return data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test harness: synthetic phase map that wraps around multiple times.
    width, height = 100, 50
    # Create a synthetic phase pattern that increases linearly and wraps.
    x = np.linspace(0, 4 * math.pi, width)  # 0 to 4π
    base_phase = np.tile(x, (height, 1))
    # Wrap the phase to [0, 2π)
    wrapped_phase = np.mod(base_phase, 2 * math.pi)
    # Choose wavelength and f0 for testing
    wavelength = 0.5e-6  # 0.5 micrometers (500 nm)
    f0 = 1.0
    # Unwrap and convert to depth
    depth = phase_to_depth(wrapped_phase, wavelength, f0)
    # Verify correctness: depth should increase linearly with the unwrapped phase
    print("Sample phase values (wrapped) [0..10]:", wrapped_phase[0, :10])
    print("Sample phase values (unwrapped) [0..10]:", np.round(base_phase[0, :10], 3))
    print("Sample depth values [0..10]:", np.round(depth[0, :10] * 1e6, 3), "micrometers")
    # Simulate loading from a file by saving and loading .npy
    np.save("test_phase.npy", wrapped_phase)
    loaded = load_phase_data("test_phase.npy")
    depth2 = phase_to_depth(loaded, wavelength, f0)
    print("Depth difference between direct and loaded approach:", np.abs(depth2 - depth).max())