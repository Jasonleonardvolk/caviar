"""
AMPLITUDE LENSING AMPLIFIER
Focuses and amplifies psi-amplitude fields through geometric lensing

This module implements gravitational lensing analogs for memory amplification,
creating focused signal pathways through curvature-driven enhancement.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import convolve2d
from scipy.interpolate import RegularGridInterpolator

# JIT support
try:
    import numba
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    njit = jit
    prange = range

logger = logging.getLogger("vision.amplitude_lens")


@dataclass
class LensingProfile:
    """Defines the lensing properties at a point"""
    magnification: float  # Amplification factor
    shear: np.ndarray  # Shear tensor (2x2)
    convergence: float  # Convergence parameter
    caustic_distance: float  # Distance to nearest caustic


@dataclass
class AmplitudeFocus:
    """Represents a focused amplitude region"""
    center: np.ndarray  # Focus center coordinates
    strength: float  # Focus strength
    extent: float  # Spatial extent
    profile: LensingProfile
    amplified_field: Optional[np.ndarray] = None


class AmplitudeLensing:
    """
    Implements gravitational lensing analogs for amplitude enhancement
    
    Key concepts:
    - Strong lensing near singularities creates caustics
    - Weak lensing provides distributed amplification
    - Multiple images form at critical curves
    """
    
    def __init__(self, resolution: int = 100):
        self.resolution = resolution
        self.lensing_planes: List[np.ndarray] = []
        self.caustic_curves: List[np.ndarray] = []
        self.critical_curves: List[np.ndarray] = []
        self.focus_points: List[AmplitudeFocus] = []
        
        logger.info("Amplitude Lensing system initialized")
    
    @staticmethod
    @njit(cache=True, parallel=True)
    def _compute_deflection_field_jit(mass_distribution: np.ndarray, 
                                     scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        JIT-compiled deflection angle computation
        
        Based on gravitational lensing equation:
        alpha = (4GM/c²) * integral(d²x' * Σ(x') * (x-x')/|x-x'|²)
        """
        rows, cols = mass_distribution.shape
        alpha_x = np.zeros((rows, cols))
        alpha_y = np.zeros((rows, cols))
        
        # Compute deflection angles
        for i in prange(rows):
            for j in range(cols):
                # Skip if no mass
                if mass_distribution[i, j] < 1e-10:
                    continue
                
                # Compute deflection contribution
                for ii in range(rows):
                    for jj in range(cols):
                        if ii == i and jj == j:
                            continue
                        
                        dx = (ii - i) * scale
                        dy = (jj - j) * scale
                        r2 = dx*dx + dy*dy
                        
                        if r2 > 0:
                            # Deflection proportional to mass/r
                            factor = mass_distribution[ii, jj] / r2
                            alpha_x[i, j] += factor * dx
                            alpha_y[i, j] += factor * dy
        
        # Normalize
        alpha_x *= 4.0 * scale  # Include constants
        alpha_y *= 4.0 * scale
        
        return alpha_x, alpha_y
    
    @staticmethod
    @njit(cache=True)
    def _compute_magnification_jit(convergence: np.ndarray, 
                                  shear1: np.ndarray,
                                  shear2: np.ndarray) -> np.ndarray:
        """
        JIT-compiled magnification computation
        
        Magnification = 1 / |det(A)|
        where A = [[1-κ-γ₁, -γ₂], [-γ₂, 1-κ+γ₁]]
        """
        rows, cols = convergence.shape
        magnification = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                kappa = convergence[i, j]
                gamma1 = shear1[i, j]
                gamma2 = shear2[i, j]
                
                # Compute determinant of amplification matrix
                det_A = (1 - kappa)**2 - gamma1**2 - gamma2**2
                
                # Magnification is inverse of determinant
                if abs(det_A) > 1e-10:
                    magnification[i, j] = 1.0 / abs(det_A)
                else:
                    # Near caustic - very high magnification
                    magnification[i, j] = 1000.0
        
        return magnification
    
    def create_mass_lens(self, 
                        curvature_field: np.ndarray,
                        einstein_radius: float = 5.0) -> np.ndarray:
        """
        Convert curvature field to effective mass distribution for lensing
        
        Args:
            curvature_field: Kretschmann scalar or similar
            einstein_radius: Characteristic lensing scale
            
        Returns:
            Mass distribution for lensing calculations
        """
        # Normalize curvature to mass density
        # Higher curvature = more "mass" = stronger lensing
        mass_field = np.abs(curvature_field)
        
        # Apply Einstein radius scaling
        mass_field = mass_field * (einstein_radius / np.max(mass_field + 1e-10))
        
        # Smooth to avoid numerical issues
        mass_field = gaussian_filter(mass_field, sigma=1.0)
        
        return mass_field
    
    def compute_lensing_potential(self, mass_distribution: np.ndarray) -> np.ndarray:
        """
        Compute the lensing potential psi
        
        The potential satisfies: ∇²ψ = 2κ (convergence)
        """
        # Solve Poisson equation for potential
        # Using FFT method for efficiency
        rows, cols = mass_distribution.shape
        
        # Fourier transform
        mass_fft = np.fft.fft2(mass_distribution)
        
        # Create k-space grid
        kx = np.fft.fftfreq(cols, d=1.0).reshape(1, -1)
        ky = np.fft.fftfreq(rows, d=1.0).reshape(-1, 1)
        k2 = kx**2 + ky**2
        
        # Avoid division by zero
        k2[0, 0] = 1.0
        
        # Solve in Fourier space: psi_k = -2 * mass_k / k²
        psi_fft = -2.0 * mass_fft / k2
        psi_fft[0, 0] = 0  # Set DC component to zero
        
        # Inverse transform
        potential = np.real(np.fft.ifft2(psi_fft))
        
        return potential
    
    def compute_lensing_fields(self, 
                              mass_distribution: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute all lensing quantities from mass distribution
        
        Returns:
            Dictionary with:
            - deflection: Deflection angle field
            - convergence: Convergence (κ)
            - shear: Shear components (γ₁, γ₂)
            - magnification: Magnification factor
        """
        # Compute potential
        potential = self.compute_lensing_potential(mass_distribution)
        
        # Compute derivatives
        psi_y, psi_x = np.gradient(potential)
        psi_yy, psi_yx = np.gradient(psi_y)
        psi_xy, psi_xx = np.gradient(psi_x)
        
        # Convergence κ = 1/2 * (∂²ψ/∂x² + ∂²ψ/∂y²)
        convergence = 0.5 * (psi_xx + psi_yy)
        
        # Shear components
        shear1 = 0.5 * (psi_xx - psi_yy)  # γ₁
        shear2 = psi_xy  # γ₂
        
        # Deflection angles
        if NUMBA_AVAILABLE:
            alpha_x, alpha_y = self._compute_deflection_field_jit(mass_distribution)
        else:
            # Simple gradient approximation
            alpha_y, alpha_x = np.gradient(potential)
            alpha_x *= -1
            alpha_y *= -1
        
        # Magnification
        if NUMBA_AVAILABLE:
            magnification = self._compute_magnification_jit(convergence, shear1, shear2)
        else:
            # Approximate magnification
            det_A = (1 - convergence)**2 - shear1**2 - shear2**2
            magnification = np.where(np.abs(det_A) > 1e-10, 
                                   1.0 / np.abs(det_A), 
                                   1000.0)
        
        return {
            'deflection_x': alpha_x,
            'deflection_y': alpha_y,
            'convergence': convergence,
            'shear1': shear1,
            'shear2': shear2,
            'magnification': magnification,
            'potential': potential
        }
    
    def apply_lensing_to_amplitude(self,
                                  amplitude_field: np.ndarray,
                                  curvature_field: np.ndarray,
                                  einstein_radius: float = 5.0,
                                  max_magnification: float = 100.0) -> np.ndarray:
        """
        Apply gravitational lensing effects to amplitude field
        
        Args:
            amplitude_field: Input psi-amplitude
            curvature_field: Spacetime curvature (e.g., Kretschmann)
            einstein_radius: Characteristic lensing scale
            max_magnification: Cap on magnification to avoid singularities
            
        Returns:
            Lensed amplitude field
        """
        logger.info("Applying gravitational lensing to amplitude field")
        
        # Ensure 2D
        if amplitude_field.ndim == 1:
            size = int(np.sqrt(len(amplitude_field)))
            amplitude_field = amplitude_field.reshape((size, size))
        
        if curvature_field.ndim == 1:
            size = int(np.sqrt(len(curvature_field)))
            curvature_field = curvature_field.reshape((size, size))
        
        # Create mass distribution from curvature
        mass_dist = self.create_mass_lens(curvature_field, einstein_radius)
        
        # Compute lensing fields
        lensing = self.compute_lensing_fields(mass_dist)
        
        # Apply magnification to amplitude
        magnification = lensing['magnification']
        
        # Cap magnification
        magnification = np.clip(magnification, 0.1, max_magnification)
        
        # Amplify the field
        lensed_amplitude = amplitude_field * magnification
        
        # Ray-trace for strong lensing effects
        lensed_amplitude = self._apply_ray_tracing(
            lensed_amplitude,
            lensing['deflection_x'],
            lensing['deflection_y']
        )
        
        # Find and mark caustics
        self._detect_caustics(lensing)
        
        logger.info(f"Lensing applied: max magnification = {np.max(magnification):.2f}")
        
        return lensed_amplitude
    
    def _apply_ray_tracing(self,
                          field: np.ndarray,
                          deflection_x: np.ndarray,
                          deflection_y: np.ndarray) -> np.ndarray:
        """
        Apply ray-tracing to account for deflection
        
        Maps source plane to image plane via lens equation:
        β = θ - α(θ)
        """
        rows, cols = field.shape
        
        # Create coordinate grids
        y, x = np.mgrid[0:rows, 0:cols]
        
        # Apply lens equation (backward ray-tracing)
        source_x = x - deflection_x
        source_y = y - deflection_y
        
        # Ensure within bounds
        source_x = np.clip(source_x, 0, cols-1)
        source_y = np.clip(source_y, 0, rows-1)
        
        # Interpolate from source plane
        coords = np.array([np.arange(rows), np.arange(cols)])
        interpolator = RegularGridInterpolator(coords, field, 
                                             bounds_error=False, 
                                             fill_value=0)
        
        # Sample at deflected positions
        positions = np.column_stack([source_y.ravel(), source_x.ravel()])
        lensed = interpolator(positions).reshape(field.shape)
        
        return lensed
    
    def _detect_caustics(self, lensing_fields: Dict[str, np.ndarray]):
        """
        Detect caustic curves where magnification diverges
        
        Caustics occur where det(A) = 0
        """
        magnification = lensing_fields['magnification']
        
        # High magnification indicates proximity to caustic
        caustic_mask = magnification > 50.0
        
        # Find contours
        from scipy import ndimage
        labeled, num_features = ndimage.label(caustic_mask)
        
        # Store caustic curves
        self.caustic_curves = []
        for i in range(1, num_features + 1):
            points = np.argwhere(labeled == i)
            if len(points) > 5:  # Significant caustic
                self.caustic_curves.append(points)
        
        logger.info(f"Detected {len(self.caustic_curves)} caustic curves")
    
    def create_focused_beam(self,
                           center: Tuple[float, float],
                           width: float = 10.0,
                           amplitude: float = 1.0,
                           shape: str = "gaussian") -> AmplitudeFocus:
        """
        Create a focused amplitude beam
        
        Args:
            center: Beam center coordinates
            width: Beam width
            amplitude: Peak amplitude
            shape: Beam profile ("gaussian", "airy", "bessel")
            
        Returns:
            AmplitudeFocus object
        """
        center_arr = np.array(center)
        
        # Create beam profile
        if shape == "gaussian":
            profile = self._gaussian_beam(center_arr, width, amplitude)
        elif shape == "airy":
            profile = self._airy_beam(center_arr, width, amplitude)
        elif shape == "bessel":
            profile = self._bessel_beam(center_arr, width, amplitude)
        else:
            raise ValueError(f"Unknown beam shape: {shape}")
        
        # Create lensing profile
        lens_profile = LensingProfile(
            magnification=amplitude,
            shear=np.zeros((2, 2)),
            convergence=0.0,
            caustic_distance=np.inf
        )
        
        focus = AmplitudeFocus(
            center=center_arr,
            strength=amplitude,
            extent=width,
            profile=lens_profile,
            amplified_field=profile
        )
        
        self.focus_points.append(focus)
        
        return focus
    
    def _gaussian_beam(self, center: np.ndarray, width: float, amplitude: float) -> np.ndarray:
        """Create Gaussian beam profile"""
        y, x = np.mgrid[0:self.resolution, 0:self.resolution]
        
        r2 = (x - center[0])**2 + (y - center[1])**2
        beam = amplitude * np.exp(-r2 / (2 * width**2))
        
        return beam
    
    def _airy_beam(self, center: np.ndarray, width: float, amplitude: float) -> np.ndarray:
        """Create Airy disk beam profile"""
        from scipy.special import j1
        
        y, x = np.mgrid[0:self.resolution, 0:self.resolution]
        
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r_scaled = r / width
        
        # Airy function: (2 * J1(x) / x)²
        with np.errstate(divide='ignore', invalid='ignore'):
            airy = np.where(r_scaled > 0,
                          (2 * j1(r_scaled) / r_scaled)**2,
                          1.0)
        
        beam = amplitude * airy
        
        return beam
    
    def _bessel_beam(self, center: np.ndarray, width: float, amplitude: float) -> np.ndarray:
        """Create Bessel beam profile (non-diffracting)"""
        from scipy.special import j0
        
        y, x = np.mgrid[0:self.resolution, 0:self.resolution]
        
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r_scaled = r / width
        
        # Bessel function of first kind
        beam = amplitude * j0(r_scaled)**2
        
        return beam
    
    def combine_multiple_lenses(self,
                               lens_positions: List[Tuple[float, float]],
                               lens_strengths: List[float]) -> np.ndarray:
        """
        Combine multiple point lenses into effective potential
        
        Used for creating complex lensing patterns
        """
        potential = np.zeros((self.resolution, self.resolution))
        
        for pos, strength in zip(lens_positions, lens_strengths):
            y, x = np.mgrid[0:self.resolution, 0:self.resolution]
            
            r = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            
            # Point lens potential: ψ = strength * log(r)
            with np.errstate(divide='ignore'):
                lens_potential = strength * np.log(r + 1e-10)
            
            potential += lens_potential
        
        return potential
    
    def adaptive_focus(self,
                      amplitude_field: np.ndarray,
                      target_snr: float = 10.0) -> np.ndarray:
        """
        Adaptively focus amplitude to achieve target signal-to-noise ratio
        
        Args:
            amplitude_field: Input amplitude
            target_snr: Desired SNR
            
        Returns:
            Optimally focused amplitude field
        """
        # Estimate noise level
        noise_level = np.std(amplitude_field[amplitude_field < np.percentile(amplitude_field, 10)])
        
        # Find signal regions
        signal_mask = amplitude_field > 3 * noise_level
        
        # Current SNR
        if noise_level > 0:
            current_snr = np.mean(amplitude_field[signal_mask]) / noise_level
        else:
            current_snr = np.inf
        
        # Required amplification
        if current_snr < target_snr and current_snr > 0:
            required_amp = target_snr / current_snr
            
            # Apply adaptive amplification
            focused = amplitude_field.copy()
            focused[signal_mask] *= required_amp
            
            logger.info(f"Applied adaptive focus: amplification = {required_amp:.2f}")
        else:
            focused = amplitude_field
        
        return focused
    
    def export_lensing_data(self, filename: str):
        """Export lensing configuration and results"""
        export_data = {
            'resolution': self.resolution,
            'focus_points': [
                {
                    'center': focus.center.tolist(),
                    'strength': focus.strength,
                    'extent': focus.extent
                }
                for focus in self.focus_points
            ],
            'caustic_curves': [
                curve.tolist() for curve in self.caustic_curves
            ],
            'num_lensing_planes': len(self.lensing_planes)
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported lensing data to {filename}")


# Convenience functions
def amplify_with_lensing(amplitude: Union[np.ndarray, float],
                        curvature: Union[np.ndarray, float],
                        einstein_radius: float = 5.0) -> np.ndarray:
    """
    Quick function to apply lensing amplification
    
    Args:
        amplitude: Input amplitude field or scalar
        curvature: Curvature field or scalar
        einstein_radius: Lensing scale
        
    Returns:
        Lensed amplitude
    """
    # Convert scalars to arrays if needed
    if isinstance(amplitude, (int, float)):
        amplitude = np.full((50, 50), amplitude)
    if isinstance(curvature, (int, float)):
        curvature = np.full((50, 50), curvature)
    
    lensing = AmplitudeLensing(resolution=amplitude.shape[0])
    return lensing.apply_lensing_to_amplitude(amplitude, curvature, einstein_radius)


def create_caustic_network(num_lenses: int = 5,
                          field_size: int = 100) -> np.ndarray:
    """
    Create a network of caustics from multiple lenses
    
    Returns:
        Magnification field with caustic network
    """
    # Random lens positions
    positions = [
        (np.random.rand() * field_size, np.random.rand() * field_size)
        for _ in range(num_lenses)
    ]
    
    # Random strengths
    strengths = np.random.uniform(0.5, 2.0, num_lenses)
    
    lensing = AmplitudeLensing(resolution=field_size)
    potential = lensing.combine_multiple_lenses(positions, strengths.tolist())
    
    # Create uniform mass and compute lensing
    mass = np.ones((field_size, field_size))
    fields = lensing.compute_lensing_fields(mass)
    
    return fields['magnification']


# Example usage
if __name__ == "__main__":
    # Create amplitude field
    amplitude = np.random.rand(100, 100) * 0.1 + 0.5
    
    # Create curvature field (e.g., from black hole)
    y, x = np.mgrid[0:100, 0:100]
    r = np.sqrt((x - 50)**2 + (y - 50)**2) + 1e-6
    curvature = 48.0 / r**6  # Schwarzschild
    
    # Apply lensing
    lensing = AmplitudeLensing(resolution=100)
    lensed = lensing.apply_lensing_to_amplitude(amplitude, curvature)
    
    print(f"Original amplitude range: [{np.min(amplitude):.3f}, {np.max(amplitude):.3f}]")
    print(f"Lensed amplitude range: [{np.min(lensed):.3f}, {np.max(lensed):.3f}]")
    print(f"Maximum amplification: {np.max(lensed) / np.max(amplitude):.2f}x")
    
    # Create focused beam
    focus = lensing.create_focused_beam(center=(50, 50), width=15.0, amplitude=5.0)
    print(f"Created focused beam at {focus.center}")
