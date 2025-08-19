#!/usr/bin/env python3
"""
Nimble FDTD Dark-Soliton Engine
Supports 128x128 lattice with minimal post-processing
Uses NumPy + Numba for performance
"""

import numpy as np
import numba
from numba import jit, prange
import yaml
from typing import Dict, Any, Tuple, Optional
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# FDTD parameters
DEFAULT_DT = 0.01
DEFAULT_DX = 1.0
STABILITY_FACTOR = 0.5  # Courant condition

@jit(nopython=True, parallel=True, cache=True)
def fdtd_step_dark_soliton(
    u_real: np.ndarray,
    u_imag: np.ndarray, 
    laplacian_real: np.ndarray,
    laplacian_imag: np.ndarray,
    dt: float,
    dx: float,
    nonlinearity: float,
    dispersion: float,
    damping: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single FDTD step for dark soliton propagation
    Solves: i∂ψ/∂t + ∇²ψ + |ψ|²ψ = 0
    
    Uses split-step method with Numba acceleration
    """
    nx, ny = u_real.shape
    
    # Compute Laplacian using finite differences
    for i in prange(1, nx-1):
        for j in range(1, ny-1):
            # 5-point stencil
            laplacian_real[i,j] = (
                u_real[i+1,j] + u_real[i-1,j] + 
                u_real[i,j+1] + u_real[i,j-1] - 
                4*u_real[i,j]
            ) / (dx * dx)
            
            laplacian_imag[i,j] = (
                u_imag[i+1,j] + u_imag[i-1,j] + 
                u_imag[i,j+1] + u_imag[i,j-1] - 
                4*u_imag[i,j]
            ) / (dx * dx)
    
    # Linear step (dispersion)
    u_real_new = u_real.copy()
    u_imag_new = u_imag.copy()
    
    for i in prange(nx):
        for j in range(ny):
            # Time evolution: i∂ψ/∂t = -∇²ψ
            u_real_new[i,j] = u_real[i,j] - dt * dispersion * laplacian_imag[i,j]
            u_imag_new[i,j] = u_imag[i,j] + dt * dispersion * laplacian_real[i,j]
    
    # Nonlinear step
    for i in prange(nx):
        for j in range(ny):
            amplitude_sq = u_real_new[i,j]**2 + u_imag_new[i,j]**2
            
            # Nonlinear phase rotation: exp(-i|ψ|²dt)
            phase = -nonlinearity * amplitude_sq * dt
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
            
            temp_real = u_real_new[i,j] * cos_phase - u_imag_new[i,j] * sin_phase
            temp_imag = u_real_new[i,j] * sin_phase + u_imag_new[i,j] * cos_phase
            
            # Apply damping at boundaries
            if i < 5 or i >= nx-5 or j < 5 or j >= ny-5:
                temp_real *= (1.0 - damping * dt)
                temp_imag *= (1.0 - damping * dt)
            
            u_real_new[i,j] = temp_real
            u_imag_new[i,j] = temp_imag
    
    return u_real_new, u_imag_new

class DarkSolitonSimulator:
    """
    Dark Soliton FDTD Simulator
    Manages lattice evolution with minimal overhead
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lattice_size = config.get('lattice_size', 128)
        self.dt = config.get('dt', DEFAULT_DT)
        self.dx = config.get('dx', DEFAULT_DX)
        
        # Physics parameters
        self.nonlinearity = config.get('nonlinearity', 1.0)
        self.dispersion = config.get('dispersion', 0.5)
        self.damping = config.get('damping', 0.1)
        
        # Initialize lattice
        self.u_real = np.ones((self.lattice_size, self.lattice_size), dtype=np.float64)
        self.u_imag = np.zeros((self.lattice_size, self.lattice_size), dtype=np.float64)
        
        # Work arrays
        self.laplacian_real = np.zeros_like(self.u_real)
        self.laplacian_imag = np.zeros_like(self.u_imag)
        
        # Metrics
        self.step_count = 0
        self.total_phase_drift = 0.0
        
        # Check stability
        self._check_stability()
        
    def _check_stability(self):
        """Check Courant-Friedrichs-Lewy condition"""
        max_velocity = 2 * self.dispersion / self.dx
        cfl = max_velocity * self.dt / self.dx
        
        if cfl > STABILITY_FACTOR:
            logger.warning(f"CFL condition violated: {cfl:.3f} > {STABILITY_FACTOR}")
            # Auto-adjust timestep
            self.dt = STABILITY_FACTOR * self.dx / max_velocity
            logger.info(f"Adjusted dt to {self.dt:.6f}")
    
    def create_dark_soliton(self, x0: int, y0: int, width: float = 10.0, 
                           depth: float = 0.8, angle: float = 0.0):
        """
        Create a dark soliton at position (x0, y0)
        
        Args:
            x0, y0: Center position
            width: Soliton width parameter
            depth: Darkness depth (0=no dip, 1=zero at center)
            angle: Propagation angle in radians
        """
        x = np.arange(self.lattice_size) - x0
        y = np.arange(self.lattice_size) - y0
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Rotate coordinates
        X_rot = X * np.cos(angle) + Y * np.sin(angle)
        
        # Dark soliton profile: tanh shape
        # ψ = tanh(X_rot/width) + i * sech(X_rot/width) * depth
        tanh_profile = np.tanh(X_rot / width)
        sech_profile = 1.0 / np.cosh(X_rot / width)
        
        # Amplitude dip
        amplitude = np.sqrt(1.0 - depth * sech_profile**2)
        
        # Phase jump
        phase = np.arctan2(depth * sech_profile, tanh_profile)
        
        # Set lattice values
        self.u_real = amplitude * np.cos(phase)
        self.u_imag = amplitude * np.sin(phase)
        
        logger.info(f"Created dark soliton at ({x0}, {y0}) with width={width}, depth={depth}")
    
    def add_dark_soliton(self, x0: int, y0: int, width: float = 10.0,
                        depth: float = 0.8, angle: float = 0.0):
        """Add a dark soliton to existing field"""
        # Create temporary soliton
        temp_sim = DarkSolitonSimulator({'lattice_size': self.lattice_size})
        temp_sim.create_dark_soliton(x0, y0, width, depth, angle)
        
        # Multiply fields (soliton interaction)
        new_real = self.u_real * temp_sim.u_real - self.u_imag * temp_sim.u_imag
        new_imag = self.u_real * temp_sim.u_imag + self.u_imag * temp_sim.u_real
        
        # Normalize
        amplitude = np.sqrt(new_real**2 + new_imag**2)
        max_amp = np.max(amplitude)
        if max_amp > 0:
            self.u_real = new_real / max_amp
            self.u_imag = new_imag / max_amp
    
    def step(self, n_steps: int = 1) -> Dict[str, float]:
        """
        Advance simulation by n_steps
        Returns metrics dict
        """
        initial_phase = np.angle(self.u_real + 1j * self.u_imag)
        
        for _ in range(n_steps):
            self.u_real, self.u_imag = fdtd_step_dark_soliton(
                self.u_real, self.u_imag,
                self.laplacian_real, self.laplacian_imag,
                self.dt, self.dx,
                self.nonlinearity, self.dispersion, self.damping
            )
            self.step_count += 1
        
        # Calculate phase drift
        final_phase = np.angle(self.u_real + 1j * self.u_imag)
        phase_drift = np.mean(np.abs(final_phase - initial_phase))
        self.total_phase_drift += phase_drift
        
        # Calculate other metrics
        amplitude = np.sqrt(self.u_real**2 + self.u_imag**2)
        
        metrics = {
            'step_count': self.step_count,
            'phase_drift': phase_drift,
            'total_phase_drift': self.total_phase_drift,
            'max_amplitude': np.max(amplitude),
            'min_amplitude': np.min(amplitude),
            'energy': np.sum(amplitude**2) * self.dx**2,
            'mean_amplitude': np.mean(amplitude)
        }
        
        return metrics
    
    def get_field(self) -> np.ndarray:
        """Get complex field"""
        return self.u_real + 1j * self.u_imag
    
    def get_amplitude(self) -> np.ndarray:
        """Get amplitude |ψ|"""
        return np.sqrt(self.u_real**2 + self.u_imag**2)
    
    def get_phase(self) -> np.ndarray:
        """Get phase arg(ψ)"""
        return np.angle(self.u_real + 1j * self.u_imag)
    
    def get_intensity(self) -> np.ndarray:
        """Get intensity |ψ|²"""
        return self.u_real**2 + self.u_imag**2
    
    def save_snapshot(self, filename: str):
        """Save current state to NPZ file"""
        np.savez_compressed(
            filename,
            u_real=self.u_real,
            u_imag=self.u_imag,
            step_count=self.step_count,
            config=self.config
        )
        logger.info(f"Saved snapshot to {filename}")
    
    def load_snapshot(self, filename: str):
        """Load state from NPZ file"""
        data = np.load(filename, allow_pickle=True)
        self.u_real = data['u_real']
        self.u_imag = data['u_imag']
        self.step_count = int(data['step_count'])
        logger.info(f"Loaded snapshot from {filename}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run(config_path: Optional[str] = None, config_dict: Optional[Dict] = None) -> DarkSolitonSimulator:
    """
    Main entry point for dark soliton simulation
    
    Args:
        config_path: Path to YAML config file
        config_dict: Direct config dictionary (overrides file)
    
    Returns:
        Configured and initialized simulator
    """
    # Load configuration
    if config_dict:
        config = config_dict
    elif config_path:
        config = load_config(config_path)
    else:
        # Default config
        config = {
            'lattice_size': 128,
            'dt': 0.01,
            'dx': 1.0,
            'nonlinearity': 1.0,
            'dispersion': 0.5,
            'damping': 0.1
        }
    
    # Create simulator
    sim = DarkSolitonSimulator(config)
    
    # Initialize with solitons if specified
    if 'initial_solitons' in config:
        for soliton in config['initial_solitons']:
            sim.add_dark_soliton(
                x0=soliton.get('x', 64),
                y0=soliton.get('y', 64),
                width=soliton.get('width', 10.0),
                depth=soliton.get('depth', 0.8),
                angle=soliton.get('angle', 0.0)
            )
    else:
        # Default: single centered soliton
        sim.create_dark_soliton(64, 64)
    
    # Run initial steps if specified
    if 'initial_steps' in config:
        n_steps = config['initial_steps']
        logger.info(f"Running {n_steps} initial steps...")
        metrics = sim.step(n_steps)
        logger.info(f"Initial metrics: {metrics}")
    
    return sim

# Demo usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create simulator with default config
    sim = run()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Initial state
    axes[0,0].imshow(sim.get_amplitude(), cmap='viridis')
    axes[0,0].set_title('Initial Amplitude')
    
    axes[0,1].imshow(sim.get_phase(), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0,1].set_title('Initial Phase')
    
    # Evolve
    print("Evolving for 1000 steps...")
    metrics = sim.step(1000)
    print(f"Metrics after 1000 steps: {metrics}")
    
    # Final state
    axes[1,0].imshow(sim.get_amplitude(), cmap='viridis')
    axes[1,0].set_title('Final Amplitude')
    
    axes[1,1].imshow(sim.get_phase(), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1,1].set_title('Final Phase')
    
    plt.tight_layout()
    plt.savefig('dark_soliton_evolution.png', dpi=150)
    print("Saved visualization to dark_soliton_evolution.png")
    
    # Verify phase stability
    print(f"\nPhase drift per step: {metrics['phase_drift']:.6f} rad")
    print(f"Total phase drift: {metrics['total_phase_drift']:.6f} rad")
    
    if metrics['phase_drift'] < 0.02:
        print("✓ Phase stability test PASSED")
    else:
        print("✗ Phase stability test FAILED")
