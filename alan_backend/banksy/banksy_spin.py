"""
Banksy Spin Oscillator Module

This module implements the dual-phase (θ, σ) oscillator step function
that combines Kuramoto phase dynamics with Ising-like spin dynamics.
"""

import numpy as np
import os
from typing import Dict, Tuple, List, Optional, Union
from .clock import spin_clock, get_clock_metrics
from .broadcast import emit_pcc

# Configure PCC broadcast settings
PCC_BROADCAST_INTERVAL = int(os.environ.get('PCC_BROADCAST_INTERVAL', '10'))
PCC_BROADCAST_ENABLED = os.environ.get('PCC_BROADCAST_ENABLED', 'true').lower() == 'true'

# Default parameters (can be overridden from config)
DEFAULT_COUPLING = 0.1
DEFAULT_DAMPING = 0.02
DEFAULT_NOISE = 0.001
DEFAULT_DT = 0.01


# Step counter for PCC broadcast
_step_counter = 0

def step(
    theta: np.ndarray,
    omega: np.ndarray,
    adjacency: np.ndarray,
    dt: float = DEFAULT_DT,
    coupling: float = DEFAULT_COUPLING,
    damping: float = DEFAULT_DAMPING,
    noise_level: float = DEFAULT_NOISE,
    metrics: Optional[Dict] = None,
    sigma: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a dual (θ, σ) step of the Banksy spin oscillator.
    
    This function updates both the phase (θ) using Kuramoto dynamics and 
    spin (σ) using an Ising-like model, with coupling between them.
    
    Args:
        theta: Current phase array [0, 2π)
        omega: Natural frequency array
        adjacency: Network adjacency matrix
        dt: Time step size
        coupling: Coupling strength between oscillators
        damping: Damping parameter to control convergence speed
        noise_level: Amount of random noise to add
        metrics: Optional dictionary to store runtime metrics
        sigma: Optional current spin array (±1), if None will be derived from theta
        
    Returns:
        (new_theta, new_sigma): Updated phase and spin arrays
    """
    global _step_counter
    _step_counter += 1
    # Number of oscillators
    n = len(theta)
    
    # Initialize or update sigma (spin) values based on phase
    if sigma is None:
        # Convert phase to binary spin values (±1)
        # Phases in quadrants 1 and 4 are +1, quadrants 2 and 3 are -1
        sigma = np.sign(np.cos(theta))
    
    # Update phase (theta) using Kuramoto dynamics with spin influence
    # 1. Calculate phase differences
    theta_diff = np.subtract.outer(theta, theta)
    
    # 2. Calculate phase update from network
    phase_coupling = np.sum(adjacency * np.sin(theta_diff), axis=1)
    
    # 3. Add influence from spins (cross-coupling)
    spin_influence = np.sum(adjacency * np.outer(sigma, sigma), axis=1)
    
    # 4. Calculate full update
    dtheta = (
        omega 
        + coupling * phase_coupling 
        + 0.5 * coupling * sigma * spin_influence
        - damping * np.sin(2 * theta)  # Double-well damping term
    )
    
    # 5. Add thermal noise (smaller for spins than for phases)
    noise = noise_level * np.random.normal(0, 1, n)
    
    # 6. Update theta with timestep
    new_theta = (theta + dt * dtheta + noise) % (2 * np.pi)
    
    # Update sigma (spin) with thermal flips based on local field
    # 1. Calculate local field at each spin
    local_field = np.dot(adjacency, sigma) + 0.1 * np.cos(new_theta)
    
    # 2. Calculate flip probabilities (Glauber dynamics)
    flip_prob = 1 / (1 + np.exp(2 * sigma * local_field / noise_level))
    
    # 3. Perform spin flips based on probability
    flip_mask = np.random.random(n) < flip_prob
    new_sigma = sigma.copy()
    new_sigma[flip_mask] *= -1
    
    # Calculate clock and other metrics if requested
    if metrics is not None:
        s_t = spin_clock(new_sigma)
        metrics["spin_clock"] = s_t
        
        # Calculate energy (approximate Ising energy + phase coupling)
        energy = -0.5 * np.sum(np.outer(new_sigma, new_sigma) * adjacency) - coupling * np.sum(np.cos(theta_diff) * adjacency)
        energy_normalized = energy / n  # Normalize by system size
        metrics["energy"] = energy_normalized
        
        # Store extended clock metrics
        clock_metrics = get_clock_metrics(new_sigma)
        metrics.update(clock_metrics)
    else:
        # If no metrics dictionary provided, calculate energy anyway for PCC broadcast
        energy = -0.5 * np.sum(np.outer(new_sigma, new_sigma) * adjacency) - coupling * np.sum(np.cos(theta_diff) * adjacency)
        energy_normalized = energy / n
    
    # Broadcast PCC state at the configured interval if enabled
    if PCC_BROADCAST_ENABLED and _step_counter % PCC_BROADCAST_INTERVAL == 0:
        # Take the first 64 oscillators for broadcast (limit for visualization)
        broadcast_limit = min(64, n)
        phases_subset = new_theta[:broadcast_limit]
        spins_subset = new_sigma[:broadcast_limit]
        
        # Emit state to MCP server for broadcasting
        try:
            emit_pcc(
                step=_step_counter,
                phases=phases_subset,
                spins=spins_subset,
                energy=energy_normalized
            )
        except Exception as e:
            # Failure to broadcast shouldn't affect simulation
            if metrics is not None:
                metrics["pcc_broadcast_error"] = str(e)
    
    return new_theta, new_sigma

# 🔧 CRITICAL FIX: Create oscillator_update alias for backward compatibility
# This resolves the ImportError in clustering.py
oscillator_update = step
