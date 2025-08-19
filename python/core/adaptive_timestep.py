#!/usr/bin/env python3
"""
Adaptive Timestep Control Based on Spectral Stability
Dynamically adjusts integration timestep to maintain stability
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AdaptiveTimestep:
    """
    Adaptive timestep controller based on Lyapunov exponents
    
    Strategy: dt = dt_base / (1 + κ * λ_max)
    """
    
    def __init__(self, dt_base: float = 0.01, 
                 kappa: float = 0.75,
                 dt_min: float = 1e-5,
                 dt_max: float = 0.05):
        self.dt_base = dt_base
        self.kappa = kappa
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # History for smoothing
        self.dt_history = []
        self.lambda_history = []
        
    def compute_timestep(self, lambda_max: float, 
                        energy_error: Optional[float] = None) -> float:
        """
        Compute adaptive timestep based on stability metrics
        
        Args:
            lambda_max: Maximum Lyapunov exponent
            energy_error: Optional energy conservation error
        """
        # Base timestep adjustment
        dt = self.dt_base / (1.0 + self.kappa * max(0, lambda_max))
        
        # Additional energy-based adjustment
        if energy_error is not None and energy_error > 0.01:
            dt *= 0.9  # Reduce timestep if energy not conserved
            
        # Apply bounds
        dt = np.clip(dt, self.dt_min, self.dt_max)
        
        # Smooth using exponential moving average
        if self.dt_history:
            alpha = 0.3  # Smoothing factor
            dt = alpha * dt + (1 - alpha) * self.dt_history[-1]
            
        # Update history
        self.dt_history.append(dt)
        self.lambda_history.append(lambda_max)
        
        # Keep history bounded
        if len(self.dt_history) > 100:
            self.dt_history.pop(0)
            self.lambda_history.pop(0)
            
        return dt
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get timestep adaptation metrics"""
        if not self.dt_history:
            return {'current_dt': self.dt_base, 'adaptation_ratio': 1.0}
            
        return {
            'current_dt': self.dt_history[-1],
            'average_dt': np.mean(self.dt_history),
            'adaptation_ratio': self.dt_history[-1] / self.dt_base,
            'lambda_trend': np.mean(self.lambda_history[-10:]) if len(self.lambda_history) > 10 else 0.0
        }
