#!/usr/bin/env python3
"""
Lyapunov Exporter for Real-time Stability Monitoring
Writes leading eigenvalues to JSON for dashboard/monitoring
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from python.core.bdg_solver import assemble_bdg, compute_spectrum

logger = logging.getLogger(__name__)

class LyapunovExporter:
    """Export Lyapunov exponents and stability metrics"""
    
    def __init__(self, output_path: Path = Path("data/lyapunov_watchlist.json")):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.history = []
        self.max_history = 1000  # Keep last 1000 readings
        
    def update_watchlist(self, state: Dict[str, Any]) -> float:
        """
        Update Lyapunov watchlist with current system state
        
        Args:
            state: Dictionary containing 'psi0' (wavefunction) and optional params
            
        Returns:
            lambda_max: Maximum Lyapunov exponent (imaginary part)
        """
        try:
            # Extract wavefunction
            psi0 = state.get('psi0')
            if psi0 is None:
                # Try to get from lattice state
                lattice = state.get('lattice')
                if lattice and hasattr(lattice, 'oscillators'):
                    # Construct pseudo-wavefunction from oscillator amplitudes/phases
                    amplitudes = np.array([o.amplitude for o in lattice.oscillators])
                    phases = np.array([o.phase for o in lattice.oscillators])
                    psi0 = amplitudes * np.exp(1j * phases)
                else:
                    logger.warning("No wavefunction found in state")
                    return 0.0
            
            # Parameters
            g = state.get('g', 1.0)  # Nonlinearity strength
            dx = state.get('dx', 1.0)  # Spatial discretization
            
            # Assemble BdG operator
            H_BdG = assemble_bdg(psi0, g=g, dx=dx)
            
            # Compute spectrum
            eigenvalues, _ = compute_spectrum(H_BdG, k=16, target='LI')
            
            # Extract maximum Lyapunov exponent (imaginary part)
            lambda_max = float(np.max(np.abs(eigenvalues.imag)))
            
            # Additional stability metrics
            spectral_radius = float(np.max(np.abs(eigenvalues)))
            oscillation_freq = float(np.max(np.abs(eigenvalues.real)))
            
            # Create watchlist entry
            entry = {
                'timestamp': time.time(),
                'lambda_max': lambda_max,
                'spectral_radius': spectral_radius,
                'oscillation_freq': oscillation_freq,
                'eigenvalues': [complex(ev) for ev in eigenvalues[:8]],  # Top 8
                'stability': 'stable' if lambda_max < 0.1 else 'unstable',
                'energy': float(np.sum(np.abs(psi0)**2)) if psi0 is not None else 0.0
            }
            
            # Update history
            self.history.append(entry)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # Write to file
            self._write_watchlist()
            
            return lambda_max
            
        except Exception as e:
            logger.error(f"Failed to update Lyapunov watchlist: {e}")
            return 0.0
    
    def _write_watchlist(self):
        """Write current watchlist to JSON file"""
        try:
            # Get latest metrics
            if self.history:
                latest = self.history[-1]
                
                # Compute statistics over recent history
                recent = self.history[-100:] if len(self.history) > 100 else self.history
                lambda_values = [h['lambda_max'] for h in recent]
                
                watchlist = {
                    'last_updated': time.time(),
                    'current': {
                        'lambda_max': latest['lambda_max'],
                        'spectral_radius': latest['spectral_radius'],
                        'stability': latest['stability'],
                        'energy': latest['energy']
                    },
                    'statistics': {
                        'mean_lambda': float(np.mean(lambda_values)),
                        'std_lambda': float(np.std(lambda_values)),
                        'max_lambda': float(np.max(lambda_values)),
                        'stability_ratio': sum(1 for h in recent if h['stability'] == 'stable') / len(recent)
                    },
                    'history': self.history[-20:]  # Last 20 entries for plots
                }
                
                # Atomic write
                temp_path = self.output_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(watchlist, f, indent=2, default=str)
                temp_path.replace(self.output_path)
                
        except Exception as e:
            logger.error(f"Failed to write Lyapunov watchlist: {e}")
    
    def get_current_lambda_max(self) -> float:
        """Get most recent lambda_max value"""
        if self.history:
            return self.history[-1]['lambda_max']
        return 0.0
    
    def get_stability_trend(self, window: int = 100) -> str:
        """Analyze stability trend over recent window"""
        if len(self.history) < 2:
            return 'unknown'
        
        recent = self.history[-window:]
        lambda_values = [h['lambda_max'] for h in recent]
        
        # Linear regression for trend
        x = np.arange(len(lambda_values))
        coeffs = np.polyfit(x, lambda_values, 1)
        slope = coeffs[0]
        
        if slope > 0.001:
            return 'deteriorating'
        elif slope < -0.001:
            return 'improving'
        else:
            return 'stable'

# Global singleton
_exporter: Optional[LyapunovExporter] = None

def get_lyapunov_exporter() -> LyapunovExporter:
    """Get global Lyapunov exporter instance"""
    global _exporter
    if _exporter is None:
        _exporter = LyapunovExporter()
    return _exporter

def update_watchlist(state: Dict[str, Any]) -> float:
    """Convenience function to update watchlist"""
    return get_lyapunov_exporter().update_watchlist(state)
