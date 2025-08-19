"""
Quantum Bridge Module

This module provides a bridge between the classical Banksy oscillator system
and the quantum reservoir computing module, enabling quantum-enhanced
phase dynamics prediction.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logger = logging.getLogger('quantum_bridge')

# Import PCC components if available
try:
    from pcc import QR, EntangledAttention
    HAS_QUANTUM = True
    logger.info("Quantum reservoir module available")
except ImportError:
    HAS_QUANTUM = False
    logger.warning("Quantum reservoir module not available, using classical mode only")

# Default configuration
DEFAULT_N_QUBITS = int(os.environ.get('QR_N_QUBITS', '8'))  # Keep small for performance
DEFAULT_DEPTH = int(os.environ.get('QR_DEPTH', '2'))
DEFAULT_NOISE = float(os.environ.get('QR_NOISE', '0.01'))
QUANTUM_ENABLED = os.environ.get('ENABLE_QUANTUM', 'false').lower() == 'true'


class QuantumBridge:
    """
    Bridge between classical Banksy oscillator and quantum reservoir.
    
    This class provides methods for translating between the classical phase space
    and quantum state space, enabling quantum-enhanced prediction and analysis.
    """
    
    def __init__(self, 
                 n_qubits: int = DEFAULT_N_QUBITS,
                 depth: int = DEFAULT_DEPTH,
                 noise: float = DEFAULT_NOISE,
                 enabled: bool = QUANTUM_ENABLED):
        """
        Initialize the quantum bridge.
        
        Args:
            n_qubits: Number of qubits in the quantum reservoir
            depth: Depth of the entangled attention mechanism
            noise: Noise level for the quantum reservoir
            enabled: Whether to enable quantum enhancement
        """
        self.n_qubits = n_qubits
        self.depth = depth
        self.noise = noise
        self.enabled = enabled and HAS_QUANTUM
        
        # Initialize quantum components if enabled
        if self.enabled:
            try:
                self.attention = EntangledAttention(n_qubits=n_qubits, depth=depth)
                logger.info(f"Initialized quantum bridge with {n_qubits} qubits")
            except Exception as e:
                logger.error(f"Error initializing quantum components: {e}")
                self.enabled = False
                self.attention = None
        else:
            self.attention = None
    
    def enhance_phase_prediction(self, 
                                phases: np.ndarray, 
                                spins: np.ndarray, 
                                forecast: np.ndarray) -> np.ndarray:
        """
        Enhance phase prediction using quantum reservoir.
        
        Args:
            phases: Current phase values [0, 2π)
            spins: Current spin values (±1)
            forecast: Classical forecast of phases
            
        Returns:
            Enhanced forecast incorporating quantum correlations
        """
        if not self.enabled or self.attention is None:
            # Return the original forecast if quantum enhancement is disabled
            return forecast
        
        try:
            # Limit to number of qubits available
            n = min(len(phases), self.n_qubits)
            
            # Get quantum representation of current state
            phase_features = self.attention.phase_to_qubit(phases[:n])
            spin_features = self.attention.spin_to_qubit(spins[:n])
            
            # Calculate attention weights based on quantum features
            phase_weights = self._soften(phase_features, beta=2.0)
            spin_weights = self._soften(spin_features, beta=1.5)
            
            # Combine weights (simple average for now)
            combined_weights = 0.7 * phase_weights + 0.3 * spin_weights
            
            # Apply quantum-derived weights to modulate the forecast
            # This creates small quantum-inspired corrections to the classical forecast
            enhanced_forecast = forecast.copy()
            
            # Apply only to the subset covered by quantum features
            for i in range(n):
                # Small correction based on quantum features (max ±0.1π)
                correction = 0.1 * np.pi * (combined_weights[i] - 0.5) * 2
                enhanced_forecast[i] += correction
            
            # Keep phases in [0, 2π) range
            enhanced_forecast = enhanced_forecast % (2 * np.pi)
            
            return enhanced_forecast
            
        except Exception as e:
            logger.error(f"Error in quantum enhancement: {e}")
            return forecast
    
    def get_quantum_metrics(self, 
                           phases: np.ndarray, 
                           spins: np.ndarray) -> Dict[str, Any]:
        """
        Get quantum metrics for the current state.
        
        Args:
            phases: Current phase values [0, 2π)
            spins: Current spin values (±1)
            
        Returns:
            Dictionary of quantum metrics
        """
        if not self.enabled or self.attention is None:
            return {
                "quantum_enabled": False
            }
        
        try:
            # Limit to number of qubits available
            n = min(len(phases), self.n_qubits)
            
            # Get quantum representation
            qubit_state = self.attention.phase_to_qubit(phases[:n])
            
            # Calculate basic metrics
            alignment = np.mean(qubit_state * spins[:n])
            
            return {
                "quantum_enabled": True,
                "n_qubits": self.n_qubits,
                "quantum_alignment": float(alignment),
                "quantum_features": qubit_state.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error calculating quantum metrics: {e}")
            return {
                "quantum_enabled": False,
                "error": str(e)
            }
    
    def _soften(self, values: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """
        Apply softmax-like transformation to normalize values.
        
        Args:
            values: Input values
            beta: Temperature parameter (higher = sharper)
            
        Returns:
            Normalized values that sum to 1.0
        """
        exp_values = np.exp(beta * values)
        return exp_values / np.sum(exp_values)


# Singleton instance
_bridge = None

def get_quantum_bridge() -> QuantumBridge:
    """Get or create the singleton quantum bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = QuantumBridge()
    return _bridge
