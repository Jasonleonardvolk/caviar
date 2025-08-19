#!/usr/bin/env python3
"""
Lyapunov-Gated Feedback Pump
Online Lyapunov estimator with PID control for chaos regulation
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PIDParams:
    """PID controller parameters"""
    kp: float = 1.0  # Proportional gain
    ki: float = 0.1  # Integral gain
    kd: float = 0.01  # Derivative gain
    setpoint: float = 0.0  # Target Lyapunov exponent
    
class LyapunovEstimator:
    """
    Online Lyapunov exponent estimator using QR decomposition
    """
    
    def __init__(self, dim: int, tau: float = 0.1):
        self.dim = dim
        self.tau = tau  # Time step
        self.Q = np.eye(dim)  # Orthonormal basis
        self.R_diag = np.ones(dim)  # Diagonal of R matrix
        self.lyap_sum = np.zeros(dim)
        self.steps = 0
        
    def update(self, jacobian: np.ndarray) -> np.ndarray:
        """
        Update Lyapunov estimate with new Jacobian
        
        Args:
            jacobian: System Jacobian matrix
            
        Returns:
            Current Lyapunov spectrum
        """
        # Propagate tangent vectors
        Y = jacobian @ self.Q
        
        # QR decomposition
        Q_new, R = np.linalg.qr(Y)
        
        # Extract diagonal (growth rates)
        r_diag = np.abs(np.diag(R))
        
        # Update running sum
        self.lyap_sum += np.log(r_diag + 1e-10)
        self.steps += 1
        
        # Update basis
        self.Q = Q_new
        self.R_diag = r_diag
        
        # Return current estimate
        if self.steps > 0:
            return self.lyap_sum / (self.steps * self.tau)
        else:
            return np.zeros(self.dim)
            
    def get_max_lyapunov(self) -> float:
        """Get maximum Lyapunov exponent"""
        if self.steps > 0:
            spectrum = self.lyap_sum / (self.steps * self.tau)
            return np.max(spectrum)
        return 0.0
        
    def reset(self):
        """Reset estimator"""
        self.Q = np.eye(self.dim)
        self.R_diag = np.ones(self.dim)
        self.lyap_sum = np.zeros(self.dim)
        self.steps = 0

class LyapunovGatedPump:
    """
    Feedback pump with gain controlled by Lyapunov exponent
    """
    
    def __init__(self, 
                 target_lyapunov: float = 0.0,
                 lambda_threshold: float = 0.05,
                 pid_params: Optional[PIDParams] = None):
        
        self.target_lyapunov = target_lyapunov
        self.lambda_threshold = lambda_threshold
        self.pid = pid_params or PIDParams(setpoint=target_lyapunov)
        
        # PID state
        self.error_integral = 0.0
        self.last_error = 0.0
        
        # Gain limits
        self.min_gain = 0.1
        self.max_gain = 2.0
        self.current_gain = 1.0
        
        # History for analysis
        self.gain_history = deque(maxlen=1000)
        self.lyap_history = deque(maxlen=1000)
        
    def compute_gain(self, current_lyapunov: float, dt: float = 0.01) -> float:
        """
        Compute pump gain using PID control
        
        Args:
            current_lyapunov: Current maximum Lyapunov exponent
            dt: Time step
            
        Returns:
            Pump gain value
        """
        # Calculate error
        error = self.pid.setpoint - current_lyapunov
        
        # Update integral
        self.error_integral += error * dt
        self.error_integral = np.clip(self.error_integral, -10, 10)  # Anti-windup
        
        # Calculate derivative
        if self.last_error is not None:
            error_derivative = (error - self.last_error) / dt
        else:
            error_derivative = 0.0
            
        # PID control law
        control_signal = (
            self.pid.kp * error +
            self.pid.ki * self.error_integral +
            self.pid.kd * error_derivative
        )
        
        # Update gain with safety clipping
        self.current_gain += control_signal * dt
        self.current_gain = np.clip(self.current_gain, self.min_gain, self.max_gain)
        
        # Emergency damping if Lyapunov exceeds threshold
        if current_lyapunov > self.lambda_threshold:
            self.current_gain *= 0.9  # Rapid reduction
            logger.warning(f"Emergency damping: λ={current_lyapunov:.3f} > {self.lambda_threshold}")
            
        # Store history
        self.gain_history.append(self.current_gain)
        self.lyap_history.append(current_lyapunov)
        self.last_error = error
        
        return self.current_gain
        
    def apply_feedback(self, state: np.ndarray, gain: Optional[float] = None) -> np.ndarray:
        """
        Apply feedback with current or specified gain
        
        Args:
            state: System state vector
            gain: Optional gain override
            
        Returns:
            Modified state after feedback
        """
        g = gain if gain is not None else self.current_gain
        
        # Nonlinear feedback inspired by Pyragas control
        # u = -g * (state - f(state))
        # Here we use a simplified version
        feedback = -g * np.tanh(state / 2.0)
        
        return state + feedback
        
    def get_status(self) -> dict:
        """Get pump status"""
        return {
            'current_gain': self.current_gain,
            'target_lyapunov': self.target_lyapunov,
            'recent_lyapunov': list(self.lyap_history)[-10:] if self.lyap_history else [],
            'error_integral': self.error_integral,
            'gain_range': [self.min_gain, self.max_gain]
        }
        
# Test function
def test_lyapunov_pump():
    """Test the Lyapunov-gated pump"""
    print("Testing Lyapunov-Gated Pump")
    print("=" * 50)
    
    # Create estimator and pump
    estimator = LyapunovEstimator(dim=3)
    pump = LyapunovGatedPump(target_lyapunov=0.02)
    
    # Simulate chaotic system
    state = np.random.randn(3)
    
    for i in range(100):
        # Fake Jacobian (would be real system Jacobian)
        J = np.random.randn(3, 3) * 0.1
        J += np.diag([0.05, -0.02, -0.1])  # Some eigenvalues
        
        # Update Lyapunov estimate
        spectrum = estimator.update(J)
        max_lyap = np.max(spectrum)
        
        # Compute pump gain
        gain = pump.compute_gain(max_lyap)
        
        # Apply feedback
        state = pump.apply_feedback(state, gain)
        
        if i % 20 == 0:
            print(f"Step {i}: λ_max={max_lyap:.3f}, gain={gain:.3f}")
            
    print(f"\nFinal status: {pump.get_status()}")
    
if __name__ == "__main__":
    test_lyapunov_pump()
