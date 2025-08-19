#!/usr/bin/env python3
"""
Modified eigensentry_guard.py - Enhanced with observer token emission
Production-ready implementation with comprehensive integration
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from collections import deque
import asyncio
import websockets
import json
from alan_backend.lyap_exporter import LyapunovExporter

# NEW IMPORTS for observer synthesis
from python.core.observer_synthesis import emit_token, get_observer_synthesis

logger = logging.getLogger(__name__)

# Guard parameters
BASE_THRESHOLD = 1.0
CURVATURE_SENSITIVITY = 0.5
DAMPING_RATE = 0.1
EMERGENCY_THRESHOLD = 2.0

@dataclass
class CurvatureMetrics:
    """Metrics for local soliton curvature"""
    mean_curvature: float
    gaussian_curvature: float
    principal_curvatures: Tuple[float, float]
    curvature_energy: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class CurvatureAwareGuard:
    """
    EigenSentry guard with dynamic threshold based on local soliton curvature
    Now enhanced with observer token emission for metacognitive feedback
    """
    
    def __init__(self, state_dim: int = 100):
        self.state_dim = state_dim
        
        # Threshold parameters
        self.base_threshold = BASE_THRESHOLD
        self.curvature_sensitivity = CURVATURE_SENSITIVITY
        
        # State tracking
        self.current_threshold = BASE_THRESHOLD
        self.eigenvalue_history = deque(maxlen=1000)
        self.curvature_history = deque(maxlen=1000)
        self.damping_active = False
        
        # Metrics for WebSocket
        self.metrics = {
            'current_threshold': BASE_THRESHOLD,
            'max_eigenvalue': 0.0,
            'mean_curvature': 0.0,
            'lyapunov_exponent': 0.0,
            'damping_active': False,
            'tokens_emitted': 0,
            'websocket_clients': 0
        }
        
        # WebSocket clients
        self.ws_clients = set()
        
        # BdG spectral stability integration
        self.lyap_exporter = LyapunovExporter()
        self.poll_counter = 0
        self.POLL_INTERVAL = 256  # Update every N steps
        self.nonlinearity = 1.0  # Default g parameter
        self.dx = 0.1  # Default spatial step
        
        # Observer synthesis integration
        self.observer_synthesis = get_observer_synthesis()
        self._token_emission_enabled = True
        self._significant_event_threshold = 0.8  # When to add token to context
        
        logger.info("CurvatureAwareGuard initialized with observer token emission")
        
    def compute_local_curvature(self, state: np.ndarray, 
                               lattice_shape: Optional[Tuple[int, int]] = None) -> CurvatureMetrics:
        """
        Compute local soliton curvature from state
        Uses differential geometry on the wave field
        """
        if lattice_shape is None:
            # Assume square lattice
            size = int(np.sqrt(len(state)))
            if size * size != len(state):
                # Fall back to 1D analysis
                return self._compute_1d_curvature(state)
            lattice_shape = (size, size)
            
        # Reshape to 2D if needed
        if len(state.shape) == 1:
            field = state.reshape(lattice_shape)
        else:
            field = state
            
        # Compute gradients with error handling
        try:
            dy, dx = np.gradient(field)
            
            # Second derivatives
            dyy, dyx = np.gradient(dy)
            dxy, dxx = np.gradient(dx)
            
            # Mean curvature: H = (fxx(1+fy²) - 2fxfyfxy + fyy(1+fx²)) / (1+fx²+fy²)^(3/2)
            denominator = (1 + dx**2 + dy**2)**(1.5)
            numerator = dxx * (1 + dy**2) - 2 * dx * dy * dxy + dyy * (1 + dx**2)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_curvature_field = numerator / denominator
                mean_curvature_field = np.nan_to_num(mean_curvature_field, 0.0)
                
            mean_curvature = float(np.mean(np.abs(mean_curvature_field)))
            
            # Gaussian curvature: K = (fxxfyy - fxy²) / (1+fx²+fy²)²
            gaussian_numerator = dxx * dyy - dxy**2
            gaussian_denominator = (1 + dx**2 + dy**2)**2
            
            with np.errstate(divide='ignore', invalid='ignore'):
                gaussian_curvature_field = gaussian_numerator / gaussian_denominator
                gaussian_curvature_field = np.nan_to_num(gaussian_curvature_field, 0.0)
                
            gaussian_curvature = float(np.mean(gaussian_curvature_field))
            
            # Principal curvatures from H and K
            # k1,k2 = H ± sqrt(H² - K)
            discriminant = mean_curvature**2 - gaussian_curvature
            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                k1 = mean_curvature + sqrt_disc
                k2 = mean_curvature - sqrt_disc
            else:
                k1 = k2 = mean_curvature
                
            # Curvature energy (bending energy)
            curvature_energy = float(np.sum(mean_curvature_field**2) * np.prod(lattice_shape) / len(state))
            
        except Exception as e:
            logger.error(f"Error computing 2D curvature: {e}")
            # Fallback to safe values
            mean_curvature = 0.0
            gaussian_curvature = 0.0
            k1 = k2 = 0.0
            curvature_energy = 0.0
            
        return CurvatureMetrics(
            mean_curvature=mean_curvature,
            gaussian_curvature=gaussian_curvature,
            principal_curvatures=(float(k1), float(k2)),
            curvature_energy=curvature_energy
        )
        
    def _compute_1d_curvature(self, state: np.ndarray) -> CurvatureMetrics:
        """Fallback 1D curvature computation"""
        try:
            # First derivative
            dx = np.gradient(state)
            
            # Second derivative
            dxx = np.gradient(dx)
            
            # 1D curvature: κ = |f''| / (1 + f'²)^(3/2)
            with np.errstate(divide='ignore', invalid='ignore'):
                curvature = np.abs(dxx) / (1 + dx**2)**(1.5)
                curvature = np.nan_to_num(curvature, 0.0)
                
            mean_curvature = float(np.mean(curvature))
            curvature_energy = float(np.sum(curvature**2))
            
        except Exception as e:
            logger.error(f"Error computing 1D curvature: {e}")
            mean_curvature = 0.0
            curvature_energy = 0.0
            
        return CurvatureMetrics(
            mean_curvature=mean_curvature,
            gaussian_curvature=0.0,  # Not defined in 1D
            principal_curvatures=(mean_curvature, 0.0),
            curvature_energy=curvature_energy
        )
        
    def update_threshold(self, curvature_metrics: CurvatureMetrics):
        """
        Update dynamic threshold based on local curvature
        Higher curvature → lower threshold (more protection)
        """
        # Base threshold modification
        curvature_factor = 1.0 + self.curvature_sensitivity * curvature_metrics.mean_curvature
        
        # Adaptive threshold with bounds
        self.current_threshold = self.base_threshold / curvature_factor
        self.current_threshold = np.clip(self.current_threshold, 0.5, EMERGENCY_THRESHOLD)
        
        # Store history
        self.curvature_history.append(curvature_metrics)
        
        # Update metrics
        self.metrics['current_threshold'] = float(self.current_threshold)
        self.metrics['mean_curvature'] = float(curvature_metrics.mean_curvature)
        
    def check_eigenvalues(self, eigenvalues: np.ndarray, state: np.ndarray) -> Dict[str, Any]:
        """
        Check eigenvalues against curvature-aware threshold
        Now emits observer tokens for metacognitive monitoring
        """
        # Validate inputs
        if not isinstance(eigenvalues, np.ndarray):
            eigenvalues = np.array(eigenvalues)
        if not isinstance(state, np.ndarray):
            state = np.array(state)
            
        max_eigenvalue = float(np.max(np.real(eigenvalues)))
        
        # Compute local curvature
        curvature_metrics = self.compute_local_curvature(state)
        
        # Poll BdG spectral stability
        self.poll_spectral_stability(state)
        
        # Update threshold
        self.update_threshold(curvature_metrics)
        
        # Store eigenvalue history
        self.eigenvalue_history.append({
            'timestamp': datetime.now(timezone.utc),
            'max_eigenvalue': max_eigenvalue,
            'threshold': float(self.current_threshold)
        })
        
        # Update metrics
        self.metrics['max_eigenvalue'] = max_eigenvalue
        self.metrics['websocket_clients'] = len(self.ws_clients)
        
        # Compute Lyapunov exponent estimate
        if len(self.eigenvalue_history) > 10:
            recent = list(self.eigenvalue_history)[-10:]
            lyapunov = np.mean([h['max_eigenvalue'] for h in recent])
            self.metrics['lyapunov_exponent'] = float(lyapunov)
        
        # Determine action
        action = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'max_eigenvalue': max_eigenvalue,
            'threshold': float(self.current_threshold),
            'curvature': float(curvature_metrics.mean_curvature),
            'gaussian_curvature': float(curvature_metrics.gaussian_curvature),
            'curvature_energy': float(curvature_metrics.curvature_energy),
            'action': 'none',
            'damping_factor': 0.0
        }
        
        # Determine damping action
        if max_eigenvalue > EMERGENCY_THRESHOLD:
            # Emergency damping
            action['action'] = 'emergency_damp'
            action['damping_factor'] = 0.5
            self.damping_active = True
            logger.error(f"EMERGENCY: Eigenvalue {max_eigenvalue:.3f} > {EMERGENCY_THRESHOLD}")
            
        elif max_eigenvalue > self.current_threshold:
            # Curvature-aware damping
            excess = max_eigenvalue - self.current_threshold
            damping_factor = DAMPING_RATE * (1 + excess)
            
            action['action'] = 'adaptive_damp'
            action['damping_factor'] = float(damping_factor)
            self.damping_active = True
            logger.warning(f"Adaptive damping: eigenvalue {max_eigenvalue:.3f} > threshold {self.current_threshold:.3f}")
            
        else:
            # No damping needed
            self.damping_active = False
            
        self.metrics['damping_active'] = self.damping_active
        
        # EMIT OBSERVER TOKEN for metacognitive feedback
        if self._token_emission_enabled:
            try:
                token_data = {
                    "type": "curvature",
                    "source": "eigensentry",
                    "lambda_max": max_eigenvalue,
                    "mean_curvature": float(curvature_metrics.mean_curvature),
                    "gaussian_curvature": float(curvature_metrics.gaussian_curvature),
                    "curvature_energy": float(curvature_metrics.curvature_energy),
                    "principal_curvatures": list(curvature_metrics.principal_curvatures),
                    "threshold": float(self.current_threshold),
                    "damping_active": self.damping_active,
                    "action": action['action'],
                    "damping_factor": action['damping_factor'],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                token = emit_token(token_data)
                self.metrics['tokens_emitted'] += 1
                logger.debug(f"Emitted observer token: {token[:8]}...")
                
                # Add to context if significant event
                is_significant = (
                    action['action'] != 'none' or 
                    max_eigenvalue > self._significant_event_threshold * self.current_threshold or
                    curvature_metrics.mean_curvature > 2.0
                )
                
                if is_significant and self.observer_synthesis:
                    self.observer_synthesis.add_to_context(token)
                    logger.info(f"Added significant event to observer context: {token[:8]}...")
                    
            except Exception as e:
                logger.warning(f"Failed to emit observer token: {e}")
        
        # Broadcast to WebSocket clients
        asyncio.create_task(self._broadcast_metrics())
        
        return action
        
    def apply_damping(self, state: np.ndarray, damping_factor: float) -> np.ndarray:
        """
        Apply curvature-aware damping to state
        Preserves soliton structure while reducing instability
        """
        if not isinstance(state, np.ndarray):
            state = np.array(state)
            
        # Compute local curvature for shaped damping
        curvature_metrics = self.compute_local_curvature(state)
        
        # Create damping profile based on curvature
        # High curvature regions get less damping (preserve solitons)
        size = int(np.sqrt(len(state)))
        if size * size == len(state):
            field = state.reshape((size, size))
            
            try:
                # Compute local curvature field
                dy, dx = np.gradient(field)
                dxx = np.gradient(dx, axis=1)
                dyy = np.gradient(dy, axis=0)
                
                local_curvature = np.abs(dxx) + np.abs(dyy)
                
                # Damping profile: less damping where curvature is high
                # Avoid division by zero
                if curvature_metrics.mean_curvature > 0:
                    damping_profile = 1.0 - np.tanh(local_curvature / curvature_metrics.mean_curvature)
                else:
                    damping_profile = np.ones_like(local_curvature)
                    
                damping_profile = damping_profile.flatten()
                
            except Exception as e:
                logger.warning(f"Error computing damping profile: {e}")
                damping_profile = np.ones_like(state)
        else:
            # Uniform damping for 1D or irregular shapes
            damping_profile = np.ones_like(state)
            
        # Apply shaped damping
        damped_state = state * (1.0 - damping_factor * damping_profile)
        
        # Emit token for damping action
        if self._token_emission_enabled:
            try:
                token = emit_token({
                    "type": "damping_applied",
                    "source": "eigensentry",
                    "damping_factor": float(damping_factor),
                    "mean_damping": float(np.mean(damping_profile)),
                    "state_norm_before": float(np.linalg.norm(state)),
                    "state_norm_after": float(np.linalg.norm(damped_state)),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                logger.debug(f"Emitted damping token: {token[:8]}...")
            except Exception as e:
                logger.warning(f"Failed to emit damping token: {e}")
        
        return damped_state
        
    async def _broadcast_metrics(self):
        """Broadcast metrics to WebSocket clients"""
        if not self.ws_clients:
            return
            
        message = json.dumps({
            'type': 'eigensentry_metrics',
            'data': self.metrics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Send to all connected clients
        disconnected = set()
        for client in self.ws_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.warning(f"Error broadcasting to client: {e}")
                disconnected.add(client)
                
        # Remove disconnected clients
        self.ws_clients -= disconnected
        if disconnected:
            logger.info(f"Removed {len(disconnected)} disconnected WebSocket clients")
        
    def poll_spectral_stability(self, soliton_state: np.ndarray):
        """Poll spectral stability every N steps"""
        self.poll_counter += 1
        
        if self.poll_counter % self.POLL_INTERVAL == 0:
            try:
                # Get current lattice parameters
                params = {'g': self.nonlinearity, 'dx': self.dx}
                
                # Update spectrum
                metrics = self.lyap_exporter.update_spectrum(soliton_state, params)
                
                # Update internal state
                self.metrics['lambda_max'] = metrics.get('lambda_max', 0.0)
                self.metrics['unstable_modes'] = metrics.get('unstable_count', 0)
                
                # Emit stability token
                if self._token_emission_enabled and metrics.get('unstable_count', 0) > 0:
                    token = emit_token({
                        "type": "spectral_stability",
                        "source": "eigensentry_bdg",
                        "lambda_max": metrics.get('lambda_max', 0.0),
                        "unstable_modes": metrics.get('unstable_count', 0),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    logger.debug(f"Emitted BdG stability token: {token[:8]}...")
                
                # Broadcast update
                asyncio.create_task(self._broadcast_metrics())
                
            except Exception as e:
                logger.error(f"Error polling spectral stability: {e}")
    
    def register_websocket(self, websocket):
        """Register a WebSocket client for metrics"""
        self.ws_clients.add(websocket)
        logger.info(f"WebSocket client registered. Total clients: {len(self.ws_clients)}")
        
    def unregister_websocket(self, websocket):
        """Unregister a WebSocket client"""
        self.ws_clients.discard(websocket)
        logger.info(f"WebSocket client unregistered. Total clients: {len(self.ws_clients)}")
        
    def inject_synthetic_blowup(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject synthetic blow-up wave for testing
        Returns (state, eigenvalues)
        """
        # Create state with high-curvature soliton
        size = 64
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        
        # Steep soliton (high curvature)
        R = np.sqrt(X**2 + Y**2)
        state = np.exp(-R**2 / 0.5)  # Narrow Gaussian
        state = state.flatten()
        
        # Synthetic eigenvalues with instability
        eigenvalues = np.random.randn(10) + 1j * np.random.randn(10)
        eigenvalues[0] = 2.5  # Large positive eigenvalue
        
        return state, eigenvalues
        
    def test_damping_effectiveness(self, steps: int = 300) -> bool:
        """
        Test that guard damps synthetic blow-up within specified steps
        Returns True if test passes
        """
        # Inject blow-up
        state, eigenvalues = self.inject_synthetic_blowup()
        
        damped = False
        initial_tokens = self.metrics['tokens_emitted']
        
        for step in range(steps):
            # Check eigenvalues
            action = self.check_eigenvalues(eigenvalues, state)
            
            if action['action'] != 'none':
                # Apply damping
                state = self.apply_damping(state, action['damping_factor'])
                
                # Reduce eigenvalues (simulated evolution)
                eigenvalues = eigenvalues * (1.0 - action['damping_factor'] * 0.1)
                
            # Check if damped
            if np.max(np.real(eigenvalues)) < self.current_threshold:
                logger.info(f"Blow-up damped in {step + 1} steps")
                damped = True
                break
                
        tokens_emitted = self.metrics['tokens_emitted'] - initial_tokens
        logger.info(f"Test emitted {tokens_emitted} observer tokens")
        
        return damped
    
    def set_token_emission(self, enabled: bool):
        """Enable or disable observer token emission"""
        self._token_emission_enabled = enabled
        logger.info(f"Observer token emission: {'enabled' if enabled else 'disabled'}")
    
    def get_observer_context(self) -> Dict[str, Any]:
        """Get current observer synthesis context"""
        if self.observer_synthesis:
            return self.observer_synthesis.synthesize_context()
        return {'empty': True}

# Standalone guard instance
guard = CurvatureAwareGuard()

def get_guard() -> CurvatureAwareGuard:
    """Get the global guard instance"""
    return guard

# Integration test
if __name__ == "__main__":
    # Test the enhanced guard
    print("Testing curvature-aware guard with observer tokens...")
    
    test_guard = CurvatureAwareGuard()
    
    print("\nTest 1: Normal state")
    normal_state = np.random.randn(100) * 0.1
    normal_eigenvalues = np.random.randn(10) * 0.5
    
    action1 = test_guard.check_eigenvalues(normal_eigenvalues, normal_state)
    print(f"Normal state: {action1['action']} (threshold: {action1['threshold']:.3f})")
    print(f"Tokens emitted: {test_guard.metrics['tokens_emitted']}")
    
    print("\nTest 2: High curvature state")
    x = np.linspace(-5, 5, 100)
    high_curve_state = np.exp(-x**2 / 0.1)  # Narrow peak
    
    action2 = test_guard.check_eigenvalues(normal_eigenvalues, high_curve_state)
    print(f"High curvature: {action2['action']} (threshold: {action2['threshold']:.3f})")
    print(f"Mean curvature: {action2['curvature']:.3f}")
    
    print("\nTest 3: Blow-up damping with token tracking")
    print("Testing blow-up damping...")
    passed = test_guard.test_damping_effectiveness()
    
    if passed:
        print("✓ Guard damping test PASSED")
    else:
        print("✗ Guard damping test FAILED")
        
    print(f"\nTotal tokens emitted: {test_guard.metrics['tokens_emitted']}")
    
    print("\nTest 4: Observer context")
    context = test_guard.get_observer_context()
    print(f"Observer context: {json.dumps(context, indent=2)}")
    
    print("\n✅ All tests completed!")
