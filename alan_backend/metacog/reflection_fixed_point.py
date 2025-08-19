#!/usr/bin/env python3
"""
Reflection Fixed Point with Chaos Burst Integration
Allows metacognition to request exploratory chaos bursts
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chaos_channel_controller import get_controller, BurstMetrics

logger = logging.getLogger(__name__)

@dataclass
class ReflectionState:
    """State of reflection process"""
    iteration: int
    convergence_metric: float
    fixed_point: Optional[np.ndarray]
    used_chaos: bool = False

class ReflectionFixedPoint:
    """
    Reflection layer with chaos burst capability
    Requests controlled chaos for creative exploration
    """
    
    def __init__(self, state_dim: int = 100):
        self.state_dim = state_dim
        self.chaos_controller = get_controller()
        
        # Reflection parameters
        self.momentum = 0.9
        self.learning_rate = 0.1
        self.convergence_threshold = 0.01
        
        # Chaos usage tracking
        self.chaos_burst_count = 0
        self.chaos_discoveries = []
        
    def reflect(self, initial_state: np.ndarray, 
                max_iterations: int = 100,
                allow_chaos: bool = True) -> ReflectionState:
        """
        Iterative reflection with optional chaos bursts
        
        Args:
            initial_state: Starting state vector
            max_iterations: Maximum reflection iterations
            allow_chaos: Whether to allow chaos bursts
            
        Returns:
            ReflectionState with convergence info
        """
        state = initial_state.copy()
        velocity = np.zeros_like(state)
        
        used_chaos = False
        
        for iteration in range(max_iterations):
            # Compute reflection gradient
            gradient = self._compute_gradient(state)
            
            # Check if stuck (low gradient)
            gradient_norm = np.linalg.norm(gradient)
            
            if gradient_norm < 0.1 and allow_chaos and not used_chaos:
                # Request chaos burst for exploration
                logger.info(f"Reflection stuck at iteration {iteration}, requesting chaos burst")
                
                burst_id = self.chaos_controller.trigger(
                    level=0.3,
                    duration=50,
                    purpose="reflection_exploration",
                    callback=self._chaos_complete_callback
                )
                
                if burst_id:
                    used_chaos = True
                    self.chaos_burst_count += 1
                    
                    # Run chaos burst
                    for _ in range(50):
                        self.chaos_controller.step()
                        
                    # Perturb state based on chaos
                    chaos_state = self.chaos_controller.get_lattice_state()
                    perturbation = np.real(chaos_state.flatten()[:self.state_dim])
                    perturbation = perturbation / (np.linalg.norm(perturbation) + 1e-6)
                    
                    state += 0.1 * perturbation
                    
            # Update with momentum
            velocity = self.momentum * velocity - self.learning_rate * gradient
            state += velocity
            
            # Check convergence
            convergence_metric = np.linalg.norm(velocity)
            
            if convergence_metric < self.convergence_threshold:
                logger.info(f"Reflection converged at iteration {iteration}")
                return ReflectionState(
                    iteration=iteration,
                    convergence_metric=convergence_metric,
                    fixed_point=state,
                    used_chaos=used_chaos
                )
                
        # Did not converge
        return ReflectionState(
            iteration=max_iterations,
            convergence_metric=convergence_metric,
            fixed_point=None,
            used_chaos=used_chaos
        )
        
    def _compute_gradient(self, state: np.ndarray) -> np.ndarray:
        """
        Compute reflection gradient
        Simplified - would be task-specific in production
        """
        # Example: gradient toward origin with noise
        gradient = -0.1 * state + 0.01 * np.random.randn(*state.shape)
        return gradient
        
    def _chaos_complete_callback(self, metrics: BurstMetrics):
        """Callback when chaos burst completes"""
        logger.info(f"Chaos burst {metrics.burst_id} complete")
        logger.info(f"Peak energy: {metrics.peak_energy:.3f}")
        logger.info(f"Discoveries: {len(metrics.discoveries)}")
        
        # Store discoveries
        self.chaos_discoveries.extend(metrics.discoveries)
        
    def get_chaos_usage_stats(self) -> Dict[str, Any]:
        """Get statistics on chaos usage"""
        return {
            'burst_count': self.chaos_burst_count,
            'total_discoveries': len(self.chaos_discoveries),
            'discovery_types': self._categorize_discoveries()
        }
        
    def _categorize_discoveries(self) -> Dict[str, int]:
        """Categorize discoveries by type"""
        categories = {}
        for discovery in self.chaos_discoveries:
            d_type = discovery.get('type', 'unknown')
            categories[d_type] = categories.get(d_type, 0) + 1
        return categories

# Test integration
def test_reflection_with_chaos():
    """Test reflection layer with chaos integration"""
    print("Testing reflection with chaos bursts...")
    
    # Create reflection layer
    reflection = ReflectionFixedPoint(state_dim=50)
    
    # Create difficult initial state (far from origin)
    initial_state = np.ones(50) * 5 + np.random.randn(50) * 0.1
    
    # Test without chaos
    print("\n1. Reflection without chaos:")
    result_no_chaos = reflection.reflect(initial_state, allow_chaos=False)
    print(f"   Iterations: {result_no_chaos.iteration}")
    print(f"   Converged: {result_no_chaos.fixed_point is not None}")
    print(f"   Used chaos: {result_no_chaos.used_chaos}")
    
    # Reset and test with chaos
    reflection_with_chaos = ReflectionFixedPoint(state_dim=50)
    
    print("\n2. Reflection with chaos allowed:")
    result_with_chaos = reflection_with_chaos.reflect(initial_state, allow_chaos=True)
    print(f"   Iterations: {result_with_chaos.iteration}")
    print(f"   Converged: {result_with_chaos.fixed_point is not None}")
    print(f"   Used chaos: {result_with_chaos.used_chaos}")
    
    # Get chaos stats
    stats = reflection_with_chaos.get_chaos_usage_stats()
    print(f"\n3. Chaos usage statistics:")
    print(f"   Burst count: {stats['burst_count']}")
    print(f"   Discoveries: {stats['total_discoveries']}")
    print(f"   Discovery types: {stats['discovery_types']}")
    
    # Check regression (accuracy should not drop significantly)
    if result_no_chaos.fixed_point is not None and result_with_chaos.fixed_point is not None:
        accuracy_drop = np.linalg.norm(result_with_chaos.fixed_point) / np.linalg.norm(result_no_chaos.fixed_point)
        print(f"\n4. Accuracy check:")
        print(f"   Relative norm ratio: {accuracy_drop:.3f}")
        print(f"   Accuracy drop: {abs(1 - accuracy_drop) * 100:.1f}%")
        
        if abs(1 - accuracy_drop) < 0.02:  # Less than 2% change
            print("   ✓ Regression test PASSED")
        else:
            print("   ✗ Regression test FAILED")
            
    print("\nReflection-chaos integration test complete")

if __name__ == "__main__":
    test_reflection_with_chaos()
