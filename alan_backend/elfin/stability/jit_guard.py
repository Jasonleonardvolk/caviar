"""
JIT-compiled Stability Guard for ELFIN Framework.

This module provides JIT-compiled stability guards for runtime enforcement
of stability properties. It includes real-time monitoring of Lyapunov function
values and control synthesis for stability enforcement.
"""

import os
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Union

try:
    from numba import jit
    HAVE_NUMBA = True
except ImportError:
    # Mock JIT decorator
    def jit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        else:
            return lambda x: x
    HAVE_NUMBA = False

try:
    from alan_backend.elfin.stability.lyapunov import LyapunovFunction
except ImportError:
    # Minimal implementation for standalone testing
    class LyapunovFunction:
        def __init__(self, name, domain_ids=None):
            self.name = name
            self.domain_ids = domain_ids or []
            
        def evaluate(self, x):
            return float(np.sum(x**2))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@jit(nopython=True)
def compute_delta_V(x_prev, x, Q):
    """
    Compute change in Lyapunov function value.
    
    Args:
        x_prev: Previous state
        x: Current state
        Q: Quadratic form matrix
        
    Returns:
        Change in Lyapunov function value
    """
    v_prev = x_prev.T @ Q @ x_prev
    v = x.T @ Q @ x
    return v - v_prev


class StabilityGuard:
    """
    Real-time stability guard.
    
    This class monitors the Lyapunov function value in real-time and
    triggers callbacks when stability violations are detected.
    """
    
    def __init__(
        self,
        lyap: LyapunovFunction,
        threshold: float = 0.0,
        callback: Optional[Callable] = None,
        use_jit: bool = True
    ):
        """
        Initialize stability guard.
        
        Args:
            lyap: Lyapunov function to monitor
            threshold: Threshold for stability violations
            callback: Callback function for violations
            use_jit: Whether to use JIT compilation
        """
        self.lyap = lyap
        self.threshold = threshold
        self.callback = callback
        self.use_jit = use_jit and HAVE_NUMBA
        
        # Stats
        self.violations = 0
        self.total_steps = 0
        self.last_violation_time = 0.0
        
        # JIT-compiled functions
        if hasattr(lyap, 'get_quadratic_form') and lyap.get_quadratic_form() is not None:
            self.Q = lyap.get_quadratic_form()
            if self.use_jit:
                self.delta_V_fn = compute_delta_V
            else:
                self.delta_V_fn = lambda x_prev, x, Q: (
                    x.T @ Q @ x - x_prev.T @ Q @ x_prev
                )
        else:
            self.Q = None
            self.delta_V_fn = None
    
    def step(self, x_prev: np.ndarray, x: np.ndarray) -> bool:
        """
        Check stability for a step from x_prev to x.
        
        Args:
            x_prev: Previous state
            x: Current state
            
        Returns:
            Whether the step is stable
        """
        self.total_steps += 1
        
        try:
            x_prev = np.asarray(x_prev).flatten()
            x = np.asarray(x).flatten()
            
            # Fast path for quadratic Lyapunov functions
            if self.delta_V_fn is not None and self.Q is not None:
                delta_V = self.delta_V_fn(x_prev, x, self.Q)
            else:
                # General path for any Lyapunov function
                v_prev = self.lyap.evaluate(x_prev)
                v = self.lyap.evaluate(x)
                delta_V = v - v_prev
            
            # Check stability
            if delta_V > self.threshold:
                self.violations += 1
                self.last_violation_time = time.time()
                
                # Call callback if provided
                if self.callback is not None:
                    self.callback(x_prev, x, self)
                
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in stability guard: {e}")
            return False
    
    def get_violation_rate(self) -> float:
        """
        Get violation rate.
        
        Returns:
            Violation rate (violations / total_steps)
        """
        if self.total_steps == 0:
            return 0.0
        return self.violations / self.total_steps


class CLFQuadraticProgramSolver:
    """
    Control Lyapunov Function (CLF) quadratic program solver.
    
    This class provides QP-based controllers that enforce stability
    using control Lyapunov functions.
    """
    
    def __init__(
        self,
        lyap: LyapunovFunction,
        control_dim: int,
        gamma: float = 0.1,
        relaxation_weight: float = 100.0,
        use_jit: bool = True
    ):
        """
        Initialize CLF-QP solver.
        
        Args:
            lyap: Lyapunov function
            control_dim: Control input dimension
            gamma: Convergence rate parameter
            relaxation_weight: Weight for relaxation variable
            use_jit: Whether to use JIT compilation
        """
        self.lyap = lyap
        self.control_dim = control_dim
        self.gamma = gamma
        self.relaxation_weight = relaxation_weight
        self.use_jit = use_jit and HAVE_NUMBA
        
        # Cache common matrices for efficiency
        try:
            import scipy.linalg as la
            self.have_scipy = True
        except ImportError:
            self.have_scipy = False
            logger.warning("SciPy not available, using NumPy instead")
        
        # Stats
        self.solve_times = []
        self.avg_solve_time = 0.0
    
    def step(
        self,
        x: np.ndarray,
        f_x: np.ndarray,
        g_x: np.ndarray
    ) -> np.ndarray:
        """
        Compute control input using CLF-QP.
        
        Args:
            x: Current state
            f_x: Drift dynamics f(x)
            g_x: Control dynamics g(x)
            
        Returns:
            Control input u
        """
        start_time = time.time()
        x = np.asarray(x).flatten()
        
        try:
            # Compute Lyapunov function and its gradient
            V_x = self.lyap.evaluate(x)
            
            # Approximate gradient using finite differences
            h = 1e-6
            grad_V = np.zeros_like(x)
            
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += h
                V_plus = self.lyap.evaluate(x_plus)
                grad_V[i] = (V_plus - V_x) / h
            
            # Lie derivatives: L_f V = ∇V·f, L_g V = ∇V·g
            L_f_V = grad_V @ f_x
            L_g_V = grad_V @ g_x
            
            # Set up and solve QP
            if self.have_scipy:
                u = self._solve_qp_scipy(L_f_V, L_g_V, V_x)
            else:
                # Fallback to simplified approach
                u = self._solve_simplified(L_f_V, L_g_V, V_x)
            
            # Update stats
            solve_time = time.time() - start_time
            self.solve_times.append(solve_time)
            if len(self.solve_times) > 100:
                self.solve_times.pop(0)
            self.avg_solve_time = np.mean(self.solve_times)
            
            return u
            
        except Exception as e:
            logger.error(f"Error in CLF-QP solver: {e}")
            return np.zeros(self.control_dim)
    
    def _solve_qp_scipy(
        self,
        L_f_V: float,
        L_g_V: np.ndarray,
        V_x: float
    ) -> np.ndarray:
        """
        Solve CLF-QP using SciPy.
        
        Args:
            L_f_V: Lie derivative L_f V
            L_g_V: Lie derivative L_g V
            V_x: Lyapunov function value V(x)
            
        Returns:
            Control input u
        """
        try:
            from scipy.optimize import minimize
            
            # QP objective: min_u 0.5 * u^T u + relaxation_weight * δ
            # Subject to: L_f V + L_g V u + γ V ≤ δ
            
            def objective(z):
                u = z[:-1]
                delta = z[-1]
                return 0.5 * u.T @ u + self.relaxation_weight * delta
            
            def constraint(z):
                u = z[:-1]
                delta = z[-1]
                return delta - (L_f_V + L_g_V @ u + self.gamma * V_x)
            
            # Initial guess
            z0 = np.zeros(self.control_dim + 1)
            
            # Solve
            constraints = [{'type': 'ineq', 'fun': constraint}]
            result = minimize(objective, z0, constraints=constraints)
            
            if result.success:
                return result.x[:-1]  # Return u
            else:
                logger.warning(f"QP solver failed: {result.message}")
                return np.zeros(self.control_dim)
                
        except Exception as e:
            logger.error(f"Error in QP solver: {e}")
            return np.zeros(self.control_dim)
    
    def _solve_simplified(
        self,
        L_f_V: float,
        L_g_V: np.ndarray,
        V_x: float
    ) -> np.ndarray:
        """
        Solve CLF constraint using simplified approach.
        
        Args:
            L_f_V: Lie derivative L_f V
            L_g_V: Lie derivative L_g V
            V_x: Lyapunov function value V(x)
            
        Returns:
            Control input u
        """
        if np.linalg.norm(L_g_V) < 1e-6:
            # L_g V ≈ 0, can't control in this direction
            return np.zeros(self.control_dim)
        
        # Compute minimum-norm u such that L_f V + L_g V u + γ V ≤ 0
        target = -L_f_V - self.gamma * V_x
        
        if L_f_V + self.gamma * V_x <= 0:
            # Already satisfies constraint, no control needed
            return np.zeros(self.control_dim)
        
        # Compute minimum-norm u
        u = -L_g_V * (target / np.sum(L_g_V**2))
        
        # Reshape if needed
        if len(u.shape) == 1 and self.control_dim == 1:
            u = u.reshape(-1, 1)
        
        return u


def run_demo():
    """Run a simple demonstration of JIT stability enforcement."""
    import matplotlib.pyplot as plt
    
    # Define a simple system: disturbed harmonic oscillator
    def harmonic_oscillator(x, u=None):
        """Harmonic oscillator: dx/dt = [0 1; -1 0]x + [0; 1]u."""
        A = np.array([
            [0.0, 1.0],
            [-1.0, 0.0]
        ])
        B = np.array([[0.0], [1.0]])
        
        drift = A @ x
        
        if u is not None:
            control = B @ u
            return drift + control
        
        return drift
    
    # Define quadratic Lyapunov function
    class QuadraticLyapunov(LyapunovFunction):
        def __init__(self, name, Q, domain_ids=None):
            super().__init__(name, domain_ids)
            self.Q = Q
            
        def evaluate(self, x):
            x = np.asarray(x).flatten()
            return float(x.T @ self.Q @ x)
        
        def get_quadratic_form(self):
            return self.Q
    
    # Create Lyapunov function
    Q = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    V = QuadraticLyapunov("V_harmonic", Q)
    
    # Define callback for stability violations
    def violation_callback(x_prev, x, guard):
        if guard.violations <= 3:  # Limit logging
            print(f"  Stability violation at x = {x}")
    
    # Create stability guard
    print("Creating JIT-compiled stability guard...")
    guard = StabilityGuard(
        V,
        threshold=1e-6,
        callback=violation_callback,
        use_jit=HAVE_NUMBA
    )
    
    # Warm up JIT compiler if available
    if HAVE_NUMBA:
        dummy_prev = np.array([0.1, 0.1])
        dummy = np.array([0.0, 0.0])
        _ = guard.step(dummy_prev, dummy)
        print("  JIT compilation successful")
    else:
        print("  Numba not available, using fallback implementation")
    
    # Create CLF-QP solver
    clf_qp = CLFQuadraticProgramSolver(
        V,
        control_dim=1,
        gamma=0.5,
        use_jit=HAVE_NUMBA
    )
    
    # Simulate without control
    print("\nSimulating system without control:")
    
    x = np.array([1.0, 0.0])
    uncontrolled_trajectory = [x.copy()]
    stability_status = []
    
    for i in range(50):
        # Compute next state
        x_prev = x.copy()
        
        # Add random disturbance for demonstration
        disturbance = np.random.normal(0, 0.1, size=2)
        x = x + 0.1 * harmonic_oscillator(x) + 0.02 * disturbance
        
        # Check stability
        is_stable = guard.step(x_prev, x)
        stability_status.append(is_stable)
        
        # Store state
        uncontrolled_trajectory.append(x.copy())
        
        if i % 10 == 0:
            print(f"  Step {i}: x = {x}, stable = {is_stable}")
    
    print(f"  Violation rate: {guard.get_violation_rate() * 100:.1f}%")
    
    # Reset guard stats
    guard.violations = 0
    guard.total_steps = 0
    
    # Simulate with control
    print("\nSimulating system with CLF-QP control:")
    
    x = np.array([1.0, 0.0])
    controlled_trajectory = [x.copy()]
    control_inputs = []
    controlled_stability = []
    
    for i in range(50):
        # Compute control input
        f_x = harmonic_oscillator(x)
        g_x = np.array([[0.0], [1.0]])
        
        u = clf_qp.step(x, f_x, g_x)
        control_inputs.append(u[0, 0] if u.shape == (1, 1) else u[0])
        
        # Compute next state
        x_prev = x.copy()
        
        # Add random disturbance for demonstration
        disturbance = np.random.normal(0, 0.1, size=2)
        x = x + 0.1 * harmonic_oscillator(x, u) + 0.02 * disturbance
        
        # Check stability
        is_stable = guard.step(x_prev, x)
        controlled_stability.append(is_stable)
        
        # Store state
        controlled_trajectory.append(x.copy())
        
        if i % 10 == 0:
            print(f"  Step {i}: x = {x}, u = {u.flatten()}, stable = {is_stable}")
    
    print(f"  Violation rate: {guard.get_violation_rate() * 100:.1f}%")
    print(f"  Average QP solve time: {clf_qp.avg_solve_time * 1000:.2f} ms")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Convert trajectories to arrays
    uncontrolled = np.array(uncontrolled_trajectory)
    controlled = np.array(controlled_trajectory)
    
    # Plot state trajectories
    plt.subplot(221)
    plt.plot(uncontrolled[:, 0], uncontrolled[:, 1], 'r.-', alpha=0.7, label='Uncontrolled')
    plt.plot(controlled[:, 0], controlled[:, 1], 'b.-', alpha=0.7, label='Controlled')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('State Trajectories')
    plt.grid(True)
    plt.legend()
    
    # Plot x1 over time
    plt.subplot(222)
    plt.plot(np.arange(len(uncontrolled)), uncontrolled[:, 0], 'r-', alpha=0.7, label='Uncontrolled')
    plt.plot(np.arange(len(controlled)), controlled[:, 0], 'b-', alpha=0.7, label='Controlled')
    plt.xlabel('Time Step')
    plt.ylabel('x1')
    plt.title('x1 vs Time')
    plt.grid(True)
    plt.legend()
    
    # Plot Lyapunov function values
    plt.subplot(223)
    v_uncontrolled = np.array([V.evaluate(x) for x in uncontrolled])
    v_controlled = np.array([V.evaluate(x) for x in controlled])
    
    plt.plot(np.arange(len(v_uncontrolled)), v_uncontrolled, 'r-', alpha=0.7, label='Uncontrolled')
    plt.plot(np.arange(len(v_controlled)), v_controlled, 'b-', alpha=0.7, label='Controlled')
    plt.xlabel('Time Step')
    plt.ylabel('V(x)')
    plt.title('Lyapunov Function Value')
    plt.grid(True)
    plt.legend()
    
    # Plot control inputs
    plt.subplot(224)
    plt.plot(np.arange(len(control_inputs)), control_inputs, 'g-')
    plt.xlabel('Time Step')
    plt.ylabel('Control Input u')
    plt.title('Control Inputs')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("jit_guard_demo.png")
    print("\nDemo results plotted to jit_guard_demo.png")


if __name__ == "__main__":
    run_demo()
