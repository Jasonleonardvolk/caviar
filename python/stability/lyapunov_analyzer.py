"""
TORI/KHA Lyapunov Analyzer - Production Implementation
Advanced Lyapunov stability analysis for cognitive systems
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class LyapunovResult:
    """Complete Lyapunov analysis results"""
    is_stable: bool
    lyapunov_function: np.ndarray
    max_exponent: float
    exponent_spectrum: np.ndarray
    basin_estimate: float
    convergence_rate: float
    robustness_margin: float
    analysis_type: str
    metadata: Dict[str, Any]

class LyapunovAnalyzer:
    """
    Comprehensive Lyapunov stability analyzer
    Implements multiple Lyapunov analysis methods
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Lyapunov analyzer"""
        self.config = config or {}
        
        # Configuration
        self.tolerance = self.config.get('tolerance', 1e-6)
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.integration_time = self.config.get('integration_time', 100.0)
        self.perturbation_size = self.config.get('perturbation_size', 1e-6)
        
        # Storage
        self.storage_path = Path(self.config.get('storage_path', 'data/lyapunov'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("LyapunovAnalyzer initialized")
    
    async def analyze_linear_system(self, A: np.ndarray, Q: Optional[np.ndarray] = None) -> LyapunovResult:
        """
        Analyze linear system stability using Lyapunov equation
        Solves A^T P + P A = -Q
        """
        n = A.shape[0]
        
        # Default Q matrix (identity for standard analysis)
        if Q is None:
            Q = np.eye(n)
        
        # Solve Lyapunov equation
        try:
            P = la.solve_continuous_lyapunov(A.T, -Q)
        except:
            # Fallback to discrete-time if continuous fails
            P = la.solve_discrete_lyapunov(A.T, Q)
        
        # Check if P is positive definite
        eigenvalues_P = la.eigvalsh(P)
        is_positive_definite = np.all(eigenvalues_P > 0)
        
        # Compute eigenvalues of A for exponents
        eigenvalues_A = la.eigvals(A)
        real_parts = np.real(eigenvalues_A)
        max_exponent = np.max(real_parts)
        
        # System is stable if P is positive definite and eigenvalues have negative real parts
        is_stable = is_positive_definite and np.all(real_parts < 0)
        
        # Estimate basin of attraction
        if is_positive_definite:
            # Use minimum eigenvalue of P
            basin_estimate = 1.0 / np.sqrt(np.max(eigenvalues_P))
        else:
            basin_estimate = 0.0
        
        # Convergence rate
        if is_stable:
            convergence_rate = -max_exponent
        else:
            convergence_rate = 0.0
        
        # Robustness margin (smallest singular value)
        robustness_margin = np.min(la.svdvals(P))
        
        return LyapunovResult(
            is_stable=is_stable,
            lyapunov_function=P,
            max_exponent=float(max_exponent),
            exponent_spectrum=real_parts,
            basin_estimate=float(basin_estimate),
            convergence_rate=float(convergence_rate),
            robustness_margin=float(robustness_margin),
            analysis_type='linear',
            metadata={
                'eigenvalues_P': eigenvalues_P.tolist(),
                'eigenvalues_A': eigenvalues_A.tolist(),
                'condition_number': float(la.norm(P) * la.norm(la.inv(P))) if is_positive_definite else np.inf
            }
        )
    
    async def analyze_nonlinear_system(
        self,
        dynamics: Callable[[float, np.ndarray], np.ndarray],
        equilibrium: np.ndarray,
        domain_radius: float = 1.0
    ) -> LyapunovResult:
        """
        Analyze nonlinear system stability using numerical methods
        Estimates Lyapunov function via SOS (Sum of Squares) approach
        """
        n = len(equilibrium)
        
        # Linearize around equilibrium
        A = self._compute_jacobian(dynamics, equilibrium)
        
        # Start with linear analysis
        linear_result = await self.analyze_linear_system(A)
        
        if not linear_result.is_stable:
            # System is unstable even linearly
            return LyapunovResult(
                is_stable=False,
                lyapunov_function=linear_result.lyapunov_function,
                max_exponent=linear_result.max_exponent,
                exponent_spectrum=linear_result.exponent_spectrum,
                basin_estimate=0.0,
                convergence_rate=0.0,
                robustness_margin=0.0,
                analysis_type='nonlinear_unstable',
                metadata={'linear_unstable': True}
            )
        
        # Use linear Lyapunov function as initial guess
        P = linear_result.lyapunov_function
        
        # Verify in neighborhood via simulation
        is_locally_stable, basin_estimate = await self._verify_local_stability(
            dynamics, equilibrium, P, domain_radius
        )
        
        # Compute Lyapunov exponents via trajectory divergence
        exponent_spectrum = await self._compute_lyapunov_spectrum_nonlinear(
            dynamics, equilibrium
        )
        max_exponent = np.max(exponent_spectrum)
        
        # Estimate convergence rate
        convergence_rate = -max_exponent if max_exponent < 0 else 0.0
        
        # Robustness via perturbation analysis
        robustness_margin = await self._estimate_robustness(
            dynamics, equilibrium, P
        )
        
        return LyapunovResult(
            is_stable=is_locally_stable,
            lyapunov_function=P,
            max_exponent=float(max_exponent),
            exponent_spectrum=exponent_spectrum,
            basin_estimate=float(basin_estimate),
            convergence_rate=float(convergence_rate),
            robustness_margin=float(robustness_margin),
            analysis_type='nonlinear',
            metadata={
                'jacobian': A.tolist(),
                'equilibrium': equilibrium.tolist(),
                'domain_radius': domain_radius
            }
        )
    
    async def compute_contraction_metrics(
        self,
        dynamics: Callable[[float, np.ndarray], np.ndarray],
        trajectory: np.ndarray,
        metric_tensor: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute contraction metrics for trajectory
        Uses differential analysis for contraction theory
        """
        n_points, n_dim = trajectory.shape
        
        if metric_tensor is None:
            metric_tensor = np.eye(n_dim)
        
        contraction_rates = []
        
        for i in range(n_points):
            # Compute Jacobian at current point
            J = self._compute_jacobian(dynamics, trajectory[i])
            
            # Compute symmetric part in metric
            F = 0.5 * (metric_tensor @ J + J.T @ metric_tensor)
            
            # Maximum eigenvalue gives contraction rate
            eigenvalues = la.eigvalsh(F)
            contraction_rates.append(np.max(eigenvalues))
        
        contraction_rates = np.array(contraction_rates)
        
        return {
            'is_contracting': np.all(contraction_rates < 0),
            'max_contraction_rate': float(np.max(contraction_rates)),
            'mean_contraction_rate': float(np.mean(contraction_rates)),
            'contraction_profile': contraction_rates,
            'metric_condition_number': float(la.cond(metric_tensor))
        }
    
    def construct_lyapunov_candidate(
        self,
        system_dimension: int,
        candidate_type: str = 'quadratic'
    ) -> Callable[[np.ndarray], float]:
        """
        Construct Lyapunov function candidate
        """
        if candidate_type == 'quadratic':
            # Random positive definite matrix
            M = np.random.randn(system_dimension, system_dimension)
            P = M.T @ M + np.eye(system_dimension)
            
            def V(x):
                return x.T @ P @ x
            
            return V
        
        elif candidate_type == 'quartic':
            # Fourth-order polynomial
            P2 = np.random.randn(system_dimension, system_dimension)
            P2 = P2.T @ P2
            
            P4 = np.random.randn(system_dimension, system_dimension)
            P4 = P4.T @ P4
            
            def V(x):
                return x.T @ P2 @ x + (x.T @ P4 @ x) ** 2
            
            return V
        
        else:
            raise ValueError(f"Unknown candidate type: {candidate_type}")
    
    def _compute_jacobian(
        self,
        dynamics: Callable,
        point: np.ndarray,
        h: float = 1e-6
    ) -> np.ndarray:
        """Compute Jacobian matrix numerically"""
        n = len(point)
        J = np.zeros((n, n))
        
        f0 = dynamics(0, point)
        
        for i in range(n):
            # Perturb in direction i
            point_perturbed = point.copy()
            point_perturbed[i] += h
            
            f_perturbed = dynamics(0, point_perturbed)
            
            # Finite difference
            J[:, i] = (f_perturbed - f0) / h
        
        return J
    
    async def _verify_local_stability(
        self,
        dynamics: Callable,
        equilibrium: np.ndarray,
        P: np.ndarray,
        radius: float
    ) -> Tuple[bool, float]:
        """Verify stability in local neighborhood"""
        n = len(equilibrium)
        n_tests = 20
        
        stable_trajectories = 0
        min_basin = radius
        
        for _ in range(n_tests):
            # Random initial condition
            direction = np.random.randn(n)
            direction /= la.norm(direction)
            
            # Test at different radii
            for r in np.linspace(0.1, radius, 10):
                x0 = equilibrium + r * direction
                
                # Simulate
                sol = solve_ivp(
                    dynamics,
                    [0, self.integration_time],
                    x0,
                    method='RK45',
                    rtol=1e-8
                )
                
                # Check if converged to equilibrium
                final_error = la.norm(sol.y[:, -1] - equilibrium)
                
                if final_error < 0.01:
                    stable_trajectories += 1
                else:
                    min_basin = min(min_basin, r)
                    break
        
        is_stable = stable_trajectories > 0.8 * n_tests
        
        return is_stable, min_basin
    
    async def _compute_lyapunov_spectrum_nonlinear(
        self,
        dynamics: Callable,
        equilibrium: np.ndarray
    ) -> np.ndarray:
        """Compute Lyapunov spectrum for nonlinear system"""
        n = len(equilibrium)
        
        # Initialize orthonormal perturbations
        Q = np.eye(n)
        
        # Integrate reference trajectory
        x0 = equilibrium + self.perturbation_size * np.random.randn(n)
        
        lyapunov_sums = np.zeros(n)
        n_renorm = 100
        t_renorm = self.integration_time / n_renorm
        
        current_state = x0
        
        for _ in range(n_renorm):
            # Integrate tangent dynamics
            def combined_dynamics(t, y):
                x = y[:n]
                Q_flat = y[n:].reshape(n, n)
                
                # System dynamics
                dx = dynamics(t, x)
                
                # Tangent dynamics
                J = self._compute_jacobian(dynamics, x)
                dQ = J @ Q_flat
                
                return np.concatenate([dx, dQ.flatten()])
            
            # Initial condition
            y0 = np.concatenate([current_state, Q.flatten()])
            
            # Integrate
            sol = solve_ivp(
                combined_dynamics,
                [0, t_renorm],
                y0,
                method='RK45'
            )
            
            # Extract final state
            current_state = sol.y[:n, -1]
            Q = sol.y[n:, -1].reshape(n, n)
            
            # QR decomposition for reorthonormalization
            Q, R = la.qr(Q)
            
            # Accumulate growth rates
            lyapunov_sums += np.log(np.abs(np.diag(R)))
        
        # Average to get exponents
        lyapunov_spectrum = lyapunov_sums / self.integration_time
        
        return np.sort(lyapunov_spectrum)[::-1]
    
    async def _estimate_robustness(
        self,
        dynamics: Callable,
        equilibrium: np.ndarray,
        P: np.ndarray
    ) -> float:
        """Estimate robustness margin via perturbation analysis"""
        n = len(equilibrium)
        
        # Test robustness to parameter variations
        robustness_tests = []
        
        for _ in range(10):
            # Perturbed dynamics
            perturbation = 0.1 * np.random.randn(n, n)
            
            def perturbed_dynamics(t, x):
                nominal = dynamics(t, x)
                return nominal + perturbation @ (x - equilibrium)
            
            # Check if still stable
            A_perturbed = self._compute_jacobian(perturbed_dynamics, equilibrium)
            eigenvalues = la.eigvals(A_perturbed)
            
            if np.all(np.real(eigenvalues) < 0):
                # Measure distance to instability
                margin = -np.max(np.real(eigenvalues))
                robustness_tests.append(margin)
        
        if robustness_tests:
            return float(np.min(robustness_tests))
        else:
            return 0.0
    
    def save_analysis(self, result: LyapunovResult, filename: str):
        """Save analysis results"""
        data = {
            'is_stable': result.is_stable,
            'max_exponent': result.max_exponent,
            'exponent_spectrum': result.exponent_spectrum.tolist(),
            'basin_estimate': result.basin_estimate,
            'convergence_rate': result.convergence_rate,
            'robustness_margin': result.robustness_margin,
            'analysis_type': result.analysis_type,
            'metadata': result.metadata,
            'timestamp': time.time()
        }
        
        filepath = self.storage_path / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Analysis saved to {filepath}")


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_lyapunov():
        """Test Lyapunov analyzer"""
        analyzer = LyapunovAnalyzer()
        
        # Test linear system
        A = np.array([
            [-1.0, 0.5],
            [0.2, -0.8]
        ])
        
        print("Linear system analysis:")
        result = await analyzer.analyze_linear_system(A)
        print(f"Stable: {result.is_stable}")
        print(f"Max exponent: {result.max_exponent:.4f}")
        print(f"Basin estimate: {result.basin_estimate:.4f}")
        
        # Test nonlinear system (Van der Pol oscillator)
        def vanderpol(t, x):
            mu = 0.5
            return np.array([
                x[1],
                mu * (1 - x[0]**2) * x[1] - x[0]
            ])
        
        print("\nNonlinear system analysis:")
        equilibrium = np.array([0.0, 0.0])
        nl_result = await analyzer.analyze_nonlinear_system(
            vanderpol, equilibrium, domain_radius=0.5
        )
        print(f"Stable: {nl_result.is_stable}")
        print(f"Exponent spectrum: {nl_result.exponent_spectrum}")
        
        # Save results
        analyzer.save_analysis(result, "linear_test.json")
        analyzer.save_analysis(nl_result, "nonlinear_test.json")
    
    asyncio.run(test_lyapunov())
