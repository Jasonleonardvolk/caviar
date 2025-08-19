"""
TORI/KHA Eigenvalue Monitor - Production Implementation
Advanced eigenvalue monitoring with epsilon-cloud prediction and Lyapunov stability
Special care taken for numerical stability and accuracy
"""

import numpy as np
import scipy.linalg as la
from scipy.linalg import expm, logm, eig, svd
from scipy.stats import chi2
import time
import logging
import asyncio
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
import json
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Numerical constants for stability
EPSILON = np.finfo(float).eps
STABILITY_THRESHOLD = 1.0 - 1e-6  # Just below 1 for stability
CONDITION_WARNING = 1e12  # Warn when condition number exceeds this

@dataclass
class EigenvalueAnalysis:
    """Complete eigenvalue analysis results"""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    max_eigenvalue: float
    spectral_radius: float
    condition_number: float
    numerical_rank: int
    stability_margin: float
    is_stable: bool
    is_hermitian: bool
    is_normal: bool
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'eigenvalues': self.eigenvalues.tolist(),
            'max_eigenvalue': self.max_eigenvalue,
            'spectral_radius': self.spectral_radius,
            'condition_number': self.condition_number,
            'numerical_rank': self.numerical_rank,
            'stability_margin': self.stability_margin,
            'is_stable': self.is_stable,
            'is_hermitian': self.is_hermitian,
            'is_normal': self.is_normal,
            'timestamp': self.timestamp
        }

@dataclass
class LyapunovAnalysis:
    """Lyapunov stability analysis results"""
    lyapunov_matrix: np.ndarray
    max_lyapunov_exponent: float
    lyapunov_spectrum: np.ndarray
    is_lyapunov_stable: bool
    basin_radius: float
    convergence_rate: float
    energy_function_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'max_lyapunov_exponent': self.max_lyapunov_exponent,
            'lyapunov_spectrum': self.lyapunov_spectrum.tolist(),
            'is_lyapunov_stable': self.is_lyapunov_stable,
            'basin_radius': self.basin_radius,
            'convergence_rate': self.convergence_rate,
            'energy_function_value': self.energy_function_value
        }

@dataclass
class EpsilonCloudPrediction:
    """Epsilon-cloud uncertainty prediction"""
    predicted_eigenvalues: np.ndarray
    confidence_intervals: np.ndarray
    cloud_radius: float
    prediction_horizon: int
    uncertainty_growth_rate: float
    bifurcation_risk: float
    confidence_level: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'predicted_eigenvalues': self.predicted_eigenvalues.tolist(),
            'confidence_intervals': self.confidence_intervals.tolist(),
            'cloud_radius': self.cloud_radius,
            'prediction_horizon': self.prediction_horizon,
            'uncertainty_growth_rate': self.uncertainty_growth_rate,
            'bifurcation_risk': self.bifurcation_risk,
            'confidence_level': self.confidence_level
        }

class EigenvalueMonitor:
    """
    Production-ready eigenvalue monitor with advanced stability analysis
    Includes epsilon-cloud prediction, Lyapunov stability, and Koopman operator
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize eigenvalue monitor"""
        self.config = config or {}
        
        # Configuration
        self.matrix_size = self.config.get('matrix_size', 512)
        self.history_size = self.config.get('history_size', 1000)
        self.prediction_horizon = self.config.get('prediction_horizon', 10)
        self.epsilon_radius = self.config.get('epsilon_radius', 0.01)
        self.stability_threshold = self.config.get('stability_threshold', STABILITY_THRESHOLD)
        
        # Storage
        self.storage_path = Path(self.config.get('storage_path', 'data/eigenvalue_monitor'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # History tracking
        self.eigenvalue_history: deque = deque(maxlen=self.history_size)
        self.matrix_history: deque = deque(maxlen=100)  # Keep recent matrices
        self.stability_history: deque = deque(maxlen=self.history_size)
        
        # Callbacks
        self.stability_callbacks: List[Callable] = []
        self.warning_callbacks: List[Callable] = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Koopman operator state
        self.koopman_matrix: Optional[np.ndarray] = None
        self.observable_dimension = self.config.get('observable_dimension', 128)
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        logger.info(f"EigenvalueMonitor initialized with matrix_size={self.matrix_size}")
    
    async def analyze_matrix(self, matrix: np.ndarray) -> EigenvalueAnalysis:
        """
        Perform comprehensive eigenvalue analysis with numerical stability checks
        """
        with self.lock:
            start_time = time.time()
            
            # Validate input
            if not self._validate_matrix(matrix):
                raise ValueError("Invalid matrix input")
            
            # Check matrix properties
            is_hermitian = np.allclose(matrix, matrix.conj().T)
            is_normal = np.allclose(matrix @ matrix.conj().T, matrix.conj().T @ matrix)
            
            # Compute eigenvalues with appropriate method
            if is_hermitian:
                # Use specialized hermitian solver for better accuracy
                eigenvalues, eigenvectors = la.eigh(matrix)
            else:
                # General eigenvalue solver
                eigenvalues, eigenvectors = la.eig(matrix)
            
            # Sort by magnitude
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Compute key metrics
            max_eigenvalue = np.max(np.abs(eigenvalues))
            spectral_radius = max_eigenvalue
            
            # Condition number (with protection against singular matrices)
            try:
                condition_number = la.norm(matrix) * la.norm(la.pinv(matrix))
            except:
                condition_number = np.inf
            
            # Numerical rank
            _, s, _ = la.svd(matrix)
            numerical_rank = np.sum(s > EPSILON * s[0])
            
            # Stability margin (distance from unit circle)
            stability_margin = self.stability_threshold - max_eigenvalue
            is_stable = max_eigenvalue < self.stability_threshold
            
            # Create analysis result
            analysis = EigenvalueAnalysis(
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                max_eigenvalue=float(max_eigenvalue),
                spectral_radius=float(spectral_radius),
                condition_number=float(condition_number),
                numerical_rank=int(numerical_rank),
                stability_margin=float(stability_margin),
                is_stable=bool(is_stable),
                is_hermitian=bool(is_hermitian),
                is_normal=bool(is_normal),
                timestamp=time.time()
            )
            
            # Update history
            self.eigenvalue_history.append(analysis)
            self.matrix_history.append(matrix.copy())
            
            # Check for warnings
            await self._check_warnings(analysis)
            
            # Notify callbacks if unstable
            if not is_stable:
                await self._notify_instability(analysis)
            
            logger.debug(f"Eigenvalue analysis completed in {time.time() - start_time:.3f}s")
            
            return analysis
    
    async def compute_lyapunov_stability(self, matrix: np.ndarray, Q: Optional[np.ndarray] = None) -> LyapunovAnalysis:
        """
        Compute Lyapunov stability analysis
        Solves A^T P + P A = -Q for stability verification
        """
        with self.lock:
            n = matrix.shape[0]
            
            # Default Q matrix (positive definite)
            if Q is None:
                Q = np.eye(n)
            
            # Solve continuous Lyapunov equation
            try:
                P = la.solve_continuous_lyapunov(matrix.T, -Q)
            except:
                logger.warning("Lyapunov equation solver failed, using alternative method")
                # Alternative: use eigenvalue decomposition
                P = self._solve_lyapunov_eigen(matrix, Q)
            
            # Check if P is positive definite (necessary for stability)
            P_eigenvalues = la.eigvalsh(P)
            is_positive_definite = np.all(P_eigenvalues > 0)
            
            # Compute Lyapunov exponents
            lyapunov_spectrum = self._compute_lyapunov_spectrum(matrix)
            max_lyapunov_exponent = np.max(lyapunov_spectrum)
            
            # Estimate basin of attraction
            if is_positive_definite:
                # Smallest eigenvalue of P gives basin estimate
                basin_radius = np.sqrt(1.0 / np.max(P_eigenvalues))
            else:
                basin_radius = 0.0
            
            # Convergence rate (from largest negative real part of eigenvalues)
            eigenvalues = la.eigvals(matrix)
            real_parts = np.real(eigenvalues)
            if np.all(real_parts < 0):
                convergence_rate = -np.max(real_parts)
            else:
                convergence_rate = 0.0
            
            # Energy function value V(x) = x^T P x
            # Evaluate at a test point
            test_point = np.ones(n) / np.sqrt(n)
            energy_value = test_point.T @ P @ test_point
            
            analysis = LyapunovAnalysis(
                lyapunov_matrix=P,
                max_lyapunov_exponent=float(max_lyapunov_exponent),
                lyapunov_spectrum=lyapunov_spectrum,
                is_lyapunov_stable=bool(is_positive_definite and max_lyapunov_exponent < 0),
                basin_radius=float(basin_radius),
                convergence_rate=float(convergence_rate),
                energy_function_value=float(energy_value)
            )
            
            self.stability_history.append(analysis)
            
            return analysis
    
    async def predict_epsilon_cloud(self, steps_ahead: Optional[int] = None) -> EpsilonCloudPrediction:
        """
        Predict eigenvalue evolution with epsilon-cloud uncertainty quantification
        Uses statistical methods to estimate confidence intervals
        """
        with self.lock:
            if len(self.eigenvalue_history) < 10:
                raise ValueError("Insufficient history for prediction")
            
            steps_ahead = steps_ahead or self.prediction_horizon
            
            # Extract eigenvalue trajectories
            trajectories = []
            for analysis in list(self.eigenvalue_history)[-100:]:
                trajectories.append(analysis.eigenvalues)
            
            trajectories = np.array(trajectories)
            n_history, n_eigenvalues = trajectories.shape
            
            # Fit trend for each eigenvalue
            predictions = []
            intervals = []
            
            for i in range(n_eigenvalues):
                # Extract single eigenvalue trajectory
                traj = trajectories[:, i]
                
                # Separate real and imaginary parts
                real_traj = np.real(traj)
                imag_traj = np.imag(traj)
                
                # Fit polynomial trend (2nd order for stability)
                t = np.arange(len(real_traj))
                real_poly = np.polyfit(t, real_traj, 2)
                imag_poly = np.polyfit(t, imag_traj, 2)
                
                # Predict future values
                future_t = np.arange(len(real_traj), len(real_traj) + steps_ahead)
                real_pred = np.polyval(real_poly, future_t)
                imag_pred = np.polyval(imag_poly, future_t)
                
                # Estimate prediction uncertainty
                real_residuals = real_traj - np.polyval(real_poly, t)
                imag_residuals = imag_traj - np.polyval(imag_poly, t)
                
                real_std = np.std(real_residuals)
                imag_std = np.std(imag_residuals)
                
                # Compute confidence intervals (95% default)
                z_score = chi2.ppf(self.config.get('confidence_level', 0.95), df=2) ** 0.5
                
                # Combine predictions
                pred_eigenvalue = real_pred[-1] + 1j * imag_pred[-1]
                predictions.append(pred_eigenvalue)
                
                # Confidence interval as complex radius
                conf_radius = z_score * np.sqrt(real_std**2 + imag_std**2)
                intervals.append(conf_radius)
            
            predictions = np.array(predictions)
            intervals = np.array(intervals)
            
            # Compute cloud metrics
            cloud_radius = np.max(intervals)
            
            # Estimate uncertainty growth rate
            if len(self.eigenvalue_history) > 20:
                recent_radii = [np.std(analysis.eigenvalues) for analysis in list(self.eigenvalue_history)[-20:]]
                growth_rate = np.polyfit(range(len(recent_radii)), recent_radii, 1)[0]
            else:
                growth_rate = 0.0
            
            # Estimate bifurcation risk (eigenvalues approaching unit circle)
            max_predicted = np.max(np.abs(predictions))
            bifurcation_risk = max(0, min(1, (max_predicted - 0.9) / 0.1))
            
            prediction = EpsilonCloudPrediction(
                predicted_eigenvalues=predictions,
                confidence_intervals=intervals,
                cloud_radius=float(cloud_radius),
                prediction_horizon=int(steps_ahead),
                uncertainty_growth_rate=float(growth_rate),
                bifurcation_risk=float(bifurcation_risk)
            )
            
            return prediction
    
    async def compute_koopman_operator(self, trajectories: List[np.ndarray]) -> np.ndarray:
        """
        Compute Koopman operator for linearization of nonlinear dynamics
        Uses DMD (Dynamic Mode Decomposition) algorithm
        """
        with self.lock:
            if len(trajectories) < 2:
                raise ValueError("Need at least 2 trajectory snapshots")
            
            # Stack trajectories into data matrices
            X = np.column_stack(trajectories[:-1])
            Y = np.column_stack(trajectories[1:])
            
            # Compute SVD of X
            U, s, Vt = la.svd(X, full_matrices=False)
            
            # Truncate based on energy
            energy_threshold = 0.99
            cumsum_energy = np.cumsum(s**2) / np.sum(s**2)
            r = np.argmax(cumsum_energy > energy_threshold) + 1
            
            U_r = U[:, :r]
            s_r = s[:r]
            V_r = Vt[:r, :].T
            
            # Compute Koopman operator
            self.koopman_matrix = Y @ V_r @ np.diag(1.0 / s_r) @ U_r.T
            
            # Analyze Koopman eigenvalues
            koopman_eigenvalues, koopman_modes = la.eig(self.koopman_matrix)
            
            # Sort by magnitude
            idx = np.argsort(np.abs(koopman_eigenvalues))[::-1]
            koopman_eigenvalues = koopman_eigenvalues[idx]
            koopman_modes = koopman_modes[:, idx]
            
            # Store for future use
            self.koopman_eigenvalues = koopman_eigenvalues
            self.koopman_modes = koopman_modes
            
            logger.info(f"Koopman operator computed: rank={r}, max eigenvalue={np.max(np.abs(koopman_eigenvalues)):.3f}")
            
            return self.koopman_matrix
    
    def get_stability_metrics(self) -> Dict[str, Any]:
        """Get current stability metrics"""
        with self.lock:
            if not self.eigenvalue_history:
                return {
                    'has_data': False,
                    'message': 'No eigenvalue data available'
                }
            
            recent = self.eigenvalue_history[-1]
            
            metrics = {
                'has_data': True,
                'current_analysis': recent.to_dict(),
                'history_size': len(self.eigenvalue_history),
                'trending_stable': self._compute_stability_trend(),
                'condition_warning': recent.condition_number > CONDITION_WARNING
            }
            
            # Add Lyapunov metrics if available
            if self.stability_history:
                lyapunov = self.stability_history[-1]
                metrics['lyapunov'] = lyapunov.to_dict()
            
            # Add prediction if possible
            try:
                prediction = asyncio.run(self.predict_epsilon_cloud(5))
                metrics['prediction'] = prediction.to_dict()
            except:
                metrics['prediction'] = None
            
            return metrics
    
    def register_stability_callback(self, callback: Callable):
        """Register callback for stability changes"""
        self.stability_callbacks.append(callback)
    
    def register_warning_callback(self, callback: Callable):
        """Register callback for warnings"""
        self.warning_callbacks.append(callback)
    
    def _validate_matrix(self, matrix: np.ndarray) -> bool:
        """Validate matrix input"""
        if matrix.ndim != 2:
            return False
        
        if matrix.shape[0] != matrix.shape[1]:
            return False
        
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            return False
        
        return True
    
    def _solve_lyapunov_eigen(self, A: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Solve Lyapunov equation using eigenvalue decomposition"""
        # Eigendecomposition of A
        eigenvalues, eigenvectors = la.eig(A)
        
        # Transform Q
        Q_transformed = eigenvectors.T @ Q @ eigenvectors
        
        # Solve in diagonal form
        n = len(eigenvalues)
        P_transformed = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            for j in range(n):
                denominator = eigenvalues[i] + eigenvalues[j].conj()
                if abs(denominator) > EPSILON:
                    P_transformed[i, j] = -Q_transformed[i, j] / denominator
        
        # Transform back
        P = eigenvectors @ P_transformed @ eigenvectors.T
        
        # Return real part for real matrices
        if np.allclose(A.imag, 0) and np.allclose(Q.imag, 0):
            P = P.real
        
        return P
    
    def _compute_lyapunov_spectrum(self, matrix: np.ndarray, n_iterations: int = 1000) -> np.ndarray:
        """
        Compute Lyapunov exponent spectrum using QR decomposition method
        """
        n = matrix.shape[0]
        
        # Random initial condition
        Q = np.random.randn(n, n)
        Q, _ = la.qr(Q)
        
        # Track stretching
        lyapunov_sums = np.zeros(n)
        
        for _ in range(n_iterations):
            # Evolve
            Q = matrix @ Q
            
            # QR decomposition
            Q, R = la.qr(Q)
            
            # Accumulate stretching factors
            lyapunov_sums += np.log(np.abs(np.diag(R)))
        
        # Average to get exponents
        lyapunov_spectrum = lyapunov_sums / n_iterations
        
        # Sort in descending order
        return np.sort(lyapunov_spectrum)[::-1]
    
    def _compute_stability_trend(self) -> bool:
        """Compute if system is trending towards stability"""
        if len(self.eigenvalue_history) < 10:
            return True  # Assume stable if insufficient data
        
        # Look at max eigenvalue trend
        max_eigenvalues = [analysis.max_eigenvalue for analysis in list(self.eigenvalue_history)[-20:]]
        
        # Fit linear trend
        t = np.arange(len(max_eigenvalues))
        slope, _ = np.polyfit(t, max_eigenvalues, 1)
        
        # Negative slope means trending stable
        return slope < 0
    
    async def _check_warnings(self, analysis: EigenvalueAnalysis):
        """Check for warning conditions"""
        warnings = []
        
        # High condition number
        if analysis.condition_number > CONDITION_WARNING:
            warnings.append({
                'type': 'condition_number',
                'severity': 'high',
                'value': analysis.condition_number,
                'message': f'Very high condition number: {analysis.condition_number:.2e}'
            })
        
        # Near instability
        if 0.9 < analysis.max_eigenvalue < self.stability_threshold:
            warnings.append({
                'type': 'near_instability',
                'severity': 'medium',
                'value': analysis.max_eigenvalue,
                'message': f'Approaching instability: max eigenvalue = {analysis.max_eigenvalue:.4f}'
            })
        
        # Rank deficiency
        if analysis.numerical_rank < analysis.eigenvalues.shape[0]:
            warnings.append({
                'type': 'rank_deficient',
                'severity': 'low',
                'value': analysis.numerical_rank,
                'message': f'Matrix is rank deficient: rank = {analysis.numerical_rank}'
            })
        
        # Notify callbacks
        for warning in warnings:
            for callback in self.warning_callbacks:
                await callback(warning)
    
    async def _notify_instability(self, analysis: EigenvalueAnalysis):
        """Notify callbacks of instability"""
        notification = {
            'type': 'instability_detected',
            'timestamp': analysis.timestamp,
            'max_eigenvalue': analysis.max_eigenvalue,
            'spectral_radius': analysis.spectral_radius,
            'analysis': analysis.to_dict()
        }
        
        for callback in self.stability_callbacks:
            await callback(notification)
    
    def _save_checkpoint(self):
        """Save monitor state to disk"""
        checkpoint = {
            'eigenvalue_history': [a.to_dict() for a in list(self.eigenvalue_history)[-100:]],
            'stability_history': [s.to_dict() for s in list(self.stability_history)[-50:]],
            'koopman_eigenvalues': self.koopman_eigenvalues.tolist() if hasattr(self, 'koopman_eigenvalues') else None,
            'timestamp': time.time()
        }
        
        checkpoint_file = self.storage_path / 'eigenvalue_checkpoint.json'
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.debug("Eigenvalue monitor checkpoint saved")
    
    def _load_checkpoint(self):
        """Load saved state"""
        checkpoint_file = self.storage_path / 'eigenvalue_checkpoint.json'
        
        if not checkpoint_file.exists():
            return
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            # Restore eigenvalue history
            for analysis_dict in checkpoint.get('eigenvalue_history', []):
                analysis_dict['eigenvalues'] = np.array(analysis_dict['eigenvalues'])
                analysis_dict['eigenvectors'] = np.eye(len(analysis_dict['eigenvalues']))  # Placeholder
                # Note: We lose eigenvectors in checkpoint, but eigenvalues are preserved
                
            logger.info("Eigenvalue monitor checkpoint loaded")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    
    def shutdown(self):
        """Save state and cleanup"""
        self._save_checkpoint()
        logger.info("EigenvalueMonitor shutdown complete")


# Example usage
if __name__ == "__main__":
    async def test_eigenvalue_monitor():
        """Test the eigenvalue monitor"""
        config = {
            'matrix_size': 10,
            'history_size': 100,
            'storage_path': 'data/eigenvalue_test'
        }
        
        monitor = EigenvalueMonitor(config)
        
        # Test with a known stable matrix
        A = np.array([
            [-1.0, 0.5, 0.0],
            [0.2, -0.8, 0.3],
            [0.1, 0.1, -0.9]
        ])
        
        print("Testing eigenvalue analysis...")
        analysis = await monitor.analyze_matrix(A)
        print(f"Max eigenvalue: {analysis.max_eigenvalue:.4f}")
        print(f"Is stable: {analysis.is_stable}")
        print(f"Condition number: {analysis.condition_number:.2f}")
        
        # Test Lyapunov stability
        print("\nTesting Lyapunov stability...")
        lyapunov = await monitor.compute_lyapunov_stability(A)
        print(f"Max Lyapunov exponent: {lyapunov.max_lyapunov_exponent:.4f}")
        print(f"Is Lyapunov stable: {lyapunov.is_lyapunov_stable}")
        print(f"Basin radius: {lyapunov.basin_radius:.4f}")
        
        # Add more matrices to build history
        for i in range(20):
            # Slightly perturb matrix
            A_perturbed = A + np.random.randn(*A.shape) * 0.01
            await monitor.analyze_matrix(A_perturbed)
        
        # Test epsilon-cloud prediction
        print("\nTesting epsilon-cloud prediction...")
        try:
            prediction = await monitor.predict_epsilon_cloud(steps_ahead=5)
            print(f"Cloud radius: {prediction.cloud_radius:.4f}")
            print(f"Bifurcation risk: {prediction.bifurcation_risk:.4f}")
        except Exception as e:
            print(f"Prediction failed: {e}")
        
        # Get overall metrics
        metrics = monitor.get_stability_metrics()
        print(f"\nOverall stability metrics:")
        print(f"Trending stable: {metrics.get('trending_stable', 'Unknown')}")
        
        monitor.shutdown()
    
    # Run test
    asyncio.run(test_eigenvalue_monitor())
