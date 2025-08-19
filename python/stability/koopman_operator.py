"""
TORI/KHA Koopman Operator - Production Implementation
Dynamic Mode Decomposition and Koopman analysis for nonlinear systems
"""

import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class KoopmanAnalysis:
    """Koopman operator analysis results"""
    koopman_matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenfunctions: np.ndarray
    modes: np.ndarray
    amplitudes: np.ndarray
    growth_rates: np.ndarray
    frequencies: np.ndarray
    reconstruction_error: float
    rank: int
    metadata: Dict[str, Any]

class KoopmanOperator:
    """
    Koopman operator computation and analysis
    Implements DMD, Extended DMD, and kernel-based methods
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Koopman operator analyzer"""
        self.config = config or {}
        
        # Configuration
        self.rank_threshold = self.config.get('rank_threshold', 0.99)  # Energy threshold
        self.regularization = self.config.get('regularization', 1e-10)
        self.kernel_type = self.config.get('kernel_type', 'rbf')
        self.kernel_bandwidth = self.config.get('kernel_bandwidth', 1.0)
        self.max_rank = self.config.get('max_rank', None)
        
        # Storage
        self.storage_path = Path(self.config.get('storage_path', 'data/koopman'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("KoopmanOperator initialized")
    
    async def compute_dmd(
        self,
        snapshots: np.ndarray,
        dt: float = 1.0
    ) -> KoopmanAnalysis:
        """
        Compute Dynamic Mode Decomposition (DMD)
        Standard algorithm for Koopman operator approximation
        """
        # Validate input
        if snapshots.shape[1] < 2:
            raise ValueError("Need at least 2 snapshots")
        
        # Split data into X and Y
        X = snapshots[:, :-1]
        Y = snapshots[:, 1:]
        
        # Compute SVD of X
        U, s, Vt = la.svd(X, full_matrices=False)
        V = Vt.T
        
        # Determine rank based on energy threshold
        cumsum_energy = np.cumsum(s**2) / np.sum(s**2)
        if self.max_rank:
            r = min(self.max_rank, len(s))
        else:
            r = np.argmax(cumsum_energy >= self.rank_threshold) + 1
        
        # Truncate
        U_r = U[:, :r]
        s_r = s[:r]
        V_r = V[:, :r]
        
        # Compute Koopman matrix in reduced space
        A_tilde = U_r.T @ Y @ V_r @ np.diag(1.0 / s_r)
        
        # Eigendecomposition
        eigenvalues, W = la.eig(A_tilde)
        
        # Sort by magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        W = W[:, idx]
        
        # Compute DMD modes
        modes = Y @ V_r @ np.diag(1.0 / s_r) @ W
        
        # Normalize modes
        for i in range(modes.shape[1]):
            modes[:, i] /= la.norm(modes[:, i])
        
        # Compute amplitudes
        amplitudes = la.lstsq(modes, X[:, 0], rcond=None)[0]
        
        # Growth rates and frequencies
        continuous_eigenvalues = np.log(eigenvalues) / dt
        growth_rates = np.real(continuous_eigenvalues)
        frequencies = np.imag(continuous_eigenvalues) / (2 * np.pi)
        
        # Reconstruction error
        X_reconstructed = self._reconstruct_dmd(modes, eigenvalues, amplitudes, X.shape[1])
        reconstruction_error = la.norm(X - X_reconstructed) / la.norm(X)
        
        # Full Koopman matrix (optional, for small systems)
        if X.shape[0] < 1000:
            K_full = modes @ np.diag(eigenvalues) @ la.pinv(modes)
        else:
            K_full = None
        
        return KoopmanAnalysis(
            koopman_matrix=K_full if K_full is not None else A_tilde,
            eigenvalues=eigenvalues,
            eigenfunctions=W,
            modes=modes,
            amplitudes=amplitudes,
            growth_rates=growth_rates,
            frequencies=frequencies,
            reconstruction_error=float(reconstruction_error),
            rank=r,
            metadata={
                'method': 'DMD',
                'dt': dt,
                'energy_captured': float(cumsum_energy[r-1]),
                'singular_values': s_r.tolist()
            }
        )
    
    async def compute_extended_dmd(
        self,
        snapshots: np.ndarray,
        observables: Callable[[np.ndarray], np.ndarray],
        dt: float = 1.0
    ) -> KoopmanAnalysis:
        """
        Extended Dynamic Mode Decomposition
        Uses dictionary of observables for better approximation
        """
        # Apply observables to snapshots
        n_snapshots = snapshots.shape[1]
        
        # Evaluate observables on first snapshot to get dimension
        psi_0 = observables(snapshots[:, 0])
        n_observables = len(psi_0)
        
        # Build observable data matrices
        Psi_X = np.zeros((n_observables, n_snapshots - 1))
        Psi_Y = np.zeros((n_observables, n_snapshots - 1))
        
        for i in range(n_snapshots - 1):
            Psi_X[:, i] = observables(snapshots[:, i])
            Psi_Y[:, i] = observables(snapshots[:, i + 1])
        
        # Compute Koopman matrix via least squares
        K = Psi_Y @ la.pinv(Psi_X, rcond=self.regularization)
        
        # Eigendecomposition
        eigenvalues, eigenfunctions = la.eig(K)
        
        # Sort by magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenfunctions = eigenfunctions[:, idx]
        
        # Compute modes in original space
        # Project eigenfunctions back to state space
        modes = snapshots[:, :-1] @ Psi_X.T @ eigenfunctions
        
        # Normalize
        for i in range(modes.shape[1]):
            if la.norm(modes[:, i]) > 0:
                modes[:, i] /= la.norm(modes[:, i])
        
        # Compute amplitudes
        amplitudes = la.lstsq(modes, snapshots[:, 0], rcond=None)[0]
        
        # Growth rates and frequencies
        continuous_eigenvalues = np.log(eigenvalues + 1e-10) / dt
        growth_rates = np.real(continuous_eigenvalues)
        frequencies = np.imag(continuous_eigenvalues) / (2 * np.pi)
        
        # Reconstruction error
        Psi_reconstructed = self._reconstruct_extended_dmd(
            K, Psi_X[:, 0], n_snapshots - 1
        )
        reconstruction_error = la.norm(Psi_Y - Psi_reconstructed) / la.norm(Psi_Y)
        
        return KoopmanAnalysis(
            koopman_matrix=K,
            eigenvalues=eigenvalues,
            eigenfunctions=eigenfunctions,
            modes=modes,
            amplitudes=amplitudes,
            growth_rates=growth_rates,
            frequencies=frequencies,
            reconstruction_error=float(reconstruction_error),
            rank=len(eigenvalues),
            metadata={
                'method': 'ExtendedDMD',
                'dt': dt,
                'n_observables': n_observables
            }
        )
    
    async def compute_kernel_dmd(
        self,
        snapshots: np.ndarray,
        dt: float = 1.0
    ) -> KoopmanAnalysis:
        """
        Kernel Dynamic Mode Decomposition
        Uses kernel trick for implicit feature space
        """
        X = snapshots[:, :-1]
        Y = snapshots[:, 1:]
        n = X.shape[1]
        
        # Compute kernel matrices
        K_XX = self._compute_kernel_matrix(X, X)
        K_YX = self._compute_kernel_matrix(Y, X)
        
        # Regularize
        K_XX_reg = K_XX + self.regularization * np.eye(n)
        
        # Solve for coefficients
        C = la.solve(K_XX_reg, K_YX.T).T
        
        # Eigendecomposition of C
        eigenvalues, alpha = la.eig(C)
        
        # Sort
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        alpha = alpha[:, idx]
        
        # Normalize eigenvectors
        for i in range(alpha.shape[1]):
            norm = np.sqrt(alpha[:, i].T @ K_XX @ alpha[:, i])
            if norm > 0:
                alpha[:, i] /= norm
        
        # Compute modes (approximate via linear combination)
        modes = X @ alpha
        
        # Growth rates and frequencies
        continuous_eigenvalues = np.log(eigenvalues + 1e-10) / dt
        growth_rates = np.real(continuous_eigenvalues)
        frequencies = np.imag(continuous_eigenvalues) / (2 * np.pi)
        
        # Reconstruction error (in feature space)
        K_Y_pred = K_XX @ C.T
        reconstruction_error = la.norm(K_YX - K_Y_pred.T) / la.norm(K_YX)
        
        return KoopmanAnalysis(
            koopman_matrix=C,
            eigenvalues=eigenvalues,
            eigenfunctions=alpha,
            modes=modes,
            amplitudes=alpha[0, :],  # First coefficients as amplitudes
            growth_rates=growth_rates,
            frequencies=frequencies,
            reconstruction_error=float(reconstruction_error),
            rank=len(eigenvalues),
            metadata={
                'method': 'KernelDMD',
                'dt': dt,
                'kernel_type': self.kernel_type,
                'kernel_bandwidth': self.kernel_bandwidth
            }
        )
    
    def predict_evolution(
        self,
        analysis: KoopmanAnalysis,
        initial_condition: np.ndarray,
        time_steps: int,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Predict future evolution using Koopman modes
        """
        # Project initial condition onto modes
        if analysis.modes.shape[1] > 0:
            b = la.lstsq(analysis.modes, initial_condition, rcond=None)[0]
        else:
            b = analysis.amplitudes
        
        # Time evolution
        predictions = np.zeros((len(initial_condition), time_steps))
        
        for t in range(time_steps):
            # Koopman evolution
            time_scaled = t * dt
            evolution = np.zeros_like(initial_condition, dtype=complex)
            
            for i in range(len(analysis.eigenvalues)):
                evolution += b[i] * analysis.modes[:, i] * (analysis.eigenvalues[i] ** t)
            
            predictions[:, t] = np.real(evolution)
        
        return predictions
    
    def compute_observables_polynomial(
        self,
        state: np.ndarray,
        max_degree: int = 3
    ) -> np.ndarray:
        """
        Polynomial observable dictionary
        """
        n = len(state)
        observables = [1.0]  # Constant
        
        # Linear terms
        observables.extend(state)
        
        # Higher order terms
        if max_degree >= 2:
            # Quadratic
            for i in range(n):
                for j in range(i, n):
                    observables.append(state[i] * state[j])
        
        if max_degree >= 3:
            # Cubic
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        observables.append(state[i] * state[j] * state[k])
        
        return np.array(observables)
    
    def compute_observables_rbf(
        self,
        state: np.ndarray,
        centers: np.ndarray,
        bandwidth: float = 1.0
    ) -> np.ndarray:
        """
        Radial basis function observable dictionary
        """
        distances = np.linalg.norm(state[:, np.newaxis] - centers.T, axis=0)
        return np.exp(-distances**2 / (2 * bandwidth**2))
    
    def _compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between data matrices"""
        if self.kernel_type == 'rbf':
            # RBF kernel
            n_x = X.shape[1]
            n_y = Y.shape[1]
            K = np.zeros((n_x, n_y))
            
            for i in range(n_x):
                for j in range(n_y):
                    dist = la.norm(X[:, i] - Y[:, j])
                    K[i, j] = np.exp(-dist**2 / (2 * self.kernel_bandwidth**2))
            
            return K
        
        elif self.kernel_type == 'polynomial':
            # Polynomial kernel
            degree = self.config.get('kernel_degree', 3)
            return (1 + X.T @ Y) ** degree
        
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _reconstruct_dmd(
        self,
        modes: np.ndarray,
        eigenvalues: np.ndarray,
        amplitudes: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """Reconstruct data using DMD modes"""
        reconstruction = np.zeros((modes.shape[0], n_steps), dtype=complex)
        
        for t in range(n_steps):
            for i in range(len(eigenvalues)):
                reconstruction[:, t] += amplitudes[i] * modes[:, i] * (eigenvalues[i] ** t)
        
        return np.real(reconstruction)
    
    def _reconstruct_extended_dmd(
        self,
        K: np.ndarray,
        psi_0: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """Reconstruct observable evolution"""
        reconstruction = np.zeros((len(psi_0), n_steps))
        reconstruction[:, 0] = psi_0
        
        current = psi_0.copy()
        for t in range(1, n_steps):
            current = K @ current
            reconstruction[:, t] = current
        
        return reconstruction
    
    def analyze_stability(self, analysis: KoopmanAnalysis) -> Dict[str, Any]:
        """Analyze stability from Koopman eigenvalues"""
        eigenvalues = analysis.eigenvalues
        
        # Stability metrics
        max_magnitude = np.max(np.abs(eigenvalues))
        is_stable = max_magnitude <= 1.0
        
        # Find dominant modes
        dominant_indices = np.argsort(np.abs(eigenvalues))[::-1][:5]
        dominant_eigenvalues = eigenvalues[dominant_indices]
        dominant_frequencies = analysis.frequencies[dominant_indices]
        dominant_growth_rates = analysis.growth_rates[dominant_indices]
        
        # Spectral gap (distance between largest and second largest)
        if len(eigenvalues) > 1:
            spectral_gap = np.abs(eigenvalues[dominant_indices[0]]) - np.abs(eigenvalues[dominant_indices[1]])
        else:
            spectral_gap = 0.0
        
        return {
            'is_stable': is_stable,
            'max_magnitude': float(max_magnitude),
            'spectral_gap': float(spectral_gap),
            'dominant_eigenvalues': dominant_eigenvalues.tolist(),
            'dominant_frequencies': dominant_frequencies.tolist(),
            'dominant_growth_rates': dominant_growth_rates.tolist(),
            'stability_margin': float(1.0 - max_magnitude)
        }
    
    def extract_coherent_structures(
        self,
        analysis: KoopmanAnalysis,
        threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Extract coherent structures from Koopman modes"""
        structures = []
        
        for i in range(len(analysis.eigenvalues)):
            # Skip if growth rate too large (unstable)
            if analysis.growth_rates[i] > threshold:
                continue
            
            # Skip if amplitude too small
            if np.abs(analysis.amplitudes[i]) < 1e-6:
                continue
            
            structure = {
                'mode_index': i,
                'eigenvalue': complex(analysis.eigenvalues[i]),
                'frequency': float(analysis.frequencies[i]),
                'growth_rate': float(analysis.growth_rates[i]),
                'amplitude': complex(analysis.amplitudes[i]),
                'mode': analysis.modes[:, i].tolist(),
                'period': 1.0 / analysis.frequencies[i] if analysis.frequencies[i] != 0 else np.inf
            }
            
            structures.append(structure)
        
        # Sort by amplitude (most energetic first)
        structures.sort(key=lambda x: np.abs(x['amplitude']), reverse=True)
        
        return structures
    
    def save_analysis(self, analysis: KoopmanAnalysis, filename: str):
        """Save Koopman analysis results"""
        # Convert to serializable format
        data = {
            'eigenvalues': analysis.eigenvalues.tolist(),
            'growth_rates': analysis.growth_rates.tolist(),
            'frequencies': analysis.frequencies.tolist(),
            'amplitudes': analysis.amplitudes.tolist(),
            'reconstruction_error': analysis.reconstruction_error,
            'rank': analysis.rank,
            'metadata': analysis.metadata,
            'stability': self.analyze_stability(analysis),
            'timestamp': time.time()
        }
        
        filepath = self.storage_path / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save modes separately (can be large)
        modes_file = self.storage_path / f"{filename.split('.')[0]}_modes.npy"
        np.save(modes_file, analysis.modes)
        
        logger.info(f"Koopman analysis saved to {filepath}")
    
    def load_analysis(self, filename: str) -> Dict[str, Any]:
        """Load saved Koopman analysis"""
        filepath = self.storage_path / filename
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load modes if available
        modes_file = self.storage_path / f"{filename.split('.')[0]}_modes.npy"
        if modes_file.exists():
            data['modes'] = np.load(modes_file)
        
        return data


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_koopman():
        """Test Koopman operator analysis"""
        # Create test data - nonlinear oscillator
        def simulate_vanderpol(mu=1.0, dt=0.1, n_steps=1000):
            x = np.zeros((2, n_steps))
            x[:, 0] = [2.0, 0.0]  # Initial condition
            
            for i in range(1, n_steps):
                x1, x2 = x[:, i-1]
                dx1 = x2
                dx2 = mu * (1 - x1**2) * x2 - x1
                
                x[:, i] = x[:, i-1] + dt * np.array([dx1, dx2])
            
            return x
        
        # Generate snapshots
        snapshots = simulate_vanderpol()
        
        # Initialize Koopman operator
        koopman = KoopmanOperator({
            'rank_threshold': 0.99,
            'kernel_bandwidth': 1.0
        })
        
        # Test standard DMD
        print("Computing DMD...")
        dmd_result = await koopman.compute_dmd(snapshots, dt=0.1)
        print(f"DMD rank: {dmd_result.rank}")
        print(f"Reconstruction error: {dmd_result.reconstruction_error:.4f}")
        
        # Analyze stability
        stability = koopman.analyze_stability(dmd_result)
        print(f"System stable: {stability['is_stable']}")
        print(f"Max eigenvalue magnitude: {stability['max_magnitude']:.4f}")
        
        # Test extended DMD with polynomial observables
        print("\nComputing Extended DMD...")
        
        def poly_observables(x):
            return koopman.compute_observables_polynomial(x, max_degree=3)
        
        edmd_result = await koopman.compute_extended_dmd(
            snapshots, poly_observables, dt=0.1
        )
        print(f"EDMD reconstruction error: {edmd_result.reconstruction_error:.4f}")
        
        # Test kernel DMD
        print("\nComputing Kernel DMD...")
        kdmd_result = await koopman.compute_kernel_dmd(snapshots, dt=0.1)
        print(f"KDMD reconstruction error: {kdmd_result.reconstruction_error:.4f}")
        
        # Extract coherent structures
        structures = koopman.extract_coherent_structures(dmd_result)
        print(f"\nFound {len(structures)} coherent structures")
        if structures:
            print(f"Dominant frequency: {structures[0]['frequency']:.4f} Hz")
        
        # Save results
        koopman.save_analysis(dmd_result, "vanderpol_dmd.json")
        
        # Test prediction
        print("\nTesting prediction...")
        initial = snapshots[:, 0]
        predictions = koopman.predict_evolution(dmd_result, initial, 50, dt=0.1)
        print(f"Prediction shape: {predictions.shape}")
    
    asyncio.run(test_koopman())
