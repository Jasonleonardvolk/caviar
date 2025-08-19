"""
Koopman-based Lyapunov functions.

This module provides tools for creating Lyapunov functions from
Koopman operator analysis, focusing on spectral Lyapunov functions.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

from .dictionaries import StandardDictionary
from alan_backend.elfin.stability.lyapunov import LyapunovFunction

logger = logging.getLogger(__name__)


def get_stable_modes(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    lambda_cut: float = 0.99,
    continuous_time: bool = False,
    min_modes: int = 2,
    max_modes: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Get stable modes from Koopman eigendecomposition.
    
    Args:
        eigenvalues: Eigenvalues of the Koopman operator
        eigenvectors: Eigenvectors (right) of the Koopman operator
        lambda_cut: Cutoff for selecting stable modes (continuous: Re(λ) < 0, discrete: |λ| < λ_cut)
        continuous_time: Whether the system is continuous-time or discrete-time
        min_modes: Minimum number of modes to select
        max_modes: Maximum number of modes to select
        
    Returns:
        Tuple of (stable_eigenvalues, stable_eigenvectors, stable_indices)
    """
    # Determine which modes are stable
    if continuous_time:
        # For continuous time, modes are stable if Re(λ) < 0
        stable_indices = np.where(np.real(eigenvalues) < 0)[0]
    else:
        # For discrete time, modes are stable if |λ| < λ_cut
        stable_indices = np.where(np.abs(eigenvalues) < lambda_cut)[0]
    
    # Sort by stability (most stable first)
    if continuous_time:
        # Sort by real part (most negative first)
        sort_idx = np.argsort(np.real(eigenvalues[stable_indices]))
    else:
        # Sort by magnitude (smallest first)
        sort_idx = np.argsort(np.abs(eigenvalues[stable_indices]))
    
    stable_indices = stable_indices[sort_idx]
    
    # Ensure minimum number of modes
    if len(stable_indices) < min_modes:
        if continuous_time:
            # Sort all eigenvalues by real part (most negative first)
            all_idx = np.argsort(np.real(eigenvalues))
        else:
            # Sort all eigenvalues by magnitude (smallest first)
            all_idx = np.argsort(np.abs(eigenvalues))
        
        # Take the minimum required number
        stable_indices = all_idx[:min_modes]
        
        # Warn about having to use unstable modes
        if min_modes > 0:
            logger.warning(
                f"Only {len(np.where(np.real(eigenvalues) < 0)[0] if continuous_time else np.where(np.abs(eigenvalues) < lambda_cut)[0])} "
                f"stable modes found, using {min_modes} modes including potentially unstable ones"
            )
    
    # Limit to maximum number of modes if specified
    if max_modes is not None and len(stable_indices) > max_modes:
        stable_indices = stable_indices[:max_modes]
    
    # Extract stable eigenvalues and eigenvectors
    stable_eigenvalues = eigenvalues[stable_indices]
    stable_eigenvectors = eigenvectors[:, stable_indices]
    
    return stable_eigenvalues, stable_eigenvectors, stable_indices.tolist()


class KoopmanLyapunov(LyapunovFunction):
    """
    Koopman-based Lyapunov function.
    
    This class implements a Lyapunov function based on the spectral
    decomposition of the Koopman operator. The function is of the form
    
    V(x) = \sum_{i \in S} w_i |ψ_i(x)|^2
    
    where ψ_i are Koopman eigenfunctions corresponding to stable eigenvalues,
    and w_i are weights (either uniform or based on eigenvalues).
    """
    
    def __init__(
        self,
        name: str,
        dictionary: StandardDictionary,
        k_matrix: np.ndarray,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray,
        stable_indices: List[int],
        phi_origin: Optional[np.ndarray] = None,
        is_continuous: bool = False,
        dt: float = 1.0,
        weighting: str = "uniform",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a Koopman-based Lyapunov function.
        
        Args:
            name: Name of the Lyapunov function
            dictionary: Dictionary function for mapping states to features
            k_matrix: Koopman matrix
            eigenvectors: Eigenvectors (right) of the Koopman matrix
            eigenvalues: Eigenvalues of the Koopman matrix
            stable_indices: Indices of stable modes to use
            phi_origin: Dictionary evaluated at the origin, for centering
            is_continuous: Whether the system is continuous-time or discrete-time
            dt: Time step for discrete-time systems
            metadata: Additional metadata about the Lyapunov function
        """
        super().__init__(name)
        
        self.dictionary = dictionary
        self.k_matrix = k_matrix
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.stable_indices = stable_indices
        self.is_continuous = is_continuous
        self.dt = dt
        self.metadata = metadata or {}
        
        # Compute dictionary at origin if not provided
        if phi_origin is None:
            origin = np.zeros(dictionary.params.get("dim", 2))
            self.phi_origin = dictionary(origin)
        else:
            self.phi_origin = phi_origin
            
        # Precompute projection matrix for efficiency
        # We want to compute ψ_i(x) = v_i^T (Φ(x) - Φ(0))
        self.projection_matrix = eigenvectors[:, stable_indices].T
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Lyapunov function.
        
        Args:
            x: State vector or batch of state vectors
            
        Returns:
            Lyapunov function value(s)
        """
        # Ensure x is 2D
        single_input = x.ndim == 1
        if single_input:
            x = x.reshape(1, -1)
        
        # Apply dictionary to state
        phi_x = self.dictionary(x)
        
        # Center around origin
        phi_x_centered = phi_x - self.phi_origin
        
        # Compute Koopman eigenfunctions: ψ_i(x) = v_i^T (Φ(x) - Φ(0))
        psi_x = self.projection_matrix @ phi_x_centered.T  # shape: (n_modes, n_samples)
        
        # Compute the squared magnitudes of eigenfunctions
        psi_x_squared = np.abs(psi_x) ** 2
        
        # Get weights based on weighting strategy
        if hasattr(self, 'weighting') and self.weighting == "lambda":
            # Get the real parts of eigenvalues for stable modes
            eigenvalues_real = np.array([np.real(self.eigenvalues[idx]) for idx in self.stable_indices])
            
            # For stable modes, use negative real part of eigenvalue as weight
            # For continuous time, Re(λ) < 0 means stability
            # For discrete time, |λ| < 1 means stability, but we still use -Re(λ) for weighting
            weights = -eigenvalues_real
            
            # Normalize weights so they sum to 1
            weights = weights / np.sum(weights)
            
            # Apply weights to eigenfunction squares
            v_x = np.sum(weights.reshape(-1, 1) * psi_x_squared, axis=0)
        else:
            # Uniform weighting: V(x) = \sum_{i} |ψ_i(x)|^2
            v_x = np.sum(psi_x_squared, axis=0)
        
        if single_input:
            return v_x[0]
        return v_x
        
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Lyapunov function.
        
        Args:
            x: State vector
            
        Returns:
            Gradient vector
        """
        # For complex Koopman eigenfunctions, the gradient is more complex
        # We'll use numerical differentiation for now
        epsilon = 1e-6
        dim = len(x)
        grad = np.zeros(dim)
        
        for i in range(dim):
            # Create perturbation vectors
            x_plus = x.copy()
            x_plus[i] += epsilon
            x_minus = x.copy()
            x_minus[i] -= epsilon
            
            # Compute central difference
            grad[i] = (self(x_plus) - self(x_minus)) / (2 * epsilon)
            
        return grad
    
    def evaluate_decrease(
        self, 
        x: np.ndarray, 
        dynamics_fn: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Evaluate the decrease condition V̇(x) < 0.
        
        Args:
            x: State vector or batch of state vectors
            dynamics_fn: Dynamics function mapping x to ẋ or x' (next state)
            
        Returns:
            V̇(x) values (negative means decreasing)
        """
        # Ensure x is 2D
        single_input = x.ndim == 1
        if single_input:
            x = x.reshape(1, -1)
        
        n_samples = x.shape[0]
        dot_v = np.zeros(n_samples)
        
        if self.is_continuous:
            # For continuous-time systems, compute V̇ = ∇V(x) · f(x)
            for i in range(n_samples):
                dot_v[i] = np.dot(self.gradient(x[i]), dynamics_fn(x[i]))
        else:
            # For discrete-time systems, compute V(x') - V(x)
            x_next = np.array([dynamics_fn(xi) for xi in x])
            v_x = self(x)
            v_x_next = self(x_next)
            dot_v = v_x_next - v_x
            
        if single_input:
            return dot_v[0]
        return dot_v
    
    def get_eigenfunction(self, idx: int) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get a specific Koopman eigenfunction.
        
        Args:
            idx: Index of the eigenfunction to get (in the stable_indices list)
            
        Returns:
            Function mapping states to eigenfunction values
        """
        if idx < 0 or idx >= len(self.stable_indices):
            raise ValueError(f"Invalid eigenfunction index: {idx}, must be in range [0, {len(self.stable_indices)-1}]")
        
        mode_idx = self.stable_indices[idx]
        eigenvector = self.eigenvectors[:, mode_idx]
        
        def eigenfunction(x: np.ndarray) -> np.ndarray:
            """
            Evaluate the Koopman eigenfunction.
            
            Args:
                x: State vector or batch of state vectors
                
            Returns:
                Eigenfunction value(s)
            """
            # Ensure x is 2D
            single_input = x.ndim == 1
            if single_input:
                x = x.reshape(1, -1)
            
            # Apply dictionary to state
            phi_x = self.dictionary(x)
            
            # Center around origin
            phi_x_centered = phi_x - self.phi_origin
            
            # Compute eigenfunction: ψ(x) = v^T (Φ(x) - Φ(0))
            psi_x = eigenvector @ phi_x_centered.T
            
            if single_input:
                return psi_x[0]
            return psi_x
        
        return eigenfunction
    
    def get_eigenvalue(self, idx: int) -> complex:
        """
        Get a specific Koopman eigenvalue.
        
        Args:
            idx: Index of the eigenvalue to get (in the stable_indices list)
            
        Returns:
            Eigenvalue
        """
        if idx < 0 or idx >= len(self.stable_indices):
            raise ValueError(f"Invalid eigenvalue index: {idx}, must be in range [0, {len(self.stable_indices)-1}]")
        
        mode_idx = self.stable_indices[idx]
        return self.eigenvalues[mode_idx]
    
    def get_koopman_triplet(self, idx: int) -> Tuple[complex, np.ndarray, int]:
        """
        Get a specific Koopman triplet (eigenvalue, eigenvector, index).
        
        Args:
            idx: Index of the triplet to get (in the stable_indices list)
            
        Returns:
            Tuple of (eigenvalue, eigenvector, original_index)
        """
        if idx < 0 or idx >= len(self.stable_indices):
            raise ValueError(f"Invalid triplet index: {idx}, must be in range [0, {len(self.stable_indices)-1}]")
        
        mode_idx = self.stable_indices[idx]
        return self.eigenvalues[mode_idx], self.eigenvectors[:, mode_idx], mode_idx
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Lyapunov function to a dictionary representation.
        
        Returns:
            Dictionary representation
        """
        # Create base dictionary
        lyap_dict = {
            "name": self.name,
            "type": "koopman",
            "dictionary": {
                "name": self.dictionary.name,
                "dimension": self.dictionary.dimension,
                "params": self.dictionary.params
            },
            "k_matrix_shape": self.k_matrix.shape,
            "eigenvalues": self.eigenvalues,
            "stable_indices": self.stable_indices,
            "is_continuous": self.is_continuous,
            "dt": self.dt
        }
        
        # Add metadata
        if self.metadata:
            lyap_dict["metadata"] = self.metadata
            
        return lyap_dict
        

def create_koopman_lyapunov(
    name: str,
    k_matrix: np.ndarray,
    dictionary: StandardDictionary,
    lambda_cut: float = 0.99,
    continuous_time: bool = False,
    min_modes: int = 2,
    max_modes: Optional[int] = None,
    dt: float = 1.0,
    weighting: str = "uniform",
    metadata: Optional[Dict[str, Any]] = None
) -> KoopmanLyapunov:
    """
    Create a Koopman-based Lyapunov function.
    
    Args:
        name: Name of the Lyapunov function
        k_matrix: Koopman matrix
        dictionary: Dictionary function for mapping states to features
        lambda_cut: Cutoff for selecting stable modes
        continuous_time: Whether the system is continuous-time
        min_modes: Minimum number of modes to use
        max_modes: Maximum number of modes to use
        dt: Time step for discrete-time systems
        metadata: Additional metadata
        
    Returns:
        KoopmanLyapunov function
    """
    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(k_matrix)
    
    # Get stable modes
    stable_eigenvalues, stable_eigenvectors, stable_indices = get_stable_modes(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        lambda_cut=lambda_cut,
        continuous_time=continuous_time,
        min_modes=min_modes,
        max_modes=max_modes
    )
    
    # Create Lyapunov function
    return KoopmanLyapunov(
        name=name,
        dictionary=dictionary,
        k_matrix=k_matrix,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        stable_indices=stable_indices,
        is_continuous=continuous_time,
        dt=dt,
        metadata=metadata
    )


def create_from_trajectory_data(
    name: str,
    x: np.ndarray,
    x_next: np.ndarray,
    dict_type: str = "rbf",
    dict_size: int = 100,
    lambda_cut: float = 0.99,
    continuous_time: bool = False,
    dt: float = 1.0,
    dict_params: Optional[Dict[str, Any]] = None
) -> KoopmanLyapunov:
    """
    Create a Koopman-based Lyapunov function from trajectory data.
    
    Args:
        name: Name of the Lyapunov function
        x: State data
        x_next: Next state data
        dict_type: Type of dictionary ('rbf', 'fourier', 'poly')
        dict_size: Size of the dictionary
        lambda_cut: Cutoff for selecting stable modes
        continuous_time: Whether the system is continuous-time
        dt: Time step for discrete-time systems
        dict_params: Additional parameters for the dictionary
        
    Returns:
        KoopmanLyapunov function
    """
    from .dictionaries import create_dictionary
    from .edmd import edmd_fit
    
    # Infer state dimension
    state_dim = x.shape[1]
    
    # Create dictionary
    dict_params = dict_params or {}
    
    if dict_type == "rbf":
        dictionary = create_dictionary(
            dict_type="rbf",
            dim=state_dim,
            n_centers=dict_size,
            **dict_params
        )
    elif dict_type == "fourier":
        dictionary = create_dictionary(
            dict_type="fourier",
            dim=state_dim,
            n_frequencies=dict_size,
            **dict_params
        )
    elif dict_type == "poly":
        dictionary = create_dictionary(
            dict_type="poly",
            dim=state_dim,
            degree=dict_size,
            **dict_params
        )
    else:
        valid_types = ["rbf", "fourier", "poly"]
        raise ValueError(f"Unknown dictionary type: {dict_type}. Valid types: {valid_types}")
    
    # Fit Koopman operator
    k_matrix, meta = edmd_fit(
        dictionary=dictionary,
        x=x,
        x_next=x_next
    )
    
    # Create Lyapunov function
    return create_koopman_lyapunov(
        name=name,
        k_matrix=k_matrix,
        dictionary=dictionary,
        lambda_cut=lambda_cut,
        continuous_time=continuous_time,
        dt=dt,
        metadata=meta
    )
