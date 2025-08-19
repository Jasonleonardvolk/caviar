"""
Koopman-based Lyapunov Learning for ELFIN Stability Framework.

This module provides tools for learning Lyapunov functions using Koopman
operator theory. It serves as a proof-of-concept bridge between nonlinear
dynamical systems and Lyapunov stability verification.

The key idea is to use the Koopman operator to lift nonlinear dynamics to a
higher-dimensional space where they become approximately linear. Then, a
quadratic Lyapunov function in this lifted space corresponds to a non-quadratic
Lyapunov function in the original state space.
"""

import os
import logging
import time
import math
import numpy as np
import scipy.linalg as la
from typing import Dict, List, Tuple, Callable, Optional, Any, Union, Callable
from enum import Enum, auto

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


class DictionaryType(Enum):
    """Types of dictionary functions for Koopman lifting."""
    
    MONOMIAL = auto()    # Polynomial basis functions
    RBF = auto()         # Radial basis functions
    FOURIER = auto()     # Fourier basis functions
    CUSTOM = auto()      # Custom user-defined basis


def generate_trajectory_data(
    dyn_fn: Callable,
    x0: np.ndarray,
    n_steps: int,
    noise_scale: float = 0.0
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate trajectory data from dynamical system.
    
    Args:
        dyn_fn: Dynamics function that maps x_{k} to x_{k+1}
        x0: Initial state
        n_steps: Number of steps to simulate
        noise_scale: Scale of Gaussian noise to add (0.0 for deterministic)
        
    Returns:
        Tuple of (states, next_states) lists
    """
    states = [x0.copy()]
    next_states = []
    
    x = x0.copy()
    for _ in range(n_steps):
        # Compute next state
        x_next = dyn_fn(x)
        
        # Add noise if requested
        if noise_scale > 0:
            x_next = x_next + noise_scale * np.random.randn(*x_next.shape)
            
        # Store states
        next_states.append(x_next.copy())
        
        # Update for next iteration
        x = x_next.copy()
        
        if _ < n_steps - 1:  # Don't add the last state as a starting state
            states.append(x.copy())
            
    return states, next_states


def create_monomial_dict(degree: int) -> Callable:
    """
    Create monomial dictionary function.
    
    For 2D state [x1, x2], degree=2 creates:
    [1, x1, x2, x1^2, x1*x2, x2^2]
    
    Args:
        degree: Maximum polynomial degree
        
    Returns:
        Dictionary function mapping state to feature vector
    """
    def dict_fn(x):
        """Map state to monomial features."""
        x = np.asarray(x).flatten()
        n = len(x)
        
        # Start with constant term
        features = [1.0]
        
        # Add monomials up to specified degree
        for d in range(1, degree + 1):
            # Generate all combinations of indices that sum to d
            if n == 1:
                # Special case for 1D
                features.append(x[0] ** d)
            else:
                # General case for nD
                from itertools import combinations_with_replacement
                for idx in combinations_with_replacement(range(n), d):
                    # Convert index tuple to monomial
                    term = 1.0
                    for i in idx:
                        term *= x[i]
                    features.append(term)
        
        return np.array(features)
    
    return dict_fn


def create_rbf_dict(centers: np.ndarray, sigma: float) -> Callable:
    """
    Create radial basis function dictionary.
    
    Args:
        centers: Array of center points
        sigma: RBF width parameter
        
    Returns:
        Dictionary function mapping state to feature vector
    """
    def dict_fn(x):
        """Map state to RBF features."""
        x = np.asarray(x).flatten()
        features = np.zeros(centers.shape[0] + 1)
        
        # Constant term
        features[0] = 1.0
        
        # RBF terms
        for i, center in enumerate(centers):
            dist = np.linalg.norm(x - center)
            features[i + 1] = np.exp(-dist**2 / (2 * sigma**2))
        
        return features
    
    return dict_fn


def create_fourier_dict(n_terms: int, freq_range: List[float]) -> Callable:
    """
    Create Fourier basis dictionary function.
    
    Args:
        n_terms: Number of Fourier terms
        freq_range: Range of frequencies [min_freq, max_freq]
        
    Returns:
        Dictionary function mapping state to feature vector
    """
    # Generate random frequencies in the specified range
    np.random.seed(42)  # For reproducibility
    freqs = np.random.uniform(freq_range[0], freq_range[1], (n_terms, 2))
    
    def dict_fn(x):
        """Map state to Fourier features."""
        x = np.asarray(x).flatten()
        if len(x) < 2:
            x = np.pad(x, (0, 2 - len(x)))
            
        features = np.zeros(2 * n_terms + 1)
        
        # Constant term
        features[0] = 1.0
        
        # Sine and cosine terms
        for i, freq in enumerate(freqs):
            # Sine term
            features[2*i + 1] = np.sin(freq @ x[:2])
            # Cosine term
            features[2*i + 2] = np.cos(freq @ x[:2])
        
        return features
    
    return dict_fn


def create_dict_fn(
    dict_type: Union[str, DictionaryType],
    dict_params: Optional[Dict] = None
) -> Callable:
    """
    Create dictionary function based on type and parameters.
    
    Args:
        dict_type: Type of dictionary ("monomial", "rbf", "fourier", "custom")
        dict_params: Dictionary parameters
        
    Returns:
        Dictionary function
    """
    if isinstance(dict_type, str):
        dict_type = dict_type.upper()
        try:
            dict_type = DictionaryType[dict_type]
        except KeyError:
            logger.warning(f"Unknown dictionary type: {dict_type}. Using MONOMIAL.")
            dict_type = DictionaryType.MONOMIAL
    
    params = dict_params or {}
    
    if dict_type == DictionaryType.MONOMIAL:
        degree = params.get("degree", 2)
        return create_monomial_dict(degree)
    
    elif dict_type == DictionaryType.RBF:
        n_centers = params.get("n_centers", 10)
        sigma = params.get("sigma", 1.0)
        
        # Generate random centers
        np.random.seed(42)  # For reproducibility
        state_dim = params.get("state_dim", 2)
        bounds = params.get("bounds", (-3.0, 3.0))
        centers = np.random.uniform(
            bounds[0], bounds[1], (n_centers, state_dim)
        )
        
        return create_rbf_dict(centers, sigma)
    
    elif dict_type == DictionaryType.FOURIER:
        n_terms = params.get("n_terms", 10)
        freq_range = params.get("freq_range", [0.1, 2.0])
        return create_fourier_dict(n_terms, freq_range)
    
    elif dict_type == DictionaryType.CUSTOM:
        if "dict_fn" in params:
            return params["dict_fn"]
        else:
            logger.warning("No custom dictionary function provided. Using monomial.")
            return create_monomial_dict(2)
    
    else:
        logger.warning(f"Unsupported dictionary type: {dict_type}. Using monomial.")
        return create_monomial_dict(2)


def compute_koopman(states, next_states, dict_fn, reg_param=1e-6):
    """
    Compute approximate Koopman operator from data.
    
    Args:
        states: List of states x_k
        next_states: List of next states x_{k+1}
        dict_fn: Dictionary function
        reg_param: Regularization parameter
        
    Returns:
        Koopman operator matrix K
    """
    # Convert lists to arrays
    X = np.array([dict_fn(x) for x in states])
    Y = np.array([dict_fn(y) for y in next_states])
    
    # Compute K using regression
    XTX = X.T @ X + reg_param * np.eye(X.shape[1])
    XTY = X.T @ Y
    K = np.linalg.solve(XTX, XTY)
    
    return K


def learn_lyapunov_from_koopman(K, discrete_time=True):
    """
    Learn Lyapunov function from Koopman operator.
    
    Args:
        K: Koopman operator matrix
        discrete_time: Whether the system is discrete-time
        
    Returns:
        Tuple of (eigenvalues, Q matrix, stability mask)
    """
    # For discrete-time system:
    # Compute eigendecomposition of K
    eigvals, eigvecs = np.linalg.eig(K)
    
    # Sort by magnitude
    idx = np.argsort(np.abs(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Check which eigenvalues correspond to stable modes
    if discrete_time:
        stable_mask = np.abs(eigvals) < 1.0
    else:
        stable_mask = np.real(eigvals) < 0.0
    
    # For real-valued Lyapunov function, use real part
    Q = np.real(eigvecs @ np.diag(stable_mask.astype(float)) @ eigvecs.T)
    
    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(Q))
    if min_eig <= 0:
        Q = Q - 1.1 * min_eig * np.eye(Q.shape[0])
    
    return eigvals, Q, stable_mask


class KoopmanLyapunov(LyapunovFunction):
    """
    Koopman-based Lyapunov function.
    
    This represents a Lyapunov function learned from data using Koopman
    operator theory. It takes the form:
    
    V(x) = g(x)^T Q g(x)
    
    where g(x) is a dictionary function and Q is a positive definite matrix.
    """
    
    def __init__(
        self,
        name: str,
        lam: np.ndarray,
        V: np.ndarray,
        dict_fn: Callable,
        stable_mask: np.ndarray,
        domain_ids: Optional[List[str]] = None
    ):
        """
        Initialize Koopman Lyapunov function.
        
        Args:
            name: Name of the function
            lam: Eigenvalues of the Koopman operator
            V: Eigenvectors of the Koopman operator
            dict_fn: Dictionary function
            stable_mask: Mask of stable modes
            domain_ids: IDs of concepts in the domain
        """
        super().__init__(name, domain_ids)
        self.lam = lam
        self.V = V
        self.dict_fn = dict_fn
        self.stable_mask = stable_mask
        
        if V is not None:
            # Compute Q matrix for Lyapunov function
            self.Q = np.real(V @ np.diag(stable_mask.astype(float)) @ V.T)
            
            # Ensure positive definiteness
            min_eig = np.min(np.linalg.eigvalsh(self.Q))
            if min_eig <= 0:
                self.Q = self.Q - 1.1 * min_eig * np.eye(self.Q.shape[0])
        else:
            # Placeholder for minimal implementation
            self.Q = np.eye(2)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Lyapunov function at a state x.
        
        Args:
            x: State vector
            
        Returns:
            Lyapunov function value
        """
        if self.dict_fn is None or self.Q is None:
            # Fallback for minimal implementation
            x = np.asarray(x).flatten()
            return float(x.T @ x)
            
        # Compute dictionary features
        g_x = self.dict_fn(x)
        
        # Compute Lyapunov function value
        return float(g_x.T @ self.Q @ g_x)
    
    def get_eigenvalues(self) -> np.ndarray:
        """
        Get Koopman eigenvalues.
        
        Returns:
            Eigenvalue array
        """
        return self.lam
    
    def get_stable_modes(self) -> np.ndarray:
        """
        Get stable mode mask.
        
        Returns:
            Boolean mask of stable modes
        """
        return self.stable_mask
    
    def get_quadratic_form(self) -> np.ndarray:
        """
        Get the quadratic form matrix Q.
        
        Returns:
            Q matrix
        """
        return self.Q
    
    def get_feature_dimension(self) -> int:
        """
        Get feature space dimension.
        
        Returns:
            Dimension of the lifted feature space
        """
        if self.Q is not None:
            return self.Q.shape[0]
        return 0


def create_koopman_lyapunov(
    states: List[np.ndarray],
    next_states: List[np.ndarray],
    dict_type: str = "monomial",
    dict_params: Optional[Dict] = None,
    name: str = "V_koop",
    domain_ids: Optional[List[str]] = None,
    discrete_time: bool = True
) -> KoopmanLyapunov:
    """
    Create Koopman Lyapunov function from trajectory data.
    
    Args:
        states: List of states
        next_states: List of next states
        dict_type: Type of dictionary function
        dict_params: Dictionary parameters
        name: Name of the Lyapunov function
        domain_ids: Domain concept IDs
        discrete_time: Whether the system is discrete-time
        
    Returns:
        KoopmanLyapunov function
    """
    # Create dictionary function
    dict_fn = create_dict_fn(dict_type, dict_params)
    
    # Compute Koopman operator
    K = compute_koopman(states, next_states, dict_fn)
    
    # Learn Lyapunov function
    lam, V, stable_mask = learn_lyapunov_from_koopman(K, discrete_time)
    
    # Create Lyapunov function
    lyap = KoopmanLyapunov(name, lam, V, dict_fn, stable_mask, domain_ids)
    
    return lyap


def visualize_koopman_spectrum(lyap, ax=None):
    """
    Visualize Koopman eigenvalue spectrum.
    
    Args:
        lyap: KoopmanLyapunov function
        ax: Matplotlib axis
        
    Returns:
        Matplotlib axis
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    
    # Get eigenvalues
    lam = lyap.get_eigenvalues()
    stable = lyap.get_stable_modes()
    
    # Plot unit circle for discrete-time systems
    circle = Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    ax.add_patch(circle)
    
    # Plot eigenvalues
    for l, s in zip(lam, stable):
        color = 'green' if s else 'red'
        ax.scatter(np.real(l), np.imag(l), c=color, s=100, alpha=0.7)
    
    # Set labels and limits
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imaginary part')
    ax.set_title('Koopman Eigenvalue Spectrum')
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    # Grid
    ax.grid(True)
    
    return ax


def visualize_lyapunov_function(lyap, bounds=(-3, 3), resolution=50, ax=None):
    """
    Visualize Lyapunov function level sets.
    
    Args:
        lyap: Lyapunov function
        bounds: Plot bounds
        resolution: Grid resolution
        ax: Matplotlib axis
        
    Returns:
        Matplotlib axis
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    
    # Create grid
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate Lyapunov function
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            state = np.array([X[i, j], Y[i, j]])
            Z[i, j] = lyap.evaluate(state)
    
    # Plot contours
    contours = ax.contourf(X, Y, Z, 50, cmap='viridis')
    ax.contour(X, Y, Z, 20, colors='white', alpha=0.5, linestyles='solid')
    plt.colorbar(contours, ax=ax)
    
    # Set labels
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'Lyapunov Function: {lyap.name}')
    
    return ax


def run_demo():
    """Run a simple demonstration of Koopman-based Lyapunov learning."""
    import matplotlib.pyplot as plt
    
    # Define a simple system: Van der Pol oscillator
    def van_der_pol(x, mu=1.0, dt=0.1):
        """Van der Pol oscillator with Euler integration."""
        dx1 = x[1]
        dx2 = mu * (1 - x[0]**2) * x[1] - x[0]
        return np.array([x[0] + dt * dx1, x[1] + dt * dx2])
    
    # Generate data
    x0 = np.array([1.0, 0.0])
    states, next_states = generate_trajectory_data(van_der_pol, x0, n_steps=1000)
    
    # Create Koopman Lyapunov function
    lyap = create_koopman_lyapunov(
        states, next_states, 
        dict_type="monomial", 
        dict_params={"degree": 4},
        name="V_vanderpol"
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot eigenvalue spectrum
    visualize_koopman_spectrum(lyap, axes[0])
    
    # Plot Lyapunov function
    visualize_lyapunov_function(lyap, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig("koopman_lyapunov.png")
    print("Results plotted to koopman_lyapunov.png")


if __name__ == "__main__":
    run_demo()
