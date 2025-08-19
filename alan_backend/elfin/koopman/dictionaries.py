"""
Dictionary functions for Koopman operator analysis.

This module provides functions for creating dictionary functions
(observables) that map states into a higher-dimensional space
where nonlinear dynamics can be approximated linearly.
"""

import numpy as np
from typing import Callable, List, Dict, Tuple, Union, Optional, Any
from functools import partial


class StandardDictionary:
    """
    Standard dictionary for Koopman operator analysis.
    
    This class wraps a dictionary function (observable) with metadata,
    providing a standard interface for use in EDMD and related algorithms.
    """
    
    def __init__(
        self, 
        dict_fn: Callable[[np.ndarray], np.ndarray],
        name: str,
        dimension: int,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a standard dictionary.
        
        Args:
            dict_fn: Dictionary function mapping states to features
            name: Name of the dictionary
            dimension: Dimension of the dictionary (output dimension)
            params: Dictionary parameters
        """
        self.dict_fn = dict_fn
        self.name = name
        self.dimension = dimension
        self.params = params or {}
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the dictionary function to a state.
        
        Args:
            x: State vector or matrix of state vectors
            
        Returns:
            Dictionary features
        """
        return self.dict_fn(x)
    
    def __repr__(self) -> str:
        return f"StandardDictionary({self.name}, dim={self.dimension}, params={self.params})"


def fourier_dict(
    x: np.ndarray,
    frequencies: np.ndarray,
    include_linear: bool = True,
    include_constant: bool = True
) -> np.ndarray:
    """
    Fourier dictionary function.
    
    Maps states to Fourier features using sine and cosine functions.
    
    Args:
        x: State vector or matrix of shape (n_samples, n_dims)
        frequencies: Matrix of frequencies of shape (n_features, n_dims)
        include_linear: Whether to include linear terms
        include_constant: Whether to include a constant term
        
    Returns:
        Matrix of Fourier features
    """
    # Ensure x is 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n_samples, n_dims = x.shape
    
    # Compute Fourier features
    # For each frequency vector w, compute sin(w·x) and cos(w·x)
    features = []
    
    if include_constant:
        # Add constant term
        features.append(np.ones((n_samples, 1)))
    
    if include_linear:
        # Add linear terms
        features.append(x)
    
    # Add Fourier terms
    for freq in frequencies:
        # Compute w·x for each sample
        wx = np.dot(x, freq)
        # Add sin and cos features
        features.append(np.sin(wx).reshape(-1, 1))
        features.append(np.cos(wx).reshape(-1, 1))
    
    # Concatenate all features
    return np.hstack(features)


def create_fourier_dict(
    dim: int,
    n_frequencies: int,
    max_freq: float = 2.0,
    include_linear: bool = True,
    include_constant: bool = True,
    random_seed: Optional[int] = None
) -> StandardDictionary:
    """
    Create a Fourier dictionary function.
    
    Args:
        dim: Dimension of the state space
        n_frequencies: Number of frequency vectors
        max_freq: Maximum frequency magnitude
        include_linear: Whether to include linear terms
        include_constant: Whether to include a constant term
        random_seed: Random seed for reproducibility
        
    Returns:
        StandardDictionary object
    """
    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate random frequencies
    frequencies = np.random.uniform(-max_freq, max_freq, (n_frequencies, dim))
    
    # Calculate output dimension
    output_dim = 2 * n_frequencies
    if include_linear:
        output_dim += dim
    if include_constant:
        output_dim += 1
    
    # Create the dictionary function
    dict_fn = partial(
        fourier_dict, 
        frequencies=frequencies, 
        include_linear=include_linear,
        include_constant=include_constant
    )
    
    # Create the dictionary object
    return StandardDictionary(
        dict_fn=dict_fn,
        name="fourier",
        dimension=output_dim,
        params={
            "dim": dim,
            "n_frequencies": n_frequencies,
            "max_freq": max_freq,
            "include_linear": include_linear,
            "include_constant": include_constant
        }
    )


def rbf_dict(
    x: np.ndarray,
    centers: np.ndarray,
    sigma: float = 1.0,
    include_linear: bool = True,
    include_constant: bool = True
) -> np.ndarray:
    """
    Radial basis function dictionary.
    
    Maps states to RBF features.
    
    Args:
        x: State vector or matrix of shape (n_samples, n_dims)
        centers: Matrix of RBF centers of shape (n_features, n_dims)
        sigma: Width parameter for the RBF
        include_linear: Whether to include linear terms
        include_constant: Whether to include a constant term
        
    Returns:
        Matrix of RBF features
    """
    # Ensure x is 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n_samples, n_dims = x.shape
    
    # Compute RBF features
    features = []
    
    if include_constant:
        # Add constant term
        features.append(np.ones((n_samples, 1)))
    
    if include_linear:
        # Add linear terms
        features.append(x)
    
    # Add RBF terms
    for center in centers:
        # Compute squared distance to center for each sample
        dist_sq = np.sum((x - center)**2, axis=1)
        # Add RBF feature
        features.append(np.exp(-dist_sq / (2 * sigma**2)).reshape(-1, 1))
    
    # Concatenate all features
    return np.hstack(features)


def create_rbf_dict(
    dim: int,
    n_centers: int,
    sigma: float = 1.0,
    domain: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    include_linear: bool = True,
    include_constant: bool = True,
    random_seed: Optional[int] = None
) -> StandardDictionary:
    """
    Create a radial basis function dictionary.
    
    Args:
        dim: Dimension of the state space
        n_centers: Number of RBF centers
        sigma: Width parameter for the RBF
        domain: Optional domain bounds as (lower, upper)
        include_linear: Whether to include linear terms
        include_constant: Whether to include a constant term
        random_seed: Random seed for reproducibility
        
    Returns:
        StandardDictionary object
    """
    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate random centers
    if domain is not None:
        lower, upper = domain
        centers = np.random.uniform(
            lower, upper, (n_centers, dim)
        )
    else:
        # Default domain is [-1, 1]^dim
        centers = np.random.uniform(-1, 1, (n_centers, dim))
    
    # Calculate output dimension
    output_dim = n_centers
    if include_linear:
        output_dim += dim
    if include_constant:
        output_dim += 1
    
    # Create the dictionary function
    dict_fn = partial(
        rbf_dict, 
        centers=centers, 
        sigma=sigma,
        include_linear=include_linear,
        include_constant=include_constant
    )
    
    # Create the dictionary object
    return StandardDictionary(
        dict_fn=dict_fn,
        name="rbf",
        dimension=output_dim,
        params={
            "dim": dim,
            "n_centers": n_centers,
            "sigma": sigma,
            "include_linear": include_linear,
            "include_constant": include_constant
        }
    )


def poly_dict(
    x: np.ndarray,
    degree: int,
    include_cross_terms: bool = True
) -> np.ndarray:
    """
    Polynomial dictionary function.
    
    Maps states to polynomial features up to the specified degree.
    
    Args:
        x: State vector or matrix of shape (n_samples, n_dims)
        degree: Maximum polynomial degree
        include_cross_terms: Whether to include cross terms
        
    Returns:
        Matrix of polynomial features
    """
    # Ensure x is 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n_samples, n_dims = x.shape
    
    if not include_cross_terms:
        # Only include pure powers
        features = [np.ones((n_samples, 1))]  # Constant term
        
        for d in range(1, degree + 1):
            for i in range(n_dims):
                features.append((x[:, i] ** d).reshape(-1, 1))
    else:
        # Include all monomials up to degree
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        return poly.fit_transform(x)
    
    # Concatenate all features
    return np.hstack(features)


def create_poly_dict(
    dim: int,
    degree: int,
    include_cross_terms: bool = True
) -> StandardDictionary:
    """
    Create a polynomial dictionary function.
    
    Args:
        dim: Dimension of the state space
        degree: Maximum polynomial degree
        include_cross_terms: Whether to include cross terms
        
    Returns:
        StandardDictionary object
    """
    # Calculate output dimension
    if include_cross_terms:
        from scipy.special import comb
        output_dim = int(sum(comb(dim + d, d) for d in range(degree + 1)))
    else:
        output_dim = 1 + dim * degree  # Constant term + pure powers
    
    # Create the dictionary function
    dict_fn = partial(
        poly_dict, 
        degree=degree,
        include_cross_terms=include_cross_terms
    )
    
    # Create the dictionary object
    return StandardDictionary(
        dict_fn=dict_fn,
        name="poly",
        dimension=output_dim,
        params={
            "dim": dim,
            "degree": degree,
            "include_cross_terms": include_cross_terms
        }
    )


def create_dictionary(
    dict_type: str,
    dim: int,
    **kwargs
) -> StandardDictionary:
    """
    Create a dictionary function based on type.
    
    Args:
        dict_type: Type of dictionary ('rbf', 'fourier', 'poly')
        dim: Dimension of the state space
        **kwargs: Additional parameters for the dictionary
        
    Returns:
        StandardDictionary object
    """
    if dict_type.lower() == 'rbf':
        return create_rbf_dict(dim, **kwargs)
    elif dict_type.lower() == 'fourier':
        return create_fourier_dict(dim, **kwargs)
    elif dict_type.lower() == 'poly':
        return create_poly_dict(dim, **kwargs)
    else:
        valid_types = ['rbf', 'fourier', 'poly']
        raise ValueError(f"Unknown dictionary type: {dict_type}. Valid types: {valid_types}")


# For backward compatibility and direct use
rbf_dict_fn = create_rbf_dict
fourier_dict_fn = create_fourier_dict
poly_dict_fn = create_poly_dict
