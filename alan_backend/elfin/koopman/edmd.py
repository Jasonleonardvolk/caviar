"""
Extended Dynamic Mode Decomposition (EDMD) for Koopman operator analysis.

This module provides functions for estimating the Koopman operator
from data using Extended Dynamic Mode Decomposition (EDMD).
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Callable, Any
from sklearn.model_selection import KFold

from .dictionaries import StandardDictionary


def edmd_fit(
    dictionary: StandardDictionary,
    x: np.ndarray,
    x_next: np.ndarray,
    reg_param: float = 1e-10,
    method: str = "svd"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit a Koopman operator using Extended Dynamic Mode Decomposition (EDMD).
    
    Args:
        dictionary: Dictionary function for mapping states to features
        x: State data of shape (n_samples, n_dims)
        x_next: Next state data of shape (n_samples, n_dims)
        reg_param: Regularization parameter for numerical stability
        method: Method for solving the least squares problem ('svd', 'qr', 'direct')
        
    Returns:
        Tuple of (K, metadata) where K is the Koopman matrix and metadata contains
        additional information about the fit
    """
    # Apply dictionary to states
    phi_x = dictionary(x)
    phi_x_next = dictionary(x_next)
    
    # Get dimensions
    n_samples, n_features = phi_x.shape
    
    # Compute the Koopman matrix
    if method == "svd":
        # Use SVD for improved numerical stability
        # Solve Φ(X') = Φ(X) K using SVD
        u, s, vh = np.linalg.svd(phi_x, full_matrices=False)
        
        # Apply regularization by setting small singular values to zero
        s_reg = np.where(s > reg_param, s, reg_param)
        s_inv = 1.0 / s_reg
        
        # Compute the pseudoinverse: pinv(Φ(X)) = V * S^-1 * U^T
        phi_x_pinv = vh.T @ np.diag(s_inv) @ u.T
        
        # Compute K = pinv(Φ(X)) * Φ(X')
        k_matrix = phi_x_pinv @ phi_x_next
        
        # Store metadata
        meta = {
            "singular_values": s,
            "cond_number": np.max(s) / np.min(s_reg),
            "method": "svd",
            "reg_param": reg_param,
            "rank": np.sum(s > reg_param)
        }
    
    elif method == "qr":
        # Use QR decomposition for improved numerical stability
        q, r = np.linalg.qr(phi_x)
        
        # Solve R K = Q^T Φ(X')
        k_matrix = np.linalg.solve(r, q.T @ phi_x_next)
        
        # Store metadata
        meta = {
            "method": "qr",
            "reg_param": reg_param,
            "rank": np.linalg.matrix_rank(phi_x)
        }
    
    elif method == "direct":
        # Direct solution using normal equations (least stable)
        # K = (Φ(X)^T Φ(X) + reg*I)^-1 Φ(X)^T Φ(X')
        phi_x_t_phi_x = phi_x.T @ phi_x
        reg_matrix = reg_param * np.eye(n_features)
        k_matrix = np.linalg.solve(
            phi_x_t_phi_x + reg_matrix,
            phi_x.T @ phi_x_next
        )
        
        # Store metadata
        meta = {
            "method": "direct",
            "reg_param": reg_param,
            "cond_number": np.linalg.cond(phi_x_t_phi_x + reg_matrix),
            "rank": np.linalg.matrix_rank(phi_x)
        }
    
    else:
        valid_methods = ["svd", "qr", "direct"]
        raise ValueError(f"Unknown method: {method}. Valid methods: {valid_methods}")
    
# Calculate fit quality metrics
    prediction = phi_x @ k_matrix
    residuals = phi_x_next - prediction
    mse = np.mean(np.sum(residuals**2, axis=1))
    
    # Update metadata
    meta.update({
        "n_samples": n_samples,
        "n_features": n_features,
        "mse": mse,
        "rmse": np.sqrt(mse)
    })
    
    return k_matrix, meta


def kfold_validation(
    dictionary: StandardDictionary,
    x: np.ndarray,
    x_next: np.ndarray,
    n_splits: int = 5,
    reg_param: float = 1e-10,
    method: str = "svd",
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform K-fold cross-validation for Koopman operator fitting.
    
    Args:
        dictionary: Dictionary function for mapping states to features
        x: State data of shape (n_samples, n_dims)
        x_next: Next state data of shape (n_samples, n_dims)
        n_splits: Number of splits for K-fold validation
        reg_param: Regularization parameter for numerical stability
        method: Method for solving the least squares problem
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with validation metrics
    """
    # Create K-fold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize results
    train_mses = []
    val_mses = []
    eigenvalues_list = []
    
    # Perform K-fold validation
    for train_idx, val_idx in kf.split(x):
        # Split data
        x_train, x_next_train = x[train_idx], x_next[train_idx]
        x_val, x_next_val = x[val_idx], x_next[val_idx]
        
        # Fit Koopman operator on training data
        k_matrix, train_meta = edmd_fit(
            dictionary=dictionary,
            x=x_train,
            x_next=x_next_train,
            reg_param=reg_param,
            method=method
        )
        
        # Apply dictionary to validation data
        phi_x_val = dictionary(x_val)
        phi_x_next_val = dictionary(x_val)
        
        # Compute validation MSE
        prediction = phi_x_val @ k_matrix
        residuals = phi_x_next_val - prediction
        val_mse = np.mean(np.sum(residuals**2, axis=1))
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(k_matrix)
        
        # Store results
        train_mses.append(train_meta["mse"])
        val_mses.append(val_mse)
        eigenvalues_list.append(eigenvalues)
    
    # Compute final Koopman operator on all data
    k_matrix, all_meta = edmd_fit(
        dictionary=dictionary,
        x=x,
        x_next=x_next,
        reg_param=reg_param,
        method=method
    )
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(k_matrix)
    
    # Compile results
    results = {
        "train_mse_mean": np.mean(train_mses),
        "train_mse_std": np.std(train_mses),
        "val_mse_mean": np.mean(val_mses),
        "val_mse_std": np.std(val_mses),
        "all_mse": all_meta["mse"],
        "eigenvalues": eigenvalues,
        "k_matrix": k_matrix,
        "meta": all_meta,
        "eigenvalues_folds": eigenvalues_list
    }
    
    return results


def estimate_optimal_dict_size(
    x: np.ndarray,
    x_next: np.ndarray,
    dict_type: str,
    dim: int,
    size_range: List[int],
    validation_splits: int = 5,
    verbose: bool = False,
    **dict_params
) -> Tuple[int, Dict[str, Any]]:
    """
    Estimate the optimal dictionary size for EDMD.
    
    This function performs a grid search over dictionary sizes and
    selects the one with the best validation performance.
    
    Args:
        x: State data of shape (n_samples, n_dims)
        x_next: Next state data of shape (n_samples, n_dims)
        dict_type: Type of dictionary ('rbf', 'fourier', 'poly')
        dim: Dimension of the state space
        size_range: List of dictionary sizes to try
        validation_splits: Number of splits for K-fold validation
        verbose: Whether to print progress
        **dict_params: Additional parameters for the dictionary
        
    Returns:
        Tuple of (optimal_size, results) where results contains validation
        metrics for each dictionary size
    """
    from .dictionaries import create_dictionary
    
    # Initialize results
    all_results = {}
    
    # Perform grid search
    for size in size_range:
        if verbose:
            print(f"Trying dictionary size: {size}")
        
        # Create dictionary
        if dict_type == "rbf":
            dictionary = create_dictionary(
                dict_type="rbf",
                dim=dim,
                n_centers=size,
                **dict_params
            )
        elif dict_type == "fourier":
            dictionary = create_dictionary(
                dict_type="fourier",
                dim=dim,
                n_frequencies=size,
                **dict_params
            )
        elif dict_type == "poly":
            dictionary = create_dictionary(
                dict_type="poly",
                dim=dim,
                degree=size,
                **dict_params
            )
        else:
            valid_types = ["rbf", "fourier", "poly"]
            raise ValueError(f"Unknown dictionary type: {dict_type}. Valid types: {valid_types}")
        
        # Perform K-fold validation
        results = kfold_validation(
            dictionary=dictionary,
            x=x,
            x_next=x_next,
            n_splits=validation_splits
        )
        
        # Store results
        all_results[size] = results
    
    # Find optimal size based on validation MSE
    val_mses = {size: results["val_mse_mean"] for size, results in all_results.items()}
    optimal_size = min(val_mses, key=val_mses.get)
    
    if verbose:
        print(f"Optimal dictionary size: {optimal_size}")
        print(f"Validation MSE: {val_mses[optimal_size]}")
    
    return optimal_size, all_results


def load_trajectory_data(
    file_path: str,
    state_dim: int,
    dt: Optional[float] = None,
    skip_header: int = 0,
    delimiter: str = ","
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load trajectory data from a file.
    
    The file should contain state trajectories where consecutive
    rows represent states at consecutive time steps.
    
    Args:
        file_path: Path to the data file
        state_dim: Dimension of the state space
        dt: Time step between consecutive states (if None, inferred from data)
        skip_header: Number of header rows to skip
        delimiter: Delimiter used in the file
        
    Returns:
        Tuple of (x, x_next) where x and x_next are arrays of states and next states
    """
    # Load data
    data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skip_header)
    
    # Check if first column is time
    has_time_column = False
    if data.shape[1] > state_dim:
        # Assume first column is time
        has_time_column = True
        time = data[:, 0]
        states = data[:, 1:state_dim+1]
    else:
        # No time column
        states = data[:, :state_dim]
    
    # Check if dt is provided
    if dt is None and has_time_column:
        # Infer dt from time column
        dt = np.mean(np.diff(time))
    elif dt is None:
        # Default dt = 1
        dt = 1.0
    
    # Create x and x_next
    x = states[:-1]
    x_next = states[1:]
    
    return x, x_next


def compute_dmd_spectrum(
    k_matrix: np.ndarray,
    dt: float = 1.0,
    continuous: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the DMD spectrum (eigenvalues and eigenvectors) of a Koopman matrix.
    
    Args:
        k_matrix: Koopman matrix
        dt: Time step
        continuous: Whether to convert to continuous time eigenvalues
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(k_matrix)
    
    # Convert to continuous time if requested
    if continuous:
        # λ_continuous = log(λ_discrete) / dt
        eigenvalues = np.log(eigenvalues) / dt
    
    return eigenvalues, eigenvectors
