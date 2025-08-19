"""lyapunov.py - Compute Lyapunov Predictability Metric for concept trajectories.

This module defines functions to calculate the largest Lyapunov exponent for a given 
concept sequence (time series of concept feature vectors). It implements a robust 
neighbor divergence algorithm based on Rosenstein et al. (1993), suitable 
even for short or noisy trajectories. 

Outputs from these functions (exponent values, divergence curves) are typically consumed 
by the batch processing pipeline or other analysis/visualization tools.
"""

import numpy as np
from typing import Tuple, Optional
# We use optional SciPy for potential performance improvements (KD-tree), but it's not required.
try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

def compute_lyapunov(signal: np.ndarray, 
                     k: int = 1, 
                     len_trajectory: int = 20, 
                     min_separation: int = 1, 
                     sample_step: int = 1, 
                     outlier_threshold: Optional[float] = None) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Estimate the largest Lyapunov exponent from a concept trajectory.
    
    This function computes the Lyapunov exponent by analyzing how quickly nearby states 
    in the concept trajectory diverge. It follows the approach of Rosenstein et al. to 
    compute the average divergence of nearest-neighbor trajectories and fits 
    a line to the log-divergence curve to obtain the exponent (slope).
    
    Parameters:
        signal : np.ndarray 
            Time series of concept states. Shape (T, D) for a trajectory of length T in D-dimensional concept space. 
            If D=1, can also pass a 1D array of length T.
        k : int, default 1
            Number of nearest neighbors to use for each point in the trajectory. The divergence will be averaged 
            over these k neighbors (for each reference point) to improve stability.
        len_trajectory : int, default 20
            Number of time steps to follow the trajectory pairs when computing divergence. This is the maximum k 
            (time horizon) for which we compute distances d(i, k) between a point i and its neighbor at time k.
        min_separation : int, default 1
            Minimum temporal separation between points when selecting nearest neighbors. Neighbors whose indices 
            differ by less than or equal to this value are excluded to avoid trivial adjacency.
        sample_step : int, default 1
            Step size for sampling the trajectory. Use 1 to consider every point, 2 to take every other point, etc. 
            This can be used to down-sample long trajectories for performance or to skip highly correlated points.
        outlier_threshold : float, optional
            If provided, any initial neighbor pair with distance above this threshold is ignored. This helps filter 
            out outlier pairs that are not truly "nearest" in a local sense or where a large jump might skew the results.
    
    Returns:
        exponent : float or None
            The estimated Lyapunov exponent (slope of divergence curve). Returns None if it cannot be computed 
            (e.g., trajectory too short or all divergence distances are zero).
        divergence_curve : np.ndarray or None
            Array of length L (<= len_trajectory) containing the average log divergence at each time step from 1 to L. 
            This is the data used for the linear fit. None if exponent could not be computed.
    """
    # Ensure signal is a NumPy array of float type
    signal = np.array(signal, dtype=float)
    if signal.ndim == 1:
        # Convert 1D array to 2D (T,1) for uniform handling
        signal = signal[:, None]
    T = signal.shape[0]  # length of trajectory
    if T < 2:
        # Not enough points to form any trajectory pair
        return None, None
    
    # Down-sample the signal if sample_step > 1
    if sample_step > 1:
        signal = signal[::sample_step]
        T = signal.shape[0]
    # After downsampling, re-check minimum length condition
    if T < 2:
        return None, None

    # Determine effective trajectory length to follow
    max_len = T - 1  # we can at most follow to T-1 steps (from first point to last point)
    L = min(len_trajectory, max_len)
    if L < 2:
        # If even after adjustment we cannot have at least 2 points in divergence curve, give up
        return None, None

    # Number of starting trajectory points we will use (those that have at least L steps ahead)
    # We will consider reference points 0 to N_ref-1 as starting indices for divergence measurement.
    N_ref = T - L  # if we follow L steps, the last starting index is T-L-1 (0-based)
    if N_ref < 1:
        return None, None

    # Build a data structure for neighbor search. Use KD-tree if available and beneficial.
    use_kdtree = (cKDTree is not None) and (T > 200)  # heuristic: for large T, KD-tree can be faster
    if use_kdtree:
        tree = cKDTree(signal)
    else:
        tree = None
        # Precompute pairwise distances matrix if T is not huge, else we will compute distances on the fly.
        # Note: For very large T without KDTree, on-the-fly search is O(T) per point which is O(T^2) in worst case.
        if T <= 1000:
            # Compute full distance matrix for efficiency (O(T^2) but manageable if T is up to 1000).
            dist_matrix = np.linalg.norm(signal[:, None] - signal[None, :], axis=2)
        else:
            dist_matrix = None

    # Prepare list to collect log-distances for each time step
    avg_log_divergence = []

    # We will iterate time steps from 1 to L (inclusive) for divergence.
    # Note: step 0 (initial distance) is used for filtering but we start our divergence curve at step 1.
    for step in range(1, L + 1):
        log_dists = []  # log distances at this step for all pairs
        # Iterate over each reference point i that has a neighbor trajectory of length L
        # i goes from 0 to N_ref-1
        for i in range(N_ref):
            # Find up to k nearest neighbors for point i, with index difference > min_separation
            if use_kdtree:
                # Query k + some buffer
                query_k = k + 5  # buffer to account for neighbors that might be excluded
                dists, idxs = tree.query(signal[i], k=query_k)
                # tree.query returns sorted by distance. It's possible it returns fewer than query_k if near boundaries.
                # Ensure dists, idxs are arrays (when k=1, they might be scalars).
                dists = np.atleast_1d(dists)
                idxs = np.atleast_1d(idxs)
                # Exclude itself and points within min_separation
                valid_neighbors = []
                for dist_val, j in zip(dists, idxs):
                    if j == i:
                        continue  # skip itself
                    if abs(j - i) <= min_separation:
                        continue  # skip neighbors too close in time
                    # If outlier threshold is set, skip neighbor if initial distance (step=0) is too large
                    if outlier_threshold is not None and dist_val > outlier_threshold:
                        continue
                    valid_neighbors.append((dist_val, j))
                    if len(valid_neighbors) >= k:
                        break  # we found k valid neighbors
                if len(valid_neighbors) == 0:
                    # No valid neighbor found for this point
                    continue
                # We have up to k valid neighbors; now evaluate distance at the given step for each
                for dist0, j in valid_neighbors:
                    # Only consider this neighbor if both i+step and j+step are within bounds
                    if i + step < T and j + step < T:
                        # Compute distance at this time step
                        dist_step = np.linalg.norm(signal[i + step] - signal[j + step])
                        if dist_step > 0:
                            log_dists.append(np.log(dist_step))
            else:
                # Not using KD-tree: use precomputed matrix or brute force search
                if dist_matrix is not None:
                    # Use precomputed distances for initial neighbor search
                    # Apply temporal exclusion by masking out neighbors within min_separation (including itself)
                    i_dists = dist_matrix[i]
                    # Temporarily set distances to inf for indices in [i - min_separation, i + min_separation]
                    start_excl = max(0, i - min_separation)
                    end_excl = min(T, i + min_separation + 1)
                    i_dists_excl = i_dists.copy()
                    i_dists_excl[start_excl:end_excl] = np.inf
                    # Find indices of k smallest distances
                    neighbor_idxs = np.argpartition(i_dists_excl, k)[:k]  # unordered k nearest
                    neighbor_idxs = neighbor_idxs[np.argsort(i_dists_excl[neighbor_idxs])][:k]  # sort them
                else:
                    # Large T without KD-tree or full matrix: do brute force neighbor search for point i
                    best_neighbors = []  # list of (distance, j) tuples
                    max_heap = []  # not really a heap, will use list and sort for simplicity since k is small
                    for j in range(T):
                        if j == i or abs(j - i) <= min_separation:
                            continue
                        dist_ij = np.linalg.norm(signal[i] - signal[j])
                        if outlier_threshold is not None and dist_ij > outlier_threshold:
                            continue
                        # Collect potential neighbor
                        best_neighbors.append((dist_ij, j))
                    if not best_neighbors:
                        continue
                    # Sort by distance and take k
                    best_neighbors.sort(key=lambda x: x[0])
                    neighbor_idxs = [bj for _, bj in best_neighbors[:k]]
                # Now we have neighbor indices for point i
                for j in neighbor_idxs:
                    if i + step < T and j + step < T:
                        dist_step = np.linalg.norm(signal[i + step] - signal[j + step])
                        if dist_step > 0:
                            log_dists.append(np.log(dist_step))
        # After iterating all i, compute average log distance for this step
        if len(log_dists) == 0:
            # No valid divergence data at this step (e.g., maybe outlier_threshold filtered everything)
            avg_log_divergence.append(-np.inf)  # mark as invalid
        else:
            avg_log_divergence.append(np.mean(log_dists))
    # Convert to numpy array for easier handling
    avg_log_divergence = np.array(avg_log_divergence)
    # Remove any trailing -inf entries (if some of the later steps had no data)
    valid_mask = np.isfinite(avg_log_divergence)
    if valid_mask.sum() < 2:
        # Fewer than 2 valid points to fit a line (can't compute a reliable slope)
        return None, avg_log_divergence if valid_mask.sum() > 0 else None
    times = np.arange(1, len(avg_log_divergence) + 1)[valid_mask]  # 1-indexed time steps for each average log-divergence
    divergence_vals = avg_log_divergence[valid_mask]
    # Linear fit (least squares) to get slope
    # We use numpy polyfit for simplicity (degree 1 polynomial fit)
    slope, intercept = np.polyfit(times, divergence_vals, 1)
    exponent = float(slope)
    return exponent, avg_log_divergence

def find_concept_sequences(labels: list, blocks_indices: list) -> dict:
    """
    Extract temporal sequences for each concept as they appear through the document.
    
    Args:
        labels: Cluster labels for each text block
        blocks_indices: Original indices of blocks in document order
    
    Returns:
        Dictionary mapping concept IDs to ordered sequences of embeddings indices
    """
    sequences = {}
    # Sort indices by their original position in document
    sorted_pairs = sorted(zip(blocks_indices, range(len(labels))), key=lambda x: x[0])
    # Group by concept ID in document order
    for orig_idx, emb_idx in sorted_pairs:
        concept_id = labels[emb_idx]
        if concept_id not in sequences:
            sequences[concept_id] = []
        sequences[concept_id].append(emb_idx)
    return sequences

def concept_predictability(
    labels: list, 
    emb: np.ndarray, 
    blocks_indices: list,
    min_sequence: int = 5,
    k: int = 3,
    len_trajectory: int = 15,
    min_separation: int = 1
) -> dict:
    """
    Calculate predictability scores for each concept using Lyapunov exponents.
    
    Args:
        labels: Cluster labels for each text block
        emb: Embedding vectors for text blocks
        blocks_indices: Original position indices of blocks in the document
        min_sequence: Minimum sequence length to calculate Lyapunov exponent
        k: Number of nearest neighbors to consider
        len_trajectory: Maximum trajectory length for divergence calculation
        min_separation: Minimum temporal separation between points
    
    Returns:
        Dictionary mapping concept IDs to predictability scores (1.0 = predictable, 0.0 = chaotic)
    """
    concept_sequences = find_concept_sequences(labels, blocks_indices)
    predictability_scores = {}
    
    for concept_id, indices in concept_sequences.items():
        if len(indices) < min_sequence:
            predictability_scores[concept_id] = 0.5  # Neutral for short sequences
            continue
            
        # Extract embedding sequence for this concept
        sequence = [emb[idx] for idx in indices]
        
        # Calculate Lyapunov exponent using the improved algorithm
        lyapunov, _ = compute_lyapunov(
            signal=np.array(sequence),
            k=k,
            len_trajectory=len_trajectory,
            min_separation=min_separation
        )
        
        # Convert to predictability score (bounded 0-1)
        # Negative Lyapunov = predictable, Positive = chaotic
        if lyapunov is None:
            # Not enough data, use neutral score
            predictability_scores[concept_id] = 0.5
            continue
            
        if lyapunov > 1.0:
            lyapunov = 1.0
        if lyapunov < -1.0:
            lyapunov = -1.0
            
        # Transform [-1, 1] to [1, 0] range (high predictability to low)
        predictability = (1.0 - (lyapunov + 1.0) / 2.0)
        predictability_scores[concept_id] = predictability
        
    return predictability_scores

def document_chaos_profile(
    labels: list, 
    emb: np.ndarray, 
    blocks_indices: list,
    window_size: int = 10,
    k: int = 3,
    len_trajectory: int = 10
) -> list:
    """
    Generate a chaos profile of the document as it progresses.
    
    Args:
        labels: Cluster labels for each text block
        emb: Embedding vectors for text blocks
        blocks_indices: Original position indices of blocks
        window_size: Sliding window size for local analysis
        k: Number of nearest neighbors for Lyapunov calculation
        len_trajectory: Trajectory length for divergence calculation
    
    Returns:
        List of chaos values for each window position
    """
    # Sort blocks by their original order
    sorted_indices = [i for _, i in sorted(zip(blocks_indices, range(len(blocks_indices))))]
    sorted_emb = emb[sorted_indices]
    
    # Calculate local Lyapunov exponents in sliding windows
    chaos_profile = []
    for i in range(0, len(sorted_indices) - window_size + 1):
        window = sorted_emb[i:i+window_size]
        local_lyap, _ = compute_lyapunov(window, k=k, len_trajectory=len_trajectory, min_separation=1)
        
        # If we couldn't compute Lyapunov, use a neutral value
        if local_lyap is None:
            chaos_profile.append(0.5)
            continue
            
        # Map Lyapunov to chaos score (0-1)
        # Higher positive Lyapunov = more chaotic
        if local_lyap > 1.0:
            local_lyap = 1.0
        if local_lyap < -1.0:
            local_lyap = -1.0
            
        chaos = (local_lyap + 1.0) / 2.0  # Transform [-1,1] to [0,1]
        chaos_profile.append(chaos)
        
    return chaos_profile
