import math
import time
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional
from alan_backend.banksy import oscillator_update

def cluster_cohesion(emb: np.ndarray, members: List[int]) -> float:
    """Calculate cluster cohesion using cosine similarity."""
    if len(members) < 2:
        return 0.0
    sub = emb[members]
    sim = (sub @ sub.T) / ((np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8) * (np.linalg.norm(sub, axis=1, keepdims=True).T + 1e-8))
    return (sim.sum() - len(sub)) / max(len(sub)*(len(sub)-1), 1)

def run_oscillator_clustering_enhanced(
    emb: np.ndarray, 
    steps: int = 60, 
    tol: float = 1e-3, 
    cohesion_threshold: float = 0.15,
    enable_logging: bool = False
) -> Dict[str, Any]:
    """
    Enhanced oscillator clustering with detailed logging and benchmarking capabilities.
    
    Returns:
        Dictionary containing labels, clusters, cohesion scores, runtime, and convergence info
    """
    start_time = time.perf_counter()
    
    n = emb.shape[0]
    if n == 1:
        return {
            "labels": [0],
            "clusters": {0: [0]},
            "cohesion_scores": {0: 0.0},
            "runtime": time.perf_counter() - start_time,
            "convergence_step": 0,
            "total_steps": 0,
            "phase_variance": 0.0,
            "singleton_merges": 0,
            "orphan_reassignments": 0,
            "removed_low_cohesion": 0
        }
    
    # Build cosine similarity matrix
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    K = (emb @ emb.T) / (norms * norms.T)
    np.fill_diagonal(K, 0)
    
    # Initialize random phases
    theta = np.random.uniform(0, 2 * math.pi, n)
    initial_phases = theta.copy()
    
    convergence_step = steps
    phase_variances = []
    
    # Oscillator evolution
    for step in range(steps):
        nxt = oscillator_update(theta, np.ones(n), 0.25, 0.05, K, 0)
        phase_diff = np.linalg.norm(nxt - theta)
        phase_variances.append(np.var(nxt))
        
        if phase_diff < tol:
            convergence_step = step
            break
        theta = nxt
    
    # Bucket phases
    buckets: Dict[int, List[int]] = defaultdict(list)
    for i, th in enumerate(theta):
        buckets[int((th % (2 * math.pi)) / 0.25)].append(i)
    
    # Assign initial cluster labels
    labels = [-1] * n
    cid_map = {}
    for cid, (_, mem) in enumerate(sorted(buckets.items(), key=lambda kv: -len(kv[1]))):
        for m in mem:
            labels[m] = cid
        cid_map[cid] = mem
    
    # Track modifications
    singleton_merges = 0
    orphan_reassignments = 0
    removed_low_cohesion = 0
    
    # Handle singletons - merge with nearest cluster
    for cid, mem in list(cid_map.items()):
        if len(mem) == 1:
            idx = mem[0]
            # Find nearest other cluster by cosine similarity
            best_cid, best_sim = None, -1e9
            for ocid, omem in cid_map.items():
                if ocid == cid or len(omem) < 1:
                    continue
                sim = float(emb[idx] @ emb[omem[0]] / (np.linalg.norm(emb[idx])*np.linalg.norm(emb[omem[0]])+1e-8))
                if sim > best_sim:
                    best_cid, best_sim = ocid, sim
            if best_cid is not None and best_sim > 0.4:
                # Merge singleton into nearest cluster
                labels[idx] = best_cid
                cid_map[best_cid].append(idx)
                cid_map[cid] = []
                singleton_merges += 1
    
    # Remove clusters with low cohesion
    for cid, mem in list(cid_map.items()):
        if len(mem) < 2:
            continue
        coh = cluster_cohesion(emb, mem)
        if coh < cohesion_threshold:
            for idx in mem:
                labels[idx] = -1  # Mark as unassigned
            cid_map[cid] = []
            removed_low_cohesion += 1
    
    # Reassign unassigned to nearest cluster
    for idx, l in enumerate(labels):
        if l == -1:
            best_cid, best_sim = None, -1e9
            for cid, mem in cid_map.items():
                if len(mem) == 0:
                    continue
                sim = float(emb[idx] @ emb[mem[0]] / (np.linalg.norm(emb[idx])*np.linalg.norm(emb[mem[0]])+1e-8))
                if sim > best_sim:
                    best_cid, best_sim = cid, sim
            if best_cid is not None and best_sim > 0.3:
                labels[idx] = best_cid
                cid_map[best_cid].append(idx)
                orphan_reassignments += 1
    
    # Calculate final cohesion scores
    cohesion_scores = {}
    for cid, mem in cid_map.items():
        if len(mem) > 0:
            cohesion_scores[cid] = cluster_cohesion(emb, mem)
    
    runtime = time.perf_counter() - start_time
    
    result = {
        "labels": labels,
        "clusters": {cid: mem for cid, mem in cid_map.items() if len(mem) > 0},
        "cohesion_scores": cohesion_scores,
        "runtime": runtime,
        "convergence_step": convergence_step,
        "total_steps": steps,
        "phase_variance": np.mean(phase_variances) if phase_variances else 0.0,
        "final_phase_variance": np.var(theta),
        "similarity_matrix_density": np.mean(K[K > 0]) if np.any(K > 0) else 0.0,
        "singleton_merges": singleton_merges,
        "orphan_reassignments": orphan_reassignments,
        "removed_low_cohesion": removed_low_cohesion,
        "initial_phases": initial_phases.tolist() if enable_logging else None,
        "final_phases": theta.tolist() if enable_logging else None
    }
    
    if enable_logging:
        print(f"Oscillator Clustering Results:")
        print(f"  Runtime: {runtime:.3f}s")
        print(f"  Convergence: {convergence_step}/{steps} steps")
        print(f"  Clusters found: {len(result['clusters'])}")
        print(f"  Singleton merges: {singleton_merges}")
        print(f"  Orphan reassignments: {orphan_reassignments}")
        print(f"  Low cohesion removed: {removed_low_cohesion}")
        print(f"  Average cohesion: {np.mean(list(cohesion_scores.values())):.3f}")
    
    return result

def run_oscillator_clustering(emb: np.ndarray, steps: int = 60, tol: float = 1e-3, cohesion_threshold: float = 0.15) -> List[int]:
    """
    Original oscillator clustering function for backward compatibility.
    """
    result = run_oscillator_clustering_enhanced(emb, steps, tol, cohesion_threshold, False)
    return result["labels"]

# Additional clustering algorithms for benchmarking

def kmeans_clustering(emb: np.ndarray, k: Optional[int] = None, max_iter: int = 100) -> Dict[str, Any]:
    """K-means clustering with automatic k selection if not provided."""
    start_time = time.perf_counter()
    
    n = emb.shape[0]
    if k is None:
        k = max(1, int(np.sqrt(n / 2)))
    
    if n <= k:
        labels = list(range(n))
        centroids = emb.tolist()
    else:
        # Initialize centroids randomly
        indices = np.random.choice(n, k, replace=False)
        centroids = emb[indices].copy()
        
        labels = np.zeros(n, dtype=int)
        
        for iteration in range(max_iter):
            # Assign points to nearest centroid
            distances = np.linalg.norm(emb[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
            new_labels = np.argmin(distances, axis=1)
            
            if np.array_equal(labels, new_labels):
                break
                
            labels = new_labels
            
            # Update centroids
            for i in range(k):
                mask = labels == i
                if np.any(mask):
                    centroids[i] = emb[mask].mean(axis=0)
    
    # Calculate cohesion scores
    cohesion_scores = {}
    clusters = {}
    for i in range(k):
        mask = labels == i
        if np.any(mask):
            cluster_indices = np.where(mask)[0].tolist()
            clusters[i] = cluster_indices
            cohesion_scores[i] = cluster_cohesion(emb, cluster_indices)
    
    runtime = time.perf_counter() - start_time
    
    return {
        "labels": labels.tolist(),
        "clusters": clusters,
        "cohesion_scores": cohesion_scores,
        "runtime": runtime,
        "centroids": centroids.tolist(),
        "iterations": iteration + 1
    }

def hdbscan_clustering(emb: np.ndarray, min_cluster_size: int = 2) -> Dict[str, Any]:
    """HDBSCAN clustering implementation."""
    try:
        import hdbscan
    except ImportError:
        raise ImportError("HDBSCAN requires 'hdbscan' package. Install with: pip install hdbscan")
    
    start_time = time.perf_counter()
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='cosine')
    cluster_labels = clusterer.fit_predict(emb)
    
    # Convert -1 (noise) to separate clusters
    unique_labels = np.unique(cluster_labels)
    noise_points = np.where(cluster_labels == -1)[0]
    
    # Assign noise points their own cluster IDs
    max_label = max(unique_labels) if len(unique_labels) > 0 else -1
    for i, noise_idx in enumerate(noise_points):
        cluster_labels[noise_idx] = max_label + 1 + i
    
    # Build clusters dict
    clusters = {}
    for label in np.unique(cluster_labels):
        clusters[int(label)] = np.where(cluster_labels == label)[0].tolist()
    
    # Calculate cohesion scores
    cohesion_scores = {}
    for cid, members in clusters.items():
        cohesion_scores[cid] = cluster_cohesion(emb, members)
    
    runtime = time.perf_counter() - start_time
    
    return {
        "labels": cluster_labels.tolist(),
        "clusters": clusters,
        "cohesion_scores": cohesion_scores,
        "runtime": runtime,
        "noise_points": len(noise_points),
        "cluster_probabilities": clusterer.probabilities_.tolist() if hasattr(clusterer, 'probabilities_') else None
    }

def affinity_propagation_clustering(emb: np.ndarray, preference: Optional[float] = None) -> Dict[str, Any]:
    """Affinity Propagation clustering implementation."""
    try:
        from sklearn.cluster import AffinityPropagation
    except ImportError:
        raise ImportError("Affinity Propagation requires 'sklearn'. Install with: pip install scikit-learn")
    
    start_time = time.perf_counter()
    
    # Use cosine similarity
    similarity_matrix = (emb @ emb.T) / (np.linalg.norm(emb, axis=1, keepdims=True) * np.linalg.norm(emb, axis=1, keepdims=True).T + 1e-8)
    
    if preference is None:
        preference = np.median(similarity_matrix)
    
    clusterer = AffinityPropagation(affinity='precomputed', preference=preference, random_state=42)
    cluster_labels = clusterer.fit_predict(similarity_matrix)
    
    # Build clusters dict
    clusters = {}
    for label in np.unique(cluster_labels):
        clusters[int(label)] = np.where(cluster_labels == label)[0].tolist()
    
    # Calculate cohesion scores
    cohesion_scores = {}
    for cid, members in clusters.items():
        cohesion_scores[cid] = cluster_cohesion(emb, members)
    
    runtime = time.perf_counter() - start_time
    
    return {
        "labels": cluster_labels.tolist(),
        "clusters": clusters,
        "cohesion_scores": cohesion_scores,
        "runtime": runtime,
        "exemplars": clusterer.cluster_centers_indices_.tolist(),
        "n_iterations": clusterer.n_iter_
    }

def benchmark_all_clustering_methods(emb: np.ndarray, methods: List[str] = None, enable_logging: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark all available clustering methods on the given embeddings.
    
    Args:
        emb: Embedding matrix (n_samples x n_features)
        methods: List of methods to benchmark. If None, uses all available methods.
        enable_logging: Whether to print detailed results
        
    Returns:
        Dictionary with results for each method
    """
    if methods is None:
        methods = ['oscillator', 'kmeans', 'hdbscan', 'affinity_propagation']
    
    results = {}
    
    for method in methods:
        try:
            if method == 'oscillator':
                result = run_oscillator_clustering_enhanced(emb, enable_logging=enable_logging)
            elif method == 'kmeans':
                result = kmeans_clustering(emb)
            elif method == 'hdbscan':
                result = hdbscan_clustering(emb)
            elif method == 'affinity_propagation':
                result = affinity_propagation_clustering(emb)
            else:
                print(f"Unknown method: {method}")
                continue
                
            # Add common metrics
            result['n_clusters'] = len(result['clusters'])
            result['avg_cohesion'] = np.mean(list(result['cohesion_scores'].values())) if result['cohesion_scores'] else 0.0
            result['method'] = method
            
            results[method] = result
            
            if enable_logging:
                print(f"\n{method.upper()} Results:")
                print(f"  Clusters: {result['n_clusters']}")
                print(f"  Avg Cohesion: {result['avg_cohesion']:.3f}")
                print(f"  Runtime: {result['runtime']:.3f}s")
                
        except Exception as e:
            print(f"Error running {method}: {e}")
            results[method] = {"error": str(e)}
    
    return results

def compute_silhouette_score(emb: np.ndarray, labels: List[int]) -> float:
    """Compute silhouette score for clustering evaluation."""
    try:
        from sklearn.metrics import silhouette_score
        return silhouette_score(emb, labels, metric='cosine')
    except ImportError:
        # Fallback implementation
        n = len(labels)
        if len(set(labels)) <= 1:
            return 0.0
            
        silhouette_scores = []
        for i in range(n):
            # Same cluster distances
            same_cluster = [j for j in range(n) if labels[j] == labels[i] and j != i]
            if len(same_cluster) == 0:
                a = 0
            else:
                a = np.mean([np.linalg.norm(emb[i] - emb[j]) for j in same_cluster])
            
            # Other cluster distances
            other_clusters = {}
            for j in range(n):
                if labels[j] != labels[i]:
                    if labels[j] not in other_clusters:
                        other_clusters[labels[j]] = []
                    other_clusters[labels[j]].append(j)
            
            if not other_clusters:
                b = 0
            else:
                cluster_dists = []
                for cluster_indices in other_clusters.values():
                    cluster_dists.append(np.mean([np.linalg.norm(emb[i] - emb[j]) for j in cluster_indices]))
                b = min(cluster_dists)
            
            if max(a, b) == 0:
                s = 0
            else:
                s = (b - a) / max(a, b)
            silhouette_scores.append(s)
        
        return np.mean(silhouette_scores)
