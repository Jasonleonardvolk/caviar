from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# {PROJECT_ROOT}\cog\transfer.py

"""
Transfer Morphism Implementation
===============================

Implements transfer learning via persistent homology and topological
data analysis.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import warnings

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    warnings.warn("GUDHI library not available. TransferMorphism functionality limited.")


class TransferMorphism:
    """
    Transfer morphism τ via persistent homology.
    
    Extracts topological features from cognitive state trajectories
    to enable transfer learning across different cognitive contexts.
    
    Attributes:
        homology_max_edge: Maximum edge length for Rips complex
        homology_dim: Maximum homological dimension to compute
        persistence_threshold: Threshold for filtering persistence features
    """
    
    def __init__(self, 
                 homology_max_edge: float = 1.0,
                 homology_dim: int = 2,
                 persistence_threshold: float = 0.01):
        """
        Initialize transfer morphism.
        
        Args:
            homology_max_edge: Maximum edge length for Rips complex
            homology_dim: Maximum dimension for homology computation
            persistence_threshold: Minimum persistence for features
        """
        if not GUDHI_AVAILABLE:
            raise ImportError("GUDHI library required for TransferMorphism. "
                            "Install with: pip install gudhi")
        
        self.max_edge = homology_max_edge
        self.dim = homology_dim
        self.persistence_threshold = persistence_threshold
        self.cached_features = {}

    def transfer(self, point_cloud: np.ndarray) -> List[Tuple]:
        """
        Compute persistent homology of cognitive trajectory.
        
        Args:
            point_cloud: Array of cognitive states (n_points × n_features)
            
        Returns:
            List of persistence pairs (dimension, (birth, death))
        """
        # Validate input
        if len(point_cloud) < 2:
            return []
        
        # Create Rips complex
        rips = gudhi.RipsComplex(points=point_cloud, max_edge_length=self.max_edge)
        
        # Create simplex tree
        st = rips.create_simplex_tree(max_dimension=self.dim)
        
        # Compute persistence
        persistence = st.persistence()
        
        # Filter by threshold
        filtered = [(dim, (birth, death)) for dim, (birth, death) in persistence
                   if abs(death - birth) > self.persistence_threshold]
        
        return filtered

    def extract_features(self, point_cloud: np.ndarray) -> Dict[str, Any]:
        """
        Extract topological features for transfer learning.
        
        Args:
            point_cloud: Cognitive state trajectory
            
        Returns:
            Dictionary of topological features
        """
        # Compute persistence
        persistence = self.transfer(point_cloud)
        
        # Initialize feature dictionary
        features = {
            'betti_numbers': {},
            'persistence_entropy': {},
            'average_persistence': {},
            'max_persistence': {},
            'persistence_landscape': {}
        }
        
        # Compute features for each dimension
        for dim in range(self.dim + 1):
            dim_pairs = [(b, d) for d, (b, d) in persistence if d == dim]
            
            if dim_pairs:
                births = np.array([b for b, _ in dim_pairs])
                deaths = np.array([d for _, d in dim_pairs])
                lifetimes = deaths - births
                
                # Betti number
                features['betti_numbers'][dim] = len(dim_pairs)
                
                # Persistence entropy
                if len(lifetimes) > 0 and np.sum(lifetimes) > 0:
                    probs = lifetimes / np.sum(lifetimes)
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    features['persistence_entropy'][dim] = entropy
                else:
                    features['persistence_entropy'][dim] = 0.0
                
                # Average and max persistence
                features['average_persistence'][dim] = np.mean(lifetimes)
                features['max_persistence'][dim] = np.max(lifetimes)
                
                # Persistence landscape (simplified)
                features['persistence_landscape'][dim] = self._compute_landscape(dim_pairs)
            else:
                # No features in this dimension
                features['betti_numbers'][dim] = 0
                features['persistence_entropy'][dim] = 0.0
                features['average_persistence'][dim] = 0.0
                features['max_persistence'][dim] = 0.0
                features['persistence_landscape'][dim] = np.zeros(10)
        
        return features

    def _compute_landscape(self, persistence_pairs: List[Tuple[float, float]], 
                          n_points: int = 10) -> np.ndarray:
        """
        Compute simplified persistence landscape.
        
        Args:
            persistence_pairs: List of (birth, death) pairs
            n_points: Number of landscape points
            
        Returns:
            Landscape values
        """
        if not persistence_pairs:
            return np.zeros(n_points)
        
        # Get range
        all_values = []
        for b, d in persistence_pairs:
            all_values.extend([b, d])
        
        min_val = min(all_values)
        max_val = max(all_values)
        
        # Sample points
        t_values = np.linspace(min_val, max_val, n_points)
        landscape = np.zeros(n_points)
        
        # Compute landscape
        for i, t in enumerate(t_values):
            values = []
            for b, d in persistence_pairs:
                if b <= t <= d:
                    values.append(min(t - b, d - t))
            
            if values:
                landscape[i] = max(values)
        
        return landscape

    def compare_topologies(self, 
                          trajectory1: np.ndarray,
                          trajectory2: np.ndarray) -> float:
        """
        Compare topological similarity between trajectories.
        
        Args:
            trajectory1: First cognitive trajectory
            trajectory2: Second cognitive trajectory
            
        Returns:
            Similarity score in [0, 1]
        """
        # Extract features
        features1 = self.extract_features(trajectory1)
        features2 = self.extract_features(trajectory2)
        
        # Compute similarity for each feature type
        similarities = []
        
        # Betti number similarity
        for dim in range(self.dim + 1):
            b1 = features1['betti_numbers'].get(dim, 0)
            b2 = features2['betti_numbers'].get(dim, 0)
            sim = 1.0 / (1.0 + abs(b1 - b2))
            similarities.append(sim)
        
        # Persistence entropy similarity
        for dim in range(self.dim + 1):
            e1 = features1['persistence_entropy'].get(dim, 0)
            e2 = features2['persistence_entropy'].get(dim, 0)
            sim = 1.0 / (1.0 + abs(e1 - e2))
            similarities.append(sim)
        
        # Landscape similarity
        for dim in range(self.dim + 1):
            l1 = features1['persistence_landscape'].get(dim, np.zeros(10))
            l2 = features2['persistence_landscape'].get(dim, np.zeros(10))
            if len(l1) == len(l2):
                sim = 1.0 / (1.0 + np.linalg.norm(l1 - l2))
                similarities.append(sim)
        
        # Average similarity
        return np.mean(similarities) if similarities else 0.0

    def transfer_knowledge(self,
                          source_trajectory: np.ndarray,
                          target_initial: np.ndarray,
                          n_steps: int = 10) -> np.ndarray:
        """
        Transfer topological structure from source to target.
        
        Args:
            source_trajectory: Source cognitive trajectory
            target_initial: Initial state for target
            n_steps: Number of transfer steps
            
        Returns:
            Transferred trajectory
        """
        # Extract source topology
        source_features = self.extract_features(source_trajectory)
        
        # Initialize target trajectory
        target_trajectory = [target_initial.copy()]
        
        # Generate trajectory preserving topological features
        for step in range(n_steps - 1):
            current = target_trajectory[-1]
            
            # Compute next state to match source topology
            # This is a simplified version - in practice would use optimization
            noise = np.random.randn(*current.shape) * 0.1
            next_state = current + noise
            
            # Add some drift towards matching Betti numbers
            if step > 0 and step % 5 == 0:
                current_features = self.extract_features(np.array(target_trajectory))
                for dim in range(min(2, self.dim + 1)):
                    source_betti = source_features['betti_numbers'].get(dim, 0)
                    current_betti = current_features['betti_numbers'].get(dim, 0)
                    if current_betti < source_betti:
                        # Add some structure
                        next_state += np.random.randn(*current.shape) * 0.05
            
            target_trajectory.append(next_state)
        
        return np.array(target_trajectory)