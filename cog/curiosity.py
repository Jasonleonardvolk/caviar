from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# {PROJECT_ROOT}\cog\curiosity.py

"""
Curiosity Functional Implementation
==================================

Implements curiosity-driven exploration through mutual information
and novelty detection mechanisms.
"""

import numpy as np
from sklearn.metrics import mutual_info_score
from typing import List, Optional, Dict, Tuple
from .manifold import MetaCognitiveManifold


class CuriosityFunctional:
    """
    Implements C(s,s') and advanced curiosity mechanisms.
    
    Curiosity drives exploration by quantifying information gain
    and novelty of cognitive states.
    
    Attributes:
        manifold: Cognitive manifold for distance calculations
        decay_const: Exponential decay constant for distance weighting
        bins: Number of bins for mutual information estimation
        exploration_bonus: Weight for novelty-based exploration
        memory_capacity: Maximum size of memory buffer
    """
    
    def __init__(self, 
                 manifold: MetaCognitiveManifold,
                 decay_const: float = 1.0,
                 bins: int = 10,
                 exploration_bonus: float = 1.0,
                 memory_capacity: int = 1000):
        """
        Initialize curiosity functional.
        
        Args:
            manifold: Cognitive manifold
            decay_const: Decay constant for distance-based weighting
            bins: Number of bins for MI estimation
            exploration_bonus: Bonus weight for novel states
            memory_capacity: Maximum memory buffer size
        """
        self.manifold = manifold
        self.decay = decay_const
        self.bins = bins
        self.exploration_bonus = exploration_bonus
        self.memory_capacity = memory_capacity
        self.memory_buffer = []
        self.visit_counts = {}

    def compute(self, s: np.ndarray, s_prime: np.ndarray) -> float:
        """
        Compute curiosity between two cognitive states.
        
        C(s,s') = MI(s,s') * exp(-λ * d(s,s'))
        
        Args:
            s: First cognitive state
            s_prime: Second cognitive state
            
        Returns:
            Curiosity value
        """
        # Prepare data for MI estimation
        data = np.vstack([s, s_prime]).flatten()
        bin_edges = np.histogram_bin_edges(data, bins=self.bins)
        
        # Discretize states
        a = np.digitize(s, bin_edges[:-1])
        b = np.digitize(s_prime, bin_edges[:-1])
        
        # Compute mutual information
        mi = mutual_info_score(a, b)
        
        # Apply distance-based decay
        dist = self.manifold.distance(s, s_prime)
        curiosity = mi * np.exp(-self.decay * dist)
        
        return curiosity

    def compute_advanced(self, 
                        s: np.ndarray, 
                        environment: np.ndarray,
                        memory_buffer: Optional[List[np.ndarray]] = None) -> float:
        """
        Compute advanced curiosity with information gain and novelty.
        
        Args:
            s: Current cognitive state
            environment: Environmental state/observations
            memory_buffer: Optional external memory buffer
            
        Returns:
            Advanced curiosity score
        """
        # Use provided buffer or internal one
        buffer = memory_buffer if memory_buffer is not None else self.memory_buffer
        
        # Compute information gain
        ig = self._information_gain(s, environment)
        
        # Compute novelty
        novelty = self._novelty_score(s, buffer)
        
        # Compute state visitation bonus
        visit_bonus = self._visitation_bonus(s)
        
        # Combined curiosity
        curiosity = ig + self.exploration_bonus * novelty + 0.5 * visit_bonus
        
        # Update memory
        self._update_memory(s)
        
        return curiosity

    def _information_gain(self, s: np.ndarray, env: np.ndarray) -> float:
        """
        Compute information gain from state about environment.
        
        IG(s,env) = H(env) - H(env|s)
        
        Args:
            s: Cognitive state
            env: Environmental state
            
        Returns:
            Information gain
        """
        # Normalize to probability distributions
        env_norm = np.abs(env) / (np.sum(np.abs(env)) + 1e-10)
        s_norm = np.abs(s) / (np.sum(np.abs(s)) + 1e-10)
        
        # Entropy of environment
        H_env = -np.sum(env_norm * np.log(env_norm + 1e-10))
        
        # Estimate conditional entropy via correlation
        # Higher correlation = lower conditional entropy
        if len(s) == len(env):
            corr = abs(np.corrcoef(s, env)[0, 1])
        else:
            # Handle different dimensions by padding or truncating
            min_len = min(len(s), len(env))
            corr = abs(np.corrcoef(s[:min_len], env[:min_len])[0, 1])
        
        H_env_given_s = H_env * (1 - corr)
        
        return max(0, H_env - H_env_given_s)

    def _novelty_score(self, s: np.ndarray, memory_buffer: List[np.ndarray]) -> float:
        """
        Compute novelty as distance to nearest memory.
        
        Args:
            s: Cognitive state
            memory_buffer: List of previous states
            
        Returns:
            Novelty score in [0, 1]
        """
        if not memory_buffer:
            return 1.0
        
        # Find minimum distance to any memory
        min_dist = float('inf')
        for mem in memory_buffer:
            if len(mem) == len(s):  # Only compare states of same dimension
                dist = self.manifold.distance(s, mem)
                if dist < min_dist:
                    min_dist = dist
        
        # Convert to novelty score with exponential decay
        novelty = 1.0 - np.exp(-min_dist)
        
        return novelty

    def _visitation_bonus(self, s: np.ndarray, resolution: float = 0.1) -> float:
        """
        Compute exploration bonus based on state visitation frequency.
        
        Args:
            s: Cognitive state
            resolution: Discretization resolution
            
        Returns:
            Visitation bonus (higher for less visited states)
        """
        # Discretize state for counting
        s_discrete = tuple(np.round(s / resolution).astype(int))
        
        # Get visit count
        count = self.visit_counts.get(s_discrete, 0)
        
        # Compute bonus: 1/sqrt(n+1)
        bonus = 1.0 / np.sqrt(count + 1)
        
        return bonus

    def _update_memory(self, s: np.ndarray):
        """
        Update memory buffer and visit counts.
        
        Args:
            s: State to add to memory
        """
        # Add to memory buffer
        self.memory_buffer.append(s.copy())
        
        # Maintain capacity limit (FIFO)
        if len(self.memory_buffer) > self.memory_capacity:
            self.memory_buffer.pop(0)
        
        # Update visit counts
        s_discrete = tuple(np.round(s / 0.1).astype(int))
        self.visit_counts[s_discrete] = self.visit_counts.get(s_discrete, 0) + 1
    
    def get_exploration_heatmap(self, bounds: Tuple[float, float], 
                               resolution: int = 50) -> np.ndarray:
        """
        Generate 2D heatmap of exploration for visualization.
        
        Args:
            bounds: (min, max) bounds for each dimension
            resolution: Grid resolution
            
        Returns:
            2D array of visitation counts
        """
        if self.manifold.dimension != 2:
            raise ValueError("Heatmap only supported for 2D manifolds")
        
        heatmap = np.zeros((resolution, resolution))
        x_range = np.linspace(bounds[0], bounds[1], resolution)
        y_range = np.linspace(bounds[0], bounds[1], resolution)
        
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                state = np.array([x, y])
                s_discrete = tuple(np.round(state / 0.1).astype(int))
                heatmap[j, i] = self.visit_counts.get(s_discrete, 0)
        
        return heatmap
    
    def reset_memory(self):
        """Reset memory buffer and visit counts."""
        self.memory_buffer = []
        self.visit_counts = {}