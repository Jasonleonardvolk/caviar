#!/usr/bin/env python3
"""
TorusCells - Topology-aware memory module
Thin facade around topology computation libraries
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path

# Try to import topology libraries
GUDHI_AVAILABLE = False
RIPSER_AVAILABLE = False

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    pass

try:
    import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    pass

from .torus_registry import TorusRegistry, get_torus_registry

logger = logging.getLogger(__name__)

class TorusCells:
    """
    Topology-aware memory cells that track persistent homology
    Uses gudhi or ripser for Betti number computation
    """
    
    def __init__(self, registry: Optional[TorusRegistry] = None,
                 max_dimension: int = 2,
                 persistence_threshold: float = 0.01):
        """
        Initialize TorusCells
        
        Args:
            registry: TorusRegistry instance (uses default if None)
            max_dimension: Maximum homology dimension to compute
            persistence_threshold: Minimum persistence for features
        """
        self.registry = registry or get_torus_registry()
        self.max_dimension = max_dimension
        self.persistence_threshold = persistence_threshold
        
        # Check available backends
        if not (GUDHI_AVAILABLE or RIPSER_AVAILABLE):
            logger.warning("No topology backend available. Install gudhi or ripser-py")
            logger.warning("Falling back to simple connected components counting")
            
        self.backend = self._select_backend()
        logger.info(f"TorusCells initialized with backend: {self.backend}")
        
        # Cache for topologically protected ideas
        self.protected_ideas = {}  # braid_id -> idea_data
        
    def _select_backend(self) -> str:
        """Select best available topology backend"""
        if GUDHI_AVAILABLE:
            return "gudhi"
        elif RIPSER_AVAILABLE:
            return "ripser"
        else:
            return "simple"
    
    def betti_update(self, idea_id: str, vertices: np.ndarray, 
                    coherence_band: str = "local",
                    metadata: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """
        Update Betti numbers for a cognitive structure
        
        Args:
            idea_id: Unique identifier for the idea/concept
            vertices: Point cloud or vertices defining the structure
            coherence_band: Current coherence state
            metadata: Additional metadata
            
        Returns:
            (betti0, betti1) tuple
        """
        # Compute Betti numbers
        betti_numbers = self.compute_betti(vertices)
        
        # Record in registry
        shape_id = self.registry.record_shape(
            vertices=vertices,
            betti_numbers=betti_numbers,
            coherence_band=coherence_band,
            metadata={
                'idea_id': idea_id,
                **(metadata or {})
            }
        )
        
        # Check if idea should be topologically protected
        if self._is_topologically_protected(betti_numbers, coherence_band):
            self._mark_protected(idea_id, shape_id, betti_numbers)
            
        return betti_numbers[0], betti_numbers[1]
    
    def compute_betti(self, vertices: np.ndarray) -> List[float]:
        """
        Compute Betti numbers using available backend
        
        Args:
            vertices: Point cloud (n_points x n_dims)
            
        Returns:
            List of Betti numbers [b0, b1, b2, ...]
        """
        if self.backend == "gudhi":
            return self._compute_betti_gudhi(vertices)
        elif self.backend == "ripser":
            return self._compute_betti_ripser(vertices)
        else:
            return self._compute_betti_simple(vertices)
    
    def _compute_betti_gudhi(self, vertices: np.ndarray) -> List[float]:
        """Compute using GUDHI"""
        # Create Rips complex
        rips = gudhi.RipsComplex(points=vertices, max_edge_length=2.0)
        simplex_tree = rips.create_simplex_tree(max_dimension=self.max_dimension + 1)
        
        # Compute persistence
        persistence = simplex_tree.compute_persistence()
        
        # Extract Betti numbers
        betti = []
        for dim in range(self.max_dimension + 1):
            # Count features with persistence above threshold
            persistent_features = [
                p for p in simplex_tree.persistence_intervals_in_dimension(dim)
                if p[1] - p[0] > self.persistence_threshold
            ]
            betti.append(float(len(persistent_features)))
            
        return betti
    
    def _compute_betti_ripser(self, vertices: np.ndarray) -> List[float]:
        """Compute using Ripser"""
        # Run ripser
        result = ripser.ripser(vertices, maxdim=self.max_dimension, thresh=2.0)
        
        # Extract Betti numbers
        betti = []
        for dim in range(self.max_dimension + 1):
            if f'dgms' in result and dim < len(result['dgms']):
                dgm = result['dgms'][dim]
                # Count features with persistence above threshold
                persistent = sum(1 for birth, death in dgm 
                               if death - birth > self.persistence_threshold)
                betti.append(float(persistent))
            else:
                betti.append(0.0)
                
        return betti
    
    def _compute_betti_simple(self, vertices: np.ndarray) -> List[float]:
        """Simple fallback: connected components only"""
        # Just count connected components (b0) approximately
        n_points = len(vertices)
        
        # Simple distance threshold clustering
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(vertices))
        connected = distances < 1.0  # Threshold
        
        # Count components using DFS
        visited = set()
        n_components = 0
        
        for i in range(n_points):
            if i not in visited:
                # DFS to find component
                stack = [i]
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        neighbors = np.where(connected[node])[0]
                        stack.extend(neighbors)
                n_components += 1
                
        # Return simplified Betti numbers
        return [float(n_components), 0.0, 0.0]
    
    def _is_topologically_protected(self, betti_numbers: List[float], 
                                   coherence_band: str) -> bool:
        """
        Determine if an idea should be topologically protected
        
        An idea is protected if:
        - It has non-trivial topology (b1 > 0 or b2 > 0)
        - It persists across coherence bands
        """
        # Non-trivial topology check
        has_loops = len(betti_numbers) > 1 and betti_numbers[1] > 0
        has_voids = len(betti_numbers) > 2 and betti_numbers[2] > 0
        
        # Protection criteria
        return (has_loops or has_voids) and coherence_band != "local"
    
    def _mark_protected(self, idea_id: str, shape_id: str, 
                       betti_numbers: List[float]):
        """Mark an idea as topologically protected"""
        self.protected_ideas[idea_id] = {
            'shape_id': shape_id,
            'betti': betti_numbers,
            'braid_id': f"braid_{shape_id}",  # For routing
            'protection_level': 'high' if betti_numbers[1] > 1 else 'medium'
        }
        logger.info(f"Idea {idea_id} marked as topologically protected "
                   f"(b0={betti_numbers[0]}, b1={betti_numbers[1]})")
    
    def get_protected_ideas(self) -> Dict[str, Any]:
        """Get all topologically protected ideas"""
        return self.protected_ideas.copy()
    
    def tunnel_idea(self, idea_id: str, target_coherence: str) -> bool:
        """
        Tunnel a protected idea to a different coherence band
        
        Args:
            idea_id: ID of protected idea
            target_coherence: Target coherence band
            
        Returns:
            True if tunneling successful
        """
        if idea_id not in self.protected_ideas:
            logger.warning(f"Idea {idea_id} is not protected")
            return False
            
        # Log tunneling event
        protection = self.protected_ideas[idea_id]
        logger.info(f"Tunneling idea {idea_id} to {target_coherence} band")
        
        # Update registry with new coherence
        # (In real implementation, this would involve more complex routing)
        self.registry.record_shape(
            vertices=np.array([]),  # Placeholder
            betti_numbers=protection['betti'],
            coherence_band=target_coherence,
            metadata={
                'idea_id': idea_id,
                'tunneled': True,
                'braid_id': protection['braid_id']
            }
        )
        
        return True
    
    def compute_homology_persistence(self, time_series: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compute persistence of homology classes across time
        
        Args:
            time_series: List of vertex arrays over time
            
        Returns:
            Persistence diagram and statistics
        """
        persistence_data = []
        
        for t, vertices in enumerate(time_series):
            betti = self.compute_betti(vertices)
            persistence_data.append({
                'time': t,
                'betti': betti
            })
            
        # Analyze persistence
        b0_series = [p['betti'][0] for p in persistence_data]
        b1_series = [p['betti'][1] for p in persistence_data if len(p['betti']) > 1]
        
        return {
            'persistence_data': persistence_data,
            'statistics': {
                'b0_mean': float(np.mean(b0_series)),
                'b0_std': float(np.std(b0_series)),
                'b1_mean': float(np.mean(b1_series)) if b1_series else 0.0,
                'b1_std': float(np.std(b1_series)) if b1_series else 0.0,
                'stable_features': self._count_stable_features(persistence_data)
            }
        }
    
    def _count_stable_features(self, persistence_data: List[Dict]) -> int:
        """Count features that persist across multiple timesteps"""
        if len(persistence_data) < 2:
            return 0
            
        stable_count = 0
        
        # Simple stability: feature exists in >50% of timesteps
        n_times = len(persistence_data)
        threshold = n_times * 0.5
        
        # Check b1 persistence
        b1_counts = sum(1 for p in persistence_data 
                       if len(p['betti']) > 1 and p['betti'][1] > 0)
        if b1_counts > threshold:
            stable_count += int(np.mean([p['betti'][1] for p in persistence_data 
                                        if len(p['betti']) > 1]))
            
        return stable_count

# Convenience functions
_torus_cells = None

def get_torus_cells() -> TorusCells:
    """Get or create global TorusCells instance"""
    global _torus_cells
    if _torus_cells is None:
        _torus_cells = TorusCells()
    return _torus_cells

def betti0_1(vertices: np.ndarray) -> Tuple[float, float]:
    """
    Quick helper to compute just b0 and b1
    Used by other modules that need basic topology
    """
    cells = get_torus_cells()
    betti = cells.compute_betti(vertices)
    return betti[0], betti[1] if len(betti) > 1 else 0.0
