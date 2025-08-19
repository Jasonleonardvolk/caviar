from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# {PROJECT_ROOT}\cog\architecture.py

"""
Advanced Cognitive Architecture Components
=========================================

Implements higher-order structures including:
- Metacognitive Tower (∞-categorical structure)
- Knowledge Sheaf (distributed cognition)
- Attention mechanisms
- Memory systems
"""

import numpy as np
from scipy.linalg import block_diag
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
import warnings
from .manifold import MetaCognitiveManifold
from .utils import compute_iit_phi, normalize_state


class MetacognitiveTower:
    """
    ∞-Categorical tower with dimension control and proper geometric structure.
    
    Implements a hierarchy of cognitive spaces with functorial mappings
    between levels, enabling metacognitive reflection at multiple scales.
    
    Attributes:
        base_dim: Dimension of base cognitive space
        levels: List of metacognitive levels
        metric: Manifold metric type
        max_dim_ratio: Maximum dimension growth ratio
        sparsity_threshold: Threshold for feature selection
    """
    
    def __init__(self, 
                 base_dim: int,
                 levels: int = 3,
                 metric: str = "euclidean",
                 max_dim_ratio: float = 10.0,
                 sparsity_threshold: float = 0.01):
        """
        Initialize metacognitive tower.
        
        Args:
            base_dim: Dimension of base cognitive space
            levels: Number of metacognitive levels
            metric: Manifold metric type
            max_dim_ratio: Maximum ratio of highest/base dimension
            sparsity_threshold: Threshold for sparse feature selection
        """
        self.base_dim = base_dim
        self.metric = metric
        self.max_dim_ratio = max_dim_ratio
        self.sparsity_threshold = sparsity_threshold
        self.levels = []
        
        # Build tower with controlled growth
        for i in range(levels):
            level_dim = self._compute_level_dim(i)
            self.levels.append({
                'manifold': MetaCognitiveManifold(level_dim, metric=metric),
                'dim': level_dim,
                'order': i,
                'christoffel': None,  # Cache Christoffel symbols
                'projection_matrix': None,  # Cache projection operators
                'features': self._get_level_features(i)  # Feature description
            })
    
    def _compute_level_dim(self, level: int) -> int:
        """
        Compute dimension with controlled growth and sparsity.
        
        Args:
            level: Metacognitive level
            
        Returns:
            Dimension of the level
        """
        if level == 0:
            return self.base_dim
        
        # Theoretical dimension based on level
        if level == 1:
            # State + velocity
            theoretical_dim = self.base_dim * 2
        elif level == 2:
            # State + velocity + curvature
            theoretical_dim = self.base_dim * 2 + self.base_dim * (self.base_dim + 1) // 2
        else:
            # Controlled exponential growth
            theoretical_dim = int(self.base_dim * (1.5 ** level))
        
        # Apply dimension cap
        max_allowed = int(self.base_dim * self.max_dim_ratio)
        actual_dim = min(theoretical_dim, max_allowed)
        
        if theoretical_dim > max_allowed:
            warnings.warn(f"Level {level} dimension capped at {actual_dim} "
                         f"(theoretical: {theoretical_dim})")
        
        return actual_dim
    
    def _get_level_features(self, level: int) -> Dict[str, Tuple[int, int]]:
        """
        Get feature ranges for a level.
        
        Args:
            level: Metacognitive level
            
        Returns:
            Dictionary mapping feature names to index ranges
        """
        features = {}
        
        if level == 0:
            features['state'] = (0, self.base_dim)
        elif level == 1:
            features['state'] = (0, self.base_dim)
            features['velocity'] = (self.base_dim, 2 * self.base_dim)
        elif level == 2:
            features['state'] = (0, self.base_dim)
            features['velocity'] = (self.base_dim, 2 * self.base_dim)
            start = 2 * self.base_dim
            n_curv = self.base_dim * (self.base_dim + 1) // 2
            features['curvature'] = (start, start + n_curv)
        else:
            # Higher levels have abstract features
            dim = self.levels[level]['dim']
            chunk_size = max(1, dim // 5)
            features['abstract_1'] = (0, chunk_size)
            features['abstract_2'] = (chunk_size, 2 * chunk_size)
            features['interaction'] = (2 * chunk_size, dim)
        
        return features
    
    def _get_block_structure(self, level: int) -> List[np.ndarray]:
        """
        Get block-diagonal structure for level's metric tensor.
        
        Args:
            level: Metacognitive level
            
        Returns:
            List of block matrices
        """
        if level == 0:
            return [np.eye(self.base_dim)]
        elif level == 1:
            # State block + velocity block
            return [np.eye(self.base_dim), np.eye(self.base_dim)]
        elif level == 2:
            # State + velocity + curvature blocks
            n_curv = self.base_dim * (self.base_dim + 1) // 2
            blocks = [np.eye(self.base_dim), np.eye(self.base_dim)]
            if n_curv > 0:
                blocks.append(np.eye(n_curv))
            return blocks
        else:
            # Higher levels: adaptive blocking
            dim = self.levels[level]['dim']
            block_size = max(self.base_dim, dim // 10)
            blocks = []
            for i in range(0, dim, block_size):
                size = min(block_size, dim - i)
                blocks.append(np.eye(size))
            return blocks
    
    def compute_christoffel_symbols(self, level: int, state: np.ndarray) -> np.ndarray:
        """
        Compute Christoffel symbols Γ^k_ij for Levi-Civita connection.
        
        Args:
            level: Metacognitive level
            state: Cognitive state at that level
            
        Returns:
            Christoffel symbols tensor
        """
        # Check cache
        cache_key = (level, tuple(state[:5]))  # Use first 5 elements as key
        if self.levels[level]['christoffel'] is not None:
            cached_key, cached_gamma = self.levels[level]['christoffel']
            if cache_key == cached_key:
                return cached_gamma
        
        blocks = self._get_block_structure(level)
        dim = self.levels[level]['dim']
        manifold = self.levels[level]['manifold']
        
        # Get metric tensor
        if hasattr(manifold, 'fisher_information_matrix') and self.metric == 'fisher_rao':
            g = manifold.fisher_information_matrix(state)
        else:
            g = block_diag(*blocks)
        
        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(g)
        if np.min(eigvals) < 1e-6:
            g += np.eye(dim) * (1e-6 - np.min(eigvals))
        
        # Simplified Christoffel symbols (exact computation would require metric derivatives)
        gamma = np.zeros((dim, dim, dim))
        
        # For each block, add state-dependent curvature
        for b_idx, block in enumerate(blocks):
            start = sum(b.shape[0] for b in blocks[:b_idx])
            end = start + block.shape[0]
            
            for i in range(start, end):
                for j in range(start, end):
                    for k in range(start, end):
                        # Simple model: Γ^k_ij ~ state-dependent
                        if i == j:
                            gamma[k, i, j] = 0.1 * state[k] / (1 + np.abs(state[k]))
        
        # Cache result
        self.levels[level]['christoffel'] = (cache_key, gamma)
        
        return gamma
    
    def lift(self, state: np.ndarray, from_level: int, to_level: int) -> np.ndarray:
        """
        Functorial lift preserving structure via sequential lifts.
        
        Args:
            state: Cognitive state
            from_level: Source level
            to_level: Target level (must be higher)
            
        Returns:
            Lifted state
        """
        if from_level == to_level:
            return state.copy()
        
        if from_level > to_level:
            raise ValueError("Use project() for downward mappings")
        
        current = state.copy()
        
        # Sequential lifting
        for lvl in range(from_level, to_level):
            current = self._lift_one_level(current, lvl)
        
        return current
    
    def _lift_one_level(self, state: np.ndarray, lvl: int) -> np.ndarray:
        """
        Lift state by one level.
        
        Args:
            state: Current state
            lvl: Current level
            
        Returns:
            Lifted state
        """
        next_dim = self.levels[lvl + 1]['dim']
        lifted = np.zeros(next_dim)
        
        if lvl == 0:
            # Level 0 → 1: Add velocity
            lifted[:self.base_dim] = state
            lifted[self.base_dim:2*self.base_dim] = self._compute_cognitive_velocity(state)
            
        elif lvl == 1:
            # Level 1 → 2: Add curvature
            state_dim = self.levels[1]['dim']
            lifted[:state_dim] = state
            
            # Extract components
            base_state = state[:self.base_dim]
            velocity = state[self.base_dim:2*self.base_dim]
            
            # Compute curvature tensor
            idx = state_dim
            for i in range(self.base_dim):
                for j in range(i, self.base_dim):
                    if idx < next_dim:
                        lifted[idx] = self._compute_awareness_curvature(
                            base_state, velocity, i, j
                        )
                        idx += 1
        else:
            # Higher levels: Abstract lifting
            curr_dim = self.levels[lvl]['dim']
            lifted[:curr_dim] = state
            
            # Extract meta-patterns
            meta_features = self._extract_meta_patterns(state, lvl)
            
            # Select most informative features
            remaining_dim = next_dim - curr_dim
            selected = self._select_sparse_features(meta_features, remaining_dim)
            
            lifted[curr_dim:curr_dim + len(selected)] = selected
        
        return lifted
    
    def project(self, state: np.ndarray, from_level: int, to_level: int) -> np.ndarray:
        """
        Structure-preserving projection via sequential steps.
        
        Args:
            state: Cognitive state
            from_level: Source level
            to_level: Target level (must be lower)
            
        Returns:
            Projected state
        """
        if from_level == to_level:
            return state.copy()
        
        if from_level < to_level:
            raise ValueError("Use lift() for upward mappings")
        
        current = state.copy()
        
        # Sequential projection
        for lvl in range(from_level, to_level, -1):
            current = self._project_one_level(current, lvl)
        
        return current
    
    def _project_one_level(self, state: np.ndarray, lvl: int) -> np.ndarray:
        """
        Project state down one level.
        
        Args:
            state: Current state
            lvl: Current level
            
        Returns:
            Projected state
        """
        target_dim = self.levels[lvl - 1]['dim']
        
        if lvl == 1:
            # Level 1 → 0: Keep only base state
            return state[:self.base_dim]
        
        elif lvl == 2:
            # Level 2 → 1: Keep state and velocity
            return state[:2 * self.base_dim]
        
        else:
            # Higher levels: Use SVD for optimal projection
            # Reshape state for better structure preservation
            matrix_shape = (min(len(state), 10), -1)
            if np.prod(matrix_shape) <= len(state):
                state_matrix = state[:np.prod(matrix_shape)].reshape(matrix_shape)
            else:
                state_matrix = state.reshape(-1, 1)
            
            # SVD projection
            U, s, Vt = np.linalg.svd(state_matrix, full_matrices=False)
            
            # Reconstruct with limited components
            n_components = min(target_dim, len(s))
            projected_matrix = U[:, :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]
            
            # Flatten and truncate
            projected = projected_matrix.flatten()[:target_dim]
            
            # Ensure correct dimension
            if len(projected) < target_dim:
                projected = np.pad(projected, (0, target_dim - len(projected)))
            
            return projected
    
    def _compute_cognitive_velocity(self, state: np.ndarray) -> np.ndarray:
        """
        Compute cognitive velocity from state.
        
        Models the rate of change in cognitive state.
        
        Args:
            state: Base cognitive state
            
        Returns:
            Velocity vector
        """
        # Magnitude from state energy
        magnitude = np.abs(state)
        
        # Phase from state interactions
        phase = np.angle(state + 1j * np.roll(state, 1))
        
        # Velocity as magnitude-weighted phase gradient
        velocity = magnitude * np.gradient(phase)
        
        # Add nonlinear coupling
        velocity += 0.1 * state * np.roll(state, 1)
        
        return velocity
    
    def _compute_awareness_curvature(self, 
                                   state: np.ndarray,
                                   velocity: np.ndarray,
                                   i: int, j: int) -> float:
        """
        Compute awareness curvature tensor component.
        
        Represents second-order metacognitive awareness.
        
        Args:
            state: Base state
            velocity: Cognitive velocity
            i, j: Tensor indices
            
        Returns:
            Curvature component R_ij
        """
        # Curvature from velocity interactions
        curvature = velocity[i] * velocity[j] / (1 + np.abs(state[i] * state[j]))
        
        # Add torsion term
        if i != j:
            torsion = (state[i] * velocity[j] - state[j] * velocity[i]) / (
                1 + np.linalg.norm(state)
            )
            curvature += 0.5 * torsion
        
        return curvature
    
    def _extract_meta_patterns(self, state: np.ndarray, level: int) -> np.ndarray:
        """
        Extract meta-cognitive patterns from state.
        
        Args:
            state: Cognitive state
            level: Current level
            
        Returns:
            Array of meta-pattern features
        """
        patterns = []
        
        # Power features (capture different scales)
        for p in range(1, min(level + 1, 4)):
            patterns.extend(state[:10] ** p)
        
        # Interaction features
        n_interact = min(5, len(state))
        for i in range(n_interact):
            for j in range(i + 1, n_interact):
                # Multiplicative interactions
                patterns.append(state[i] * state[j])
                
                # Phase interactions
                phase_diff = np.angle(state[i] + 1j * state[j])
                patterns.append(np.sin(level * phase_diff))
        
        # Statistical features
        if len(state) > 10:
            patterns.extend([
                np.mean(state),
                np.std(state),
                np.max(np.abs(state)),
                np.sum(state > 0) / len(state)  # Activation ratio
            ])
        
        # Fourier features (capture periodicity)
        if len(state) > 20:
            fft = np.fft.fft(state)
            patterns.extend(np.abs(fft[:5]))
        
        return np.array(patterns)
    
    def _select_sparse_features(self, features: np.ndarray, n_select: int) -> np.ndarray:
        """
        Select most informative features using sparsity criterion.
        
        Args:
            features: Candidate features
            n_select: Number to select
            
        Returns:
            Selected features
        """
        if len(features) <= n_select:
            return features
        
        # Compute feature importance (variance-based)
        importance = np.abs(features - np.mean(features))
        
        # Add small noise to break ties
        importance += np.random.randn(len(importance)) * 1e-10
        
        # Select top features
        indices = np.argsort(importance)[-n_select:]
        selected = features[indices]
        
        # Apply sparsity threshold
        mask = np.abs(selected) > self.sparsity_threshold
        selected[~mask] = 0
        
        return selected
    
    def parallel_transport(self,
                         vector: np.ndarray,
                         path: List[np.ndarray],
                         level: int) -> np.ndarray:
        """
        Parallel transport vector along path using connection.
        
        Args:
            vector: Tangent vector to transport
            path: List of states forming a path
            level: Metacognitive level
            
        Returns:
            Transported vector
        """
        if len(path) < 2:
            return vector.copy()
        
        transported = vector.copy()
        
        for k in range(len(path) - 1):
            # Get Christoffel symbols at current point
            gamma = self.compute_christoffel_symbols(level, path[k])
            
            # Tangent to path
            tangent = path[k + 1] - path[k]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-10:
                tangent /= tangent_norm
            
            # Parallel transport equation: ∇_T V = 0
            # dV^i/dt + Γ^i_jk T^j V^k = 0
            correction = np.zeros_like(transported)
            
            for i in range(len(transported)):
                for j in range(len(tangent)):
                    for k in range(len(transported)):
                        correction[i] += gamma[i, j, k] * tangent[j] * transported[k]
            
            # Update transported vector
            transported -= correction * tangent_norm * 0.1  # Small step
            
            # Normalize to preserve magnitude
            transported *= np.linalg.norm(vector) / (np.linalg.norm(transported) + 1e-10)
        
        return transported
    
    def holonomy(self, loop: List[np.ndarray], level: int) -> np.ndarray:
        """
        Compute holonomy around a closed loop.
        
        Measures curvature by parallel transporting around loop.
        
        Args:
            loop: Closed path (first and last states should match)
            level: Metacognitive level
            
        Returns:
            Holonomy transformation matrix
        """
        dim = self.levels[level]['dim']
        
        # Start with identity
        holonomy_matrix = np.eye(dim)
        
        # Transport each basis vector
        for i in range(dim):
            basis_vector = np.zeros(dim)
            basis_vector[i] = 1.0
            
            # Transport around loop
            transported = self.parallel_transport(basis_vector, loop, level)
            
            holonomy_matrix[:, i] = transported
        
        return holonomy_matrix
    
    def get_level_summary(self, level: int) -> Dict[str, Any]:
        """
        Get summary information about a level.
        
        Args:
            level: Level index
            
        Returns:
            Dictionary of level information
        """
        if level >= len(self.levels):
            raise ValueError(f"Level {level} does not exist")
        
        lvl = self.levels[level]
        
        return {
            'order': lvl['order'],
            'dimension': lvl['dim'],
            'features': lvl['features'],
            'metric': self.metric,
            'blocks': len(self._get_block_structure(level)),
            'theoretical_dim': self._compute_level_dim(level)
        }
    
    def validate_functoriality(self,
                             state: np.ndarray,
                             max_level: int = None,
                             tolerance: float = 1e-6) -> bool:
        """
        Validate functorial properties of lift/project.
        
        Checks that project ∘ lift ≈ identity.
        
        Args:
            state: Test state at level 0
            max_level: Maximum level to test
            tolerance: Error tolerance
            
        Returns:
            True if functorial properties hold
        """
        if max_level is None:
            max_level = len(self.levels) - 1
        
        current = state.copy()
        
        for level in range(min(max_level, len(self.levels) - 1)):
            # Lift then project
            lifted = self.lift(current, level, level + 1)
            recovered = self.project(lifted, level + 1, level)
            
            # Check recovery error
            error = np.linalg.norm(recovered - current) / (np.linalg.norm(current) + 1e-10)
            
            if error > tolerance:
                warnings.warn(f"Functoriality violated at level {level}: error = {error}")
                return False
            
            # Move to next level
            current = lifted[:self.levels[level + 1]['dim']]
        
        return True


class KnowledgeSheaf:
    """
    Knowledge sheaf for distributed cognitive representation.
    
    Implements a sheaf structure over cognitive regions enabling
    local-to-global knowledge integration.
    
    Attributes:
        manifold: Base cognitive manifold
        topology: Open cover of cognitive space
        sections: Local knowledge sections
        restrictions: Restriction maps between overlapping regions
        consistency_threshold: Threshold for gluing consistency
    """
    
    def __init__(self,
                 manifold: MetaCognitiveManifold,
                 topology: Optional[Dict[str, List[str]]] = None,
                 auto_restrictions: bool = True,
                 consistency_threshold: float = 1e-3):
        """
        Initialize knowledge sheaf.
        
        Args:
            manifold: Cognitive manifold
            topology: Dictionary mapping regions to their neighborhoods
            auto_restrictions: Automatically compute restriction maps
            consistency_threshold: Threshold for consistency checks
        """
        self.manifold = manifold
        self.topology = topology or self._default_topology()
        self.sections = {}
        self.restrictions = {}
        self.auto_restrictions = auto_restrictions
        self.consistency_threshold = consistency_threshold
        
        # Precompute overlap structure
        self._compute_overlaps()
        
        # Cache for gluing computations
        self._gluing_cache = {}
    
    def _default_topology(self) -> Dict[str, List[str]]:
        """
        Create default topology with cognitive regions.
        
        Returns:
            Topology dictionary
        """
        return {
            'perception': ['reasoning', 'memory'],
            'reasoning': ['perception', 'action'],
            'memory': ['perception', 'action'],
            'action': ['reasoning', 'memory']
        }
    
    def _compute_overlaps(self):
        """Precompute pairwise and triple overlaps."""
        self.pairwise_overlaps = {}
        self.triple_overlaps = []
        
        regions = list(self.topology.keys())
        
        # Pairwise overlaps
        for i, r1 in enumerate(regions):
            for r2 in regions[i+1:]:
                if r2 in self.topology[r1]:
                    overlap_key = tuple(sorted([r1, r2]))
                    self.pairwise_overlaps[overlap_key] = {
                        'regions': overlap_key,
                        'restriction_computed': False
                    }
        
        # Triple overlaps (for cocycle condition)
        for i, r1 in enumerate(regions):
            for j, r2 in enumerate(regions[i+1:], i+1):
                for r3 in regions[j+1:]:
                    if (r2 in self.topology[r1] and 
                        r3 in self.topology[r2] and 
                        r3 in self.topology[r1]):
                        self.triple_overlaps.append((r1, r2, r3))
    
    def add_section(self,
                   region: str,
                   knowledge_func: Callable[[np.ndarray], Any],
                   domain: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Add local knowledge section.
        
        Args:
            region: Region name
            knowledge_func: Function mapping states to knowledge
            domain: Optional (min, max) bounds for region
        """
        if region not in self.topology:
            raise ValueError(f"Region {region} not in topology")
        
        self.sections[region] = {
            'function': knowledge_func,
            'domain': domain,
            'cache': {}
        }
        
        # Compute restrictions if auto mode
        if self.auto_restrictions:
            self._compute_restrictions_for_region(region)
    
    def _compute_restrictions_for_region(self, region: str):
        """
        Compute restriction maps for a region.
        
        Args:
            region: Region to compute restrictions for
        """
        for neighbor in self.topology[region]:
            if neighbor in self.sections:
                overlap_key = tuple(sorted([region, neighbor]))
                
                # Define restriction as projection/interpolation
                def make_restriction(r1, r2):
                    def restriction(knowledge, state):
                        # Simple model: restrictions preserve structure
                        if isinstance(knowledge, np.ndarray):
                            return knowledge  # Identity for vectors
                        elif isinstance(knowledge, dict):
                            # Project dictionary knowledge
                            return {k: v for k, v in knowledge.items()
                                   if not k.startswith('_local_')}
                        else:
                            return knowledge
                    return restriction
                
                self.restrictions[overlap_key] = make_restriction(region, neighbor)
    
    def get_section_value(self, region: str, state: np.ndarray) -> Any:
        """
        Evaluate section at a state.
        
        Args:
            region: Region name
            state: Cognitive state
            
        Returns:
            Knowledge value at state
        """
        if region not in self.sections:
            raise ValueError(f"No section defined for region {region}")
        
        section = self.sections[region]
        
        # Check domain
        if section['domain'] is not None:
            min_bound, max_bound = section['domain']
            if np.any(state < min_bound) or np.any(state > max_bound):
                warnings.warn(f"State outside domain for region {region}")
        
        # Check cache
        state_key = tuple(np.round(state, 4))
        if state_key in section['cache']:
            return section['cache'][state_key]
        
        # Compute value
        value = section['function'](state)
        
        # Cache result
        section['cache'][state_key] = value
        
        return value
    
    def check_consistency(self, state: np.ndarray, regions: List[str]) -> bool:
        """
        Check if sections agree on overlaps.
        
        Args:
            state: Cognitive state
            regions: Regions to check
            
        Returns:
            True if consistent
        """
        if len(regions) < 2:
            return True
        
        values = []
        
        for region in regions:
            if region in self.sections:
                value = self.get_section_value(region, state)
                values.append(value)
        
        if len(values) < 2:
            return True
        
        # Check consistency based on value type
        if isinstance(values[0], np.ndarray):
            # Vector consistency
            for i in range(1, len(values)):
                if np.linalg.norm(values[i] - values[0]) > self.consistency_threshold:
                    return False
        
        elif isinstance(values[0], (int, float)):
            # Scalar consistency
            for i in range(1, len(values)):
                if abs(values[i] - values[0]) > self.consistency_threshold:
                    return False
        
        else:
            # For other types, require exact equality
            for i in range(1, len(values)):
                if values[i] != values[0]:
                    return False
        
        return True
    
    def glue_sections(self,
                     state: np.ndarray,
                     method: str = 'average') -> Optional[Any]:
        """
        Glue local sections into global knowledge.
        
        Args:
            state: Cognitive state
            method: Gluing method ('average', 'max', 'consensus')
            
        Returns:
            Global knowledge or None if inconsistent
        """
        # Find regions containing state
        active_regions = []
        values = []
        
        for region in self.topology:
            if region in self.sections:
                try:
                    value = self.get_section_value(region, state)
                    active_regions.append(region)
                    values.append(value)
                except:
                    continue
        
        if not values:
            return None
        
        # Check consistency
        if not self.check_consistency(state, active_regions):
            warnings.warn("Inconsistent sections, cannot glue")
            return None
        
        # Glue based on method
        if method == 'average':
            if isinstance(values[0], np.ndarray):
                return np.mean(values, axis=0)
            elif isinstance(values[0], (int, float)):
                return np.mean(values)
            else:
                # Return first value for non-numeric types
                return values[0]
        
        elif method == 'max':
            if isinstance(values[0], np.ndarray):
                return values[np.argmax([np.linalg.norm(v) for v in values])]
            elif isinstance(values[0], (int, float)):
                return max(values)
            else:
                return values[0]
        
        elif method == 'consensus':
            # Use voting for consensus
            if isinstance(values[0], np.ndarray):
                # Vector consensus via median
                return np.median(values, axis=0)
            else:
                # Most common value
                from collections import Counter
                counts = Counter(values)
                return counts.most_common(1)[0][0]
        
        else:
            raise ValueError(f"Unknown gluing method: {method}")
    
    def verify_cocycle_condition(self,
                               state: np.ndarray,
                               triple: Tuple[str, str, str]) -> bool:
        """
        Verify cocycle condition for triple overlap.
        
        The cocycle condition ensures consistency of restrictions:
        ρ_AC = ρ_BC ∘ ρ_AB
        
        Args:
            state: Cognitive state
            triple: Three regions (A, B, C)
            
        Returns:
            True if cocycle condition holds
        """
        r1, r2, r3 = triple
        
        # Get section values
        try:
            val1 = self.get_section_value(r1, state)
            val2 = self.get_section_value(r2, state)
            val3 = self.get_section_value(r3, state)
        except:
            return True  # Skip if any section undefined
        
        # Get restrictions
        r12_key = tuple(sorted([r1, r2]))
        r23_key = tuple(sorted([r2, r3]))
        r13_key = tuple(sorted([r1, r3]))
        
        # Apply restrictions
        if r12_key in self.restrictions and r23_key in self.restrictions:
            # Path 1: r1 -> r2 -> r3
            val1_to_2 = self.restrictions[r12_key](val1, state)
            val2_to_3 = self.restrictions[r23_key](val1_to_2, state)
            
            # Path 2: r1 -> r3 directly
            if r13_key in self.restrictions:
                val1_to_3_direct = self.restrictions[r13_key](val1, state)
                
                # Check consistency
                if isinstance(val2_to_3, np.ndarray):
                    error = np.linalg.norm(val2_to_3 - val1_to_3_direct)
                    return error < self.consistency_threshold
                else:
                    return val2_to_3 == val1_to_3_direct
        
        return True
    
    def extend_section(self,
                      from_region: str,
                      to_region: str,
                      extension_method: str = 'smooth') -> bool:
        """
        Extend section from one region to another.
        
        Args:
            from_region: Source region
            to_region: Target region
            extension_method: Method for extension
            
        Returns:
            Success flag
        """
        if from_region not in self.sections:
            return False
        
        if to_region not in self.topology:
            return False
        
        source_section = self.sections[from_region]
        
        if extension_method == 'smooth':
            # Smooth extension using Gaussian process-like interpolation
            def extended_func(state):
                # Get nearest boundary point
                if source_section['domain'] is not None:
                    min_b, max_b = source_section['domain']
                    # Project to domain
                    proj_state = np.clip(state, min_b, max_b)
                    
                    # Compute distance to domain
                    dist = self.manifold.distance(state, proj_state)
                    
                    # Get value at boundary
                    boundary_val = source_section['function'](proj_state)
                    
                    # Smooth decay
                    if isinstance(boundary_val, np.ndarray):
                        decay = np.exp(-dist)
                        return boundary_val * decay
                    else:
                        return boundary_val
                else:
                    # No domain restriction, use original function
                    return source_section['function'](state)
            
            self.add_section(to_region, extended_func)
            return True
        
        elif extension_method == 'constant':
            # Constant extension
            reference_val = None
            
            def constant_func(state):
                nonlocal reference_val
                if reference_val is None:
                    # Compute once at first call
                    reference_state = np.zeros(self.manifold.dimension)
                    reference_val = source_section['function'](reference_state)
                return reference_val
            
            self.add_section(to_region, constant_func)
            return True
        
        else:
            return False
    
    def compute_cohomology(self, degree: int = 0) -> Dict[str, Any]:
        """
        Compute sheaf cohomology (simplified version).
        
        Args:
            degree: Cohomological degree (0 or 1)
            
        Returns:
            Cohomology information
        """
        if degree == 0:
            # H^0: Global sections
            # Count consistent global sections
            
            test_states = [
                np.random.randn(self.manifold.dimension) for _ in range(10)
            ]
            
            consistent_count = 0
            for state in test_states:
                if self.glue_sections(state) is not None:
                    consistent_count += 1
            
            return {
                'degree': 0,
                'dimension': consistent_count,
                'interpretation': 'Global sections',
                'consistency_rate': consistent_count / len(test_states)
            }
        
        elif degree == 1:
            # H^1: Obstruction to gluing
            # Check cocycle conditions
            
            violations = 0
            total_checks = 0
            
            for triple in self.triple_overlaps:
                test_state = np.random.randn(self.manifold.dimension)
                if not self.verify_cocycle_condition(test_state, triple):
                    violations += 1
                total_checks += 1
            
            return {
                'degree': 1,
                'dimension': violations,
                'interpretation': 'Gluing obstructions',
                'violation_rate': violations / max(1, total_checks)
            }
        
        else:
            return {
                'degree': degree,
                'dimension': 0,
                'interpretation': 'Not computed'
            }
    
    def visualize_structure(self) -> Dict[str, Any]:
        """
        Get visualization data for sheaf structure.
        
        Returns:
            Dictionary with visualization information
        """
        viz_data = {
            'regions': list(self.topology.keys()),
            'edges': [],
            'sections_defined': list(self.sections.keys()),
            'overlaps': []
        }
        
        # Add edges from topology
        for region, neighbors in self.topology.items():
            for neighbor in neighbors:
                edge = tuple(sorted([region, neighbor]))
                if edge not in viz_data['edges']:
                    viz_data['edges'].append(edge)
        
        # Add overlap information
        for overlap_key in self.pairwise_overlaps:
            viz_data['overlaps'].append({
                'regions': overlap_key,
                'has_restriction': overlap_key in self.restrictions
            })
        
        return viz_data


class AttentionMechanism:
    """
    Attention mechanism for cognitive focus.
    
    Implements various attention mechanisms including self-attention,
    cross-attention, and memory-based attention.
    
    Attributes:
        dim: Dimension of attention space
        n_heads: Number of attention heads
        temperature: Softmax temperature
    """
    
    def __init__(self,
                 dim: int,
                 n_heads: int = 4,
                 temperature: float = 1.0,
                 dropout: float = 0.0):
        """
        Initialize attention mechanism.
        
        Args:
            dim: Dimension of states
            n_heads: Number of attention heads
            temperature: Softmax temperature
            dropout: Dropout probability
        """
        self.dim = dim
        self.n_heads = n_heads
        self.temperature = temperature
        self.dropout = dropout
        
        # Initialize projection matrices
        self.head_dim = dim // n_heads
        self._init_projections()
    
    def _init_projections(self):
        """Initialize random projection matrices."""
        # Query, Key, Value projections for each head
        self.W_q = []
        self.W_k = []
        self.W_v = []
        
        for _ in range(self.n_heads):
            self.W_q.append(np.random.randn(self.dim, self.head_dim) * 0.1)
            self.W_k.append(np.random.randn(self.dim, self.head_dim) * 0.1)
            self.W_v.append(np.random.randn(self.dim, self.head_dim) * 0.1)
        
        # Output projection
        self.W_o = np.random.randn(self.n_heads * self.head_dim, self.dim) * 0.1
    
    def self_attention(self, states: np.ndarray) -> np.ndarray:
        """
        Apply self-attention to states.
        
        Args:
            states: Array of states (n_states × dim)
            
        Returns:
            Attended states
        """
        n_states = len(states)
        
        # Multi-head attention
        head_outputs = []
        
        for h in range(self.n_heads):
            # Project to Q, K, V
            Q = states @ self.W_q[h]  # (n_states × head_dim)
            K = states @ self.W_k[h]
            V = states @ self.W_v[h]
            
            # Compute attention scores
            scores = Q @ K.T / np.sqrt(self.head_dim)
            scores /= self.temperature
            
            # Softmax
            attention_weights = softmax(scores, axis=1)
            
            # Apply dropout
            if self.dropout > 0:
                mask = np.random.rand(*attention_weights.shape) > self.dropout
                attention_weights *= mask
                attention_weights /= (np.sum(attention_weights, axis=1, keepdims=True) + 1e-10)
            
            # Apply attention
            head_output = attention_weights @ V
            head_outputs.append(head_output)
        
        # Concatenate heads
        concat_output = np.hstack(head_outputs)
        
        # Final projection
        output = concat_output @ self.W_o
        
        # Residual connection
        output += states
        
        return output
    
    def cross_attention(self,
                       queries: np.ndarray,
                       keys: np.ndarray,
                       values: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply cross-attention between different sequences.
        
        Args:
            queries: Query states
            keys: Key states
            values: Value states (defaults to keys)
            
        Returns:
            Attended query states
        """
        if values is None:
            values = keys
        
        n_queries = len(queries)
        n_keys = len(keys)
        
        output = np.zeros_like(queries)
        
        for h in range(self.n_heads):
            # Project
            Q = queries @ self.W_q[h]
            K = keys @ self.W_k[h]
            V = values @ self.W_v[h]
            
            # Attention scores
            scores = Q @ K.T / np.sqrt(self.head_dim)
            scores /= self.temperature
            
            # Softmax
            attention_weights = softmax(scores, axis=1)
            
            # Apply attention
            head_output = attention_weights @ V
            
            # Add to output (simplified - should concatenate and project)
            output += head_output @ self.W_v[h].T
        
        return output / self.n_heads
    
    def memory_attention(self,
                        query: np.ndarray,
                        memory: List[np.ndarray],
                        memory_values: Optional[List[Any]] = None) -> Tuple[Any, np.ndarray]:
        """
        Retrieve from memory using attention.
        
        Args:
            query: Query state
            memory: List of memory states
            memory_values: Optional values associated with memories
            
        Returns:
            Retrieved value and attention weights
        """
        if not memory:
            return None, np.array([])
        
        # Stack memory
        memory_matrix = np.vstack(memory)
        
        # Compute attention
        scores = []
        for mem_state in memory:
            # Dot product attention
            score = np.dot(query, mem_state) / (
                np.linalg.norm(query) * np.linalg.norm(mem_state) + 1e-10
            )
            scores.append(score / self.temperature)
        
        # Softmax
        attention_weights = softmax(np.array(scores))
        
        # Retrieve value
        if memory_values is not None:
            # Weighted combination based on value type
            if isinstance(memory_values[0], np.ndarray):
                retrieved = np.sum([w * v for w, v in zip(attention_weights, memory_values)], axis=0)
            else:
                # Return highest weighted value
                idx = np.argmax(attention_weights)
                retrieved = memory_values[idx]
        else:
            # Return weighted combination of states
            retrieved = np.sum([w * s for w, s in zip(attention_weights, memory)], axis=0)
        
        return retrieved, attention_weights


class CognitiveMemorySystem:
    """
    Hierarchical memory system for cognitive architectures.
    
    Implements working memory, episodic memory, and semantic memory
    with consolidation and retrieval mechanisms.
    
    Attributes:
        working_capacity: Size of working memory
        episodic_capacity: Size of episodic memory
        consolidation_threshold: Threshold for memory consolidation
    """
    
    def __init__(self,
                 manifold: MetaCognitiveManifold,
                 working_capacity: int = 7,
                 episodic_capacity: int = 1000,
                 consolidation_threshold: float = 0.8):
        """
        Initialize memory system.
        
        Args:
            manifold: Cognitive manifold
            working_capacity: Working memory size (Miller's 7±2)
            episodic_capacity: Episodic memory size
            consolidation_threshold: Threshold for consolidation
        """
        self.manifold = manifold
        self.working_capacity = working_capacity
        self.episodic_capacity = episodic_capacity
        self.consolidation_threshold = consolidation_threshold
        
        # Memory stores
        self.working_memory = deque(maxlen=working_capacity)
        self.episodic_memory = deque(maxlen=episodic_capacity)
        self.semantic_memory = {}  # Concept -> prototype mapping
        
        # Memory metadata
        self.access_counts = {}
        self.timestamps = {}
        self.importance_scores = {}
        
        # Attention mechanism for retrieval
        self.attention = AttentionMechanism(manifold.dimension)
    
    def encode(self, state: np.ndarray, context: Optional[Dict] = None) -> str:
        """
        Encode state into memory.
        
        Args:
            state: Cognitive state
            context: Optional context information
            
        Returns:
            Memory ID
        """
        # Generate unique ID
        memory_id = f"mem_{len(self.timestamps)}"
        
        # Store in working memory
        memory_item = {
            'id': memory_id,
            'state': state.copy(),
            'context': context or {},
            'timestamp': len(self.timestamps)
        }
        
        self.working_memory.append(memory_item)
        
        # Update metadata
        self.access_counts[memory_id] = 1
        self.timestamps[memory_id] = len(self.timestamps)
        self.importance_scores[memory_id] = self._compute_importance(state, context)
        
        # Check for consolidation
        self._consolidate_if_needed()
        
        return memory_id
    
    def retrieve(self,
                query: np.ndarray,
                n_items: int = 1,
                memory_type: str = 'all') -> List[Dict]:
        """
        Retrieve memories similar to query.
        
        Args:
            query: Query state
            n_items: Number of items to retrieve
            memory_type: 'working', 'episodic', 'semantic', or 'all'
            
        Returns:
            List of retrieved memories
        """
        candidates = []
        
        # Gather candidates based on memory type
        if memory_type in ['working', 'all']:
            candidates.extend(self.working_memory)
        
        if memory_type in ['episodic', 'all']:
            candidates.extend(self.episodic_memory)
        
        if memory_type in ['semantic', 'all']:
            # Convert semantic memories to candidate format
            for concept, prototype in self.semantic_memory.items():
                candidates.append({
                    'id': f"semantic_{concept}",
                    'state': prototype,
                    'context': {'type': 'semantic', 'concept': concept}
                })
        
        if not candidates:
            return []
        
        # Use attention mechanism for retrieval
        memory_states = [item['state'] for item in candidates]
        _, attention_weights = self.attention.memory_attention(query, memory_states)
        
        # Sort by attention weight
        scored_items = list(zip(attention_weights, candidates))
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        # Update access counts
        retrieved = []
        for score, item in scored_items[:n_items]:
            if item['id'] in self.access_counts:
                self.access_counts[item['id']] += 1
            retrieved.append(item)
        
        return retrieved
    
    def _compute_importance(self, state: np.ndarray, context: Optional[Dict]) -> float:
        """
        Compute importance score for memory.
        
        Args:
            state: Cognitive state
            context: Context information
            
        Returns:
            Importance score
        """
        # Base importance from state properties
        importance = np.linalg.norm(state) / (self.manifold.dimension ** 0.5)
        
        # Boost for high IIT states (conscious moments)
        phi = compute_iit_phi(state)
        importance += phi
        
        # Context-based adjustments
        if context:
            if 'reward' in context:
                importance += abs(context['reward'])
            if 'surprise' in context:
                importance += context['surprise']
            if 'emotional_valence' in context:
                importance += abs(context['emotional_valence'])
        
        return importance
    
    def _consolidate_if_needed(self):
        """
        Consolidate working memory to episodic/semantic memory.
        """
        if len(self.working_memory) < self.working_capacity:
            return
        
        # Find items to consolidate
        consolidation_candidates = []
        
        for item in self.working_memory:
            importance = self.importance_scores.get(item['id'], 0)
            access_count = self.access_counts.get(item['id'], 1)
            
            # Consolidation score
            score = importance * np.log(access_count + 1)
            
            if score > self.consolidation_threshold:
                consolidation_candidates.append((score, item))
        
        # Consolidate top candidates
        consolidation_candidates.sort(key=lambda x: x[0], reverse=True)
        
        for score, item in consolidation_candidates[:2]:  # Consolidate top 2
            # Move to episodic memory
            self.episodic_memory.append(item)
            
            # Check for semantic extraction
            self._extract_semantic_memory(item)
    
    def _extract_semantic_memory(self, memory_item: Dict):
        """
        Extract semantic knowledge from episodic memory.
        
        Args:
            memory_item: Memory to process
        """
        state = memory_item['state']
        context = memory_item.get('context', {})
        
        # Simple clustering approach for concept extraction
        # Find if this state is similar to existing semantic memories
        
        best_concept = None
        best_similarity = 0
        
        for concept, prototype in self.semantic_memory.items():
            similarity = 1.0 / (1.0 + self.manifold.distance(state, prototype))
            if similarity > best_similarity:
                best_similarity = similarity
                best_concept = concept
        
        if best_similarity > 0.7:
            # Update existing concept (running average)
            old_prototype = self.semantic_memory[best_concept]
            self.semantic_memory[best_concept] = 0.9 * old_prototype + 0.1 * state
        else:
            # Create new concept
            new_concept = f"concept_{len(self.semantic_memory)}"
            self.semantic_memory[new_concept] = state.copy()
    
    def forget(self, decay_rate: float = 0.01):
        """
        Apply forgetting to reduce memory load.
        
        Args:
            decay_rate: Rate of importance decay
        """
        # Decay importance scores
        for memory_id in list(self.importance_scores.keys()):
            self.importance_scores[memory_id] *= (1 - decay_rate)
            
            # Remove if importance too low
            if self.importance_scores[memory_id] < 0.1:
                del self.importance_scores[memory_id]
                del self.access_counts[memory_id]
                del self.timestamps[memory_id]
        
        # Remove low-importance items from episodic memory
        self.episodic_memory = deque(
            [item for item in self.episodic_memory 
             if self.importance_scores.get(item['id'], 0) > 0.1],
            maxlen=self.episodic_capacity
        )
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about memory system.
        
        Returns:
            Dictionary of memory statistics
        """
        return {
            'working_memory_size': len(self.working_memory),
            'working_memory_utilization': len(self.working_memory) / self.working_capacity,
            'episodic_memory_size': len(self.episodic_memory),
            'semantic_concepts': len(self.semantic_memory),
            'total_memories_encoded': len(self.timestamps),
            'average_importance': np.mean(list(self.importance_scores.values())) 
                                if self.importance_scores else 0,
            'most_accessed': max(self.access_counts.items(), key=lambda x: x[1])
                           if self.access_counts else None
        }