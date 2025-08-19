"""
Hopfield-on-spin memory implementation for ALAN.

This module provides an implementation of a Hopfield network adapted to work with
spin vectors from the ALAN oscillator system. It combines traditional Hopfield
dynamics with antiferromagnetic couplings that map directly to hardware.

Key features:
- Energy-based memory recall
- Antiferromagnetic coupling with hardware mapping
- Binary and continuous activation modes
- Adaptive temperature for simulated annealing
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HopfieldConfig:
    """Configuration parameters for the Spin-Hopfield memory."""
    
    # Inverse temperature parameter (β), controls sharpness of activation
    beta: float = 1.0
    
    # Whether to use binary or continuous state values
    binary: bool = True
    
    # Maximum number of iterations for recall
    max_iterations: int = 100
    
    # Convergence threshold for energy change
    energy_threshold: float = 1e-6
    
    # Learning rate for weight updates
    learning_rate: float = 0.01
    
    # Whether to normalize weights after learning
    normalize_weights: bool = True
    
    # Whether to use asyncronous (one unit at a time) updates
    asynchronous: bool = True
    
    # Annealing schedule parameters
    use_annealing: bool = False
    annealing_init_temp: float = 5.0
    annealing_final_temp: float = 0.1
    annealing_steps: int = 50


class SpinHopfieldMemory:
    """Hopfield network memory using spin states.
    
    This implementation supports both binary and continuous state values,
    with antiferromagnetic couplings that can be mapped to hardware.
    """
    
    def __init__(self, size: int, config: Optional[HopfieldConfig] = None):
        """Initialize a new Hopfield memory network.
        
        Args:
            size: Number of memory units
            config: Configuration parameters (or None for defaults)
        """
        self.size = size
        self.config = config or HopfieldConfig()
        
        # Weight matrix (initially zeros)
        self.weights = np.zeros((size, size), dtype=np.float64)
        np.fill_diagonal(self.weights, 0.0)  # No self-connections
        
        # Current state
        self.state = np.zeros(size, dtype=np.float64)
        
        # Stored patterns
        self.patterns = []
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function to inputs.
        
        For binary mode, this is sign(β*x)
        For continuous mode, this is tanh(β*x)
        
        Args:
            x: Input values
            
        Returns:
            Activated values
        """
        beta = self.config.beta
        
        if self.config.binary:
            return np.sign(beta * x)
        else:
            return np.tanh(beta * x)
    
    def store(self, patterns: Union[np.ndarray, List[np.ndarray]]) -> None:
        """Store patterns in the memory.
        
        This uses a Hebbian learning rule: W_ij += η * x_i * x_j
        
        Args:
            patterns: Single pattern (1D array) or list of patterns to store
        """
        if isinstance(patterns, np.ndarray) and patterns.ndim == 1:
            # Single pattern
            patterns = [patterns]
        
        # Add to stored patterns list
        self.patterns.extend([p.copy() for p in patterns])
        
        # Hebbian learning rule
        eta = self.config.learning_rate
        for pattern in patterns:
            if len(pattern) != self.size:
                raise ValueError(f"Pattern size {len(pattern)} doesn't match network size {self.size}")
            
            # Outer product of pattern with itself, excluding diagonal
            weight_update = np.outer(pattern, pattern)
            np.fill_diagonal(weight_update, 0.0)
            
            # Update weights
            self.weights += eta * weight_update
        
        # Normalize weights if configured
        if self.config.normalize_weights:
            # Scale to [-1, 1] range
            max_abs_weight = np.max(np.abs(self.weights))
            if max_abs_weight > 0:
                self.weights /= max_abs_weight
    
    def _is_continuous_mode(self, state: np.ndarray) -> bool:
        """Detect if we're working with continuous or binary spins.
        
        Args:
            state: Current state vector
            
        Returns:
            True if continuous mode, False if binary
        """
        # Check if all values are exactly -1 or 1 (binary)
        # or if there are values between (continuous)
        if np.all(np.isin(state, [-1, 1])):
            return False
        return True
    
    def _get_effective_weights(self, state: np.ndarray) -> np.ndarray:
        """Get effective weights based on spin mode.
        
        For binary spins (σ∈{±1}): use antiferromagnetic weights (W < 0)
        For continuous spins (σ∈[-1, 1]): use ferromagnetic weights (|W| > 0)
        
        Args:
            state: Current state vector
            
        Returns:
            Effective weight matrix for the current mode
        """
        if self._is_continuous_mode(state):
            # Continuous mode: make weights ferromagnetic (positive)
            return np.abs(self.weights)
        else:
            # Binary mode: keep antiferromagnetic (negative)
            return -np.abs(self.weights)
    
    def _compute_lambda(self) -> float:
        """Compute optimal regularization strength based on weight matrix properties.
        
        This uses a power iteration approach to approximate the largest eigenvalue 
        of the weight matrix, then scales lambda proportionally to ensure stability
        for networks of any size.
        
        Returns:
            Regularization strength λ
        """
        if not hasattr(self, '_lambda') or self._lambda is None:
            # Use power iteration to estimate largest eigenvalue magnitude
            n = self.size
            if n == 0:
                return 0.05  # Default for empty network
                
            # Get absolute weights (we care about magnitude)
            abs_weights = np.abs(self.weights)
            
            # Initialize random vector
            v = np.random.rand(n)
            v = v / np.linalg.norm(v)
            
            # Power iteration (20 steps is usually sufficient)
            for _ in range(20):
                v_new = abs_weights @ v
                norm = np.linalg.norm(v_new)
                if norm > 1e-10:  # Prevent division by zero
                    v = v_new / norm
                else:
                    break
            
            # Estimate the largest eigenvalue
            lambda_max = v @ (abs_weights @ v)
            
            # Guard against singular matrices (all zeros) with a minimum lambda
            if lambda_max < 1e-6:
                self._lambda = 0.01  # Safe default for singular matrices
                logger.debug(f"Matrix may be singular, using safe λ = {self._lambda}")
            else:
                # Set regularization to 5% of max eigenvalue to ensure stability
                # This ensures energy is bounded regardless of network size
                self._lambda = max(0.05, 0.05 * lambda_max)
                
            logger.debug(f"Computed lambda = {self._lambda:.6f} (max_eig ≈ {lambda_max:.6f})")
        
        return self._lambda
    
    def _energy(self, state: np.ndarray) -> float:
        """Calculate the energy of a state.
        
        E = -0.5 * sum_ij W_ij * s_i * s_j + λ * sum_i s_i^2
        Lower energy = more stable state
        
        Sign convention changes based on spin mode:
        - Binary spins (σ∈{±1}): antiferromagnetic (W < 0)
        - Continuous spins (σ∈[-1, 1]): ferromagnetic (W > 0)
        
        The L2 regularization term ensures the energy is bounded below
        even for continuous spins, preventing divergence to -∞.
        
        Args:
            state: State to compute energy for
            
        Returns:
            Energy value (lower is more stable)
        """
        # Get effective weights based on spin mode
        effective_weights = self._get_effective_weights(state)
        
        # Calculate quadratic energy term
        quadratic_term = -0.5 * np.sum(np.outer(state, state) * effective_weights)
        
        # Add L2 regularization term with auto-scaled lambda
        regularization_strength = self._compute_lambda()
        regularization_term = regularization_strength * np.sum(state**2)
        
        return quadratic_term + regularization_term
    
    def _update_unit(self, i: int, state: np.ndarray, temp: float = 1.0) -> Tuple[np.ndarray, bool]:
        """Update a single unit in the network.
        
        Args:
            i: Index of unit to update
            state: Current state
            temp: Current temperature (for annealing)
            
        Returns:
            (new_state, changed): Updated state and whether the unit changed
        """
        # Get effective weights based on spin mode
        effective_weights = self._get_effective_weights(state)
        
        # Calculate input to unit i using effective weights
        input_i = np.sum(effective_weights[i, :] * state)
        
        # Current value
        current_value = state[i]
        
        # New value with temperature scaling
        beta = self.config.beta / temp
        if self.config.binary:
            # Add small random fluctuation for annealing
            if temp > 1.0:
                input_i += np.random.normal(0, temp * 0.1)
            new_value = np.sign(beta * input_i)
        else:
            new_value = np.tanh(beta * input_i)
        
        # Update state
        new_state = state.copy()
        new_state[i] = new_value
        
        # Check if changed
        if self.config.binary:
            changed = new_value != current_value
        else:
            changed = abs(new_value - current_value) > 1e-6
        
        return new_state, changed
    
    def _update_all_synchronous(self, state: np.ndarray, temp: float = 1.0) -> np.ndarray:
        """Update all units synchronously.
        
        Args:
            state: Current state
            temp: Current temperature (for annealing)
            
        Returns:
            new_state: Updated state
        """
        # Get effective weights based on spin mode
        effective_weights = self._get_effective_weights(state)
        
        # Calculate input to all units using effective weights
        inputs = np.dot(effective_weights, state)
        
        # Apply activation function with temperature scaling
        beta = self.config.beta / temp
        if self.config.binary:
            # Add small random fluctuations for annealing
            if temp > 1.0:
                inputs += np.random.normal(0, temp * 0.1, size=inputs.shape)
            new_state = np.sign(beta * inputs)
        else:
            new_state = np.tanh(beta * inputs)
        
        return new_state
    
    def recall(self, initial_state: np.ndarray, callback: Optional[Callable[[np.ndarray, float], None]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform memory recall from an initial state.
        
        Args:
            initial_state: Starting state for recall
            callback: Optional function called after each iteration with (state, energy)
            
        Returns:
            (final_state, info): Final state after recall and information dict
        """
        if len(initial_state) != self.size:
            raise ValueError(f"Initial state size {len(initial_state)} doesn't match network size {self.size}")
        
        # Initialize state
        state = initial_state.copy()
        
        # Set up annealing schedule if used
        if self.config.use_annealing:
            temps = np.linspace(
                self.config.annealing_init_temp,
                self.config.annealing_final_temp,
                self.config.annealing_steps
            )
            annealing_iter = max(1, self.config.max_iterations // len(temps))
        else:
            temps = [1.0]  # No annealing
            annealing_iter = self.config.max_iterations
        
        # Initialize tracking
        energies = [self._energy(state)]
        states = [state.copy()]
        
        # Main recall loop
        iterations = 0
        converged = False
        
        for temp in temps:
            for _ in range(annealing_iter):
                if self.config.asynchronous:
                    # Update units one at a time in random order
                    indices = np.random.permutation(self.size)
                    changed = False
                    
                    for i in indices:
                        state, unit_changed = self._update_unit(i, state, temp)
                        changed = changed or unit_changed
                else:
                    # Update all units synchronously
                    new_state = self._update_all_synchronous(state, temp)
                    changed = not np.array_equal(state, new_state)
                    state = new_state
                
                # Calculate new energy
                energy = self._energy(state)
                energies.append(energy)
                states.append(state.copy())
                
                # Call callback if provided
                if callback is not None:
                    callback(state, energy)
                
                iterations += 1
                
                # Check for convergence
                if not changed or (iterations >= 2 and abs(energies[-1] - energies[-2]) < self.config.energy_threshold):
                    converged = True
                    break
                
                if iterations >= self.config.max_iterations:
                    break
            
            if converged or iterations >= self.config.max_iterations:
                break
        
        # Store final state
        self.state = state.copy()
        
        # Return result and info
        info = {
            'iterations': iterations,
            'converged': converged,
            'energy': energies[-1],
            'energy_history': energies,
            'state_history': states,
        }
        
        return state, info
    
    def get_state(self) -> np.ndarray:
        """Get the current network state.
        
        Returns:
            Current state vector
        """
        return self.state.copy()
    
    def get_weights(self) -> np.ndarray:
        """Get the weight matrix.
        
        Returns:
            Weight matrix
        """
        return self.weights.copy()
    
    def set_weights(self, weights: np.ndarray) -> None:
        """Set the weight matrix directly.
        
        This can be used to load pre-trained weights or
        to set up specific connectivity patterns.
        
        Args:
            weights: New weight matrix
        """
        if weights.shape != (self.size, self.size):
            raise ValueError(f"Weight matrix must have shape ({self.size}, {self.size})")
        
        self.weights = weights.copy()
        np.fill_diagonal(self.weights, 0.0)  # Ensure no self-connections
    
    def get_memory_capacity(self) -> int:
        """Estimate the theoretical memory capacity.
        
        For binary Hopfield networks, capacity is approximately 0.14*N,
        where N is the number of units.
        
        Returns:
            Estimated capacity (number of patterns)
        """
        return int(0.14 * self.size)
    
    def compute_overlap(self, state: np.ndarray, pattern_idx: Optional[int] = None) -> Union[float, List[float]]:
        """Compute overlap between state and stored pattern(s).
        
        Overlap = 1/N * sum_i (s_i * p_i)
        
        Args:
            state: State to compare
            pattern_idx: Index of pattern to compare with, or None for all patterns
            
        Returns:
            Overlap value(s) in range [-1, 1]
        """
        if not self.patterns:
            return 0.0
        
        if pattern_idx is not None:
            # Compare with specific pattern
            pattern = self.patterns[pattern_idx]
            return np.mean(state * pattern)
        else:
            # Compare with all patterns
            return [np.mean(state * pattern) for pattern in self.patterns]


class SpinHopfieldNetwork:
    """A network of Hopfield memories with hierarchical organization.
    
    This provides a higher-level interface for working with multiple
    Hopfield memories organized into modules.
    """
    
    def __init__(self, modules_config: List[Tuple[int, Optional[HopfieldConfig]]]):
        """Initialize a hierarchical Hopfield network.
        
        Args:
            modules_config: List of (size, config) tuples for each module
        """
        self.modules = []
        self.module_sizes = []
        
        for size, config in modules_config:
            module = SpinHopfieldMemory(size, config)
            self.modules.append(module)
            self.module_sizes.append(size)
    
    @property
    def total_size(self) -> int:
        """Get the total number of units across all modules."""
        return sum(self.module_sizes)
    
    def store_pattern(self, pattern: np.ndarray, module_idx: Optional[int] = None) -> None:
        """Store a pattern in one or all modules.
        
        Args:
            pattern: Pattern to store
            module_idx: Index of module to store in, or None for all
        """
        if module_idx is not None:
            # Store in specific module
            if len(pattern) != self.module_sizes[module_idx]:
                raise ValueError(f"Pattern size {len(pattern)} doesn't match module size {self.module_sizes[module_idx]}")
            self.modules[module_idx].store([pattern])
        else:
            # Split pattern and store in all modules
            start = 0
            for i, size in enumerate(self.module_sizes):
                self.modules[i].store([pattern[start:start+size]])
                start += size
    
    def recall(self, initial_state: np.ndarray, module_idx: Optional[int] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Perform recall on one or all modules.
        
        Args:
            initial_state: Initial state for recall
            module_idx: Index of module to recall in, or None for all
            
        Returns:
            (final_state, infos): Final state and info dicts for each module
        """
        if module_idx is not None:
            # Recall in specific module
            if len(initial_state) != self.module_sizes[module_idx]:
                raise ValueError(f"Initial state size {len(initial_state)} doesn't match module size {self.module_sizes[module_idx]}")
            final_state, info = self.modules[module_idx].recall(initial_state)
            return final_state, [info]
        else:
            # Split initial state and recall in all modules
            start = 0
            final_states = []
            infos = []
            
            for i, size in enumerate(self.module_sizes):
                module_state = initial_state[start:start+size]
                final_module_state, info = self.modules[i].recall(module_state)
                final_states.append(final_module_state)
                infos.append(info)
                start += size
            
            return np.concatenate(final_states), infos
    
    def get_module(self, module_idx: int) -> SpinHopfieldMemory:
        """Get a specific module.
        
        Args:
            module_idx: Index of module to get
            
        Returns:
            The selected module
        """
        return self.modules[module_idx]


if __name__ == "__main__":
    # Simple example/test of the Hopfield memory
    import matplotlib.pyplot as plt
    
    # Create a simple 10x10 grid pattern
    size = 100  # 10x10 grid
    
    # Create some patterns (simple geometric shapes)
    def create_grid_pattern(pattern_func):
        pattern = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                pattern[i, j] = pattern_func(i, j)
        return pattern.flatten()
    
    # Horizontal line
    horizontal = create_grid_pattern(lambda i, j: 1 if i == 4 else -1)
    
    # Vertical line
    vertical = create_grid_pattern(lambda i, j: 1 if j == 4 else -1)
    
    # Diagonal
    diagonal = create_grid_pattern(lambda i, j: 1 if i == j else -1)
    
    # Cross
    cross = create_grid_pattern(lambda i, j: 1 if i == 4 or j == 4 else -1)
    
    # Create memory and store patterns
    config = HopfieldConfig(beta=2.0, binary=True, asynchronous=True, max_iterations=1000)
    memory = SpinHopfieldMemory(size, config)
    memory.store([horizontal, vertical, diagonal, cross])
    
    # Create a noisy version of one pattern
    noise_level = 0.3
    noisy_pattern = horizontal.copy()
    noise_mask = np.random.choice([0, 1], size=size, p=[1-noise_level, noise_level])
    noisy_pattern[noise_mask == 1] *= -1
    
    # Perform recall
    iterations_history = []
    
    def callback(state, energy):
        iterations_history.append((state.copy(), energy))
    
    recalled, info = memory.recall(noisy_pattern, callback)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot original pattern
    plt.subplot(141)
    plt.imshow(horizontal.reshape(10, 10), cmap='binary', interpolation='nearest')
    plt.title('Original Pattern')
    plt.axis('off')
    
    # Plot noisy pattern
    plt.subplot(142)
    plt.imshow(noisy_pattern.reshape(10, 10), cmap='binary', interpolation='nearest')
    plt.title(f'Noisy Pattern ({noise_level*100:.0f}% noise)')
    plt.axis('off')
    
    # Plot recalled pattern
    plt.subplot(143)
    plt.imshow(recalled.reshape(10, 10), cmap='binary', interpolation='nearest')
    plt.title(f'Recalled Pattern ({info["iterations"]} iterations)')
    plt.axis('off')
    
    # Plot energy during recall
    plt.subplot(144)
    plt.plot(info['energy_history'])
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Energy During Recall')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate overlap with stored patterns
    overlaps = memory.compute_overlap(recalled)
    pattern_names = ['Horizontal', 'Vertical', 'Diagonal', 'Cross']
    
    print("Pattern recall complete")
    print(f"Iterations: {info['iterations']}")
    print(f"Converged: {info['converged']}")
    print(f"Final energy: {info['energy']:.4f}")
    print("\nOverlap with stored patterns:")
    for name, overlap in zip(pattern_names, overlaps):
        print(f"{name}: {overlap:.4f}")
