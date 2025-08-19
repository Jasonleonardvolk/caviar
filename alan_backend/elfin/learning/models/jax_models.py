"""
JAX Models for Neural Barrier and Lyapunov Functions

This module provides JAX implementations of neural networks for representing
barrier and Lyapunov functions.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import flax.linen as nn
from flax.training import train_state
import optax
from typing import List, Tuple, Optional, Dict, Any, Union, Callable, Sequence
import numpy as np


class MLP(nn.Module):
    """Multi-layer perceptron implemented in Flax/JAX."""
    
    features: Sequence[int]
    activation: str = "tanh"
    
    @nn.compact
    def __call__(self, x):
        """Forward pass through the network."""
        # Get activation function
        if self.activation == "relu":
            act_fn = nn.relu
        elif self.activation == "tanh":
            act_fn = nn.tanh
        elif self.activation == "sigmoid":
            act_fn = nn.sigmoid
        elif self.activation == "swish":
            act_fn = nn.swish
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        
        # Pass through hidden layers
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = act_fn(x)
        
        # Output layer (no activation)
        x = nn.Dense(self.features[-1])(x)
        return x


class JAXBarrierNetwork:
    """
    Neural network for representing barrier functions using JAX/Flax.
    
    This network computes B(x), where B(x) > 0 in the safe set and B(x) <= 0 in the unsafe set.
    The network also provides methods for computing gradients of B with respect to inputs,
    which is necessary for verifying the barrier certificate condition:
        ∇B(x) · f(x, u) >= -α(B(x)) for all x in the safe set
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_layers: List[int] = [64, 64],
        activation: str = "tanh",
        output_activation: Optional[str] = None,
        init_type: str = "normal",
        init_params: Optional[Dict[str, float]] = None,
        seed: int = 0,
    ):
        """
        Initialize a neural barrier function network.
        
        Args:
            state_dim: Dimension of the state space
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'swish')
            output_activation: Optional output activation function
            init_type: Weight initialization type ('normal', 'uniform', 'xavier', 'kaiming')
            init_params: Initialization parameters (mean, std for normal, etc.)
            seed: Random seed for initialization
        """
        self.state_dim = state_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_activation = output_activation
        self.init_type = init_type
        self.init_params = init_params or {}
        self.seed = seed
        
        # Define network architecture
        all_layers = hidden_layers + [1]  # Output layer with 1 neuron
        self.model = MLP(features=all_layers, activation=activation)
        
        # Initialize parameters
        self.key = random.PRNGKey(seed)
        self.params = self._init_params()
        
        # JIT-compile forward function and gradient
        self.forward_jit = jit(self.forward)
        self.gradient_jit = jit(self.gradient)
    
    def _init_params(self):
        """Initialize network parameters."""
        # Create dummy input for initialization
        dummy_input = jnp.ones((1, self.state_dim))
        
        # Initialize with default initialization
        return self.model.init(self.key, dummy_input)
    
    def forward(self, params, x):
        """
        Compute the barrier function value.
        
        Args:
            params: Network parameters
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Barrier function values, shape (batch_size, 1)
        """
        outputs = self.model.apply(params, x)
        
        # Apply output activation if specified
        if self.output_activation == "relu":
            outputs = nn.relu(outputs)
        elif self.output_activation == "sigmoid":
            outputs = nn.sigmoid(outputs)
        elif self.output_activation == "tanh":
            outputs = nn.tanh(outputs)
        
        return outputs
    
    def __call__(self, x):
        """
        Compute barrier function value for given inputs.
        
        Args:
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Barrier function values, shape (batch_size, 1)
        """
        # Ensure x is a JAX array
        x = jnp.asarray(x)
        
        # Call forward function
        return self.forward_jit(self.params, x)
    
    def gradient(self, params, x):
        """
        Compute the gradient of the barrier function with respect to the input.
        
        Args:
            params: Network parameters
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Gradient array, shape (batch_size, state_dim)
        """
        # Define a function that takes a single input vector and returns a scalar
        def single_forward(x_single):
            return self.forward(params, x_single.reshape(1, -1))[0, 0]
        
        # Vectorize the gradient computation
        batch_grad_fn = vmap(grad(single_forward))
        
        return batch_grad_fn(x)
    
    def get_gradient(self, x):
        """
        Compute gradient for given inputs.
        
        Args:
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Gradient array, shape (batch_size, state_dim)
        """
        # Ensure x is a JAX array
        x = jnp.asarray(x)
        
        # Call gradient function
        return self.gradient_jit(self.params, x)
    
    def verify_condition(
        self,
        x: jnp.ndarray,
        dynamics_fn: Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray],
        u: Optional[jnp.ndarray] = None,
        alpha_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """
        Verify the barrier certificate condition: ∇B(x) · f(x, u) >= -α(B(x))
        
        Args:
            x: State array, shape (batch_size, state_dim)
            dynamics_fn: Function mapping (x, u) to state derivatives
            u: Control inputs (optional), shape (batch_size, input_dim)
            alpha_fn: Class-K function (optional), defaults to identity
            
        Returns:
            Boolean array indicating whether the condition is satisfied for each state
        """
        # Ensure x is a JAX array
        x = jnp.asarray(x)
        
        # Default alpha function (identity)
        if alpha_fn is None:
            alpha_fn = lambda b: b
        
        # Compute barrier value
        b = self(x)
        
        # Compute gradient
        grad_b = self.get_gradient(x)
        
        # Compute dynamics
        if u is None:
            # If no control input is provided, assume autonomous system
            f = dynamics_fn(x, None)
        else:
            # Ensure u is a JAX array
            u = jnp.asarray(u)
            f = dynamics_fn(x, u)
        
        # Compute Lie derivative: ∇B(x) · f(x, u)
        lie_derivatives = jnp.sum(grad_b * f, axis=1, keepdims=True)
        
        # Compute α(B(x))
        alpha_b = alpha_fn(b)
        
        # Check condition: ∇B(x) · f(x, u) >= -α(B(x))
        condition = lie_derivatives >= -alpha_b
        
        return condition
    
    def save(self, filepath: str):
        """Save the model parameters to a file."""
        # Convert params to numpy arrays for saving
        from flax.serialization import to_state_dict, msgpack_serialize
        import io
        
        with open(filepath, 'wb') as f:
            f.write(msgpack_serialize(to_state_dict(self.params)))
    
    @classmethod
    def load(cls, filepath: str, state_dim: int, hidden_layers: List[int] = [64, 64], 
             activation: str = "tanh", output_activation: Optional[str] = None):
        """Load the model from a file."""
        from flax.serialization import from_state_dict, msgpack_restore
        import io
        
        # Create a new instance
        model = cls(
            state_dim=state_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            output_activation=output_activation
        )
        
        # Load params
        with open(filepath, 'rb') as f:
            data = f.read()
            model.params = from_state_dict(model.params, msgpack_restore(data))
        
        return model
    
    def to_numpy(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the barrier function on numpy arrays.
        
        Args:
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Barrier function values, shape (batch_size, 1)
        """
        # Convert to JAX array, compute, and convert back
        x_jax = jnp.asarray(x)
        b = self(x_jax)
        return np.asarray(b)


class JAXLyapunovNetwork(JAXBarrierNetwork):
    """
    Neural network for representing Lyapunov functions using JAX/Flax.
    
    This network extends JAXBarrierNetwork with specialized functionality for
    Lyapunov functions, which have slightly different properties than barrier functions.
    
    A Lyapunov function V(x) must satisfy:
    1. V(x) > 0 for all x != 0
    2. V(0) = 0 (the equilibrium point is at the origin)
    3. ∇V(x) · f(x) < 0 for all x != 0 (the derivative along system trajectories is negative)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_layers: List[int] = [64, 64],
        activation: str = "tanh",
        init_type: str = "normal",
        init_params: Optional[Dict[str, float]] = None,
        seed: int = 0,
    ):
        """
        Initialize a neural Lyapunov function network.
        
        Note: For Lyapunov functions, we ensure positive definiteness and
        V(0) = 0 through the forward function.
        """
        super(JAXLyapunovNetwork, self).__init__(
            state_dim=state_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            output_activation=None,  # We'll add special output handling in forward
            init_type=init_type,
            init_params=init_params,
            seed=seed,
        )
        
        # Initialize origin parameter (will be learned to represent the equilibrium point)
        self.origin = jnp.zeros((state_dim,))
    
    def forward(self, params, x):
        """
        Compute the Lyapunov function value.
        
        Modified to ensure V(0) = 0 and V(x) > 0 for x != 0.
        
        Args:
            params: Network parameters
            x: State array, shape (batch_size, state_dim)
            
        Returns:
            Lyapunov function values, shape (batch_size, 1)
        """
        # Shift input by the origin
        x_shifted = x - self.origin
        
        # Get raw network output
        raw_output = self.model.apply(params, x_shifted)
        
        # Square the output to ensure positive definiteness
        v = raw_output ** 2
        
        # Scale by the norm of x_shifted to ensure V(x) grows with distance from origin
        norms = jnp.linalg.norm(x_shifted, axis=1, keepdims=True)
        v = v * norms
        
        return v
    
    def verify_condition(
        self,
        x: jnp.ndarray,
        dynamics_fn: Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray],
        u: Optional[jnp.ndarray] = None,
        alpha_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """
        Verify the Lyapunov condition: ∇V(x) · f(x, u) < 0 for all x != 0
        
        For controlled systems, this can be relaxed to:
        ∇V(x) · f(x, u) <= -α(||x||) for all x != 0
        
        Args:
            x: State array, shape (batch_size, state_dim)
            dynamics_fn: Function mapping (x, u) to state derivatives
            u: Control inputs (optional), shape (batch_size, input_dim)
            alpha_fn: Class-K function (optional), defaults to small fraction of norm
            
        Returns:
            Boolean array indicating whether the condition is satisfied for each state
        """
        # Ensure x is a JAX array
        x = jnp.asarray(x)
        
        # Default alpha function (small fraction of norm)
        if alpha_fn is None:
            alpha_fn = lambda x_norm: 0.1 * x_norm
        
        # Compute Lyapunov value
        v = self(x)
        
        # Compute gradient
        grad_v = self.get_gradient(x)
        
        # Compute dynamics
        if u is None:
            # If no control input is provided, assume autonomous system
            f = dynamics_fn(x, None)
        else:
            # Ensure u is a JAX array
            u = jnp.asarray(u)
            f = dynamics_fn(x, u)
        
        # Compute Lie derivative: ∇V(x) · f(x, u)
        lie_derivatives = jnp.sum(grad_v * f, axis=1, keepdims=True)
        
        # Compute norm of x for the alpha function
        x_norm = jnp.linalg.norm(x - self.origin, axis=1, keepdims=True)
        
        # Compute α(||x||)
        alpha_x = alpha_fn(x_norm)
        
        # Check condition: ∇V(x) · f(x, u) <= -α(||x||)
        # For x very close to origin, we relax the condition to avoid numerical issues
        near_origin = x_norm < 1e-6
        condition = (near_origin) | (lie_derivatives <= -alpha_x)
        
        return condition
