"""
PyTorch Models for Neural Barrier and Lyapunov Functions

This module provides PyTorch implementations of neural networks for representing
barrier and Lyapunov functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
import numpy as np


class TorchBarrierNetwork(nn.Module):
    """
    Neural network for representing barrier functions using PyTorch.

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
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
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
            dtype: Data type for network parameters
            device: Device to use for computations (CPU or GPU)
        """
        super(TorchBarrierNetwork, self).__init__()

        self.state_dim = state_dim
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.init_type = init_type
        self.init_params = init_params or {}
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Define activation function
        self.activation = self._get_activation(activation)
        self.output_activation = (
            self._get_activation(output_activation) if output_activation else None
        )

        # Build network layers
        layers = []
        prev_dim = state_dim
        for dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(self.activation)
            prev_dim = dim

        # Output layer (scalar output for barrier function)
        layers.append(nn.Linear(prev_dim, 1))

        # Apply output activation if specified
        if self.output_activation:
            layers.append(self.output_activation)

        # Create sequential model
        self.model = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

        # Move model to specified device and data type
        self.to(device=self.device, dtype=self.dtype)

    def _get_activation(self, name: Optional[str]) -> Optional[nn.Module]:
        """Get activation function module by name."""
        if name is None:
            return None
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "swish": nn.SiLU(),  # SiLU is the same as Swish
        }
        if name.lower() not in activations:
            raise ValueError(f"Unsupported activation function: {name}")
        return activations[name.lower()]

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == "normal":
                    mean = self.init_params.get("mean", 0.0)
                    std = self.init_params.get("std", 0.1)
                    nn.init.normal_(m.weight, mean=mean, std=std)
                    nn.init.zeros_(m.bias)
                elif self.init_type == "uniform":
                    a = self.init_params.get("a", -0.1)
                    b = self.init_params.get("b", 0.1)
                    nn.init.uniform_(m.weight, a=a, b=b)
                    nn.init.zeros_(m.bias)
                elif self.init_type == "xavier":
                    gain = self.init_params.get("gain", 1.0)
                    nn.init.xavier_normal_(m.weight, gain=gain)
                    nn.init.zeros_(m.bias)
                elif self.init_type == "kaiming":
                    mode = self.init_params.get("mode", "fan_in")
                    nonlinearity = self.init_params.get("nonlinearity", "relu")
                    nn.init.kaiming_normal_(
                        m.weight, mode=mode, nonlinearity=nonlinearity
                    )
                    nn.init.zeros_(m.bias)
                else:
                    raise ValueError(f"Unsupported initialization type: {self.init_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the barrier function value.

        Args:
            x: State tensor, shape (batch_size, state_dim)

        Returns:
            Barrier function values, shape (batch_size, 1)
        """
        return self.model(x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the barrier function with respect to the input.

        Args:
            x: State tensor, shape (batch_size, state_dim), requires_grad=True

        Returns:
            Gradient tensor, shape (batch_size, state_dim)
        """
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)

        # Forward pass
        b = self.forward(x)

        # Compute gradients
        gradients = []
        for i in range(b.shape[0]):
            # Create a gradient vector with a 1 at position i
            grad_output = torch.zeros_like(b)
            grad_output[i] = 1.0

            # Compute gradient of output with respect to input
            grad_x, = torch.autograd.grad(
                b, x, grad_outputs=grad_output, create_graph=True, retain_graph=True
            )
            gradients.append(grad_x[i].unsqueeze(0))

        return torch.cat(gradients, dim=0)

    def verify_condition(
        self,
        x: torch.Tensor,
        dynamics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        u: Optional[torch.Tensor] = None,
        alpha_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Verify the barrier certificate condition: ∇B(x) · f(x, u) >= -α(B(x))

        Args:
            x: State tensor, shape (batch_size, state_dim)
            dynamics_fn: Function mapping (x, u) to state derivatives
            u: Control inputs (optional), shape (batch_size, input_dim)
            alpha_fn: Class-K function (optional), defaults to identity

        Returns:
            Boolean tensor indicating whether the condition is satisfied for each state
        """
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)

        # Default alpha function (identity)
        if alpha_fn is None:
            alpha_fn = lambda b: b

        # Compute barrier value
        b = self.forward(x)

        # Compute gradient
        grad_b = self.gradient(x)

        # Compute dynamics
        if u is None:
            # If no control input is provided, assume autonomous system
            f = dynamics_fn(x, None)
        else:
            f = dynamics_fn(x, u)

        # Compute Lie derivative: ∇B(x) · f(x, u)
        lie_derivatives = torch.sum(grad_b * f, dim=1, keepdim=True)

        # Compute α(B(x))
        alpha_b = alpha_fn(b)

        # Check condition: ∇B(x) · f(x, u) >= -α(B(x))
        condition = lie_derivatives >= -alpha_b

        return condition

    def save(self, filepath: str):
        """Save the model to a file."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "state_dim": self.state_dim,
                "hidden_layers": self.hidden_layers,
                "activation": self.activation_name,
                "output_activation": self.output_activation_name,
                "init_type": self.init_type,
                "init_params": self.init_params,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str, device: Optional[Union[str, torch.device]] = None):
        """Load the model from a file."""
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            state_dim=checkpoint["state_dim"],
            hidden_layers=checkpoint["hidden_layers"],
            activation=checkpoint["activation"],
            output_activation=checkpoint["output_activation"],
            init_type=checkpoint["init_type"],
            init_params=checkpoint["init_params"],
            device=device,
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def to_numpy(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the barrier function on numpy arrays.

        Args:
            x: State array, shape (batch_size, state_dim)

        Returns:
            Barrier function values, shape (batch_size, 1)
        """
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=self.dtype, device=self.device)
            b = self.forward(x_tensor)
            return b.cpu().numpy()


class TorchLyapunovNetwork(TorchBarrierNetwork):
    """
    Neural network for representing Lyapunov functions using PyTorch.
    
    This network extends TorchBarrierNetwork with specialized functionality for
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
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize a neural Lyapunov function network.
        
        Note: For Lyapunov functions, we typically use a quadratic or softplus output
        activation to ensure positive definiteness (V(x) > 0 for x != 0).
        """
        # Initialize with squared output to ensure positive definiteness
        super(TorchLyapunovNetwork, self).__init__(
            state_dim=state_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            output_activation=None,  # We'll add special output handling in forward
            init_type=init_type,
            init_params=init_params,
            dtype=dtype,
            device=device,
        )
        
        # Replace the final layer to include origin correction
        self.origin_shift = nn.Parameter(torch.zeros(state_dim, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Lyapunov function value.
        
        Modified to ensure V(0) = 0 and V(x) > 0 for x != 0.
        
        Args:
            x: State tensor, shape (batch_size, state_dim)
            
        Returns:
            Lyapunov function values, shape (batch_size, 1)
        """
        # Shift input by the origin parameter
        x_shifted = x - self.origin_shift
        
        # Pass through the base network
        features = self.model[:-1](x_shifted)  # All layers except the last
        raw_output = self.model[-1](features)  # Last layer
        
        # Square the output to ensure positive definiteness
        v = raw_output ** 2
        
        # Scale by the norm of x_shifted to ensure V(x) grows with distance from origin
        norms = torch.norm(x_shifted, dim=1, keepdim=True)
        v = v * norms
        
        return v
    
    def verify_condition(
        self,
        x: torch.Tensor,
        dynamics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        u: Optional[torch.Tensor] = None,
        alpha_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Verify the Lyapunov condition: ∇V(x) · f(x, u) < 0 for all x != 0
        
        For controlled systems, this can be relaxed to:
        ∇V(x) · f(x, u) <= -α(||x||) for all x != 0
        
        Args:
            x: State tensor, shape (batch_size, state_dim)
            dynamics_fn: Function mapping (x, u) to state derivatives
            u: Control inputs (optional), shape (batch_size, input_dim)
            alpha_fn: Class-K function (optional), defaults to small fraction of norm
            
        Returns:
            Boolean tensor indicating whether the condition is satisfied for each state
        """
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        
        # Default alpha function (small fraction of norm)
        if alpha_fn is None:
            alpha_fn = lambda x_norm: 0.1 * x_norm
        
        # Compute Lyapunov value
        v = self.forward(x)
        
        # Compute gradient
        grad_v = self.gradient(x)
        
        # Compute dynamics
        if u is None:
            # If no control input is provided, assume autonomous system
            f = dynamics_fn(x, None)
        else:
            f = dynamics_fn(x, u)
        
        # Compute Lie derivative: ∇V(x) · f(x, u)
        lie_derivatives = torch.sum(grad_v * f, dim=1, keepdim=True)
        
        # Compute norm of x for the alpha function
        x_norm = torch.norm(x - self.origin_shift, dim=1, keepdim=True)
        
        # Compute α(||x||)
        alpha_x = alpha_fn(x_norm)
        
        # Check condition: ∇V(x) · f(x, u) <= -α(||x||)
        # For x very close to origin, we relax the condition to avoid numerical issues
        near_origin = x_norm < 1e-6
        condition = (near_origin) | (lie_derivatives <= -alpha_x)
        
        return condition
