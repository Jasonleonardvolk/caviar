"""
Neural Lyapunov Learning Module.

This module provides functionality for learning neural network Lyapunov functions
that satisfy stability conditions for dynamical systems.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class LyapunovNetwork(nn.Module):
    """Neural network for Lyapunov function approximation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 64],
        activation=nn.ReLU(),
        output_activation=nn.Softplus()
    ):
        """
        Initialize a neural Lyapunov network.
        
        Args:
            input_dim: Dimension of the state space
            hidden_dims: Dimensions of hidden layers
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
                               (ensures positive output)
        """
        super().__init__()
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if i < len(dims) - 2:
                layers.append(activation)
            else:
                layers.append(output_activation)
                
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: State vector(s) of shape (batch_size, input_dim)
            
        Returns:
            V(x) of shape (batch_size, 1)
        """
        return self.net(x)
    
    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient ∇V(x) using autograd.
        
        Args:
            x: State vector(s) of shape (batch_size, input_dim)
            
        Returns:
            ∇V(x) of shape (batch_size, input_dim)
        """
        x = x.clone().detach().requires_grad_(True)
        V = self.forward(x)
        
        # Compute batch gradients
        batch_size = x.shape[0]
        grads = torch.zeros(batch_size, self.input_dim, device=x.device)
        
        for i in range(batch_size):
            self.zero_grad()
            V[i].backward(retain_graph=(i < batch_size - 1))
            grads[i] = x.grad[i].clone()
            
        return grads
    
    def zero_condition(self, eps: float = 1e-6) -> float:
        """
        Compute V(0) to verify V(0) = 0 condition.
        
        Args:
            eps: Small epsilon for numerical stability
            
        Returns:
            Value of V(0)
        """
        with torch.no_grad():
            zero_state = torch.zeros(1, self.input_dim, device=next(self.parameters()).device)
            return self.forward(zero_state).item()
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str, device: str = 'cpu') -> 'LyapunovNetwork':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from
            device: Device to load the model to
            
        Returns:
            Loaded LyapunovNetwork
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dims=checkpoint['hidden_dims']
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        
        return model


@dataclass
class DynamicsModel:
    """
    Model of system dynamics dx/dt = f(x) or dx/dt = f(x, u).
    
    This class provides a PyTorch-compatible wrapper around
    system dynamics functions.
    """
    
    forward_fn: Callable
    input_dim: int
    control_dim: int = 0
    name: str = "dynamics"
    
    def __call__(
        self,
        x: torch.Tensor,
        u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Evaluate dynamics function.
        
        Args:
            x: State vector(s) of shape (batch_size, input_dim)
            u: Optional control input(s) of shape (batch_size, control_dim)
            
        Returns:
            dx/dt of shape (batch_size, input_dim)
        """
        if self.control_dim > 0 and u is None:
            raise ValueError("Control input required for controlled system")
            
        # Convert to NumPy if the forward function doesn't accept tensors
        if not isinstance(self.forward_fn, torch.nn.Module):
            x_np = x.detach().cpu().numpy()
            
            if u is not None:
                u_np = u.detach().cpu().numpy()
                
                # Batch processing
                results = []
                for i in range(x_np.shape[0]):
                    results.append(self.forward_fn(x_np[i], u_np[i]))
                    
                result_np = np.stack(results)
            else:
                # Batch processing
                results = []
                for i in range(x_np.shape[0]):
                    results.append(self.forward_fn(x_np[i]))
                    
                result_np = np.stack(results)
                
            return torch.tensor(
                result_np,
                dtype=torch.float32,
                device=x.device
            )
        else:
            # PyTorch module
            if u is not None:
                return self.forward_fn(x, u)
            else:
                return self.forward_fn(x)


class NeuralLyapunovLearner:
    """
    Trainer for neural network Lyapunov functions.
    
    This class provides functionality for learning neural network Lyapunov
    functions that satisfy stability conditions for dynamical systems.
    """
    
    def __init__(
        self,
        dynamics: DynamicsModel,
        state_dim: int,
        hidden_dims: List[int] = [64, 64],
        lr: float = 1e-3,
        epsilon: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_interval: int = 100,
        sampling_radius: float = 5.0,
        zero_penalty: float = 1.0
    ):
        """
        Initialize the neural Lyapunov learner.
        
        Args:
            dynamics: Dynamics model
            state_dim: Dimension of the state space
            hidden_dims: Dimensions of hidden layers
            lr: Learning rate
            epsilon: Margin for Lyapunov conditions
            device: Device to use for training
            log_interval: Interval for logging
            sampling_radius: Radius for sampling states
            zero_penalty: Weight for V(0) = 0 penalty
        """
        self.dynamics = dynamics
        self.state_dim = state_dim
        self.device = device
        self.epsilon = epsilon
        self.log_interval = log_interval
        self.sampling_radius = sampling_radius
        self.zero_penalty = zero_penalty
        
        # Initialize network and optimizer
        self.network = LyapunovNetwork(state_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Learning history
        self.history = {
            "total_loss": [],
            "pd_loss": [],
            "decreasing_loss": [],
            "zero_loss": []
        }
        
    def sample_states(
        self,
        n_samples: int,
        include_origin: bool = False
    ) -> torch.Tensor:
        """
        Sample states from the state space.
        
        Args:
            n_samples: Number of samples
            include_origin: Whether to include the origin
            
        Returns:
            Tensor of shape (n_samples, state_dim)
        """
        # Sample from unit ball and scale
        states = torch.randn(n_samples, self.state_dim, device=self.device)
        
        # Normalize and scale by random radius
        norms = torch.norm(states, dim=1, keepdim=True)
        states = states / norms * torch.rand_like(norms) * self.sampling_radius
        
        if include_origin:
            # Replace one sample with the origin
            states[0] = torch.zeros(self.state_dim, device=self.device)
            
        return states
    
    def compute_lie_derivative(
        self,
        x: torch.Tensor,
        u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the Lie derivative dV/dt = ∇V(x)·f(x).
        
        Args:
            x: State vector(s) of shape (batch_size, state_dim)
            u: Optional control input of shape (batch_size, control_dim)
            
        Returns:
            dV/dt of shape (batch_size, 1)
        """
        grads = self.network.grad(x)  # Shape: (batch_size, state_dim)
        
        # Compute f(x) or f(x, u)
        if u is not None:
            f_x = self.dynamics(x, u)  # Shape: (batch_size, state_dim)
        else:
            f_x = self.dynamics(x)  # Shape: (batch_size, state_dim)
        
        # Compute Lie derivative ∇V(x)·f(x)
        lie_derivative = torch.sum(grads * f_x, dim=1, keepdim=True)
        
        return lie_derivative
    
    def train_step(
        self,
        n_samples: int = 1000,
        include_origin: bool = False
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            n_samples: Number of samples
            include_origin: Whether to include the origin
            
        Returns:
            Dictionary of loss values
        """
        self.optimizer.zero_grad()
        
        # Sample states
        states = self.sample_states(n_samples, include_origin)
        
        # Compute V(x) and ∇V(x)
        V = self.network(states)
        
        # Compute dV/dt
        dVdt = self.compute_lie_derivative(states)
        
        # Compute loss for V(x) > 0 for x ≠ 0
        # Mask out the origin if included
        if include_origin:
            non_origin_mask = ~torch.all(states == 0, dim=1, keepdim=True)
            pd_loss = torch.relu(-V[non_origin_mask] + 1e-6).mean()
        else:
            # Exclude points very close to the origin
            radius = torch.norm(states, dim=1, keepdim=True)
            non_origin_mask = (radius > 1e-6)
            pd_loss = torch.relu(-V[non_origin_mask] + 1e-6).mean()
        
        # Compute loss for dV/dt < 0 for x ≠ 0
        decreasing_loss = torch.relu(dVdt[non_origin_mask] + self.epsilon).mean()
        
        # Compute loss for V(0) = 0
        zero_loss = self.network.zero_condition() ** 2 * self.zero_penalty
        
        # Total loss
        loss = pd_loss + decreasing_loss + zero_loss
        
        # Backward and optimize
        loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": loss.item(),
            "pd_loss": pd_loss.item(),
            "decreasing_loss": decreasing_loss.item(),
            "zero_loss": zero_loss
        }
    
    def train(
        self,
        n_epochs: int = 1000,
        samples_per_epoch: int = 1000,
        include_origin: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the neural Lyapunov function.
        
        Args:
            n_epochs: Number of epochs
            samples_per_epoch: Number of samples per epoch
            include_origin: Whether to include the origin in sampling
            
        Returns:
            Dictionary of loss histories
        """
        start_time = time.time()
        
        for epoch in range(n_epochs):
            metrics = self.train_step(samples_per_epoch, include_origin)
            
            # Store metrics
            for k, v in metrics.items():
                self.history[k].append(v)
            
            # Log progress
            if epoch % self.log_interval == 0:
                elapsed = time.time() - start_time
                logger.info(f"Epoch {epoch}/{n_epochs} "
                          f"[{elapsed:.1f}s] "
                          f"Loss: {metrics['total_loss']:.6f} "
                          f"(PD: {metrics['pd_loss']:.6f}, "
                          f"Decr: {metrics['decreasing_loss']:.6f}, "
                          f"Zero: {metrics['zero_loss']:.6f})")
        
        logger.info(f"Training completed in {time.time() - start_time:.1f}s")
        
        return self.history
    
    def save(self, filepath: str) -> None:
        """
        Save the learned Lyapunov function.
        
        Args:
            filepath: Path to save the model
        """
        self.network.save(filepath)
        
    def load(self, filepath: str) -> None:
        """
        Load a learned Lyapunov function.
        
        Args:
            filepath: Path to load the model from
        """
        self.network = LyapunovNetwork.load(filepath, self.device)
    
    def verify_around_equilibrium(
        self,
        radius: float = 1.0,
        n_samples: int = 1000,
        epsilon: float = 0.0
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify Lyapunov conditions around an equilibrium point.
        
        Args:
            radius: Radius around the equilibrium
            n_samples: Number of samples for verification
            epsilon: Margin for decreasing condition
            
        Returns:
            Tuple of (success, details)
        """
        # Sample states
        states = self.sample_states(n_samples)
        
        # Limit to radius
        norms = torch.norm(states, dim=1, keepdim=True)
        states = states * radius / norms
        
        # Compute V(x)
        with torch.no_grad():
            V = self.network(states)
            
            # Check positive definiteness
            pd_violations = (V <= 0).sum().item()
            
            # Compute dV/dt
            dVdt = self.compute_lie_derivative(states)
            
            # Check decreasing property
            decreasing_violations = (dVdt >= -epsilon).sum().item()
            
            # Overall success
            success = (pd_violations == 0) and (decreasing_violations == 0)
            
            details = {
                "pd_violations": pd_violations,
                "decreasing_violations": decreasing_violations,
                "pd_violation_rate": pd_violations / n_samples,
                "decreasing_violation_rate": decreasing_violations / n_samples,
                "V_min": V.min().item(),
                "V_max": V.max().item(),
                "dVdt_min": dVdt.min().item(),
                "dVdt_max": dVdt.max().item()
            }
            
            return success, details
