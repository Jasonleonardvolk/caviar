"""
Loss Functions for Neural Barrier and Lyapunov Networks

This module provides custom loss functions for training neural barrier and
Lyapunov functions to satisfy their respective mathematical conditions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
import jax
import jax.numpy as jnp


class BarrierLoss(nn.Module):
    """
    Loss function for training neural barrier functions using PyTorch.
    
    The loss function has three components:
    1. Classification loss: Ensures B(x) > 0 for safe states and B(x) <= 0 for unsafe states
    2. Gradient loss: Ensures ∇B(x) · f(x, u) >= -α(B(x)) for safe states
    3. Smoothness loss: Encourages smoothness of the barrier function
    """
    
    def __init__(
        self,
        dynamics_fn: Optional[Callable] = None,
        alpha_fn: Optional[Callable] = None,
        classification_weight: float = 1.0,
        gradient_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        margin: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize barrier loss function.
        
        Args:
            dynamics_fn: Function mapping (x, u) to state derivatives
            alpha_fn: Class-K function for barrier certificate condition
            classification_weight: Weight for classification loss
            gradient_weight: Weight for gradient loss
            smoothness_weight: Weight for smoothness loss
            margin: Margin for classification loss
            device: Device to use for computations
        """
        super(BarrierLoss, self).__init__()
        
        self.dynamics_fn = dynamics_fn
        self.alpha_fn = alpha_fn or (lambda b: 0.1 * torch.abs(b))
        self.classification_weight = classification_weight
        self.gradient_weight = gradient_weight
        self.smoothness_weight = smoothness_weight
        self.margin = margin
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(
        self,
        barrier_network: nn.Module,
        states: torch.Tensor,
        labels: torch.Tensor,
        control_inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute barrier loss.
        
        Args:
            barrier_network: Neural network representing barrier function
            states: Batch of states, shape (batch_size, state_dim)
            labels: Batch of labels (1 for safe, 0 for unsafe), shape (batch_size, 1)
            control_inputs: Batch of control inputs, shape (batch_size, input_dim)
            
        Returns:
            Dictionary of loss components
        """
        # Ensure states have requires_grad=True
        states = states.clone().detach().to(self.device).requires_grad_(True)
        labels = labels.to(self.device)
        
        if control_inputs is not None:
            control_inputs = control_inputs.to(self.device)
        
        # Compute barrier values
        barrier_values = barrier_network(states)
        
        # Classification loss
        # We want B(x) > margin for safe states and B(x) < -margin for unsafe states
        # For safe states (label=1): max(0, margin - B(x))
        # For unsafe states (label=0): max(0, margin + B(x))
        safe_mask = labels > 0.5
        unsafe_mask = ~safe_mask
        
        safe_loss = F.relu(self.margin - barrier_values[safe_mask]).mean() if safe_mask.any() else 0
        unsafe_loss = F.relu(self.margin + barrier_values[unsafe_mask]).mean() if unsafe_mask.any() else 0
        
        classification_loss = safe_loss + unsafe_loss
        
        # Gradient loss for verifying barrier condition
        # Only compute for safe states
        gradient_loss = torch.tensor(0.0).to(self.device)
        
        if self.dynamics_fn is not None and safe_mask.any() and self.gradient_weight > 0:
            # Extract safe states
            safe_states = states[safe_mask]
            safe_barrier_values = barrier_values[safe_mask]
            
            # Compute gradients
            gradients = torch.autograd.grad(
                safe_barrier_values.sum(),
                safe_states,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Compute dynamics for safe states
            if control_inputs is not None:
                safe_inputs = control_inputs[safe_mask]
                dynamics = self.dynamics_fn(safe_states, safe_inputs)
            else:
                dynamics = self.dynamics_fn(safe_states, None)
            
            # Compute Lie derivatives: ∇B(x) · f(x, u)
            lie_derivatives = torch.sum(gradients * dynamics, dim=1, keepdim=True)
            
            # Compute α(B(x))
            alpha_values = self.alpha_fn(safe_barrier_values)
            
            # Lie derivative should be >= -α(B(x))
            # Loss: max(0, -α(B(x)) - Lie derivative)
            gradient_loss = F.relu(-alpha_values - lie_derivatives).mean()
        
        # Smoothness loss
        # Use L2 regularization on gradients
        smoothness_loss = torch.tensor(0.0).to(self.device)
        
        if self.smoothness_weight > 0:
            # Compute gradients with respect to all states
            all_gradients = torch.autograd.grad(
                barrier_values.sum(),
                states,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # L2 regularization on gradients
            smoothness_loss = torch.mean(torch.sum(all_gradients**2, dim=1))
        
        # Compute total loss
        total_loss = (
            self.classification_weight * classification_loss +
            self.gradient_weight * gradient_loss +
            self.smoothness_weight * smoothness_loss
        )
        
        return {
            "total_loss": total_loss,
            "classification_loss": classification_loss,
            "gradient_loss": gradient_loss,
            "smoothness_loss": smoothness_loss,
            "safe_loss": safe_loss if isinstance(safe_loss, torch.Tensor) else torch.tensor(safe_loss).to(self.device),
            "unsafe_loss": unsafe_loss if isinstance(unsafe_loss, torch.Tensor) else torch.tensor(unsafe_loss).to(self.device)
        }


class LyapunovLoss(nn.Module):
    """
    Loss function for training neural Lyapunov functions using PyTorch.
    
    The loss function has three components:
    1. Positive definiteness loss: Ensures V(x) > 0 for all x != 0 and V(0) = 0
    2. Derivative loss: Ensures ∇V(x) · f(x, u) < 0 for all x != 0
    3. Boundedness loss: Encourages boundedness of the Lyapunov function
    """
    
    def __init__(
        self,
        dynamics_fn: Optional[Callable] = None,
        origin: Optional[torch.Tensor] = None,
        positive_weight: float = 1.0,
        derivative_weight: float = 1.0,
        boundedness_weight: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize Lyapunov loss function.
        
        Args:
            dynamics_fn: Function mapping (x, u) to state derivatives
            origin: Equilibrium point, shape (state_dim,)
            positive_weight: Weight for positive definiteness loss
            derivative_weight: Weight for derivative loss
            boundedness_weight: Weight for boundedness loss
            device: Device to use for computations
        """
        super(LyapunovLoss, self).__init__()
        
        self.dynamics_fn = dynamics_fn
        self.positive_weight = positive_weight
        self.derivative_weight = derivative_weight
        self.boundedness_weight = boundedness_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set origin to zero if not provided
        self.origin = origin if origin is not None else torch.zeros(1, dtype=torch.float32).to(self.device)
    
    def forward(
        self,
        lyapunov_network: nn.Module,
        states: torch.Tensor,
        control_inputs: Optional[torch.Tensor] = None,
        origin: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Lyapunov loss.
        
        Args:
            lyapunov_network: Neural network representing Lyapunov function
            states: Batch of states, shape (batch_size, state_dim)
            control_inputs: Batch of control inputs, shape (batch_size, input_dim)
            origin: Equilibrium point, shape (state_dim,)
            
        Returns:
            Dictionary of loss components
        """
        # Ensure states have requires_grad=True
        states = states.clone().detach().to(self.device).requires_grad_(True)
        
        if control_inputs is not None:
            control_inputs = control_inputs.to(self.device)
        
        # Use provided origin or default
        origin_point = origin if origin is not None else self.origin
        origin_point = origin_point.to(self.device)
        
        # Compute Lyapunov values
        lyapunov_values = lyapunov_network(states)
        
        # Positive definiteness loss
        # We want V(x) > 0 for x != 0 and V(0) = 0
        # Compute distance from origin
        if origin_point.dim() == 1:
            origin_point = origin_point.unsqueeze(0)
        
        distance_from_origin = torch.norm(states - origin_point, dim=1, keepdim=True)
        
        # Compute origin value (should be 0)
        origin_value = lyapunov_network(origin_point)
        
        # Compute positive definiteness loss
        # For states close to origin, V(x) should be close to 0
        # For states far from origin, V(x) should be positive and increasing with distance
        near_origin_mask = distance_from_origin < 1e-2
        far_from_origin_mask = ~near_origin_mask
        
        # Origin value should be 0
        origin_loss = torch.abs(origin_value).mean()
        
        # For states near origin, V(x) should be small and positive
        near_origin_loss = F.relu(0.1 - lyapunov_values[near_origin_mask]).mean() if near_origin_mask.any() else 0
        
        # For states far from origin, V(x) should be positive and increasing with distance
        expected_values = 0.1 * distance_from_origin[far_from_origin_mask]
        far_from_origin_loss = F.relu(expected_values - lyapunov_values[far_from_origin_mask]).mean() if far_from_origin_mask.any() else 0
        
        positive_loss = origin_loss + near_origin_loss + far_from_origin_loss
        
        # Derivative loss for verifying Lyapunov condition
        # ∇V(x) · f(x, u) < 0 for all x != 0
        derivative_loss = torch.tensor(0.0).to(self.device)
        
        if self.dynamics_fn is not None and self.derivative_weight > 0:
            # Exclude states very close to origin
            valid_mask = distance_from_origin > 1e-4
            
            if valid_mask.any():
                valid_states = states[valid_mask]
                valid_values = lyapunov_values[valid_mask]
                
                # Compute gradients
                gradients = torch.autograd.grad(
                    valid_values.sum(),
                    valid_states,
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                # Compute dynamics
                if control_inputs is not None:
                    valid_inputs = control_inputs[valid_mask]
                    dynamics = self.dynamics_fn(valid_states, valid_inputs)
                else:
                    dynamics = self.dynamics_fn(valid_states, None)
                
                # Compute Lie derivatives: ∇V(x) · f(x, u)
                lie_derivatives = torch.sum(gradients * dynamics, dim=1, keepdim=True)
                
                # Compute target derivative (negative and proportional to distance)
                valid_distances = distance_from_origin[valid_mask]
                target_derivatives = -0.1 * valid_distances
                
                # Loss: max(0, Lie derivative - target)
                derivative_loss = F.relu(lie_derivatives - target_derivatives).mean()
        
        # Boundedness loss
        # Use L2 regularization on Lyapunov values to prevent explosion
        boundedness_loss = torch.mean(lyapunov_values**2)
        
        # Compute total loss
        total_loss = (
            self.positive_weight * positive_loss +
            self.derivative_weight * derivative_loss +
            self.boundedness_weight * boundedness_loss
        )
        
        return {
            "total_loss": total_loss,
            "positive_loss": positive_loss,
            "derivative_loss": derivative_loss,
            "boundedness_loss": boundedness_loss,
            "origin_loss": origin_loss,
            "near_origin_loss": near_origin_loss if isinstance(near_origin_loss, torch.Tensor) else torch.tensor(near_origin_loss).to(self.device),
            "far_from_origin_loss": far_from_origin_loss if isinstance(far_from_origin_loss, torch.Tensor) else torch.tensor(far_from_origin_loss).to(self.device)
        }


class JAXBarrierLoss:
    """
    Loss function for training neural barrier functions using JAX.
    
    The loss function has three components:
    1. Classification loss: Ensures B(x) > 0 for safe states and B(x) <= 0 for unsafe states
    2. Gradient loss: Ensures ∇B(x) · f(x, u) >= -α(B(x)) for safe states
    3. Smoothness loss: Encourages smoothness of the barrier function
    """
    
    def __init__(
        self,
        dynamics_fn: Optional[Callable] = None,
        alpha_fn: Optional[Callable] = None,
        classification_weight: float = 1.0,
        gradient_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        margin: float = 0.1,
    ):
        """
        Initialize barrier loss function.
        
        Args:
            dynamics_fn: Function mapping (params, x, u) to state derivatives
            alpha_fn: Class-K function for barrier certificate condition
            classification_weight: Weight for classification loss
            gradient_weight: Weight for gradient loss
            smoothness_weight: Weight for smoothness loss
            margin: Margin for classification loss
        """
        self.dynamics_fn = dynamics_fn
        self.alpha_fn = alpha_fn or (lambda b: 0.1 * jnp.abs(b))
        self.classification_weight = classification_weight
        self.gradient_weight = gradient_weight
        self.smoothness_weight = smoothness_weight
        self.margin = margin
    
    def __call__(
        self,
        params: Any,
        barrier_network: Callable,
        states: jnp.ndarray,
        labels: jnp.ndarray,
        control_inputs: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute barrier loss.
        
        Args:
            params: Parameters for the barrier network
            barrier_network: Neural network forward function
            states: Batch of states, shape (batch_size, state_dim)
            labels: Batch of labels (1 for safe, 0 for unsafe), shape (batch_size, 1)
            control_inputs: Batch of control inputs, shape (batch_size, input_dim)
            
        Returns:
            Dictionary of loss components
        """
        # Compute barrier values
        barrier_values = barrier_network(params, states)
        
        # Classification loss
        # We want B(x) > margin for safe states and B(x) < -margin for unsafe states
        safe_mask = labels > 0.5
        unsafe_mask = jnp.logical_not(safe_mask)
        
        # Compute losses for safe and unsafe states
        safe_loss = jnp.mean(jnp.maximum(0, self.margin - barrier_values) * safe_mask)
        unsafe_loss = jnp.mean(jnp.maximum(0, self.margin + barrier_values) * unsafe_mask)
        
        classification_loss = safe_loss + unsafe_loss
        
        # Gradient loss for verifying barrier condition
        # Only compute for safe states
        if self.dynamics_fn is not None and jnp.any(safe_mask) and self.gradient_weight > 0:
            # Extract safe states
            safe_indices = jnp.where(safe_mask[:, 0])[0]
            safe_states = states[safe_indices]
            safe_barrier_values = barrier_values[safe_indices]
            
            # Compute gradients using JAX
            barrier_grad_fn = jax.vmap(
                lambda x: jax.grad(lambda x_single: barrier_network(params, x_single.reshape(1, -1))[0, 0])(x)
            )
            gradients = barrier_grad_fn(safe_states)
            
            # Compute dynamics for safe states
            if control_inputs is not None:
                safe_inputs = control_inputs[safe_indices]
                dynamics = self.dynamics_fn(params, safe_states, safe_inputs)
            else:
                dynamics = self.dynamics_fn(params, safe_states, None)
            
            # Compute Lie derivatives: ∇B(x) · f(x, u)
            lie_derivatives = jnp.sum(gradients * dynamics, axis=1, keepdims=True)
            
            # Compute α(B(x))
            alpha_values = self.alpha_fn(safe_barrier_values)
            
            # Lie derivative should be >= -α(B(x))
            # Loss: max(0, -α(B(x)) - Lie derivative)
            gradient_loss = jnp.mean(jnp.maximum(0, -alpha_values - lie_derivatives))
        else:
            gradient_loss = jnp.array(0.0)
        
        # Smoothness loss
        # Use L2 regularization on gradients
        if self.smoothness_weight > 0:
            # Compute gradients with respect to all states
            all_gradients = jax.vmap(
                lambda x: jax.grad(lambda x_single: barrier_network(params, x_single.reshape(1, -1))[0, 0])(x)
            )(states)
            
            # L2 regularization on gradients
            smoothness_loss = jnp.mean(jnp.sum(all_gradients**2, axis=1))
        else:
            smoothness_loss = jnp.array(0.0)
        
        # Compute total loss
        total_loss = (
            self.classification_weight * classification_loss +
            self.gradient_weight * gradient_loss +
            self.smoothness_weight * smoothness_loss
        )
        
        return {
            "total_loss": total_loss,
            "classification_loss": classification_loss,
            "gradient_loss": gradient_loss,
            "smoothness_loss": smoothness_loss,
            "safe_loss": safe_loss,
            "unsafe_loss": unsafe_loss
        }


class JAXLyapunovLoss:
    """
    Loss function for training neural Lyapunov functions using JAX.
    
    The loss function has three components:
    1. Positive definiteness loss: Ensures V(x) > 0 for all x != 0 and V(0) = 0
    2. Derivative loss: Ensures ∇V(x) · f(x, u) < 0 for all x != 0
    3. Boundedness loss: Encourages boundedness of the Lyapunov function
    """
    
    def __init__(
        self,
        dynamics_fn: Optional[Callable] = None,
        origin: Optional[jnp.ndarray] = None,
        positive_weight: float = 1.0,
        derivative_weight: float = 1.0,
        boundedness_weight: float = 0.1,
    ):
        """
        Initialize Lyapunov loss function.
        
        Args:
            dynamics_fn: Function mapping (params, x, u) to state derivatives
            origin: Equilibrium point, shape (state_dim,)
            positive_weight: Weight for positive definiteness loss
            derivative_weight: Weight for derivative loss
            boundedness_weight: Weight for boundedness loss
        """
        self.dynamics_fn = dynamics_fn
        self.positive_weight = positive_weight
        self.derivative_weight = derivative_weight
        self.boundedness_weight = boundedness_weight
        
        # Set origin to zero if not provided
        self.origin = origin if origin is not None else jnp.zeros(1)
    
    def __call__(
        self,
        params: Any,
        lyapunov_network: Callable,
        states: jnp.ndarray,
        control_inputs: Optional[jnp.ndarray] = None,
        origin: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute Lyapunov loss.
        
        Args:
            params: Parameters for the Lyapunov network
            lyapunov_network: Neural network forward function
            states: Batch of states, shape (batch_size, state_dim)
            control_inputs: Batch of control inputs, shape (batch_size, input_dim)
            origin: Equilibrium point, shape (state_dim,)
            
        Returns:
            Dictionary of loss components
        """
        # Use provided origin or default
        origin_point = origin if origin is not None else self.origin
        
        # Ensure origin is a batch
        if origin_point.ndim == 1:
            origin_point = origin_point.reshape(1, -1)
        
        # Compute Lyapunov values
        lyapunov_values = lyapunov_network(params, states)
        
        # Compute distance from origin
        distance_from_origin = jnp.linalg.norm(states - origin_point, axis=1, keepdims=True)
        
        # Compute origin value (should be 0)
        origin_value = lyapunov_network(params, origin_point)
        
        # Compute positive definiteness loss
        # For states close to origin, V(x) should be close to 0
        # For states far from origin, V(x) should be positive and increasing with distance
        near_origin_mask = distance_from_origin < 1e-2
        far_from_origin_mask = jnp.logical_not(near_origin_mask)
        
        # Origin value should be 0
        origin_loss = jnp.abs(origin_value).mean()
        
        # For states near origin, V(x) should be small and positive
        near_origin_loss = jnp.mean(jnp.maximum(0, 0.1 - lyapunov_values) * near_origin_mask)
        
        # For states far from origin, V(x) should be positive and increasing with distance
        expected_values = 0.1 * distance_from_origin
        far_from_origin_loss = jnp.mean(jnp.maximum(0, expected_values - lyapunov_values) * far_from_origin_mask)
        
        positive_loss = origin_loss + near_origin_loss + far_from_origin_loss
        
        # Derivative loss for verifying Lyapunov condition
        # ∇V(x) · f(x, u) < 0 for all x != 0
        if self.dynamics_fn is not None and self.derivative_weight > 0:
            # Exclude states very close to origin
            valid_mask = distance_from_origin > 1e-4
            
            if jnp.any(valid_mask):
                valid_indices = jnp.where(valid_mask[:, 0])[0]
                valid_states = states[valid_indices]
                
                # Compute gradients using JAX
                lyapunov_grad_fn = jax.vmap(
                    lambda x: jax.grad(lambda x_single: lyapunov_network(params, x_single.reshape(1, -1))[0, 0])(x)
                )
                gradients = lyapunov_grad_fn(valid_states)
                
                # Compute dynamics
                if control_inputs is not None:
                    valid_inputs = control_inputs[valid_indices]
                    dynamics = self.dynamics_fn(params, valid_states, valid_inputs)
                else:
                    dynamics = self.dynamics_fn(params, valid_states, None)
                
                # Compute Lie derivatives: ∇V(x) · f(x, u)
                lie_derivatives = jnp.sum(gradients * dynamics, axis=1, keepdims=True)
                
                # Compute target derivative (negative and proportional to distance)
                valid_distances = distance_from_origin[valid_indices]
                target_derivatives = -0.1 * valid_distances
                
                # Loss: max(0, Lie derivative - target)
                derivative_loss = jnp.mean(jnp.maximum(0, lie_derivatives - target_derivatives))
            else:
                derivative_loss = jnp.array(0.0)
        else:
            derivative_loss = jnp.array(0.0)
        
        # Boundedness loss
        # Use L2 regularization on Lyapunov values to prevent explosion
        boundedness_loss = jnp.mean(lyapunov_values**2)
        
        # Compute total loss
        total_loss = (
            self.positive_weight * positive_loss +
            self.derivative_weight * derivative_loss +
            self.boundedness_weight * boundedness_loss
        )
        
        return {
            "total_loss": total_loss,
            "positive_loss": positive_loss,
            "derivative_loss": derivative_loss,
            "boundedness_loss": boundedness_loss,
            "origin_loss": origin_loss,
            "near_origin_loss": near_origin_loss,
            "far_from_origin_loss": far_from_origin_loss
        }
