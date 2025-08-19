"""
Neural Lyapunov function training module.

This module implements the Lyapunov-Net architecture and training algorithm
for learning verifiable Lyapunov functions using neural networks.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from ..samplers.trajectory_sampler import TrajectorySampler
from .alpha_scheduler import AlphaScheduler, ExponentialAlphaScheduler

# Configure logging
logger = logging.getLogger(__name__)

class LyapunovNet(nn.Module):
    """
    Lyapunov-Net architecture for learning verifiable Lyapunov functions.
    
    Implements the approach from Gaby et al. where V(x) = |phi(x) - phi(0)| + alpha*||x||.
    This construction guarantees V(0)=0 and V(x)>0 for x≠0, ensuring positive
    definiteness by design.
    
    Attributes:
        phi: Neural network mapping states to scalar values
        alpha: Small positive constant for the norm term
        hidden_dims: Dimensions of hidden layers
        activation: Activation function used in hidden layers
    """
    
    def __init__(
        self, 
        dim: int, 
        hidden_dims: Tuple[int, ...] = (64, 64),
        alpha: float = 1e-3,
        activation: nn.Module = nn.Tanh()
    ):
        """
        Initialize the LyapunovNet.
        
        Args:
            dim: Dimension of the state space
            hidden_dims: Dimensions of hidden layers
            alpha: Small positive constant for the norm term
            activation: Activation function used in hidden layers
        """
        super().__init__()
        
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
            
        self.alpha = alpha
        self.hidden_dims = hidden_dims
        self.activation_type = activation.__class__.__name__
        
        # Build the neural network phi(x)
        layers = []
        in_dim = dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation)
            in_dim = h_dim
            
        # Final output layer (scalar)
        layers.append(nn.Linear(in_dim, 1))
        
        # Create sequential model
        self.phi = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Created LyapunovNet with hidden dims {hidden_dims}, "
                   f"alpha={alpha}, activation={self.activation_type}")
    
    def _initialize_weights(self):
        """Initialize network weights for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization
                nn.init.kaiming_uniform_(m.weight, nonlinearity='tanh')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        """
        Compute the Lyapunov function value V(x).
        
        Args:
            x: Batch of state vectors, shape (..., dim)
            alpha: Optional override for the alpha parameter. If None, uses self.alpha.
            
        Returns:
            V(x): Lyapunov function values, shape (..., 1)
        """
        # Create a zero vector with the same batch shape as x
        zeros = torch.zeros_like(x)
        
        # Compute |phi(x) - phi(0)|
        phi_diff = torch.abs(self.phi(x) - self.phi(zeros))
        
        # Compute the norm term alpha*||x||
        use_alpha = alpha if alpha is not None else self.alpha
        norm_term = use_alpha * torch.norm(x, dim=-1, keepdim=True)
        
        # Return V(x) = |phi(x) - phi(0)| + alpha*||x||
        return phi_diff + norm_term
        
    def update_alpha(self, alpha: float) -> None:
        """
        Update the alpha parameter.
        
        Args:
            alpha: New value for alpha
        """
        if alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {alpha}")
            
        self.alpha = alpha
        logger.debug(f"Updated LyapunovNet alpha to {alpha:.6f}")
    
    def compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of V(x) with respect to x.
        
        Args:
            x: Batch of state vectors, shape (batch_size, dim)
            
        Returns:
            Gradient of V(x), shape (batch_size, dim)
        """
        # Ensure x requires gradient
        x = x.detach().clone().requires_grad_(True)
        
        # Forward pass
        v = self(x)
        
        # Compute gradient
        grad_outputs = torch.ones_like(v)
        gradients = torch.autograd.grad(
            outputs=v, 
            inputs=x, 
            grad_outputs=grad_outputs, 
            create_graph=True
        )[0]
        
        return gradients
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'state_dict': self.state_dict(),
            'hidden_dims': self.hidden_dims,
            'alpha': self.alpha,
            'activation_type': self.activation_type
        }, path)
        
        logger.info(f"Saved LyapunovNet to {path}")
    
    @classmethod
    def load(cls, path: str, dim: int) -> 'LyapunovNet':
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model
            dim: Dimension of the state space
            
        Returns:
            Loaded LyapunovNet
        """
        # Load checkpoint
        checkpoint = torch.load(path)
        
        # Create activation based on saved type
        activation_map = {
            'ReLU': nn.ReLU(),
            'Tanh': nn.Tanh(),
            'Sigmoid': nn.Sigmoid(),
            'ELU': nn.ELU(),
            'LeakyReLU': nn.LeakyReLU()
        }
        
        activation_type = checkpoint.get('activation_type', 'Tanh')
        activation = activation_map.get(activation_type, nn.Tanh())
        
        # Create model
        model = cls(
            dim=dim,
            hidden_dims=checkpoint.get('hidden_dims', (64, 64)),
            alpha=checkpoint.get('alpha', 1e-3),
            activation=activation
        )
        
        # Load state dictionary
        model.load_state_dict(checkpoint['state_dict'])
        
        logger.info(f"Loaded LyapunovNet from {path}")
        return model


class NeuralLyapunovTrainer:
    """
    Trainer for neural Lyapunov functions.
    
    Trains a LyapunovNet using the provided sampler to generate training data.
    Uses the decrease condition as the loss function, as the LyapunovNet
    architecture guarantees positive definiteness by construction.
    
    Attributes:
        net: The LyapunovNet being trained
        sampler: TrajectorySampler for generating training data
        optimizer: PyTorch optimizer
        device: Device to run training on (CPU or GPU)
        gamma: Margin for the decrease condition
        history: Training history
    """
    
    def __init__(
        self, 
        model: LyapunovNet, 
        sampler: TrajectorySampler,
        learning_rate: float = 1e-3,
        gamma: float = 0.0,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
        alpha_scheduler: Optional[AlphaScheduler] = None
    ):
        """
        Initialize the Lyapunov function trainer.
        
        Args:
            model: LyapunovNet model to train
            sampler: TrajectorySampler for generating training data
            learning_rate: Learning rate for optimizer
            gamma: Margin for decrease condition: V̇(x) ≤ -gamma*||x||
            weight_decay: L2 regularization coefficient
            device: Device to run training on (CPU or GPU)
            alpha_scheduler: Optional scheduler for alpha parameter decay
        """
        self.net = model
        self.sampler = sampler
        
        # Create alpha scheduler if not provided
        if alpha_scheduler is None:
            self.alpha_scheduler = ExponentialAlphaScheduler(
                initial_alpha=model.alpha,
                min_alpha=1e-3,
                decay_steps=2000,
                step_size=100
            )
        else:
            self.alpha_scheduler = alpha_scheduler
        
        # Set device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.net.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.net.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize gradient scaler for mixed precision training
        self.use_amp = self.device.type == 'cuda'  # Only use AMP on CUDA devices
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Set margin for decrease condition
        self.gamma = gamma
        
        # Training history
        self.history = {
            'loss': [],
            'steps': [],
            'decrease_violations': [],
            'time': []
        }
        
        logger.info(f"Initialized {self.__class__.__name__} with lr={learning_rate}, "
                   f"gamma={gamma}, device={self.device}")
    
    def train_step(self, use_counterexamples: bool = True) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            use_counterexamples: Whether to use the balanced batch with counterexamples
                               or just random sampling
        
        Returns:
            Dict containing metrics for this step
        """
        self.net.train()
        
        # Generate batch
        if use_counterexamples:
            x_np, xdot_np = self.sampler.balanced_batch()
        else:
            x_np, xdot_np = self.sampler.random_batch()
        
        # Convert to tensors
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        xdot = torch.tensor(xdot_np, dtype=torch.float32, device=self.device)
        
        # Use mixed precision training if on CUDA
        with autocast(enabled=self.use_amp):
            # Compute gradient of V with respect to x
            x.requires_grad_(True)
            V = self.net(x)
            
            # Compute ∇V (jacobian of V with respect to x)
            gradV = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
            
            # Compute V̇ = ∇V · ẋ (time derivative of V)
            Vdot = torch.sum(gradV * xdot, dim=1, keepdim=True)
            
            # Compute margin term (optional)
            if self.gamma > 0:
                margin = self.gamma * torch.norm(x, dim=1, keepdim=True)
            else:
                margin = 0.0
            
            # Loss: hinge on decrease condition V̇(x) ≤ -margin
            # Only penalize where V̇(x) > -margin
            loss = torch.relu(Vdot + margin).mean()
        
        # Count violations of strict decrease condition
        with torch.no_grad():
            violations = torch.sum(Vdot > 0).item()
        
        # Backpropagation with gradient scaling for mixed precision
        self.optimizer.zero_grad()
        
        if self.use_amp:
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            
            # Update with scaled gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            
            # Regular update
            self.optimizer.step()
        
        # Return metrics
        return {
            'loss': loss.item(),
            'decrease_violations': violations,
            'violation_rate': violations / len(x)
        }
    
    def fit(
        self, 
        steps: int = 2000, 
        log_every: int = 100,
        save_path: Optional[str] = None,
        save_every: Optional[int] = None,
        counterexample_schedule: Optional[List[int]] = None
    ) -> Dict[str, List]:
        """
        Train the model for the specified number of steps.
        
        Args:
            steps: Number of training steps
            log_every: How often to log training metrics
            save_path: Path to save the model (if None, model is not saved)
            save_every: How often to save the model (if None, only saved at the end)
            counterexample_schedule: List of steps at which to use counterexamples
                                   (if None, always use counterexamples)
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {steps} steps")
        start_time = time.time()
        
        # Add alpha history to track changes
        if 'alpha' not in self.history:
            self.history['alpha'] = []
            
        for step in range(1, steps + 1):
            # Update alpha using scheduler
            current_alpha = self.alpha_scheduler.step()
            self.net.update_alpha(current_alpha)
            
            # Determine whether to use counterexamples
            use_ce = True
            if counterexample_schedule is not None:
                use_ce = step in counterexample_schedule
            
            # Execute training step
            metrics = self.train_step(use_counterexamples=use_ce)
            
            # Update history
            self.history['loss'].append(metrics['loss'])
            self.history['decrease_violations'].append(metrics['decrease_violations'])
            self.history['steps'].append(step)
            self.history['time'].append(time.time() - start_time)
            self.history['alpha'].append(current_alpha)
            
            # Log progress
            if step % log_every == 0 or step == steps:
                elapsed = time.time() - start_time
                logger.info(
                    f"Step {step}/{steps} ({step/steps*100:.1f}%) - "
                    f"Loss: {metrics['loss']:.6f}, "
                    f"Violations: {metrics['decrease_violations']} "
                    f"({metrics['violation_rate']*100:.2f}%), "
                    f"Time: {elapsed:.2f}s"
                )
            
            # Save intermediate model
            if save_path and save_every and step % save_every == 0:
                save_file = f"{save_path}.step{step}"
                self.net.save(save_file)
        
        # Save final model
        if save_path:
            self.net.save(save_path)
            
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        return self.history

    def evaluate(self, n_samples: int = 1000) -> Dict[str, float]:
        """
        Evaluate the current model on a test batch.
        
        Args:
            n_samples: Number of samples to evaluate on
            
        Returns:
            Dict containing evaluation metrics
        """
        self.net.eval()
        
        # Generate samples
        batch_size = self.sampler.batch_size
        original_batch_size = self.sampler.batch_size
        
        # Temporarily set batch size
        self.sampler.batch_size = n_samples
        x_np, xdot_np = self.sampler.random_batch()
        self.sampler.batch_size = original_batch_size
        
        # Convert to tensors
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        xdot = torch.tensor(xdot_np, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            # Compute V(x)
            V = self.net(x)
            
            # Compute ∇V manually since we're in no_grad context
            x.requires_grad_(True)
            V_grad = self.net(x)
            gradV = torch.autograd.grad(V_grad.sum(), x)[0]
            
            # Compute V̇ = ∇V · ẋ
            Vdot = torch.sum(gradV * xdot, dim=1, keepdim=True)
            
            # Compute margin term
            if self.gamma > 0:
                margin = self.gamma * torch.norm(x, dim=1, keepdim=True)
            else:
                margin = 0.0
            
            # Calculate metrics
            decrease_violations = torch.sum(Vdot > 0).item()
            margin_violations = torch.sum(Vdot + margin > 0).item()
            
            # Calculate statistics
            v_min = V.min().item()
            v_max = V.max().item()
            v_mean = V.mean().item()
            vdot_min = Vdot.min().item()
            vdot_max = Vdot.max().item()
            vdot_mean = Vdot.mean().item()
        
        # Return metrics
        return {
            'decrease_violations': decrease_violations,
            'decrease_violation_rate': decrease_violations / n_samples,
            'margin_violations': margin_violations,
            'margin_violation_rate': margin_violations / n_samples,
            'V_min': v_min,
            'V_max': v_max,
            'V_mean': v_mean,
            'Vdot_min': vdot_min,
            'Vdot_max': vdot_max,
            'Vdot_mean': vdot_mean
        }
