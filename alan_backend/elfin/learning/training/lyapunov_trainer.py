"""
Lyapunov Function Trainer

This module provides training functionality for neural Lyapunov functions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..models.torch_models import TorchLyapunovNetwork
from .losses import LyapunovLoss


class LyapunovTrainer:
    """
    Trainer for neural Lyapunov functions using PyTorch.
    
    This class handles training neural Lyapunov functions, including
    data management, optimization, and evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dynamics_fn: Optional[Callable] = None,
        origin: Optional[torch.Tensor] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        positive_weight: float = 1.0,
        derivative_weight: float = 1.0,
        boundedness_weight: float = 0.1,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Initialize the Lyapunov trainer.
        
        Args:
            model: Neural network model for Lyapunov function
            dynamics_fn: Function mapping (x, u) to state derivatives
            origin: Equilibrium point, shape (state_dim,)
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            positive_weight: Weight for positive definiteness loss
            derivative_weight: Weight for derivative loss
            boundedness_weight: Weight for boundedness loss
            device: Device to use for computations
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        # Convert origin to device
        if origin is not None:
            origin = origin.to(self.device)
        
        # Set up loss function
        self.loss_fn = LyapunovLoss(
            dynamics_fn=dynamics_fn,
            origin=origin,
            positive_weight=positive_weight,
            derivative_weight=derivative_weight,
            boundedness_weight=boundedness_weight,
            device=self.device
        )
        
        # Set up optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize training history
        self.history = {
            "total_loss": [],
            "positive_loss": [],
            "derivative_loss": [],
            "boundedness_loss": [],
            "origin_loss": [],
            "near_origin_loss": [],
            "far_from_origin_loss": []
        }
    
    def train(
        self,
        states: np.ndarray,
        control_inputs: Optional[np.ndarray] = None,
        batch_size: int = 64,
        epochs: int = 100,
        validation_states: Optional[np.ndarray] = None,
        validation_inputs: Optional[np.ndarray] = None,
        eval_every: int = 1,
        verbose: bool = True,
        early_stopping: bool = False,
        patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the neural Lyapunov function.
        
        Args:
            states: Training states, shape (num_samples, state_dim)
            control_inputs: Control inputs, shape (num_samples, input_dim)
            batch_size: Batch size for training
            epochs: Number of training epochs
            validation_states: Validation states, shape (num_val, state_dim)
            validation_inputs: Validation inputs, shape (num_val, input_dim)
            eval_every: Evaluate model every n epochs
            verbose: Whether to print training progress
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            Training history
        """
        # Convert data to PyTorch tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        
        if control_inputs is not None:
            control_inputs_tensor = torch.tensor(control_inputs, dtype=torch.float32)
        else:
            control_inputs_tensor = None
        
        # Create data loaders
        if control_inputs_tensor is not None:
            train_dataset = TensorDataset(states_tensor, control_inputs_tensor)
        else:
            train_dataset = TensorDataset(states_tensor)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Set up validation data
        if validation_states is not None:
            val_states = torch.tensor(validation_states, dtype=torch.float32).to(self.device)
            
            if validation_inputs is not None:
                val_inputs = torch.tensor(validation_inputs, dtype=torch.float32).to(self.device)
            else:
                val_inputs = None
        else:
            val_states = None
            val_inputs = None
        
        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = {
                "total_loss": 0.0,
                "positive_loss": 0.0,
                "derivative_loss": 0.0,
                "boundedness_loss": 0.0,
                "origin_loss": 0.0,
                "near_origin_loss": 0.0,
                "far_from_origin_loss": 0.0
            }
            
            for batch_idx, batch_data in enumerate(train_loader):
                # Get batch data
                if len(batch_data) == 2:
                    batch_states, batch_inputs = batch_data
                    batch_inputs = batch_inputs.to(self.device)
                else:
                    batch_states = batch_data[0]
                    batch_inputs = None
                
                batch_states = batch_states.to(self.device)
                
                # Forward pass
                loss_dict = self.loss_fn(
                    self.model,
                    batch_states,
                    batch_inputs
                )
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                self.optimizer.step()
                
                # Update training losses
                for key in train_losses.keys():
                    train_losses[key] += loss_dict[key].item()
            
            # Compute average losses
            for key in train_losses.keys():
                train_losses[key] /= len(train_loader)
                self.history[key].append(train_losses[key])
            
            # Evaluation
            if epoch % eval_every == 0:
                if val_states is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_loss_dict = self.loss_fn(
                            self.model,
                            val_states,
                            val_inputs
                        )
                        val_total_loss = val_loss_dict["total_loss"].item()
                else:
                    val_total_loss = train_losses["total_loss"]
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {train_losses['total_loss']:.4f} - "
                          f"Val Loss: {val_total_loss:.4f}")
                
                # Early stopping
                if early_stopping:
                    if val_total_loss < best_val_loss:
                        best_val_loss = val_total_loss
                        best_epoch = epoch
                    elif epoch - best_epoch >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
        
        return self.history
    
    def verify_lyapunov_condition(
        self,
        states: Union[np.ndarray, torch.Tensor],
        control_inputs: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Verify if states satisfy the Lyapunov condition.
        
        Args:
            states: States to verify, shape (num_samples, state_dim)
            control_inputs: Control inputs, shape (num_samples, input_dim)
            
        Returns:
            Boolean tensor indicating which states satisfy the condition
        """
        # Convert data to PyTorch tensors if needed
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32)
        
        if control_inputs is not None and isinstance(control_inputs, np.ndarray):
            control_inputs = torch.tensor(control_inputs, dtype=torch.float32)
        
        # Move data to device
        states = states.to(self.device)
        
        if control_inputs is not None:
            control_inputs = control_inputs.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make states require gradients
        states = states.clone().detach().requires_grad_(True)
        
        # Compute Lyapunov values
        lyapunov_values = self.model(states)
        
        # Initialize result tensor (all False)
        verification_result = torch.zeros_like(lyapunov_values, dtype=torch.bool)
        
        # Verify condition for all states (except those very close to origin)
        if self.loss_fn.dynamics_fn is not None:
            # Compute distance from origin
            if hasattr(self.loss_fn, "origin"):
                origin = self.loss_fn.origin
                if origin.dim() == 1:
                    origin = origin.unsqueeze(0)
                
                distance_from_origin = torch.norm(states - origin, dim=1, keepdim=True)
            else:
                distance_from_origin = torch.norm(states, dim=1, keepdim=True)
            
            # Exclude states very close to origin
            valid_mask = distance_from_origin > 1e-4
            
            if valid_mask.any():
                # Extract valid states
                valid_indices = valid_mask.nonzero().squeeze()
                
                # Handle single index case
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                
                valid_states = states[valid_indices]
                
                # Compute gradients
                valid_values = self.model(valid_states)
                gradients = torch.autograd.grad(
                    valid_values.sum(),
                    valid_states,
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                # Compute dynamics
                if control_inputs is not None:
                    valid_inputs = control_inputs[valid_indices]
                    dynamics = self.loss_fn.dynamics_fn(valid_states, valid_inputs)
                else:
                    dynamics = self.loss_fn.dynamics_fn(valid_states, None)
                
                # Compute Lie derivatives
                lie_derivatives = torch.sum(gradients * dynamics, dim=1, keepdim=True)
                
                # Verify condition: Lie derivative < 0
                condition_satisfied = lie_derivatives < 0
                
                # Update verification result for valid states
                verification_result[valid_indices] = condition_satisfied
                
                # All states very close to origin are considered stable
                verification_result[~valid_mask] = True
        
        return verification_result
    
    def save(self, filepath: str):
        """
        Save the model and training history.
        
        Args:
            filepath: Path to save the model and history
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history
        }, filepath)
    
    def load(self, filepath: str):
        """
        Load the model and training history.
        
        Args:
            filepath: Path to the saved model and history
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
    
    def visualize_training(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize training history.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot total loss
        axes[0, 0].plot(self.history["total_loss"])
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)
        
        # Plot component losses
        axes[0, 1].plot(self.history["positive_loss"], label="Positive")
        axes[0, 1].plot(self.history["derivative_loss"], label="Derivative")
        axes[0, 1].plot(self.history["boundedness_loss"], label="Boundedness")
        axes[0, 1].set_title("Component Losses")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot positive definiteness losses
        axes[1, 0].plot(self.history["origin_loss"], label="Origin")
        axes[1, 0].plot(self.history["near_origin_loss"], label="Near Origin")
        axes[1, 0].plot(self.history["far_from_origin_loss"], label="Far from Origin")
        axes[1, 0].set_title("Positive Definiteness Losses")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Keep one slot empty or use for another visualization
        axes[1, 1].set_visible(False)
        
        fig.tight_layout()
        return fig
    
    def visualize_lyapunov(
        self,
        states: np.ndarray,
        dims: Tuple[int, int] = (0, 1),
        resolution: int = 100,
        figsize: Tuple[int, int] = (12, 8),
        show_stability: bool = False,
        control_inputs: Optional[np.ndarray] = None
    ):
        """
        Visualize the learned Lyapunov function.
        
        Args:
            states: States for visualization, shape (num_samples, state_dim)
            dims: Dimensions to visualize
            resolution: Resolution for contour plot
            figsize: Figure size
            show_stability: Whether to show stability information
            control_inputs: Control inputs, shape (num_samples, input_dim)
            
        Returns:
            Matplotlib figure
        """
        # Extract dimensions to visualize
        x_dim, y_dim = dims
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Generate grid for contour plot
        x_min, x_max = np.min(states[:, x_dim]), np.max(states[:, x_dim])
        y_min, y_max = np.min(states[:, y_dim]), np.max(states[:, y_dim])
        
        # Add margin
        margin = 0.1 * max(x_max - x_min, y_max - y_min)
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        # Generate grid
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Create grid points
        grid_points = np.zeros((resolution * resolution, states.shape[1]))
        grid_points[:, x_dim] = xx.ravel()
        grid_points[:, y_dim] = yy.ravel()
        
        # Evaluate Lyapunov function on grid
        with torch.no_grad():
            grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(self.device)
            lyapunov_values = self.model(grid_tensor).cpu().numpy().reshape(resolution, resolution)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot contour
        contour = ax.contourf(xx, yy, lyapunov_values, levels=20, cmap='viridis', alpha=0.7)
        fig.colorbar(contour, ax=ax, label='Lyapunov Value')
        
        # Plot level sets
        level_sets = np.linspace(np.min(lyapunov_values) + 0.1, np.max(lyapunov_values), 10)
        cs = ax.contour(xx, yy, lyapunov_values, levels=level_sets, colors='k', linestyles='-', linewidths=1, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=8)
        
        # If showing stability, color regions by stability
        if show_stability and self.loss_fn.dynamics_fn is not None:
            # Verify stability on grid points
            stability = self.verify_lyapunov_condition(
                grid_tensor,
                None if control_inputs is None else torch.zeros_like(grid_tensor[:, :control_inputs.shape[1]])
            ).cpu().numpy().reshape(resolution, resolution)
            
            # Plot stability regions
            ax.contourf(xx, yy, stability, levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.3)
        
        # Plot states
        ax.scatter(
            states[:, x_dim],
            states[:, y_dim],
            c='black',
            marker='o',
            s=10,
            alpha=0.5
        )
        
        # Plot origin
        if hasattr(self.loss_fn, "origin"):
            origin = self.loss_fn.origin.cpu().numpy()
            if origin.ndim == 1:
                ax.scatter(
                    origin[x_dim],
                    origin[y_dim],
                    c='red',
                    marker='x',
                    s=100,
                    linewidth=2,
                    label='Equilibrium Point'
                )
        else:
            ax.scatter(
                0, 0,
                c='red',
                marker='x',
                s=100,
                linewidth=2,
                label='Equilibrium Point'
            )
        
        # Set plot properties
        ax.set_xlabel(f'Dimension {x_dim}')
        ax.set_ylabel(f'Dimension {y_dim}')
        ax.set_title('Learned Lyapunov Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_vector_field(
        self,
        states: np.ndarray,
        dims: Tuple[int, int] = (0, 1),
        resolution: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        controller: Optional[Callable] = None
    ):
        """
        Visualize the vector field of the system dynamics.
        
        Args:
            states: States for visualization, shape (num_samples, state_dim)
            dims: Dimensions to visualize
            resolution: Resolution for vector field
            figsize: Figure size
            controller: Controller function mapping state to control input
            
        Returns:
            Matplotlib figure
        """
        if self.loss_fn.dynamics_fn is None:
            raise ValueError("Dynamics function is required for vector field visualization")
        
        # Extract dimensions to visualize
        x_dim, y_dim = dims
        
        # Generate grid for vector field
        x_min, x_max = np.min(states[:, x_dim]), np.max(states[:, x_dim])
        y_min, y_max = np.min(states[:, y_dim]), np.max(states[:, y_dim])
        
        # Add margin
        margin = 0.1 * max(x_max - x_min, y_max - y_min)
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        # Generate grid
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Create grid points
        grid_points = np.zeros((resolution * resolution, states.shape[1]))
        grid_points[:, x_dim] = xx.ravel()
        grid_points[:, y_dim] = yy.ravel()
        
        # Convert to tensor
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(self.device)
        
        # Compute control inputs if controller is provided
        if controller is not None:
            control_inputs = torch.tensor(
                np.array([controller(state) for state in grid_points]),
                dtype=torch.float32
            ).to(self.device)
        else:
            control_inputs = None
        
        # Compute dynamics
        with torch.no_grad():
            if control_inputs is not None:
                dynamics = self.loss_fn.dynamics_fn(grid_tensor, control_inputs)
            else:
                dynamics = self.loss_fn.dynamics_fn(grid_tensor, None)
            
            dynamics = dynamics.cpu().numpy()
        
        # Extract vector field components
        u = dynamics[:, x_dim].reshape(resolution, resolution)
        v = dynamics[:, y_dim].reshape(resolution, resolution)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Evaluate Lyapunov function on grid
        with torch.no_grad():
            lyapunov_values = self.model(grid_tensor).cpu().numpy().reshape(resolution, resolution)
        
        # Plot contour of Lyapunov function
        contour = ax.contourf(xx, yy, lyapunov_values, levels=20, cmap='viridis', alpha=0.5)
        fig.colorbar(contour, ax=ax, label='Lyapunov Value')
        
        # Plot vector field
        ax.quiver(xx, yy, u, v, angles='xy', scale_units='xy', scale=10, width=0.002, color='black')
        
        # Plot origin
        if hasattr(self.loss_fn, "origin"):
            origin = self.loss_fn.origin.cpu().numpy()
            if origin.ndim == 1:
                ax.scatter(
                    origin[x_dim],
                    origin[y_dim],
                    c='red',
                    marker='x',
                    s=100,
                    linewidth=2,
                    label='Equilibrium Point'
                )
        else:
            ax.scatter(
                0, 0,
                c='red',
                marker='x',
                s=100,
                linewidth=2,
                label='Equilibrium Point'
            )
        
        # Set plot properties
        ax.set_xlabel(f'Dimension {x_dim}')
        ax.set_ylabel(f'Dimension {y_dim}')
        ax.set_title('Vector Field and Lyapunov Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
