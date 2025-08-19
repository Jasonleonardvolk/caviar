"""
Barrier Function Trainer

This module provides training functionality for neural barrier functions.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..models.torch_models import TorchBarrierNetwork
from .losses import BarrierLoss


class BarrierTrainer:
    """
    Trainer for neural barrier functions using PyTorch.
    
    This class handles training neural barrier functions, including
    data management, optimization, and evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dynamics_fn: Optional[Callable] = None,
        alpha_fn: Optional[Callable] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        classification_weight: float = 1.0,
        gradient_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        margin: float = 0.1,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Initialize the barrier trainer.
        
        Args:
            model: Neural network model for barrier function
            dynamics_fn: Function mapping (x, u) to state derivatives
            alpha_fn: Class-K function for barrier certificate condition
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            classification_weight: Weight for classification loss
            gradient_weight: Weight for gradient loss
            smoothness_weight: Weight for smoothness loss
            margin: Margin for classification loss
            device: Device to use for computations
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        # Set up loss function
        self.loss_fn = BarrierLoss(
            dynamics_fn=dynamics_fn,
            alpha_fn=alpha_fn,
            classification_weight=classification_weight,
            gradient_weight=gradient_weight,
            smoothness_weight=smoothness_weight,
            margin=margin,
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
            "classification_loss": [],
            "gradient_loss": [],
            "smoothness_loss": [],
            "safe_loss": [],
            "unsafe_loss": [],
            "validation_accuracy": []
        }
    
    def train(
        self,
        states: np.ndarray,
        labels: np.ndarray,
        control_inputs: Optional[np.ndarray] = None,
        batch_size: int = 64,
        epochs: int = 100,
        validation_split: float = 0.2,
        eval_every: int = 1,
        verbose: bool = True,
        early_stopping: bool = False,
        patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the neural barrier function.
        
        Args:
            states: Training states, shape (num_samples, state_dim)
            labels: Training labels (1 for safe, 0 for unsafe), shape (num_samples, 1)
            control_inputs: Control inputs, shape (num_samples, input_dim)
            batch_size: Batch size for training
            epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            eval_every: Evaluate model every n epochs
            verbose: Whether to print training progress
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            Training history
        """
        # Convert data to PyTorch tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        if control_inputs is not None:
            control_inputs_tensor = torch.tensor(control_inputs, dtype=torch.float32)
        else:
            control_inputs_tensor = None
        
        # Split data into training and validation sets
        n_samples = len(states)
        n_val = int(validation_split * n_samples)
        n_train = n_samples - n_val
        
        # Create indices for random split
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Split data
        train_states = states_tensor[train_indices]
        train_labels = labels_tensor[train_indices]
        
        val_states = states_tensor[val_indices]
        val_labels = labels_tensor[val_indices]
        
        if control_inputs_tensor is not None:
            train_inputs = control_inputs_tensor[train_indices]
            val_inputs = control_inputs_tensor[val_indices]
        else:
            train_inputs = None
            val_inputs = None
        
        # Create data loaders
        if train_inputs is not None:
            train_dataset = TensorDataset(train_states, train_labels, train_inputs)
        else:
            train_dataset = TensorDataset(train_states, train_labels)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Training loop
        best_val_acc = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = {
                "total_loss": 0.0,
                "classification_loss": 0.0,
                "gradient_loss": 0.0,
                "smoothness_loss": 0.0,
                "safe_loss": 0.0,
                "unsafe_loss": 0.0
            }
            
            for batch_idx, batch_data in enumerate(train_loader):
                # Get batch data
                if len(batch_data) == 3:
                    batch_states, batch_labels, batch_inputs = batch_data
                    batch_inputs = batch_inputs.to(self.device)
                else:
                    batch_states, batch_labels = batch_data
                    batch_inputs = None
                
                batch_states = batch_states.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                loss_dict = self.loss_fn(
                    self.model,
                    batch_states,
                    batch_labels,
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
                val_acc = self.evaluate(val_states, val_labels, val_inputs)
                self.history["validation_accuracy"].append(val_acc)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {train_losses['total_loss']:.4f} - "
                          f"Val Acc: {val_acc:.4f}")
                
                # Early stopping
                if early_stopping:
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = epoch
                    elif epoch - best_epoch >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
        
        return self.history
    
    def evaluate(
        self,
        states: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        control_inputs: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> float:
        """
        Evaluate the neural barrier function.
        
        Args:
            states: Evaluation states, shape (num_samples, state_dim)
            labels: Evaluation labels (1 for safe, 0 for unsafe), shape (num_samples, 1)
            control_inputs: Control inputs, shape (num_samples, input_dim)
            
        Returns:
            Classification accuracy
        """
        # Convert data to PyTorch tensors if needed
        if isinstance(states, np.ndarray):
            states = torch.tensor(states, dtype=torch.float32)
        
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.float32)
        
        if control_inputs is not None and isinstance(control_inputs, np.ndarray):
            control_inputs = torch.tensor(control_inputs, dtype=torch.float32)
        
        # Move data to device
        states = states.to(self.device)
        labels = labels.to(self.device)
        
        if control_inputs is not None:
            control_inputs = control_inputs.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Compute predictions
        with torch.no_grad():
            barrier_values = self.model(states)
            predictions = (barrier_values > 0).float()
            
            # Compute accuracy
            correct = (predictions == labels).float().sum().item()
            total = labels.numel()
            accuracy = correct / total
        
        return accuracy
    
    def verify_barrier_condition(
        self,
        states: Union[np.ndarray, torch.Tensor],
        control_inputs: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Verify if states satisfy the barrier condition.
        
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
        
        # Compute barrier values
        barrier_values = self.model(states)
        
        # Filter for states in the safe set (B(x) > 0)
        safe_mask = barrier_values > 0
        
        # Initialize result tensor (all False)
        verification_result = torch.zeros_like(safe_mask, dtype=torch.bool)
        
        # Verify condition for safe states
        if self.loss_fn.dynamics_fn is not None and safe_mask.any():
            # Extract safe states
            safe_indices = safe_mask.nonzero().squeeze()
            
            # Handle single index case
            if safe_indices.dim() == 0:
                safe_indices = safe_indices.unsqueeze(0)
            
            safe_states = states[safe_indices]
            safe_barrier_values = barrier_values[safe_indices]
            
            # Get control inputs for safe states
            if control_inputs is not None:
                safe_inputs = control_inputs[safe_indices]
            else:
                safe_inputs = None
            
            # Compute gradients
            gradients = torch.autograd.grad(
                safe_barrier_values.sum(),
                safe_states,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Compute dynamics
            if safe_inputs is not None:
                dynamics = self.loss_fn.dynamics_fn(safe_states, safe_inputs)
            else:
                dynamics = self.loss_fn.dynamics_fn(safe_states, None)
            
            # Compute Lie derivatives
            lie_derivatives = torch.sum(gradients * dynamics, dim=1, keepdim=True)
            
            # Compute α(B(x))
            alpha_values = self.loss_fn.alpha_fn(safe_barrier_values)
            
            # Verify condition: Lie derivative >= -α(B(x))
            condition_satisfied = lie_derivatives >= -alpha_values
            
            # Update verification result for safe states
            verification_result[safe_indices] = condition_satisfied
        
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
        axes[0, 1].plot(self.history["classification_loss"], label="Classification")
        axes[0, 1].plot(self.history["gradient_loss"], label="Gradient")
        axes[0, 1].plot(self.history["smoothness_loss"], label="Smoothness")
        axes[0, 1].set_title("Component Losses")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot safe and unsafe losses
        axes[1, 0].plot(self.history["safe_loss"], label="Safe")
        axes[1, 0].plot(self.history["unsafe_loss"], label="Unsafe")
        axes[1, 0].set_title("Safe and Unsafe Losses")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot validation accuracy
        axes[1, 1].plot(self.history["validation_accuracy"])
        axes[1, 1].set_title("Validation Accuracy")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].grid(True)
        
        fig.tight_layout()
        return fig
    
    def visualize_barrier(
        self,
        states: np.ndarray,
        labels: np.ndarray,
        dims: Tuple[int, int] = (0, 1),
        resolution: int = 100,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Visualize the learned barrier function.
        
        Args:
            states: States for visualization, shape (num_samples, state_dim)
            labels: Labels for states (1 for safe, 0 for unsafe), shape (num_samples, 1)
            dims: Dimensions to visualize
            resolution: Resolution for contour plot
            figsize: Figure size
            
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
        
        # Evaluate barrier function on grid
        with torch.no_grad():
            grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(self.device)
            barrier_values = self.model(grid_tensor).cpu().numpy().reshape(resolution, resolution)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot contour
        contour = ax.contourf(xx, yy, barrier_values, levels=20, cmap='RdBu_r', alpha=0.7)
        fig.colorbar(contour, ax=ax, label='Barrier Value')
        
        # Plot zero levelset
        ax.contour(xx, yy, barrier_values, levels=[0], colors='k', linestyles='-', linewidths=2)
        
        # Plot states
        safe_mask = labels.squeeze() > 0.5
        ax.scatter(
            states[safe_mask, x_dim],
            states[safe_mask, y_dim],
            c='g',
            marker='o',
            label='Safe',
            alpha=0.7
        )
        ax.scatter(
            states[~safe_mask, x_dim],
            states[~safe_mask, y_dim],
            c='r',
            marker='x',
            label='Unsafe',
            alpha=0.7
        )
        
        # Set plot properties
        ax.set_xlabel(f'Dimension {x_dim}')
        ax.set_ylabel(f'Dimension {y_dim}')
        ax.set_title('Learned Barrier Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
