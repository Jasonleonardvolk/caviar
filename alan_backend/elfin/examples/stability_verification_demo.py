"""
ELFIN Stability Verification Demo.

This script demonstrates how to use the ELFIN stability verification framework
to verify Lyapunov stability for a simple dynamical system.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import time
import os
from pathlib import Path

import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from alan_backend.elfin.stability import (
    ConstraintIR, VerificationResult, VerificationStatus, ConstraintType,
    ProofCache, LyapunovNetwork, DynamicsModel, NeuralLyapunovLearner
)
from alan_backend.elfin.stability.backends import SOSVerifier

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_dynamics(x):
    """
    Example dynamics dx/dt = -x for a simple linear system.
    
    Args:
        x: State vector
        
    Returns:
        Derivative dx/dt
    """
    return -x


def create_constraint(state_dim=3):
    """
    Create a constraint for positive definiteness verification.
    
    Args:
        state_dim: Dimension of the state space
        
    Returns:
        ConstraintIR instance
    """
    # Create a random positive definite matrix
    q_matrix = np.random.randn(state_dim, state_dim)
    q_matrix = q_matrix.T @ q_matrix + np.eye(state_dim) * 0.1
    
    # Create variables
    variables = [f"x{i}" for i in range(state_dim)]
    
    # Create constraint for positive definiteness
    constraint = ConstraintIR(
        id="pd_test",
        variables=variables,
        expression="V(x) > 0",  # Symbolic representation
        constraint_type=ConstraintType.POSITIVE,
        context={
            "q_matrix": q_matrix.tolist(),
            "dimension": state_dim,
            "lyapunov_type": "polynomial"
        },
        solver_hint="sos",
        proof_needed=True
    )
    
    return constraint, q_matrix


def verify_with_sos(constraint, q_matrix):
    """
    Verify a constraint using SOS.
    
    Args:
        constraint: Constraint to verify
        q_matrix: Q matrix for verification
        
    Returns:
        Verification result
    """
    logger.info("Verifying with SOS...")
    
    # Create SOS verifier
    verifier = SOSVerifier()
    
    # Verify positive definiteness
    start_time = time.time()
    success, details = verifier.verify_pd(q_matrix)
    verification_time = time.time() - start_time
    
    # Create verification result
    status = VerificationStatus.VERIFIED if success else VerificationStatus.REFUTED
    
    result = VerificationResult(
        constraint_id=constraint.id,
        status=status,
        proof_hash=constraint.compute_hash(),
        verification_time=verification_time,
        certificate=details.get("certificate"),
        counterexample=details.get("counterexample"),
        solver_info={"solver": "sos_verifier"}
    )
    
    logger.info(f"SOS verification result: {status.name} in {verification_time:.3f}s")
    
    return result


def train_neural_lyapunov(state_dim=3, epochs=500):
    """
    Train a neural Lyapunov function.
    
    Args:
        state_dim: Dimension of the state space
        epochs: Number of training epochs
        
    Returns:
        Trained neural Lyapunov function
    """
    logger.info(f"Training neural Lyapunov function ({epochs} epochs)...")
    
    # Create dynamics model
    dynamics = DynamicsModel(
        forward_fn=example_dynamics,
        input_dim=state_dim
    )
    
    # Create neural Lyapunov learner
    learner = NeuralLyapunovLearner(
        dynamics=dynamics,
        state_dim=state_dim,
        hidden_dims=[32, 32],
        lr=1e-3,
        epsilon=0.1,
        log_interval=100,
        zero_penalty=10.0
    )
    
    # Train
    history = learner.train(
        n_epochs=epochs,
        samples_per_epoch=1000,
        include_origin=True
    )
    
    # Verify
    success, details = learner.verify_around_equilibrium(
        radius=1.0,
        n_samples=5000,
        epsilon=0.0
    )
    
    logger.info(f"Neural Lyapunov verification: {'Success' if success else 'Failed'}")
    logger.info(f"  PD violations: {details['pd_violations']}/{5000}")
    logger.info(f"  Decreasing violations: {details['decreasing_violations']}/{5000}")
    
    return learner, history


def plot_lyapunov_values(lyapunov_fn, title="Lyapunov Function", 
                       x_range=(-2, 2), y_range=(-2, 2), num_points=50):
    """
    Plot Lyapunov function values in 2D.
    
    Args:
        lyapunov_fn: Lyapunov function
        title: Plot title
        x_range: Range for x-axis
        y_range: Range for y-axis
        num_points: Number of points per dimension
    """
    # Create meshgrid
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate Lyapunov function
    Z = np.zeros_like(X)
    
    for i in range(num_points):
        for j in range(num_points):
            state = np.array([X[i, j], Y[i, j], 0.0])  # Assume 3D state with z=0
            
            if isinstance(lyapunov_fn, np.ndarray):
                # Quadratic form
                Z[i, j] = state @ lyapunov_fn @ state
            else:
                # Neural network
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    Z[i, j] = lyapunov_fn(state_tensor).item()
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Lyapunov Value')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    
    # Add a quiver plot for the dynamics
    dx = np.zeros_like(X)
    dy = np.zeros_like(Y)
    for i in range(num_points):
        for j in range(num_points):
            state = np.array([X[i, j], Y[i, j], 0.0])  # Assume 3D state with z=0
            dstate = example_dynamics(state)
            dx[i, j] = dstate[0]
            dy[i, j] = dstate[1]
    
    # Skip some points for clarity
    stride = 5
    plt.quiver(X[::stride, ::stride], Y[::stride, ::stride], 
              dx[::stride, ::stride], dy[::stride, ::stride],
              color='white', alpha=0.8)
    
    plt.tight_layout()


def plot_training_history(history):
    """
    Plot training history for neural Lyapunov function.
    
    Args:
        history: Training history
    """
    plt.figure(figsize=(12, 6))
    
    # Plot total loss
    plt.subplot(1, 2, 1)
    plt.semilogy(history['total_loss'], label='Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.grid(True)
    
    # Plot component losses
    plt.subplot(1, 2, 2)
    plt.semilogy(history['pd_loss'], label='PD Loss')
    plt.semilogy(history['decreasing_loss'], label='Decreasing Loss')
    plt.semilogy(history['zero_loss'], label='Zero Loss')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()


def main():
    """Run the demonstration."""
    state_dim = 3
    
    # Create constraint for SOS verification
    constraint, q_matrix = create_constraint(state_dim)
    
    # Verify with SOS
    verify_with_sos(constraint, q_matrix)
    
    # Train neural Lyapunov function
    learner, history = train_neural_lyapunov(state_dim, epochs=500)
    
    # Plot Lyapunov functions
    plot_lyapunov_values(q_matrix, title="Quadratic Lyapunov Function")
    plot_lyapunov_values(learner.network, title="Neural Lyapunov Function")
    
    # Plot training history
    plot_training_history(history)
    
    plt.show()


if __name__ == "__main__":
    main()
