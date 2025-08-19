"""
Mobile Robot Example with Neural Barrier and Lyapunov Functions

This example demonstrates how to:
1. Create and train neural barrier functions for a mobile robot
2. Export trained networks to ELFIN format
3. Import ELFIN models back to neural networks
4. Benchmark and verify barrier certificates

The mobile robot is modeled as a simple 2D system with states (x, y, θ, v) representing
position, orientation, and velocity.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Callable

# Add parent directory to path for imports when running as a script
import sys
import os
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from alan_backend.elfin.learning.models.torch_models import TorchBarrierNetwork, TorchLyapunovNetwork
from alan_backend.elfin.learning.training.data_generator import ObstacleDataGenerator
from alan_backend.elfin.learning.training.barrier_trainer import BarrierTrainer
from alan_backend.elfin.learning.training.lyapunov_trainer import LyapunovTrainer
from alan_backend.elfin.learning.integration.export import export_to_elfin
from alan_backend.elfin.learning.integration.import_models import import_barrier_function
from alan_backend.elfin.learning.integration.benchmark_integration import (
    NeuralBarrierSystem,
    BenchmarkIntegration
)


# Define mobile robot dynamics
def mobile_robot_dynamics(state: np.ndarray, control_input: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Mobile robot dynamics with states (x, y, θ, v) and inputs (a, ω).
    
    Args:
        state: State vector [x, y, θ, v]
        control_input: Control input [a, ω] (acceleration and angular velocity)
        
    Returns:
        State derivative [dx/dt, dy/dt, dθ/dt, dv/dt]
    """
    # Handle batch input
    if state.ndim == 1:
        state = state.reshape(1, -1)
    
    if control_input is not None and control_input.ndim == 1:
        control_input = control_input.reshape(1, -1)
    
    # Extract states
    theta = state[:, 2]
    v = state[:, 3]
    
    # Initialize derivatives
    derivatives = np.zeros_like(state)
    
    # Position derivatives
    derivatives[:, 0] = v * np.cos(theta)  # dx/dt
    derivatives[:, 1] = v * np.sin(theta)  # dy/dt
    
    # Default zero control
    a = np.zeros_like(v)
    omega = np.zeros_like(v)
    
    # Apply control if provided
    if control_input is not None:
        a = control_input[:, 0]
        omega = control_input[:, 1]
    
    # Orientation and velocity derivatives
    derivatives[:, 2] = omega  # dθ/dt
    derivatives[:, 3] = a      # dv/dt
    
    return derivatives


# Define a PyTorch version of the dynamics for barrier training
def mobile_robot_dynamics_torch(state: torch.Tensor, control_input: Optional[torch.Tensor] = None) -> torch.Tensor:
    """PyTorch implementation of mobile robot dynamics."""
    # Extract states
    theta = state[:, 2]
    v = state[:, 3]
    
    # Initialize derivatives
    derivatives = torch.zeros_like(state)
    
    # Position derivatives
    derivatives[:, 0] = v * torch.cos(theta)  # dx/dt
    derivatives[:, 1] = v * torch.sin(theta)  # dy/dt
    
    # Default zero control
    a = torch.zeros_like(v)
    omega = torch.zeros_like(v)
    
    # Apply control if provided
    if control_input is not None:
        a = control_input[:, 0]
        omega = control_input[:, 1]
    
    # Orientation and velocity derivatives
    derivatives[:, 2] = omega  # dθ/dt
    derivatives[:, 3] = a      # dv/dt
    
    return derivatives


def train_barrier_function() -> TorchBarrierNetwork:
    """
    Train a neural barrier function for obstacle avoidance.
    
    Returns:
        Trained barrier network
    """
    print("Training barrier function...")
    
    # Define obstacles in the environment
    obstacles = [
        {'type': 'circle', 'position': np.array([2.0, 2.0]), 'radius': 0.5},
        {'type': 'circle', 'position': np.array([-1.0, 1.5]), 'radius': 0.7},
        {'type': 'rectangle', 'position': np.array([0.0, -2.0]), 'width': 1.0, 'height': 0.8}
    ]
    
    # Create data generator
    data_generator = ObstacleDataGenerator(
        state_dim=4,
        input_dim=2,
        dynamics_fn=mobile_robot_dynamics,
        obstacles=obstacles,
        robot_radius=0.3,
        safety_margin=0.2,
        state_bounds=np.array([[-5.0, 5.0], [-5.0, 5.0], [-np.pi, np.pi], [0.0, 2.0]])
    )
    
    # Generate training data
    states, labels = data_generator.generate_safe_unsafe_data(
        num_samples=5000,
        balance=True
    )
    
    # Visualize environment and training data
    fig = data_generator.visualize_environment(
        states=states[:, :2],  # Only use x, y coordinates for visualization
        labels=labels
    )
    
    # Save the visualization
    output_dir = "outputs/mobile_robot"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{output_dir}/environment.png")
    plt.close(fig)
    
    # Create barrier network model
    model = TorchBarrierNetwork(
        state_dim=4,
        hidden_layers=[64, 64, 32],
        activation="tanh"
    )
    
    # Create barrier trainer
    trainer = BarrierTrainer(
        model=model,
        dynamics_fn=mobile_robot_dynamics_torch,
        classification_weight=1.0,
        gradient_weight=0.5,
        smoothness_weight=0.1
    )
    
    # Train model
    trainer.train(
        states=states,
        labels=labels,
        batch_size=64,
        epochs=500,
        validation_split=0.2,
        early_stopping=True,
        patience=20,
        verbose=True
    )
    
    # Visualize training history
    fig = trainer.visualize_training()
    fig.savefig(f"{output_dir}/barrier_training.png")
    plt.close(fig)
    
    # Visualize learned barrier function
    fig = trainer.visualize_barrier(
        states=states,
        labels=labels,
        dims=(0, 1)  # Visualize in x-y space
    )
    fig.savefig(f"{output_dir}/barrier_function.png")
    plt.close(fig)
    
    # Save model
    trainer.save(f"{output_dir}/barrier_model.pt")
    
    return model


def train_lyapunov_function() -> TorchLyapunovNetwork:
    """
    Train a neural Lyapunov function for stability verification.
    
    Returns:
        Trained Lyapunov network
    """
    print("Training Lyapunov function...")
    
    # Define stabilizing controller
    def stabilizing_controller(state):
        """Simple stabilizing controller for the mobile robot."""
        # Target is the origin
        x, y, theta, v = state
        
        # Compute distance to origin
        distance = np.sqrt(x**2 + y**2)
        
        # Compute desired heading (towards origin)
        desired_theta = np.arctan2(-y, -x)
        
        # Compute errors
        theta_error = (desired_theta - theta + np.pi) % (2 * np.pi) - np.pi
        
        # Compute control inputs
        a = -0.5 * v - 0.3 * distance  # Acceleration
        omega = 1.0 * theta_error  # Angular velocity
        
        return np.array([a, omega])
    
    # Create data generator
    data_generator = ObstacleDataGenerator(
        state_dim=4,
        input_dim=2,
        dynamics_fn=mobile_robot_dynamics,
        controller_fn=stabilizing_controller,
        state_bounds=np.array([[-5.0, 5.0], [-5.0, 5.0], [-np.pi, np.pi], [0.0, 2.0]])
    )
    
    # Generate states
    states = data_generator.generate_uniform_states(5000)
    
    # Compute control inputs for each state
    control_inputs = np.array([stabilizing_controller(state) for state in states])
    
    # Create Lyapunov network model
    model = TorchLyapunovNetwork(
        state_dim=4,
        hidden_layers=[64, 64, 32],
        activation="tanh"
    )
    
    # Create PyTorch version of the controller
    def stabilizing_controller_torch(state_batch):
        """PyTorch version of the stabilizing controller."""
        # Extract states
        x = state_batch[:, 0]
        y = state_batch[:, 1]
        theta = state_batch[:, 2]
        v = state_batch[:, 3]
        
        # Compute distance to origin
        distance = torch.sqrt(x**2 + y**2)
        
        # Compute desired heading (towards origin)
        desired_theta = torch.atan2(-y, -x)
        
        # Compute errors
        theta_error = (desired_theta - theta + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Compute control inputs
        a = -0.5 * v - 0.3 * distance  # Acceleration
        omega = 1.0 * theta_error  # Angular velocity
        
        return torch.stack([a, omega], dim=1)
    
    # Create Lyapunov trainer
    trainer = LyapunovTrainer(
        model=model,
        dynamics_fn=mobile_robot_dynamics_torch,
        positive_weight=1.0,
        derivative_weight=1.0,
        boundedness_weight=0.1
    )
    
    # Train model
    trainer.train(
        states=states,
        control_inputs=control_inputs,
        batch_size=64,
        epochs=500,
        early_stopping=True,
        patience=20,
        verbose=True
    )
    
    # Visualize training history
    output_dir = "outputs/mobile_robot"
    os.makedirs(output_dir, exist_ok=True)
    
    fig = trainer.visualize_training()
    fig.savefig(f"{output_dir}/lyapunov_training.png")
    plt.close(fig)
    
    # Visualize learned Lyapunov function
    fig = trainer.visualize_lyapunov(
        states=states,
        dims=(0, 1),  # Visualize in x-y space
        show_stability=True
    )
    fig.savefig(f"{output_dir}/lyapunov_function.png")
    plt.close(fig)
    
    # Visualize vector field
    fig = trainer.visualize_vector_field(
        states=states,
        dims=(0, 1),
        controller=stabilizing_controller
    )
    fig.savefig(f"{output_dir}/vector_field.png")
    plt.close(fig)
    
    # Save model
    trainer.save(f"{output_dir}/lyapunov_model.pt")
    
    return model


def export_models_to_elfin():
    """Export trained models to ELFIN format."""
    print("Exporting models to ELFIN format...")
    
    output_dir = "outputs/mobile_robot"
    
    # Train models if they don't exist
    if not os.path.exists(f"{output_dir}/barrier_model.pt"):
        barrier_model = train_barrier_function()
    else:
        # Load existing model
        barrier_model = TorchBarrierNetwork(state_dim=4, hidden_layers=[64, 64, 32])
        checkpoint = torch.load(f"{output_dir}/barrier_model.pt")
        barrier_model.load_state_dict(checkpoint["model_state_dict"])
    
    if not os.path.exists(f"{output_dir}/lyapunov_model.pt"):
        lyapunov_model = train_lyapunov_function()
    else:
        # Load existing model
        lyapunov_model = TorchLyapunovNetwork(state_dim=4, hidden_layers=[64, 64, 32])
        checkpoint = torch.load(f"{output_dir}/lyapunov_model.pt")
        lyapunov_model.load_state_dict(checkpoint["model_state_dict"])
    
    # Export barrier function
    barrier_elfin = export_to_elfin(
        model=barrier_model,
        model_type="barrier",
        system_name="MobileRobot",
        state_dim=4,
        input_dim=2,
        state_names=["x", "y", "theta", "v"],
        input_names=["a", "omega"],
        filepath=f"{output_dir}/neural_barrier.elfin",
        approximation_method="explicit"
    )
    
    # Export Lyapunov function
    lyapunov_elfin = export_to_elfin(
        model=lyapunov_model,
        model_type="lyapunov",
        system_name="MobileRobot",
        state_dim=4,
        input_dim=2,
        state_names=["x", "y", "theta", "v"],
        input_names=["a", "omega"],
        filepath=f"{output_dir}/neural_lyapunov.elfin",
        approximation_method="polynomial",
        approximation_params={"degree": 4}
    )
    
    print(f"Models exported to {output_dir}")


def benchmark_models():
    """Benchmark the exported models."""
    print("Benchmarking neural barrier and Lyapunov functions...")
    
    output_dir = "outputs/mobile_robot"
    
    # Load trained models
    barrier_model = TorchBarrierNetwork(state_dim=4, hidden_layers=[64, 64, 32])
    checkpoint = torch.load(f"{output_dir}/barrier_model.pt")
    barrier_model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create benchmark system
    barrier_system = NeuralBarrierSystem(
        name="MobileRobotBarrier",
        barrier_network=barrier_model,
        dynamics_fn=mobile_robot_dynamics,
        state_dim=4,
        input_dim=2
    )
    
    # Run benchmark
    benchmark_results = BenchmarkIntegration.benchmark_barrier_network(
        barrier_system=barrier_system,
        metrics=["validation_success_rate", "computation_time"],
        output_dir=output_dir
    )
    
    print("Benchmark results:")
    for metric, value in benchmark_results.items():
        print(f"  {metric}: {value}")


def import_from_elfin_example():
    """Example of importing ELFIN models back to neural networks."""
    print("Importing ELFIN models back to neural networks...")
    
    output_dir = "outputs/mobile_robot"
    
    # Import barrier function
    imported_barrier = import_barrier_function(
        filepath=f"{output_dir}/neural_barrier.elfin",
        state_dim=4,
        input_dim=2,
        state_names=["x", "y", "theta", "v"],
        hidden_layers=[64, 64, 32],
        activation="tanh"
    )
    
    # Generate some test data
    test_states = np.random.uniform(
        [-5.0, -5.0, -np.pi, 0.0],
        [5.0, 5.0, np.pi, 2.0],
        size=(100, 4)
    )
    
    # Evaluate original and imported models
    original_barrier = TorchBarrierNetwork(state_dim=4, hidden_layers=[64, 64, 32])
    checkpoint = torch.load(f"{output_dir}/barrier_model.pt")
    original_barrier.load_state_dict(checkpoint["model_state_dict"])
    
    # Convert test states to PyTorch tensor
    test_tensor = torch.tensor(test_states, dtype=torch.float32)
    
    with torch.no_grad():
        original_values = original_barrier(test_tensor).cpu().numpy()
        imported_values = imported_barrier(test_tensor).cpu().numpy()
    
    # Compute error
    mean_error = np.mean(np.abs(original_values - imported_values))
    max_error = np.max(np.abs(original_values - imported_values))
    
    print(f"Mean approximation error: {mean_error:.6f}")
    print(f"Max approximation error: {max_error:.6f}")


def main():
    """Main function to run the example."""
    print("Mobile Robot Example with Neural Barrier and Lyapunov Functions")
    print("=" * 80)
    
    # Ensure output directory exists
    output_dir = "outputs/mobile_robot"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the different components
    train_barrier_function()
    train_lyapunov_function()
    export_models_to_elfin()
    benchmark_models()
    import_from_elfin_example()
    
    print("=" * 80)
    print("Example completed successfully!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
