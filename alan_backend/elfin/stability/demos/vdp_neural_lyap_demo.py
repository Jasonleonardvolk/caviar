"""
Van der Pol oscillator neural Lyapunov function demo.

This module demonstrates the process of learning and verifying a neural
Lyapunov function for the Van der Pol oscillator using the counterexample-guided
training approach.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from ..samplers.trajectory_sampler import TrajectorySampler
from ..training.neural_lyapunov_trainer import LyapunovNet, NeuralLyapunovTrainer
from ..verify.milp_verifier import MILPVerifier, GUROBI_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def vdp_dynamics(x: np.ndarray, mu: float = 1.0) -> np.ndarray:
    """
    Van der Pol oscillator dynamics.
    
    The Van der Pol oscillator is a non-conservative oscillator with nonlinear damping.
    The system has a stable limit cycle and an unstable equilibrium at the origin.
    
    Args:
        x: States with shape (batch_size, 2)
        mu: Damping parameter
        
    Returns:
        State derivatives with shape (batch_size, 2)
    """
    # Extract states
    x1 = x[:, 0]
    x2 = x[:, 1]
    
    # Compute derivatives
    dx1 = x2
    dx2 = mu * (1 - x1**2) * x2 - x1
    
    # Stack derivatives
    return np.stack([dx1, dx2], axis=1)


def certify_vdp(
    net: LyapunovNet, 
    sampler: TrajectorySampler, 
    domain: tuple,
    max_rounds: int = 5, 
    fine_tune_steps: int = 200,
    time_limit: float = 60.0
) -> bool:
    """
    Attempt to certify a neural Lyapunov function for the Van der Pol oscillator.
    
    This function implements the counterexample-guided training approach:
    1. Verify the current neural Lyapunov function.
    2. If verification succeeds, the function is certified.
    3. If verification fails, add the counterexample to the training set.
    4. Fine-tune the neural Lyapunov function and try again.
    
    Args:
        net: Neural Lyapunov function
        sampler: Trajectory sampler for generating training data
        domain: Tuple of (lower_bounds, upper_bounds) for verification domain
        max_rounds: Maximum number of certification rounds
        fine_tune_steps: Number of training steps per round
        time_limit: Time limit for MILP solver in seconds
        
    Returns:
        Whether certification succeeded
    """
    if not GUROBI_AVAILABLE:
        logger.warning("Gurobi not available, skipping certification")
        return False
        
    # Create trainer
    trainer = NeuralLyapunovTrainer(net, sampler, learning_rate=1e-4)
    
    # Create verifier
    verifier = MILPVerifier(net, domain, time_limit=time_limit)
    
    # Start certification loop
    for k in range(max_rounds):
        logger.info(f"=== Certification Round {k+1}/{max_rounds} ===")
        
        # Evaluate the current Lyapunov function
        metrics = trainer.evaluate(n_samples=1000)
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Try to find a counterexample to positive definiteness
        ce = verifier.find_pd_counterexample()
        
        if ce is None:
            logger.info("=== Certified! ===")
            return True
            
        # Found a counterexample
        logger.info(f"Counterexample found: {ce}")
        
        # Add counterexample to training set
        sampler.add_counterexamples([ce])
        
        # Fine-tune the Lyapunov function
        logger.info(f"Fine-tuning for {fine_tune_steps} steps...")
        trainer.fit(steps=fine_tune_steps, log_every=50)
    
    logger.warning(f"Failed to certify after {max_rounds} rounds")
    return False


def plot_vector_field(
    dynamics_fn,
    domain,
    lyapunov_fn=None,
    grid_size=20,
    trajectories=None,
    ax=None,
    title=None
):
    """
    Plot vector field of a dynamical system with optional Lyapunov function.
    
    Args:
        dynamics_fn: Function mapping states to derivatives
        domain: Tuple of (lower_bounds, upper_bounds)
        lyapunov_fn: Optional function mapping states to Lyapunov values
        grid_size: Number of grid points in each dimension
        trajectories: Optional list of trajectory states to plot
        ax: Matplotlib axis to plot on
        title: Plot title
    """
    # Create grid
    low, high = domain
    x1 = np.linspace(low[0], high[0], grid_size)
    x2 = np.linspace(low[1], high[1], grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Compute vector field
    X = np.stack([X1.flatten(), X2.flatten()], axis=1)
    dX = dynamics_fn(X)
    dX1 = dX[:, 0].reshape(X1.shape)
    dX2 = dX[:, 1].reshape(X2.shape)
    
    # Create plot if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot vector field
    ax.streamplot(X1, X2, dX1, dX2, color='gray', density=1.0, linewidth=0.5)
    
    # Plot Lyapunov function if provided
    if lyapunov_fn is not None:
        try:
            # Evaluate Lyapunov function
            V = lyapunov_fn(torch.tensor(X, dtype=torch.float32)).detach().cpu().numpy()
            V = V.reshape(X1.shape)
            
            # Plot level sets
            levels = np.linspace(0, np.percentile(V, 95), 10)
            contour = ax.contour(X1, X2, V, levels=levels, cmap='viridis')
            ax.clabel(contour, inline=True, fontsize=8)
            
            # Colorize based on Lyapunov function
            im = ax.pcolormesh(X1, X2, V, shading='auto', alpha=0.3, cmap='viridis')
            plt.colorbar(im, ax=ax, label='V(x)')
        except Exception as e:
            logger.warning(f"Error plotting Lyapunov function: {e}")
    
    # Plot trajectories if provided
    if trajectories is not None:
        for traj in trajectories:
            ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=1.5)
            ax.plot(traj[0, 0], traj[0, 1], 'ro')  # Start point
    
    # Set axis labels and title
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    if title is not None:
        ax.set_title(title)
    
    # Set axis limits
    ax.set_xlim([low[0], high[0]])
    ax.set_ylim([low[1], high[1]])
    
    return ax


def run_vdp_demo(
    batch_size: int = 1024,
    hidden_dims: tuple = (64, 64),
    initial_steps: int = 3000,
    certification_rounds: int = 5,
    fine_tune_steps: int = 200,
    save_model: bool = True,
    save_plot: bool = True,
    output_dir: str = None,
    mu: float = 1.0,
    domain_size: float = 3.0,
    gpu: bool = False
):
    """
    Run the Van der Pol oscillator neural Lyapunov demo.
    
    Args:
        batch_size: Batch size for training
        hidden_dims: Hidden layer dimensions for LyapunovNet
        initial_steps: Number of initial training steps
        certification_rounds: Maximum number of certification rounds
        fine_tune_steps: Number of training steps per certification round
        save_model: Whether to save the trained model
        save_plot: Whether to save the plot of vector field and Lyapunov function
        output_dir: Directory to save outputs (defaults to ./outputs)
        mu: Damping parameter for Van der Pol oscillator
        domain_size: Size of domain for training and verification
        gpu: Whether to use GPU for training
    """
    # Create output directory if needed
    if save_model or save_plot:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set domain
    domain = (np.array([-domain_size, -domain_size]), np.array([domain_size, domain_size]))
    
    # Create Van der Pol dynamics function with the specified mu
    vdp_fn = lambda x: vdp_dynamics(x, mu=mu)
    
    # Create sampler
    logger.info("Creating trajectory sampler...")
    sampler = TrajectorySampler(
        dynamics_fn=vdp_fn,
        dim=2,
        domain=domain,
        batch_size=batch_size
    )
    
    # Create LyapunovNet
    logger.info("Creating LyapunovNet...")
    net = LyapunovNet(
        dim=2,
        hidden_dims=hidden_dims,
        alpha=1e-3,
        activation=torch.nn.Tanh()
    )
    net = net.to(device)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = NeuralLyapunovTrainer(
        model=net,
        sampler=sampler,
        learning_rate=1e-3,
        gamma=0.1,  # Require strict decrease with margin
        device=device
    )
    
    # Train the model
    logger.info(f"Training for {initial_steps} steps...")
    history = trainer.fit(
        steps=initial_steps,
        log_every=100,
        save_path=os.path.join(output_dir, 'vdp_lyapunov.pt') if save_model else None
    )
    
    # Evaluate the trained model
    logger.info("Evaluating trained model...")
    metrics = trainer.evaluate(n_samples=1000)
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")
    
    # Try to certify the Lyapunov function
    logger.info("Attempting to certify Lyapunov function...")
    certified = certify_vdp(
        net=net,
        sampler=sampler,
        domain=domain,
        max_rounds=certification_rounds,
        fine_tune_steps=fine_tune_steps
    )
    
    if certified:
        logger.info("Successfully certified Lyapunov function!")
    else:
        logger.warning("Failed to certify Lyapunov function.")
    
    # Simulate some trajectories
    logger.info("Simulating trajectories...")
    trajectories = []
    initial_states = [
        np.array([1.0, 1.0]),
        np.array([-1.0, 1.0]),
        np.array([1.0, -1.0]),
        np.array([-1.0, -1.0]),
        np.array([2.0, 0.0]),
        np.array([0.0, 2.0])
    ]
    
    for x0 in initial_states:
        states, _ = sampler.simulate_trajectory(x0, steps=100, dt=0.1)
        trajectories.append(states)
    
    # Create a plotting function for the Lyapunov function
    def lyap_fn(x):
        with torch.no_grad():
            return net(x.to(device)).cpu()
    
    # Plot results
    logger.info("Creating plots...")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_vector_field(
        dynamics_fn=vdp_fn,
        domain=domain,
        lyapunov_fn=lyap_fn,
        grid_size=30,
        trajectories=trajectories,
        ax=ax,
        title=f"Van der Pol Oscillator (μ={mu}) with Neural Lyapunov Function"
    )
    
    # Save plot if requested
    if save_plot:
        plot_path = os.path.join(output_dir, 'vdp_lyapunov.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {plot_path}")
    
    plt.tight_layout()
    plt.show()
    
    # Return the trained model and metrics
    return {
        'net': net,
        'metrics': metrics,
        'certified': certified,
        'history': history
    }


def main():
    """Run the demo from command line."""
    parser = argparse.ArgumentParser(description='Van der Pol Neural Lyapunov Demo')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 64], help='Hidden layer dimensions')
    parser.add_argument('--initial-steps', type=int, default=3000, help='Number of initial training steps')
    parser.add_argument('--cert-rounds', type=int, default=5, help='Maximum number of certification rounds')
    parser.add_argument('--fine-tune-steps', type=int, default=200, help='Steps per certification round')
    parser.add_argument('--no-save-model', action='store_true', help='Do not save the trained model')
    parser.add_argument('--no-save-plot', action='store_true', help='Do not save the plot')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save outputs')
    parser.add_argument('--mu', type=float, default=1.0, help='Damping parameter for VDP oscillator')
    parser.add_argument('--domain-size', type=float, default=3.0, help='Size of domain for training/verification')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    args = parser.parse_args()
    
    # Run demo
    run_vdp_demo(
        batch_size=args.batch_size,
        hidden_dims=tuple(args.hidden_dims),
        initial_steps=args.initial_steps,
        certification_rounds=args.cert_rounds,
        fine_tune_steps=args.fine_tune_steps,
        save_model=not args.no_save_model,
        save_plot=not args.no_save_plot,
        output_dir=args.output_dir,
        mu=args.mu,
        domain_size=args.domain_size,
        gpu=args.gpu
    )


if __name__ == "__main__":
    main()
