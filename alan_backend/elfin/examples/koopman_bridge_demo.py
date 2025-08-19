#!/usr/bin/env python3
"""
Koopman Bridge Demo for ELFIN.

This script demonstrates how to use the Koopman Bridge Agent to learn
Lyapunov functions from trajectory data and verify stability properties.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time
import argparse
import logging
from typing import List, Dict, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("elfin.koopman_demo")

# Import ELFIN components
from alan_backend.elfin.koopman import (
    create_dictionary,
    edmd_fit,
    create_koopman_lyapunov,
    KoopmanLyapunov
)
from alan_backend.elfin.koopman.koopman_bridge_agent import (
    KoopmanBridgeAgent,
    create_pendulum_agent,
    create_vdp_agent
)


def plot_lyapunov_function(
    lyap_fn: KoopmanLyapunov,
    domain: Tuple[np.ndarray, np.ndarray],
    title: str = "Lyapunov Function",
    resolution: int = 50,
    show_modes: bool = False,
    save_path: Optional[str] = None
):
    """
    Plot a Lyapunov function.
    
    Args:
        lyap_fn: Lyapunov function to plot
        domain: Domain for plotting as (lower, upper)
        title: Plot title
        resolution: Resolution of the plot
        show_modes: Whether to also plot individual Koopman modes
        save_path: Path to save the plot (if None, display)
    """
    lower, upper = domain
    x_min, y_min = lower
    x_max, y_max = upper
    
    # Create grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Reshape for evaluation
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Evaluate Lyapunov function
    V = lyap_fn(grid_points)
    Z = V.reshape(X.shape)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot Lyapunov function
    plt.subplot(221)
    contour = plt.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(contour)
    plt.title(f"{title}\nV(x)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    # Plot 3D surface
    ax = plt.subplot(222, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_title("3D Surface View")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("V(x)")
    
    # If showing modes, plot a few Koopman eigenfunctions
    if show_modes and len(lyap_fn.stable_indices) > 0:
        # Get a few eigenfunctions
        n_modes = min(4, len(lyap_fn.stable_indices))
        
        for i in range(n_modes):
            # Get eigenfunction
            eigenfunction = lyap_fn.get_eigenfunction(i)
            eigenvalue = lyap_fn.get_eigenvalue(i)
            
            # Evaluate eigenfunction
            psi = eigenfunction(grid_points)
            psi_real = np.real(psi).reshape(X.shape)
            
            # Plot real part of eigenfunction
            plt.subplot(2, 3, i + 4)
            plt.contourf(X, Y, psi_real, 50, cmap='coolwarm')
            plt.colorbar()
            plt.title(f"Re(ψ_{i}) (λ={eigenvalue:.3f})")
            plt.xlabel("x1")
            plt.ylabel("x2")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_verifier_results(
    system_name: str,
    lyap_fn: KoopmanLyapunov,
    domain: Tuple[np.ndarray, np.ndarray],
    results: Dict[str, Any],
    save_path: Optional[str] = None
):
    """
    Plot verification results.
    
    Args:
        system_name: Name of the system
        lyap_fn: Verified Lyapunov function
        domain: Domain for verification
        results: Verification results
        save_path: Path to save the plot (if None, display)
    """
    lower, upper = domain
    x_min, y_min = lower
    x_max, y_max = upper
    
    # Create grid
    resolution = 50
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Reshape for evaluation
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Evaluate Lyapunov function
    V = lyap_fn(grid_points)
    Z = V.reshape(X.shape)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot Lyapunov function
    plt.subplot(221)
    contour = plt.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(contour)
    plt.title(f"Koopman Lyapunov Function: {system_name}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    # Plot system phase portrait
    plt.subplot(222)
    
    # Define system dynamics
    if system_name == "pendulum":
        def f(x, alpha=0.1):
            """Pendulum dynamics"""
            theta, omega = x
            return np.array([omega, -np.sin(theta) - alpha*omega])
    elif system_name == "vdp":
        def f(x, mu=1.0):
            """Van der Pol dynamics"""
            return np.array([x[1], mu*(1-x[0]**2)*x[1] - x[0]])
    else:
        # Generic stable system
        def f(x):
            """Generic stable system"""
            return np.array([-x[0], -x[1]])
    
    # Create vector field
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(len(x)):
        for j in range(len(y)):
            dx = f(np.array([X[i, j], Y[i, j]]))
            U[i, j] = dx[0]
            V[i, j] = dx[1]
    
    # Plot vector field
    plt.quiver(X, Y, U, V, scale=25, alpha=0.7)
    
    # Add contour lines for Lyapunov function
    plt.contour(X, Y, Z, 10, colors='k', alpha=0.5)
    
    plt.title(f"Phase Portrait and Level Sets")
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    # Mark verification status
    if results["status"] == "VERIFIED":
        status_str = f"✓ VERIFIED\nSolve time: {results.get('solve_time', 0):.3f}s"
        plt.subplot(223)
        plt.text(0.5, 0.5, status_str, fontsize=20, ha='center', va='center',
                 bbox=dict(facecolor='green', alpha=0.3))
        plt.axis('off')
    else:
        status_str = f"✗ VERIFICATION FAILED\nSolve time: {results.get('solve_time', 0):.3f}s"
        plt.subplot(223)
        plt.text(0.5, 0.5, status_str, fontsize=20, ha='center', va='center',
                 bbox=dict(facecolor='red', alpha=0.3))
        plt.axis('off')
        
        # Plot counterexample if available
        if "counterexample" in results:
            ce = results["counterexample"]
            plt.subplot(224)
            
            # Evaluate V in counterexample neighborhood
            ce_x, ce_y = ce
            
            # Create small grid around counterexample
            margin = 0.2 * np.maximum(np.abs(upper - lower), 0.1)
            x_local = np.linspace(ce_x - margin[0], ce_x + margin[0], resolution)
            y_local = np.linspace(ce_y - margin[1], ce_y + margin[1], resolution)
            X_local, Y_local = np.meshgrid(x_local, y_local)
            
            # Evaluate at grid points
            grid_local = np.column_stack([X_local.ravel(), Y_local.ravel()])
            V_local = lyap_fn(grid_local).reshape(X_local.shape)
            
            # Plot local behavior
            plt.contourf(X_local, Y_local, V_local, 50, cmap='viridis')
            plt.colorbar()
            plt.plot(ce_x, ce_y, 'rx', markersize=10, label='Counterexample')
            
            # Add vector at counterexample
            dx = f(np.array([ce_x, ce_y]))
            plt.arrow(ce_x, ce_y, dx[0]/10, dx[1]/10, width=0.01, head_width=0.05,
                    head_length=0.05, fc='red', ec='red')
            
            plt.title(f"Counterexample at ({ce_x:.3f}, {ce_y:.3f})")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def demo_pendulum(dict_type: str = "rbf", dict_size: int = 100, plot_results: bool = True):
    """Run the pendulum demo."""
    print("\n" + "="*80)
    print(" Koopman Bridge Demo: Pendulum System ".center(80, "="))
    print("="*80)
    
    # Create pendulum agent
    print("\nCreating pendulum agent...")
    
    start_time = time.time()
    agent, lyap_fn = create_pendulum_agent(
        dict_type=dict_type,
        dict_size=dict_size,
        verify=False
    )
    
    # Define verification domain
    domain = (np.array([-np.pi/2, -1.0]), np.array([np.pi/2, 1.0]))
    
    # Verify Lyapunov function
    print("\nVerifying Lyapunov function...")
    result = agent.verify(lyap_fn, "pendulum", domain)
    
    # Print elapsed time
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")
    
    # Print a summary
    print("\nSummary:")
    print(agent.get_summary())
    
    # Print verification result
    print("\nVerification result:")
    print(f"  Status: {result['status']}")
    print(f"  Solve time: {result.get('solve_time', 0):.3f}s")
    
    if result["status"] != "VERIFIED":
        if "counterexample" in result:
            print(f"  Counterexample: {result['counterexample']}")
            
    # Plot results
    if plot_results:
        print("\nPlotting results...")
        plot_verifier_results("pendulum", lyap_fn, domain, result)
    
    print("\n" + "="*80)
    print(" Demo Complete ".center(80, "="))
    print("="*80)
    
    return agent, lyap_fn, result


def demo_vdp(dict_type: str = "rbf", dict_size: int = 100, plot_results: bool = True):
    """Run the Van der Pol oscillator demo."""
    print("\n" + "="*80)
    print(" Koopman Bridge Demo: Van der Pol System ".center(80, "="))
    print("="*80)
    
    # Create VdP agent
    print("\nCreating Van der Pol agent...")
    
    start_time = time.time()
    agent, lyap_fn = create_vdp_agent(
        dict_type=dict_type,
        dict_size=dict_size,
        verify=False
    )
    
    # Define verification domain
    domain = (np.array([-2.0, -2.0]), np.array([2.0, 2.0]))
    
    # Verify Lyapunov function
    print("\nVerifying Lyapunov function...")
    result = agent.verify(lyap_fn, "vdp", domain)
    
    # Print elapsed time
    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")
    
    # Print a summary
    print("\nSummary:")
    print(agent.get_summary())
    
    # Print verification result
    print("\nVerification result:")
    print(f"  Status: {result['status']}")
    print(f"  Solve time: {result.get('solve_time', 0):.3f}s")
    
    if result["status"] != "VERIFIED":
        if "counterexample" in result:
            print(f"  Counterexample: {result['counterexample']}")
            
    # Plot results
    if plot_results:
        print("\nPlotting results...")
        plot_verifier_results("vdp", lyap_fn, domain, result)
    
    print("\n" + "="*80)
    print(" Demo Complete ".center(80, "="))
    print("="*80)
    
    return agent, lyap_fn, result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Koopman Bridge Demo"
    )
    parser.add_argument(
        "--system",
        type=str,
        default="pendulum",
        choices=["pendulum", "vdp", "both"],
        help="System to demonstrate"
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="rbf",
        choices=["rbf", "fourier", "poly"],
        help="Dictionary type"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Dictionary size"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting"
    )
    
    args = parser.parse_args()
    
    # Run demo(s)
    if args.system == "pendulum" or args.system == "both":
        demo_pendulum(
            dict_type=args.dictionary,
            dict_size=args.size,
            plot_results=not args.no_plot
        )
    
    if args.system == "vdp" or args.system == "both":
        demo_vdp(
            dict_type=args.dictionary,
            dict_size=args.size,
            plot_results=not args.no_plot
        )


if __name__ == "__main__":
    main()
