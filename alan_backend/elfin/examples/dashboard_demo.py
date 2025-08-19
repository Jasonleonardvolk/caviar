"""
ELFIN Stability Dashboard Demo

This script demonstrates the visualization dashboard for the ELFIN stability framework.
It creates a simple simulation with pendulum dynamics and shows real-time monitoring
of Lyapunov function values and stability properties.
"""

import os
import sys
import time
import threading
import logging
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from alan_backend.elfin.stability.lyapunov import PolynomialLyapunov
    from alan_backend.elfin.stability.jit_guard import StabilityGuard
    from alan_backend.elfin.visualization.dashboard import DashboardServer, create_dashboard_files
except ImportError:
    print("Error: Required modules not found.")
    print("Please ensure the ELFIN stability framework is installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def pendulum_dynamics(x, dt=0.01):
    """
    Simple pendulum dynamics.
    
    Args:
        x: State [theta, theta_dot]
        dt: Time step
        
    Returns:
        New state
    """
    # Parameters
    m = 1.0     # mass
    l = 1.0     # length
    g = 9.81    # gravity
    b = 0.1     # damping
    
    # Extract state
    theta, theta_dot = x
    
    # Compute acceleration
    theta_ddot = -(g / l) * np.sin(theta) - (b / (m * l**2)) * theta_dot
    
    # Euler integration
    theta_new = theta + dt * theta_dot
    theta_dot_new = theta_dot + dt * theta_ddot
    
    return np.array([theta_new, theta_dot_new])


def cart_pole_dynamics(x, dt=0.01, u=0.0):
    """
    Cart-pole dynamics.
    
    Args:
        x: State [cart_pos, theta, cart_vel, theta_dot]
        dt: Time step
        u: Control input
        
    Returns:
        New state
    """
    # Parameters
    g = 9.81        # gravity
    mc = 1.0        # cart mass
    mp = 0.1        # pole mass
    l = 0.5         # pole length
    b = 0.1         # friction
    
    # Extract state
    cart_pos, theta, cart_vel, theta_dot = x
    
    # Compute accelerations
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Intermediate calculations
    temp = (u + mp * l * theta_dot**2 * sin_theta - b * cart_vel) / (mc + mp)
    temp2 = (g * sin_theta - cos_theta * temp) / (l * (4.0/3.0 - mp * cos_theta**2 / (mc + mp)))
    cart_acc = temp + mp * l * temp2 * cos_theta / (mc + mp)
    theta_acc = temp2
    
    # Euler integration
    cart_pos_new = cart_pos + dt * cart_vel
    theta_new = theta + dt * theta_dot
    cart_vel_new = cart_vel + dt * cart_acc
    theta_dot_new = theta_dot + dt * theta_acc
    
    return np.array([cart_pos_new, theta_new, cart_vel_new, theta_dot_new])


def stability_violation_callback(x_prev, x, guard):
    """
    Callback function for stability violations.
    
    Args:
        x_prev: Previous state
        x: Current state
        guard: Stability guard
    """
    logger.warning(f"Stability violation detected!")
    logger.warning(f"  Previous state: {x_prev}")
    logger.warning(f"  Current state: {x}")
    logger.warning(f"  Total violations: {guard.violations}")


def run_simulation(dashboard, duration=60.0):
    """
    Run simulation for the specified duration.
    
    Args:
        dashboard: Dashboard server
        duration: Simulation duration (seconds)
    """
    logger.info("Starting simulation...")
    
    # Initial states
    pendulum_state = np.array([0.2, 0.0])  # Initial pendulum state [theta, theta_dot]
    cart_pole_state = np.array([0.0, 0.1, 0.0, 0.0])  # Initial cart-pole state
    
    # Create Lyapunov functions
    V_pendulum = PolynomialLyapunov(
        name="V_pendulum",
        Q=np.array([[1.0, 0.0], [0.0, 0.5]]),
        domain_ids=["pendulum"]
    )
    
    V_cart_pole = PolynomialLyapunov(
        name="V_cart_pole",
        Q=np.diag([1.0, 2.0, 0.5, 0.5]),
        domain_ids=["cart_pole"]
    )
    
    # Create stability guards
    pendulum_guard = StabilityGuard(
        lyap=V_pendulum,
        threshold=0.0,
        callback=stability_violation_callback
    )
    
    cart_pole_guard = StabilityGuard(
        lyap=V_cart_pole,
        threshold=0.0,
        callback=stability_violation_callback
    )
    
    # Register with dashboard
    dashboard.register_lyapunov_function(V_pendulum)
    dashboard.register_lyapunov_function(V_cart_pole)
    
    dashboard.register_stability_guard(pendulum_guard, "pendulum")
    dashboard.register_stability_guard(cart_pole_guard, "cart_pole")
    
    # Add initial state to dashboard
    dashboard.update_system_state(pendulum_state, "pendulum")
    dashboard.update_system_state(cart_pole_state, "cart_pole")
    
    # Simulation parameters
    dt = 0.01
    steps = int(duration / dt)
    perturbation_time = 10.0  # Time for perturbation
    perturbation_steps = int(perturbation_time / dt)
    
    # Log initial values
    logger.info(f"Initial pendulum state: {pendulum_state}")
    logger.info(f"Initial pendulum Lyapunov value: {V_pendulum.evaluate(pendulum_state):.4f}")
    logger.info(f"Initial cart-pole state: {cart_pole_state}")
    logger.info(f"Initial cart-pole Lyapunov value: {V_cart_pole.evaluate(cart_pole_state):.4f}")
    
    # Simulation loop
    for i in range(steps):
        # Store previous state
        pendulum_prev = pendulum_state.copy()
        cart_pole_prev = cart_pole_state.copy()
        
        # Apply perturbation at specific time
        if i == perturbation_steps:
            logger.info("Applying perturbation to pendulum...")
            pendulum_state[1] += 1.0  # Add velocity
            
            logger.info("Applying perturbation to cart-pole...")
            cart_pole_state[2] += 0.5  # Add cart velocity
        
        # Update dynamics
        pendulum_state = pendulum_dynamics(pendulum_state, dt)
        cart_pole_state = cart_pole_dynamics(cart_pole_state, dt)
        
        # Check stability
        pendulum_stable = pendulum_guard.step(pendulum_prev, pendulum_state)
        cart_pole_stable = cart_pole_guard.step(cart_pole_prev, cart_pole_state)
        
        # Update dashboard
        dashboard.update_system_state(pendulum_state, "pendulum")
        dashboard.update_system_state(cart_pole_state, "cart_pole")
        
        # Record stability violations
        if not pendulum_stable:
            dashboard.record_stability_violation("pendulum", pendulum_prev, pendulum_state)
        
        if not cart_pole_stable:
            dashboard.record_stability_violation("cart_pole", cart_pole_prev, cart_pole_state)
        
        # Log periodic status
        if i % 100 == 0:
            logger.info(f"Step {i}/{steps}")
            logger.info(f"  Pendulum state: {pendulum_state}")
            logger.info(f"  Pendulum Lyapunov value: {V_pendulum.evaluate(pendulum_state):.4f}")
            logger.info(f"  Cart-pole state: {cart_pole_state}")
            logger.info(f"  Cart-pole Lyapunov value: {V_cart_pole.evaluate(cart_pole_state):.4f}")
        
        # Short sleep to avoid CPU overload
        time.sleep(0.001)
    
    logger.info("Simulation complete.")


def main():
    """Main function."""
    print("ELFIN Stability Dashboard Demo")
    print("=" * 80)
    
    # Create dashboard files
    create_dashboard_files()
    
    # Create dashboard server
    dashboard = DashboardServer(host="localhost", port=5000)
    
    # Start dashboard server
    dashboard.start(debug=False)
    print(f"Dashboard server started at http://localhost:5000")
    print("Please open this URL in your browser to view the dashboard.")
    print()
    
    # Wait for server to start
    time.sleep(1.0)
    
    try:
        # Run simulation
        run_simulation(dashboard, duration=60.0)
        
        print()
        print("Simulation complete. The dashboard will remain active.")
        print("Press Ctrl+C to exit.")
        
        # Keep main thread alive
        while True:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Stop dashboard server
        dashboard.stop()
        print("Dashboard server stopped.")


if __name__ == "__main__":
    main()
