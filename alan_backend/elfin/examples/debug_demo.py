"""
ELFIN Debug Demo

This example demonstrates the use of ELFIN's debug features including
Lyapunov stability monitoring and barrier function verification.
"""

import time
import math
import random
import argparse
import numpy as np
from threading import Thread

from alan_backend.elfin.debug.lyapunov_monitor import lyapunov_monitor
from alan_backend.elfin.debug.unit_checker import unit_checker

# Simple pendulum example
class Pendulum:
    """A simple pendulum system for demonstration."""
    
    def __init__(self, mass=1.0, length=1.0, damping=0.1, g=9.81):
        """
        Initialize the pendulum.
        
        Args:
            mass: Pendulum mass (kg)
            length: Pendulum length (m)
            damping: Damping coefficient
            g: Gravitational acceleration (m/s^2)
        """
        self.mass = mass
        self.length = length
        self.damping = damping
        self.g = g
        
        # State: [theta, omega]
        self.state = np.array([0.0, 0.0])
        
        # Control input
        self.torque = 0.0
        
        # Register with Lyapunov monitor
        self._register_with_monitor()
    
    def _register_with_monitor(self):
        """Register with the Lyapunov monitor."""
        # Register state variables with units
        lyapunov_monitor.register_state_var("theta", "rad")
        lyapunov_monitor.register_state_var("omega", "rad/s")
        lyapunov_monitor.register_state_var("energy", "J")
        
        # Set Lyapunov function (energy-based)
        lyapunov_monitor.set_lyapunov_function(
            V_func=self.energy,
            Vdot_func=self.energy_derivative
        )
        
        # Register barrier function for angle limits
        lyapunov_monitor.register_barrier_function(
            name="angle_limit",
            func=self.angle_barrier,
            threshold=0.0,
            description="Pendulum angle limit"
        )
        
        # Add breakpoints
        lyapunov_monitor.add_breakpoint(
            condition="V > 10.0",
            expression="V > 10.0"
        )
        lyapunov_monitor.add_breakpoint(
            condition="omega > 5.0",
            expression="abs(omega) > 5.0"
        )
    
    def energy(self, **kwargs):
        """
        Calculate the total energy of the pendulum (Lyapunov function).
        
        Args:
            **kwargs: State variables (ignored, using internal state)
            
        Returns:
            The energy (J)
        """
        theta, omega = self.state
        
        # Kinetic energy
        T = 0.5 * self.mass * self.length**2 * omega**2
        
        # Potential energy (zero at the bottom position)
        U = self.mass * self.g * self.length * (1 - math.cos(theta))
        
        return T + U
    
    def energy_derivative(self, **kwargs):
        """
        Calculate the time derivative of energy (Lyapunov derivative).
        
        Args:
            **kwargs: State variables (ignored, using internal state)
            
        Returns:
            The energy derivative (J/s)
        """
        _, omega = self.state
        
        # For a damped pendulum: dE/dt = -b*omega^2 + tau*omega
        return -self.damping * omega**2 + self.torque * omega
    
    def angle_barrier(self, **kwargs):
        """
        Barrier function for angle limits.
        
        Args:
            **kwargs: State variables (ignored, using internal state)
            
        Returns:
            Barrier value (positive when safe)
        """
        theta, _ = self.state
        theta_max = math.pi  # Maximum allowed angle
        
        # Barrier: B(x) = theta_max^2 - theta^2
        return theta_max**2 - theta**2
    
    def update(self, dt):
        """
        Update the pendulum state.
        
        Args:
            dt: Time step (s)
        """
        theta, omega = self.state
        
        # Pendulum dynamics
        # d(theta)/dt = omega
        # d(omega)/dt = -(g/L)*sin(theta) - (b/mL^2)*omega + tau/(mL^2)
        
        # Euler integration
        theta_dot = omega
        omega_dot = (
            -(self.g / self.length) * math.sin(theta) 
            - (self.damping / (self.mass * self.length**2)) * omega
            + self.torque / (self.mass * self.length**2)
        )
        
        theta += theta_dot * dt
        omega += omega_dot * dt
        
        # Update state
        self.state = np.array([theta, omega])
        
        # Update Lyapunov monitor with current state
        lyapunov_monitor.update(
            theta=theta,
            omega=omega,
            energy=self.energy()
        )
    
    def set_torque(self, torque):
        """Set the control torque."""
        self.torque = torque
    
    def reset(self, theta0=None, omega0=None):
        """Reset the pendulum state."""
        if theta0 is None:
            theta0 = random.uniform(-math.pi/2, math.pi/2)
        if omega0 is None:
            omega0 = random.uniform(-1.0, 1.0)
        
        self.state = np.array([theta0, omega0])
        self.torque = 0.0


def demo_controlled_pendulum():
    """Run a demo of a controlled pendulum with debug monitoring."""
    # Create pendulum
    pendulum = Pendulum(mass=1.0, length=1.0, damping=0.2)
    
    # Start Lyapunov monitor server
    lyapunov_monitor.start()
    
    # Initial state
    pendulum.reset(theta0=math.pi/4, omega0=0.0)
    
    # Run simulation
    dt = 0.01
    t = 0.0
    t_end = 60.0
    
    print(f"Running pendulum simulation for {t_end} seconds...")
    print("Connect to ws://localhost:8642/state to monitor")
    
    # Destabilizing control after 20 seconds
    def destabilize_controller():
        time.sleep(20.0)
        print("Applying destabilizing control...")
        pendulum.set_torque(2.0)
        
        # Reset to stabilizing control after 10 more seconds
        time.sleep(10.0)
        print("Applying stabilizing control...")
        pendulum.set_torque(-1.0)
    
    # Start controller in a separate thread
    controller_thread = Thread(target=destabilize_controller)
    controller_thread.daemon = True
    controller_thread.start()
    
    try:
        while t < t_end:
            # Update pendulum
            pendulum.update(dt)
            
            # Sleep to simulate real-time
            time.sleep(dt)
            
            # Update time
            t += dt
            
            # Print status occasionally
            if int(t * 100) % 100 == 0:
                theta, omega = pendulum.state
                energy = pendulum.energy()
                print(f"t={t:.1f} s, θ={theta:.2f} rad, ω={omega:.2f} rad/s, E={energy:.2f} J")
    
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    
    finally:
        # Stop Lyapunov monitor
        lyapunov_monitor.stop()
        print("Simulation finished.")


def demo_unit_checking():
    """Run a demo of unit checking for ELFIN code."""
    # Example ELFIN code with unit issues
    elfin_code = """
    // Example ELFIN code with unit inconsistencies
    
    concept Robot {
        // Properties with units
        property position: m = Vector3D(0.0, 0.0, 0.0);
        property velocity: m/s = Vector3D(0.0, 0.0, 0.0);
        property acceleration: m/s^2 = Vector3D(0.0, 0.0, 0.0);
        property force: N = Vector3D(0.0, 0.0, 0.0);
        property mass: kg = 10.0;
        property time: s = 0.0;
        property angle: rad = 0.0;
        property angular_velocity: rad/s = 0.0;
        
        // Function with unit inconsistency
        function update(dt: s) {
            // Correct: position = position + velocity * dt
            position = position + velocity * dt;
            
            // Incorrect: velocity = position (unit mismatch)
            velocity = position;
            
            // Incorrect: acceleration = force (missing mass division)
            acceleration = force;
            
            // Correct: F = m * a
            force = mass * acceleration;
            
            // Correct: v = v0 + a * t
            velocity = velocity + acceleration * dt;
            
            // Incorrect: angle = time (unit mismatch)
            angle = time;
        }
    }
    """
    
    # Analyze the code
    diagnostics = unit_checker.analyze_code(elfin_code)
    
    # Print results
    print("\nUnit Checking Results:")
    print("======================")
    
    if not diagnostics:
        print("No unit inconsistencies found.")
    
    for diag in diagnostics:
        print(f"Line {diag.line}, Column {diag.column}: {diag.message}")
        
        if diag.fixes:
            print("  Suggested fixes:")
            for fix in diag.fixes:
                print(f"    - {fix.title}: '{fix.new_text}'")
        
        print()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ELFIN Debug Demo")
    parser.add_argument("--pendulum", action="store_true", help="Run pendulum demo")
    parser.add_argument("--unit-check", action="store_true", help="Run unit checking demo")
    
    args = parser.parse_args()
    
    if args.pendulum:
        demo_controlled_pendulum()
    
    if args.unit_check:
        demo_unit_checking()
    
    # If no arguments provided, run both demos
    if not (args.pendulum or args.unit_check):
        print("Running unit checking demo...\n")
        demo_unit_checking()
        
        print("\nRunning pendulum demo...\n")
        demo_controlled_pendulum()
