"""
Mobile Robot Benchmark Example

This script demonstrates how to use the ELFIN benchmark suite with the mobile robot model.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Add parent directory to path so we can import the benchmark suite
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from alan_backend.elfin.benchmarks import run_benchmark
from alan_backend.elfin.benchmarks.metrics import ValidationSuccessRate, ComputationTime, Conservativeness
from alan_backend.elfin.benchmarks.benchmark import BenchmarkSystem


class DifferentialDriveRobot(BenchmarkSystem):
    """
    Differential drive robot benchmark system (simplified from mobile_robot_controller.elfin).
    
    State variables:
        x[0]: x - robot x position (m)
        x[1]: y - robot y position (m)
        x[2]: theta - robot orientation (rad)
        x[3]: v - linear velocity (m/s)
        x[4]: omega - angular velocity (rad/s)
    
    Input variables:
        u[0]: v_l - left wheel velocity (m/s)
        u[1]: v_r - right wheel velocity (m/s)
    """
    
    def __init__(self):
        """Initialize the differential drive robot system."""
        params = {
            "wheel_radius": 0.05,    # Wheel radius (m)
            "wheel_base": 0.3,       # Distance between wheels (m)
            "max_speed": 1.0,        # Maximum linear speed (m/s)
            "max_omega": 3.0,        # Maximum angular speed (rad/s)
            "friction": 0.1,         # Friction coefficient
            "x_obs": 3.0,            # Obstacle x-coordinate
            "y_obs": 2.0,            # Obstacle y-coordinate
            "r_obs": 0.5,            # Obstacle radius
            "r_robot": 0.2,          # Robot radius
            "safety_margin": 0.1,    # Additional safety margin
        }
        
        super().__init__(
            name="DifferentialDriveRobot",
            state_dim=5,
            input_dim=2,
            params=params
        )
    
    def dynamics(self, state: np.ndarray, input_vec: np.ndarray) -> np.ndarray:
        """
        Compute system dynamics.
        
        Args:
            state: [x, y, theta, v, omega]
            input_vec: [v_l, v_r]
        
        Returns:
            State derivative [x_dot, y_dot, theta_dot, v_dot, omega_dot]
        """
        x, y, theta, v, omega = state
        v_l, v_r = input_vec
        
        # Extract parameters
        wheel_radius = self.params["wheel_radius"]
        wheel_base = self.params["wheel_base"]
        friction = self.params["friction"]
        
        # Compute derivatives
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = omega
        v_dot = (wheel_radius/2) * (v_r + v_l) - friction * v
        omega_dot = (wheel_radius/wheel_base) * (v_r - v_l) - friction * omega
        
        return np.array([x_dot, y_dot, theta_dot, v_dot, omega_dot])
    
    def barrier_function(self, state: np.ndarray) -> float:
        """
        Compute the barrier function for obstacle avoidance.
        
        Args:
            state: [x, y, theta, v, omega]
        
        Returns:
            Barrier function value
        """
        x, y, _, _, _ = state
        
        # Extract parameters
        x_obs = self.params["x_obs"]
        y_obs = self.params["y_obs"]
        r_obs = self.params["r_obs"]
        r_robot = self.params["r_robot"]
        safety_margin = self.params["safety_margin"]
        
        # Compute distance to obstacle
        r_safe = r_robot + r_obs + safety_margin
        dist_squared = (x - x_obs)**2 + (y - y_obs)**2
        
        # Barrier function (positive in safe region, negative in unsafe)
        return dist_squared - r_safe**2
    
    def is_safe(self, state: np.ndarray) -> bool:
        """
        Check if a state is safe (outside obstacle).
        
        Args:
            state: [x, y, theta, v, omega]
        
        Returns:
            True if safe, False if unsafe
        """
        return self.barrier_function(state) > 0
    
    def get_state_bounds(self) -> np.ndarray:
        """Get state space bounds."""
        return np.array([
            [-5.0, 5.0],    # x range
            [-5.0, 5.0],    # y range
            [-np.pi, np.pi], # theta range
            [-1.5, 1.5],    # v range
            [-4.5, 4.5]     # omega range
        ])
    
    def get_input_bounds(self) -> np.ndarray:
        """Get input space bounds."""
        max_wheel_speed = 2.0
        return np.array([
            [-max_wheel_speed, max_wheel_speed],  # left wheel velocity
            [-max_wheel_speed, max_wheel_speed]   # right wheel velocity
        ])


def simulate_trajectory(system, initial_state, controller, time_horizon=10.0, dt=0.1):
    """
    Simulate a trajectory for the differential drive robot.
    
    Args:
        system: The benchmark system
        initial_state: Initial state vector
        controller: Controller function mapping state to inputs
        time_horizon: Simulation time horizon
        dt: Time step
        
    Returns:
        Trajectory as a list of states and a list of inputs
    """
    t = 0.0
    state = initial_state.copy()
    states = [state.copy()]
    inputs = []
    
    while t < time_horizon:
        # Compute control input
        input_vec = controller(state)
        inputs.append(input_vec.copy())
        
        # Compute state derivative
        deriv = system.dynamics(state, input_vec)
        
        # Euler integration
        state = state + dt * deriv
        states.append(state.copy())
        
        t += dt
    
    return np.array(states), np.array(inputs)


def go_to_goal_controller(state):
    """Simple go-to-goal controller."""
    x, y, theta, v, omega = state
    
    # Goal position
    x_goal, y_goal = 4.0, 4.0
    
    # PD controller parameters
    k_p = 0.5
    k_d = 2.0
    
    # Compute heading to goal
    angle_to_goal = np.arctan2(y_goal - y, x_goal - x)
    heading_error = angle_to_goal - theta
    
    # Normalize to [-pi, pi]
    while heading_error > np.pi:
        heading_error -= 2 * np.pi
    while heading_error < -np.pi:
        heading_error += 2 * np.pi
    
    # Distance to goal
    distance = np.sqrt((x_goal - x)**2 + (y_goal - y)**2)
    
    # Compute wheel velocities
    v_desired = k_p * distance
    omega_desired = k_d * heading_error
    
    # Robot parameters
    wheel_radius = 0.05
    wheel_base = 0.3
    
    # Convert to differential drive commands
    v_l = (2*v_desired - omega_desired*wheel_base) / (2*wheel_radius)
    v_r = (2*v_desired + omega_desired*wheel_base) / (2*wheel_radius)
    
    # Clamp commands
    v_l = np.clip(v_l, -2.0, 2.0)
    v_r = np.clip(v_r, -2.0, 2.0)
    
    return np.array([v_l, v_r])


def plot_trajectory(system, states, title="Robot Trajectory"):
    """Plot the robot trajectory and obstacle."""
    plt.figure(figsize=(10, 8))
    
    # Extract obstacle info
    x_obs = system.params["x_obs"]
    y_obs = system.params["y_obs"]
    r_obs = system.params["r_obs"]
    r_robot = system.params["r_robot"]
    safety_margin = system.params["safety_margin"]
    
    # Plot obstacle
    obstacle = Circle((x_obs, y_obs), r_obs, fill=True, color='red', alpha=0.5)
    plt.gca().add_patch(obstacle)
    
    # Plot safety region
    safety = Circle((x_obs, y_obs), r_obs + r_robot + safety_margin, 
                  fill=False, linestyle='--', color='orange')
    plt.gca().add_patch(safety)
    
    # Plot robot path
    plt.plot(states[:, 0], states[:, 1], 'b-', linewidth=2)
    
    # Plot robot orientation at intervals
    step = max(1, len(states) // 20)
    for i in range(0, len(states), step):
        x, y, theta = states[i, 0:3]
        dx = 0.2 * np.cos(theta)
        dy = 0.2 * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.15, fc='blue', ec='blue')
    
    # Plot start and goal
    plt.plot(states[0, 0], states[0, 1], 'go', markersize=10, label='Start')
    plt.plot(4.0, 4.0, 'g*', markersize=15, label='Goal')
    
    # Set plot properties
    plt.grid(True)
    plt.axis('equal')
    plt.title(title)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    
    return plt


def main():
    """Run the mobile robot benchmark example."""
    print("Running Mobile Robot Benchmark Example...")
    
    # Create the system
    system = DifferentialDriveRobot()
    
    # Define metrics
    metrics = [
        ValidationSuccessRate(samples=1000),
        ComputationTime(samples=100, repetitions=5),
        Conservativeness(samples=1000)
    ]
    
    # Run benchmark
    result = run_benchmark(
        system,
        metrics=metrics,
        output_dir="benchmark_results/mobile_robot"
    )
    
    # Print benchmark results
    print("\nBenchmark Results:")
    for metric_name, value in result.metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Simulate a trajectory
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    states, inputs = simulate_trajectory(system, initial_state, go_to_goal_controller)
    
    # Plot the trajectory
    plt = plot_trajectory(system, states)
    plt.savefig("benchmark_results/mobile_robot/trajectory.png")
    plt.show()
    
    print("\nTrajectory simulation and visualization complete.")
    print("Results saved to benchmark_results/mobile_robot/")


if __name__ == "__main__":
    # Create output directory
    os.makedirs("benchmark_results/mobile_robot", exist_ok=True)
    main()
