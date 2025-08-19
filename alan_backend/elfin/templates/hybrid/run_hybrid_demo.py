#!/usr/bin/env python3
"""
Hybrid System Demonstration Script

This script demonstrates how to use the ELFIN hybrid system templates.
It loads the hybrid controller templates, parses them, and runs a simple simulation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the ELFIN package to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from elfin.parser import parse_elfin_file
    from elfin.simulation import simulate_hybrid_system
except ImportError:
    print("Error: Could not import ELFIN modules. Make sure ELFIN is installed.")
    sys.exit(1)

def run_bouncing_ball_demo():
    """Run the bouncing ball demo from the hybrid controller template."""
    print("Running Bouncing Ball Demo...")
    
    # Parse the hybrid controller file
    file_path = Path(__file__).parent / "src" / "hybrid_controller.elfin"
    parsed_model = parse_elfin_file(file_path)
    
    # Extract the bouncing ball system and controller
    system = parsed_model.get_system("BouncingBall")
    controller = parsed_model.get_controller("BouncingBallController")
    
    # Set up initial conditions
    initial_state = {
        "h": 10.0,
        "v": 0.0,
        "bounce_count": 0
    }
    
    # Simulation parameters
    t_end = 10.0
    dt = 0.01
    
    # Run the simulation
    times, states, events = simulate_hybrid_system(
        system, controller, initial_state, t_end, dt
    )
    
    # Extract state variables
    heights = [state["h"] for state in states]
    velocities = [state["v"] for state in states]
    bounce_counts = [state["bounce_count"] for state in states]
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(times, heights)
    plt.ylabel('Height (m)')
    plt.title('Bouncing Ball Simulation')
    
    plt.subplot(3, 1, 2)
    plt.plot(times, velocities)
    plt.ylabel('Velocity (m/s)')
    
    plt.subplot(3, 1, 3)
    plt.plot(times, bounce_counts, 'r-')
    plt.ylabel('Bounce Count')
    plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig("bouncing_ball_simulation.png")
    plt.show()
    
    print(f"Simulation completed with {bounce_counts[-1]} bounces.")
    print(f"Results saved to bouncing_ball_simulation.png")

def run_thermostat_demo():
    """Run the thermostat demo from the hybrid controller template."""
    print("Running Thermostat Demo...")
    
    # Parse the hybrid controller file
    file_path = Path(__file__).parent / "src" / "hybrid_controller.elfin"
    parsed_model = parse_elfin_file(file_path)
    
    # Extract the thermostat system and controller
    system = parsed_model.get_system("Thermostat")
    controller = parsed_model.get_controller("ThermostatController")
    
    # Set up initial conditions
    initial_state = {
        "T": 15.0,
        "mode": 0  # Start with heater off
    }
    
    # Simulation parameters
    t_end = 60.0  # One minute simulation
    dt = 0.1
    
    # Target temperature
    inputs = {
        "T_target": 22.0  # Target temperature in degrees C
    }
    
    # Run the simulation
    times, states, events = simulate_hybrid_system(
        system, controller, initial_state, t_end, dt, inputs
    )
    
    # Extract state variables
    temperatures = [state["T"] for state in states]
    modes = [state["mode"] for state in states]
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(times, temperatures)
    plt.axhline(y=inputs["T_target"], color='r', linestyle='--', label='Target')
    plt.axhline(y=inputs["T_target"] + system.params["hysteresis"], color='g', linestyle='--', label='Upper Bound')
    plt.axhline(y=inputs["T_target"] - system.params["hysteresis"], color='g', linestyle='--', label='Lower Bound')
    plt.ylabel('Temperature (°C)')
    plt.title('Thermostat Simulation')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.step(times, modes, where='post')
    plt.ylabel('Heater Mode')
    plt.xlabel('Time (s)')
    plt.yticks([0, 1], ['Off', 'On'])
    
    plt.tight_layout()
    plt.savefig("thermostat_simulation.png")
    plt.show()
    
    print(f"Simulation completed with final temperature: {temperatures[-1]:.2f}°C")
    print(f"Results saved to thermostat_simulation.png")

def run_lane_changing_demo():
    """Run the lane changing demo from the hybrid controller template."""
    print("Running Lane Changing Demo...")
    
    # Parse the hybrid controller file
    file_path = Path(__file__).parent / "src" / "hybrid_controller.elfin"
    parsed_model = parse_elfin_file(file_path)
    
    # Extract the lane changing system and controller
    system = parsed_model.get_system("LaneChanging")
    controller = parsed_model.get_controller("LaneChangingController")
    
    # Set up initial conditions
    initial_state = {
        "x": 0.0,
        "y": 0.0,
        "v": 20.0,
        "theta": 0.0,
        "lane_mode": 0  # Start in cruising mode
    }
    
    # External states
    external_states = {
        "change_requested": 0,
        "safe_to_change": 1,
        "obstacle_detected": 0,
        "obstacle_distance": 100.0
    }
    
    # Simulation parameters
    t_end = 20.0
    dt = 0.05
    
    # Events during simulation
    events = [
        {"time": 5.0, "state_updates": {"change_requested": 1}},
        {"time": 15.0, "state_updates": {"obstacle_detected": 1, "obstacle_distance": 30.0}}
    ]
    
    # Run the simulation
    times, states, event_logs = simulate_hybrid_system(
        system, controller, initial_state, t_end, dt, 
        external_states=external_states, events=events
    )
    
    # Extract state variables
    x_values = [state["x"] for state in states]
    y_values = [state["y"] for state in states]
    velocities = [state["v"] for state in states]
    thetas = [state["theta"] for state in states]
    modes = [state["lane_mode"] for state in states]
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(x_values, y_values)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Vehicle Trajectory')
    
    plt.subplot(3, 2, 2)
    plt.plot(times, velocities)
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Profile')
    
    plt.subplot(3, 2, 3)
    plt.step(times, modes, where='post')
    plt.ylabel('Lane Mode')
    plt.title('Lane Mode')
    plt.yticks([0, 1, 2], ['Cruising', 'Changing', 'Emergency'])
    
    plt.subplot(3, 2, 4)
    plt.plot(times, thetas)
    plt.ylabel('Heading (rad)')
    plt.title('Vehicle Heading')
    
    plt.subplot(3, 2, 5)
    plt.plot(times, x_values)
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.title('X Position vs Time')
    
    plt.subplot(3, 2, 6)
    plt.plot(times, y_values)
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.title('Y Position vs Time')
    
    plt.tight_layout()
    plt.savefig("lane_changing_simulation.png")
    plt.show()
    
    print(f"Simulation completed with final position: ({x_values[-1]:.2f}, {y_values[-1]:.2f})")
    print(f"Results saved to lane_changing_simulation.png")

def main():
    """Main entry point for the demo script."""
    print("ELFIN Hybrid System Demo")
    print("=======================")
    print("\nThis script demonstrates the hybrid system templates in ELFIN.")
    
    while True:
        print("\nAvailable demos:")
        print("1. Bouncing Ball")
        print("2. Thermostat")
        print("3. Lane Changing")
        print("q. Quit")
        
        choice = input("\nSelect a demo to run: ")
        
        if choice == '1':
            run_bouncing_ball_demo()
        elif choice == '2':
            run_thermostat_demo()
        elif choice == '3':
            run_lane_changing_demo()
        elif choice.lower() == 'q':
            print("Exiting demo.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
