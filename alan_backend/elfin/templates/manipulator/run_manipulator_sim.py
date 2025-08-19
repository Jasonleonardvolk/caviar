#!/usr/bin/env python3
"""
Manipulator Simulator Runner

This script demonstrates the usage of the ManipulatorSimulator class by running
different control modes and visualizing the results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sim_manipulator import ManipulatorSimulator

def main():
    """Run demonstration of manipulator simulation with different controllers"""
    print("Initializing simulator...")
    simulator = ManipulatorSimulator()
    
    # Set non-zero initial joint configuration for more interesting motion
    initial_state = np.zeros(12)
    initial_state[0:6] = [0.3, 0.5, -0.2, 0.1, 0.0, 0.0]  # Joint positions
    
    # Define sinusoidal reference trajectory
    def reference_trajectory(t):
        """Sinusoidal reference trajectory for demonstration"""
        amplitude = 0.5
        frequency = 0.2  # Hz
        phase_shift = np.pi/3
        
        return {
            "q1_d": amplitude * np.sin(2*np.pi*frequency*t),
            "q2_d": amplitude * np.sin(2*np.pi*frequency*t + phase_shift),
            "q3_d": amplitude * np.sin(2*np.pi*frequency*t + 2*phase_shift)
        }
    
    # Choose which controller to test (or 'all' for all controllers)
    controller_mode = 'all'  # Options: 'joint_position', 'human_collaboration', 'force', 'all'
    
    # Simulation parameters
    T = 10.0  # seconds
    dt = 0.01  # timestep
    
    if controller_mode == 'all' or controller_mode == 'joint_position':
        print("\nRunning joint position control simulation...")
        sim_joint = simulator.simulate(
            controller_name='joint_position',
            T=T,
            dt=dt,
            initial_state=initial_state,
            reference_trajectory=reference_trajectory
        )
        
        # Plot joint states
        fig_joint = simulator.plot_joint_states(sim_joint)
        fig_joint.suptitle('Joint Position Control')
        
        # Plot barrier values
        fig_barrier_joint = simulator.plot_barriers(sim_joint)
        fig_barrier_joint.suptitle('Safety Barriers - Joint Position Control')
        
        # Visualize robot configuration
        fig_robot_joint = plt.figure(figsize=(10, 8))
        ax = fig_robot_joint.add_subplot(111, projection='3d')
        simulator.visualize_robot(sim_joint, ax)
        ax.set_title('Final Configuration - Joint Position Control')
    
    if controller_mode == 'all' or controller_mode == 'human_collaboration':
        print("\nRunning human collaboration control simulation...")
        # Update proximity factor for demonstration
        simulator.params['human_proximity_factor'] = 0.3
        simulator.params['x_h'] = 0.7  # Move human closer to demonstrate different behavior
        
        sim_human = simulator.simulate(
            controller_name='human_collaboration',
            T=T,
            dt=dt,
            initial_state=initial_state,
            reference_trajectory=reference_trajectory
        )
        
        # Plot joint states
        fig_human = simulator.plot_joint_states(sim_human)
        fig_human.suptitle('Human Collaboration Control')
        
        # Plot barrier values
        fig_barrier_human = simulator.plot_barriers(sim_human)
        fig_barrier_human.suptitle('Safety Barriers - Human Collaboration Control')
        
        # Visualize robot configuration
        fig_robot_human = plt.figure(figsize=(10, 8))
        ax = fig_robot_human.add_subplot(111, projection='3d')
        simulator.visualize_robot(sim_human, ax)
        ax.set_title('Final Configuration - Human Collaboration Control')
    
    if controller_mode == 'all' or controller_mode == 'force':
        print("\nRunning force control simulation...")
        # Update force measurement for demonstration
        simulator.params['F_measured'] = 2.0
        
        sim_force = simulator.simulate(
            controller_name='force',
            T=T,
            dt=dt,
            initial_state=initial_state,
            reference_trajectory=reference_trajectory
        )
        
        # Plot joint states
        fig_force = simulator.plot_joint_states(sim_force)
        fig_force.suptitle('Force Control')
        
        # Plot barrier values
        fig_barrier_force = simulator.plot_barriers(sim_force)
        fig_barrier_force.suptitle('Safety Barriers - Force Control')
        
        # Visualize robot configuration
        fig_robot_force = plt.figure(figsize=(10, 8))
        ax = fig_robot_force.add_subplot(111, projection='3d')
        simulator.visualize_robot(sim_force, ax)
        ax.set_title('Final Configuration - Force Control')
    
    print("\nSimulation complete. Displaying plots...")
    plt.show()

if __name__ == "__main__":
    main()
