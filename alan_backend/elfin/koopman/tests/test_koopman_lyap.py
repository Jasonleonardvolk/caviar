"""
Unit tests for Koopman-based Lyapunov functions.

This module tests the functionality of the Koopman-based Lyapunov functions,
verifying their properties against known dynamical systems.
"""

import unittest
import numpy as np
import os
import pathlib
import sys
from typing import Tuple

# Add parent directory to path for imports
parent_dir = pathlib.Path(__file__).parent.parent.parent.parent
if parent_dir not in sys.path:
    sys.path.append(str(parent_dir))

from alan_backend.elfin.koopman.dictionaries import create_dictionary
from alan_backend.elfin.koopman.edmd import edmd_fit
from alan_backend.elfin.koopman.koopman_lyap import create_koopman_lyapunov
from alan_backend.elfin.koopman.koopman_bridge_agent import create_pendulum_agent


class TestKoopmanLyapunov(unittest.TestCase):
    """Test cases for Koopman-based Lyapunov functions."""
    
    def generate_pendulum_data(self, n_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate pendulum data for testing.
        
        Args:
            n_points: Number of data points to generate
            
        Returns:
            Tuple of (x, x_next) data
        """
        # Generate pendulum trajectory data
        t = np.linspace(0, 10, n_points)
        dt = t[1] - t[0]
        
        # Pendulum parameters
        alpha = 0.1  # Damping coefficient
        
        # Initial conditions (multiple trajectories)
        x0_list = [
            np.array([np.pi/4, 0.0]),  # Small angle
            np.array([np.pi/2, 0.0]),  # Medium angle
            np.array([0.0, 0.5]),      # Zero angle, nonzero velocity
            np.array([np.pi/4, 0.5])   # Small angle, positive velocity
        ]
        
        # Pendulum dynamics
        def pendulum_dynamics(x, alpha=0.1):
            """Pendulum dynamics: x' = [x[1], -sin(x[0]) - alpha*x[1]]"""
            theta, omega = x
            return np.array([omega, -np.sin(theta) - alpha*omega])
        
        # Generate trajectory data
        all_x = []
        all_x_next = []
        
        for x0 in x0_list:
            # Simulate trajectory
            x_traj = [x0]
            for i in range(1, len(t)):
                # Simple Euler integration
                x_prev = x_traj[-1]
                x_next = x_prev + dt * pendulum_dynamics(x_prev, alpha)
                x_traj.append(x_next)
            
            # Convert to numpy array
            x_traj = np.array(x_traj)
            
            # Extract x and x_next
            x = x_traj[:-1]
            x_next = x_traj[1:]
            
            # Append to data
            all_x.append(x)
            all_x_next.append(x_next)
        
        # Concatenate data
        x = np.vstack(all_x)
        x_next = np.vstack(all_x_next)
        
        return x, x_next
    
    def test_pendulum_stable_modes(self):
        """Test that pendulum EDMD produces at least 3 strictly stable modes."""
        # Generate pendulum data
        x, x_next = self.generate_pendulum_data()
        
        # Create dictionary
        state_dim = x.shape[1]
        dict_size = 50
        dictionary = create_dictionary(
            dict_type="rbf",
            dim=state_dim,
            n_centers=dict_size
        )
        
        # Fit Koopman operator
        k_matrix, meta = edmd_fit(
            dictionary=dictionary,
            x=x,
            x_next=x_next
        )
        
        # Create Lyapunov function
        lyap_fn = create_koopman_lyapunov(
            name="pendulum_test",
            k_matrix=k_matrix,
            dictionary=dictionary,
            lambda_cut=0.98,
            continuous_time=True
        )
        
        # Check that we have at least 3 stable modes
        self.assertGreaterEqual(
            len(lyap_fn.stable_indices), 
            3, 
            "Pendulum EDMD should produce at least 3 stable modes"
        )
        
        # Check that the eigenvalues have negative real parts
        for i in range(min(3, len(lyap_fn.stable_indices))):
            eigenvalue = lyap_fn.get_eigenvalue(i)
            self.assertLess(
                np.real(eigenvalue), 
                0, 
                f"Eigenvalue {i} should have negative real part, got {eigenvalue}"
            )
        
        # Print some information
        print(f"Found {len(lyap_fn.stable_indices)} stable modes out of {len(lyap_fn.eigenvalues)}")
        print(f"First 3 eigenvalues: {[complex(lyap_fn.get_eigenvalue(i)) for i in range(min(3, len(lyap_fn.stable_indices)))]}")
    
    def test_pendulum_agent(self):
        """Test the pendulum agent creation and verification."""
        # Create agent and Lyapunov function using convenience function
        agent, lyap_fn = create_pendulum_agent(
            name="test_pendulum_agent",
            dict_type="rbf",
            dict_size=50,
            λ_cut=0.98,
            n_points=500,
            noise_level=0.0,
            verify=False  # Don't verify to speed up tests
        )
        
        # Check that the agent was created successfully
        self.assertIsNotNone(agent)
        self.assertIsNotNone(lyap_fn)
        
        # Check that we have pendulum results
        self.assertIn("pendulum", agent.results)
        
        # Check that we have at least 3 stable modes
        self.assertGreaterEqual(
            len(lyap_fn.stable_indices), 
            3, 
            "Pendulum agent should produce at least 3 stable modes"
        )
        
        # Check that the eigenvalues have negative real parts
        for i in range(min(3, len(lyap_fn.stable_indices))):
            eigenvalue = lyap_fn.get_eigenvalue(i)
            self.assertLess(
                np.real(eigenvalue), 
                0, 
                f"Eigenvalue {i} should have negative real part, got {eigenvalue}"
            )


if __name__ == "__main__":
    unittest.main()
