#!/usr/bin/env python3
"""
Demo of the StabilityAgent for the ELFIN stability framework.

This script demonstrates how to use the StabilityAgent to verify
stability properties of a system, log interactions, and generate
rich error messages.
"""

import os
import logging
import pathlib
import sys
import time
import numpy as np
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("elfin.demo")

# Add project root to path if needed
project_root = pathlib.Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import ELFIN components
from alan_backend.elfin.stability.agents import StabilityAgent
from alan_backend.elfin.stability.training import LyapunovNet
from alan_backend.elfin.stability.core import Interaction


class SimpleOscillator:
    """
    A simple oscillator system for demonstration.
    
    This class implements a simple 2D oscillator system with adjustable
    parameters. It can be stable or unstable depending on the parameters.
    """
    
    def __init__(self, id="simple_oscillator", damping=0.1, stiffness=1.0):
        """
        Initialize the oscillator.
        
        Args:
            id: Identifier for the system
            damping: Damping coefficient (c)
            stiffness: Stiffness coefficient (k)
        """
        self.id = id
        self.damping = damping
        self.stiffness = stiffness
        
        logger.info(f"Created oscillator with damping={damping}, stiffness={stiffness}")
    
    def dynamics(self, x):
        """
        Compute the dynamics of the system.
        
        Args:
            x: State vector [position, velocity]
            
        Returns:
            Derivative of the state vector [velocity, acceleration]
        """
        # For batched input
        if len(x.shape) > 1:
            position = x[:, 0]
            velocity = x[:, 1]
            
            # Simple damped oscillator: mx'' + cx' + kx = 0 (m=1)
            acceleration = -self.stiffness * position - self.damping * velocity
            
            return np.stack([velocity, acceleration], axis=1)
        else:
            # For single input
            position, velocity = x
            
            # Simple damped oscillator: mx'' + cx' + kx = 0 (m=1)
            acceleration = -self.stiffness * position - self.damping * velocity
            
            return np.array([velocity, acceleration])
    
    def is_stable(self):
        """
        Check if the system is stable.
        
        Returns:
            True if the system is stable, False otherwise
        """
        # A simple damped oscillator is stable if damping > 0
        return self.damping > 0


def create_unstable_lyapunov_net():
    """
    Create a neural Lyapunov function that is unstable.
    
    Returns:
        LyapunovNet instance
    """
    # Create model with normal initialization
    model = LyapunovNet(
        dim=2,
        hidden_dims=(32, 32),
        alpha=0.001,
        activation=nn.Tanh()
    )
    
    # Explicitly make it unstable by setting all weights negative
    with torch.no_grad():
        for param in model.parameters():
            param.data = -torch.abs(param.data)
    
    model.id = "unstable_net"
    
    return model


def create_stable_lyapunov_net():
    """
    Create a neural Lyapunov function that is stable.
    
    Returns:
        LyapunovNet instance
    """
    # Create model with normal initialization
    model = LyapunovNet(
        dim=2,
        hidden_dims=(32, 32),
        alpha=0.1,  # Larger alpha ensures positive definiteness
        activation=nn.Tanh()
    )
    
    model.id = "stable_net"
    
    return model


def demonstrate_stability_agent():
    """
    Demonstrate the StabilityAgent functionality.
    
    This function shows:
    1. Creating a StabilityAgent
    2. Verifying stability properties
    3. Logging interactions
    4. Handling verification errors
    5. Parameter tuning
    """
    # Create a temporary directory for cache
    cache_dir = pathlib.Path("temp_cache")
    cache_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print(" ELFIN StabilityAgent Demo ".center(80, "="))
    print("="*80 + "\n")
    
    # Create a StabilityAgent
    print("Creating StabilityAgent...")
    agent = StabilityAgent("demo_agent", cache_dir)
    
    # ==== Scenario 1: Verify a stable system ====
    print("\n[Scenario 1] Verifying a stable system")
    print("-" * 50)
    
    # Create a stable oscillator
    system = SimpleOscillator(damping=0.5, stiffness=1.0)
    
    # Create a stable Lyapunov function
    lyap_net = create_stable_lyapunov_net()
    
    # Define verification domain
    domain = (np.array([-2.0, -2.0]), np.array([2.0, 2.0]))
    
    # Verify positive definiteness
    print("Verifying positive definiteness...")
    try:
        result = agent.verify(lyap_net, domain)
        
        if result["status"] == "VERIFIED":
            print(f"✅ System verified!")
            print(f"  Solve time: {result['solve_time']:.2f}s")
        else:
            print(f"⚠️ Verification failed!")
            if result["counterexample"] is not None:
                print(f"  Counterexample: x={np.round(result['counterexample'], 3)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Verify decrease condition
    print("\nVerifying decrease condition...")
    try:
        result = agent.verify_decrease(lyap_net, system.dynamics, domain, gamma=0.1)
        
        if result["status"] == "VERIFIED":
            print(f"✅ Decrease verified!")
        else:
            print(f"⚠️ Verification failed!")
            if result["counterexample"] is not None:
                print(f"  Counterexample: x={np.round(result['counterexample'], 3)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ==== Scenario 2: Verify an unstable system ====
    print("\n[Scenario 2] Verifying an unstable system")
    print("-" * 50)
    
    # Create an unstable Lyapunov function
    unstable_net = create_unstable_lyapunov_net()
    
    # Verify positive definiteness
    print("Verifying positive definiteness...")
    try:
        result = agent.verify(unstable_net, domain)
        
        if result["status"] == "VERIFIED":
            print(f"✅ System verified!")
        else:
            print(f"⚠️ Verification failed!")
            if result["counterexample"] is not None:
                print(f"  Counterexample: x={np.round(result['counterexample'], 3)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ==== Scenario 3: Parameter tuning ====
    print("\n[Scenario 3] Parameter tuning")
    print("-" * 50)
    
    # Tune parameter
    print("Tuning damping parameter...")
    try:
        old_value = system.damping
        new_value = 0.05
        
        result = agent.param_tune(system, "damping", old_value, new_value)
        
        print(f"✅ Parameter tuned: damping = {old_value} → {new_value}")
        
        # Verify decrease with new parameter
        print("\nVerifying decrease condition with new parameter...")
        result = agent.verify_decrease(lyap_net, system.dynamics, domain, gamma=0.05)
        
        if result["status"] == "VERIFIED":
            print(f"✅ Decrease verified!")
        else:
            print(f"⚠️ Verification failed!")
            if result["counterexample"] is not None:
                print(f"  Counterexample: x={np.round(result['counterexample'], 3)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ==== Scenario 4: View interaction log ====
    print("\n[Scenario 4] View interaction log")
    print("-" * 50)
    
    # Get log summary
    summary = agent.get_summary()
    print(summary)
    
    # ==== Cleanup ====
    print("\nCleaning up...")
    import shutil
    shutil.rmtree(cache_dir)
    
    print("\n" + "="*80)
    print(" Demo Complete ".center(80, "="))
    print("="*80 + "\n")


if __name__ == "__main__":
    demonstrate_stability_agent()
