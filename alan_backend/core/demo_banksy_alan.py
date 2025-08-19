"""
Demo script for ALAN core Banksy implementation.

This script demonstrates the full ALAN system with:
1. Banksy-spin oscillator substrate
2. TRS-ODE controller
3. Spin-Hopfield memory
4. Banksy fusion

It implements a simple reasoning task where the system infers
properties from partial information, demonstrating the core capabilities
of the integrated architecture.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Ensure we can import from alan_backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import ALAN components
from alan_backend.core.oscillator.banksy_oscillator import BanksyOscillator, BanksyConfig
from alan_backend.core.controller.trs_ode import TRSController, TRSConfig
from alan_backend.core.memory.spin_hopfield import SpinHopfieldMemory, HopfieldConfig
from alan_backend.core.banksy_fusion import BanksyFusion, BanksyReasoner, BanksyFusionConfig


def run_oscillator_demo():
    """Demo of the Banksy oscillator component."""
    print("\n===== Banksy Oscillator Demo =====")
    
    # Create an oscillator with 32 units
    n_oscillators = 32
    config = BanksyConfig(gamma=0.1, epsilon=0.01, eta_damp=1e-4, dt=0.01)
    oscillator = BanksyOscillator(n_oscillators, config)
    
    # Set up a modular coupling matrix (two communities)
    coupling = np.ones((n_oscillators, n_oscillators)) * 0.05
    np.fill_diagonal(coupling, 0.0)
    
    # Strengthen coupling within communities
    community_size = n_oscillators // 2
    for i in range(community_size):
        for j in range(community_size):
            if i != j:
                coupling[i, j] = 0.3
                coupling[i + community_size, j + community_size] = 0.3
    
    oscillator.set_coupling(coupling)
    
    # Run for 200 steps and collect metrics
    order_history = []
    n_eff_history = []
    
    for _ in range(200):
        oscillator.step()
        order_history.append(oscillator.order_parameter())
        n_eff_history.append(oscillator.effective_count(phase_threshold=0.7, spin_threshold=0.3))
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(order_history)
    plt.title('Order Parameter (Phase Synchronization)')
    plt.ylabel('Order Parameter')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(n_eff_history)
    plt.title('Effective Number of Synchronized Oscillators')
    plt.xlabel('Time Steps')
    plt.ylabel('N_eff')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Final metrics
    print(f"Final order parameter: {order_history[-1]:.4f}")
    print(f"Final N_eff: {n_eff_history[-1]} / {n_oscillators}")


def run_trs_controller_demo():
    """Demo of the TRS controller component."""
    print("\n===== TRS Controller Demo =====")
    
    # Create a TRS controller with a Duffing oscillator
    from alan_backend.core.controller.trs_ode import DuffingOscillator, State
    
    # Create the vector field (a double-well potential with some damping)
    vector_field = DuffingOscillator(a=1.0, b=0.3, delta=0.01)
    
    # Create a TRS controller
    config = TRSConfig(dt=0.05, train_steps=400)
    controller = TRSController(
        state_dim=1, 
        vector_field=vector_field,
        config=config,
    )
    
    # Define an initial state slightly off-center in the left well
    initial_state = State(np.array([-0.8]), np.array([0.0]))
    
    # Run simulation
    results = controller.simulate_with_trs(initial_state)
    
    # Extract trajectory data
    forward_traj = results['forward_trajectory']
    backward_traj = results['backward_trajectory']
    
    # Convert to numpy arrays for plotting
    forward_h = np.array([s.h[0] for s in forward_traj])
    forward_p = np.array([s.p[0] for s in forward_traj])
    backward_h = np.array([s.h[0] for s in backward_traj])
    backward_p = np.array([s.p[0] for s in backward_traj])
    time = np.arange(len(forward_traj)) * config.dt
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Position vs time
    plt.subplot(2, 2, 1)
    plt.plot(time, forward_h, 'b-', label='Forward')
    plt.plot(time, backward_h, 'r--', label='Backward')
    plt.xlabel('Time')
    plt.ylabel('Position h')
    plt.legend()
    plt.title('Position over Time')
    plt.grid(True)
    
    # Momentum vs time
    plt.subplot(2, 2, 2)
    plt.plot(time, forward_p, 'b-', label='Forward')
    plt.plot(time, backward_p, 'r--', label='Backward')
    plt.xlabel('Time')
    plt.ylabel('Momentum p')
    plt.legend()
    plt.title('Momentum over Time')
    plt.grid(True)
    
    # Phase space
    plt.subplot(2, 2, 3)
    plt.plot(forward_h, forward_p, 'b-', label='Forward')
    plt.plot(backward_h, backward_p, 'r--', label='Backward')
    plt.scatter(forward_h[0], forward_p[0], c='g', s=100, marker='o', label='Start')
    plt.scatter(forward_h[-1], forward_p[-1], c='m', s=100, marker='x', label='End')
    plt.xlabel('Position h')
    plt.ylabel('Momentum p')
    plt.legend()
    plt.title('Phase Space Portrait')
    plt.grid(True)
    
    # Error between forward and backward trajectories
    plt.subplot(2, 2, 4)
    h_error = np.abs(backward_h - forward_h[::-1])
    p_error = np.abs(backward_p - forward_p[::-1])
    plt.semilogy(time, h_error, 'g-', label='|h_back - h_forward|')
    plt.semilogy(time, p_error, 'm-', label='|p_back - p_forward|')
    plt.xlabel('Time')
    plt.ylabel('Error (log scale)')
    plt.legend()
    plt.title(f'TRS Error (Loss = {results["trs_loss"]:.6f})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"TRS Loss: {results['trs_loss']:.6f}")


def run_hopfield_memory_demo():
    """Demo of the Spin-Hopfield memory component."""
    print("\n===== Spin-Hopfield Memory Demo =====")
    
    # Create a set of patterns representing binary images
    size = 64  # 8x8 grid
    
    # Create patterns (simple shapes)
    def create_grid_pattern(pattern_func):
        pattern = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                pattern[i, j] = pattern_func(i, j)
        return pattern.flatten()
    
    # Letter X
    x_pattern = create_grid_pattern(lambda i, j: 1 if i==j or i==7-j else -1)
    
    # Letter O
    o_pattern = create_grid_pattern(lambda i, j: 1 if (i==0 or i==7 or j==0 or j==7) and not (i==0 and j==0) and not (i==0 and j==7) and not (i==7 and j==0) and not (i==7 and j==7) else -1)
    
    # Letter T
    t_pattern = create_grid_pattern(lambda i, j: 1 if i==0 or j==3 or j==4 else -1)
    
    # Create memory and store patterns
    config = HopfieldConfig(beta=2.0, binary=True, asynchronous=True, max_iterations=100)
    memory = SpinHopfieldMemory(size, config)
    memory.store([x_pattern, o_pattern, t_pattern])
    
    # Test pattern recall with noise
    noisy_patterns = []
    recalled_patterns = []
    pattern_names = ['X', 'O', 'T']
    original_patterns = [x_pattern, o_pattern, t_pattern]
    
    plt.figure(figsize=(12, 8))
    
    for i, (pattern, name) in enumerate(zip(original_patterns, pattern_names)):
        # Add 30% noise
        noise_level = 0.3
        noisy = pattern.copy()
        noise_mask = np.random.choice([0, 1], size=size, p=[1-noise_level, noise_level])
        noisy[noise_mask == 1] *= -1
        
        # Recall the pattern
        recalled, info = memory.recall(noisy)
        
        # Store for display
        noisy_patterns.append(noisy)
        recalled_patterns.append(recalled)
        
        # Calculate overlap
        overlap = memory.compute_overlap(recalled, i)
        
        # Display in subplot
        plt.subplot(3, 3, i*3 + 1)
        plt.imshow(pattern.reshape(8, 8), cmap='binary', interpolation='nearest')
        plt.title(f'Original {name}')
        plt.axis('off')
        
        plt.subplot(3, 3, i*3 + 2)
        plt.imshow(noisy.reshape(8, 8), cmap='binary', interpolation='nearest')
        plt.title(f'Noisy {name} (30% noise)')
        plt.axis('off')
        
        plt.subplot(3, 3, i*3 + 3)
        plt.imshow(recalled.reshape(8, 8), cmap='binary', interpolation='nearest')
        plt.title(f'Recalled ({info["iterations"]} iter, {overlap:.2f})')
        plt.axis('off')
        
        print(f"Pattern {name}: Iterations={info['iterations']}, Overlap={overlap:.4f}")
    
    plt.tight_layout()
    plt.show()


def run_full_system_demo():
    """Demo of the full ALAN Banksy fusion system."""
    print("\n===== Full ALAN Banksy System Demo =====")
    
    # Create a knowledge domain about animals and properties
    concepts = [
        "animal", "mammal", "bird", "dog", "cat", "robin", "eagle", 
        "can_fly", "has_fur", "has_feathers", "barks", "meows", "chirps", "hunts", "pet"
    ]
    
    # Create a reasoner with our concept set
    config = BanksyFusionConfig(
        oscillator=BanksyConfig(gamma=0.15, epsilon=0.02),
        controller=TRSConfig(dt=0.02, train_steps=30),
        memory=HopfieldConfig(beta=1.5, max_iterations=50),
        phase_threshold=0.6,
        spin_threshold=0.3,
        concept_threshold=0.5,
        stabilization_steps=20,
        readout_window=5
    )
    reasoner = BanksyReasoner(concepts, config)
    
    # Define concept relations
    relations = [
        # Taxonomy
        ("dog", "mammal", 0.9),
        ("cat", "mammal", 0.9),
        ("robin", "bird", 0.9),
        ("eagle", "bird", 0.9),
        ("mammal", "animal", 0.9),
        ("bird", "animal", 0.9),
        
        # Properties
        ("mammal", "has_fur", 0.8),
        ("bird", "has_feathers", 0.8),
        ("bird", "can_fly", 0.7),  # Most birds can fly
        ("dog", "barks", 0.9),
        ("cat", "meows", 0.9),
        ("robin", "chirps", 0.9),
        ("eagle", "hunts", 0.8),
        
        # Pets
        ("dog", "pet", 0.8),
        ("cat", "pet", 0.8),
        ("robin", "pet", 0.3),  # Sometimes pet birds
        ("eagle", "pet", -0.7),  # Rarely pets
        
        # Negative relations
        ("mammal", "has_feathers", -0.8),
        ("bird", "has_fur", -0.8),
        ("dog", "meows", -0.9),
        ("cat", "barks", -0.9),
    ]
    
    for src, tgt, strength in relations:
        reasoner.add_concept_relation(src, tgt, strength)
    
    # Add concept patterns (prototypical examples)
    reasoner.add_concept_pattern("dog_pattern", {
        "dog": 1.0, "mammal": 1.0, "animal": 1.0, "has_fur": 1.0, 
        "barks": 1.0, "pet": 1.0
    })
    
    reasoner.add_concept_pattern("cat_pattern", {
        "cat": 1.0, "mammal": 1.0, "animal": 1.0, "has_fur": 1.0, 
        "meows": 1.0, "pet": 1.0
    })
    
    reasoner.add_concept_pattern("robin_pattern", {
        "robin": 1.0, "bird": 1.0, "animal": 1.0, "has_feathers": 1.0, 
        "can_fly": 1.0, "chirps": 1.0
    })
    
    reasoner.add_concept_pattern("eagle_pattern", {
        "eagle": 1.0, "bird": 1.0, "animal": 1.0, "has_feathers": 1.0, 
        "can_fly": 1.0, "hunts": 1.0
    })
    
    # Run a series of reasoning queries
    queries = [
        {"name": "What has fur?", "query": {"has_fur": 1.0}},
        {"name": "What makes sounds?", "query": {"barks": 0.7, "meows": 0.7, "chirps": 0.7}},
        {"name": "What is a pet with fur?", "query": {"pet": 1.0, "has_fur": 1.0}},
        {"name": "What can fly?", "query": {"can_fly": 1.0}},
        {"name": "What is a mammal that barks?", "query": {"mammal": 1.0, "barks": 1.0}}
    ]
    
    # Run each query
    results = []
    for q in queries:
        print(f"\nQuery: {q['name']}")
        result = reasoner.reason(q['query'], steps=50)
        results.append(result)
        
        print(f"Result concepts: {result['result_concepts']}")
        print(f"Confidence: {result['confidence']:.2f}")
    
    # Plot the results of multiple queries
    plt.figure(figsize=(15, 10))
    
    for i, (query, result) in enumerate(zip(queries, results)):
        # Plot query results
        plt.subplot(len(queries), 1, i+1)
        
        concepts_list = list(result['final_state'].keys())
        activations = [result['final_state'][c] for c in concepts_list]
        
        bars = plt.bar(concepts_list, activations)
        
        # Highlight active concepts
        active_set = set(result['result_concepts'].keys())
        for j, concept in enumerate(concepts_list):
            if concept in active_set:
                bars[j].set_color('red')
        
        plt.ylim(-1.1, 1.1)
        plt.title(f"Query: {query['name']} (Confidence: {result['confidence']:.2f})")
        plt.ylabel('Activation')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the individual component demos
    run_oscillator_demo()
    run_trs_controller_demo()
    run_hopfield_memory_demo()
    
    # Run the full system demo
    run_full_system_demo()
