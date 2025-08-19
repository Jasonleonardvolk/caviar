"""
Phase-Koopman Coupled System Demo.

This example demonstrates the integration between the phase synchronization
engine and the Koopman spectral analysis pipeline - two core components of
the ALAN cognitive architecture.

The demo shows how:
1. The phase engine synchronizes concepts based on their relationships
2. The spectral analyzer detects dynamical patterns and instabilities
3. Feedback from spectral analysis modulates phase coupling
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import time
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stability.core.phase_engine import PhaseEngine
from koopman.snapshot_buffer import SnapshotBuffer
from koopman.spectral_analyzer import SpectralAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_simulation(steps=1000, spectral_feedback=True, 
                  buffer_capacity=100, spectral_update_interval=20):
    """
    Run a simulation of the coupled phase-spectral system.
    
    Args:
        steps: Number of simulation steps
        spectral_feedback: Whether to use spectral feedback to phase engine
        buffer_capacity: Capacity of the snapshot buffer
        spectral_update_interval: How often to update spectral analysis
        
    Returns:
        Tuple of (phase_engine, snapshot_buffer, analyzer, history)
    """
    # Create phase engine with moderate coupling
    engine = PhaseEngine(coupling_strength=0.15)
    
    # Create concept graph
    # Main concepts
    concepts = ["User", "Interface", "Database", "Network", "Security", 
               "Algorithm", "Storage", "Processing"]
    
    # Create concepts with random initial phases
    np.random.seed(42)  # For reproducibility
    for c in concepts:
        engine.add_concept(c, initial_phase=np.random.uniform(0, 2*np.pi))
    
    # Add edges between related concepts (symmetric)
    edges = [
        ("User", "Interface", 1.0),
        ("Interface", "Database", 0.7),
        ("Database", "Storage", 0.9),
        ("Network", "Database", 0.5),
        ("Network", "Security", 0.8),
        ("Algorithm", "Processing", 0.9),
        ("Processing", "Database", 0.6),
        ("Security", "Database", 0.7),
        # Cross-cluster connection with weak coupling
        ("User", "Algorithm", 0.1),
    ]
    
    # Add bidirectional edges
    for source, target, weight in edges:
        engine.add_edge(source, target, weight=weight)
        engine.add_edge(target, source, weight=weight)
    
    # Create snapshot buffer
    buffer = SnapshotBuffer(capacity=buffer_capacity)
    
    # Create spectral analyzer
    analyzer = SpectralAnalyzer(buffer)
    
    # Initialize history dictionary for storing results
    history = {
        'phases': {},
        'sync_ratio': [],
        'spectral_feedback': [],
        'stability_index': [],
        'timestamps': []
    }
    
    # Initialize phase history for each concept
    for c in concepts:
        history['phases'][c] = []
    
    # Run simulation
    for step in range(steps):
        # Get concept phases as dictionary
        phases = engine.phases.copy()
        
        # Capture current state
        timestamp = step * 0.1  # dt = 0.1
        buffer.add_snapshot(phases, timestamp=timestamp)
        
        # Update phase engine
        engine.step(dt=0.1)
        
        # Perform spectral analysis periodically
        if step > 0 and step % spectral_update_interval == 0 and len(buffer.buffer) > 10:
            try:
                # Perform EDMD decomposition
                result = analyzer.edmd_decompose(time_shift=1)
                
                # Get stability feedback
                stability_index = analyzer.calculate_stability_index()
                
                # Update phase engine coupling if feedback is enabled
                if spectral_feedback:
                    feedback = analyzer.get_spectral_feedback()
                    engine.set_spectral_feedback(feedback)
                
                # Log status
                n_unstable = len(analyzer.unstable_modes)
                logger.info(f"Step {step}: Sync: {engine.calculate_sync_ratio():.3f}, "
                           f"Stability: {stability_index:.3f}, "
                           f"Unstable modes: {n_unstable}")
                
            except Exception as e:
                logger.warning(f"Spectral analysis failed at step {step}: {e}")
        
        # Calculate sync ratio
        sync_ratio = engine.calculate_sync_ratio()
        
        # Record history
        for c in concepts:
            history['phases'][c].append(phases.get(c, 0.0))
        
        history['sync_ratio'].append(sync_ratio)
        history['spectral_feedback'].append(engine.spectral_feedback)
        
        if analyzer.last_result is not None:
            history['stability_index'].append(analyzer.calculate_stability_index())
        else:
            history['stability_index'].append(0.0)
            
        history['timestamps'].append(timestamp)
    
    return engine, buffer, analyzer, history


def introduce_instability(engine, source, target, new_weight=1.5, step=500):
    """
    Introduce instability by increasing coupling between two concepts.
    
    Args:
        engine: Phase engine instance
        source: Source concept
        target: Target concept
        new_weight: New coupling weight (high value can cause instability)
        step: Simulation step when to introduce the change
    """
    # Store original weights for future reference
    original_weight = engine.graph[source][target]['weight']
    
    # Function to modify weight at the specified step
    def modify(current_step):
        if current_step == step:
            logger.info(f"Introducing potential instability: {source}->{target} "
                       f"weight {original_weight} -> {new_weight}")
            # Increase coupling weight
            engine.graph[source][target]['weight'] = new_weight
            engine.graph[target][source]['weight'] = new_weight
            return True
        return False
    
    return modify


def plot_results(history, concepts, output_dir=None):
    """
    Visualize simulation results.
    
    Args:
        history: History data from simulation
        concepts: List of concept names
        output_dir: Directory to save plots (if None, plots are displayed)
    """
    # Create output directory if it doesn't exist
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot phases
    plt.figure(figsize=(12, 8))
    timestamps = history['timestamps']
    
    for c in concepts:
        plt.plot(timestamps, history['phases'][c], label=c)
    
    plt.xlabel('Time')
    plt.ylabel('Phase (radians)')
    plt.title('Concept Phase Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir is not None:
        plt.savefig(Path(output_dir) / 'phases.png')
    else:
        plt.show()
    
    # Plot synchronization and stability metrics
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, history['sync_ratio'], 'b-', label='Sync Ratio')
    plt.ylabel('Sync Ratio')
    plt.title('System Synchronization and Stability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, history['stability_index'], 'r-', label='Stability Index')
    plt.plot(timestamps, history['spectral_feedback'], 'g--', label='Spectral Feedback')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if output_dir is not None:
        plt.savefig(Path(output_dir) / 'metrics.png')
    else:
        plt.show()


def main():
    """Run the phase-spectral integration demo."""
    
    logger.info("Starting Phase-Koopman system demo")
    
    # Define scenarios
    scenarios = [
        {
            'name': 'baseline',
            'steps': 600, 
            'spectral_feedback': True,
            'instability': None,
            'title': 'Baseline System (With Spectral Feedback)'
        },
        {
            'name': 'no_feedback',
            'steps': 600, 
            'spectral_feedback': False,
            'instability': None,
            'title': 'No Spectral Feedback'
        },
        {
            'name': 'with_instability_and_feedback',
            'steps': 1000, 
            'spectral_feedback': True,
            'instability': ('Database', 'Network', 2.0, 500),
            'title': 'Instability Introduced with Spectral Feedback'
        },
        {
            'name': 'with_instability_no_feedback',
            'steps': 1000, 
            'spectral_feedback': False,
            'instability': ('Database', 'Network', 2.0, 500),
            'title': 'Instability Introduced without Feedback'
        }
    ]
    
    # Run each scenario
    results = {}
    
    for scenario in scenarios:
        logger.info(f"Running scenario: {scenario['name']}")
        
        # Set up instability trigger if specified
        instability_trigger = None
        if scenario['instability'] is not None:
            source, target, weight, step = scenario['instability']
            instability_trigger = introduce_instability(
                None, source, target, weight, step
            )
        
        # Run simulation
        engine, buffer, analyzer, history = run_simulation(
            steps=scenario['steps'],
            spectral_feedback=scenario['spectral_feedback']
        )
        
        # Apply instability if configured
        if instability_trigger is not None:
            # Rebind instability trigger to actual engine
            instability_trigger = introduce_instability(
                engine, source, target, weight, step
            )
            
            # Apply at the specified step
            for i in range(scenario['steps']):
                if instability_trigger(i):
                    break
        
        # Store results
        results[scenario['name']] = {
            'engine': engine,
            'buffer': buffer,
            'analyzer': analyzer,
            'history': history,
            'title': scenario['title']
        }
        
        # Plot results
        plot_results(
            history, 
            list(engine.phases.keys()),
            output_dir=f"./outputs/{scenario['name']}"
        )
    
    logger.info("Demo completed. Results saved to ./outputs/ directory")
    
    # Return results for interactive use
    return results


if __name__ == "__main__":
    main()
