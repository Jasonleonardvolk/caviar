"""
Test script for quantum reservoir computing module.

This script demonstrates the basic functionality of the quantum reservoir
computing module and provides a simple test case for verification.
"""

import os
import numpy as np
import logging
import time
from typing import List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pcc-test")

# Import the quantum reservoir module
try:
    from q_reservoir import QR, EntangledAttention, phases_to_hamiltonian, spins_to_hamiltonian
    logger.info("Successfully imported quantum reservoir module")
except ImportError as e:
    logger.error(f"Error importing quantum reservoir module: {e}")
    # Try relative import
    try:
        from .q_reservoir import QR, EntangledAttention, phases_to_hamiltonian, spins_to_hamiltonian
        logger.info("Successfully imported quantum reservoir module (relative import)")
    except ImportError as e:
        logger.error(f"Error importing quantum reservoir module (relative import): {e}")
        raise

def test_quantum_reservoir_basic():
    """Test basic functionality of the quantum reservoir."""
    logger.info("\n--- Testing Basic Quantum Reservoir ---")
    
    # Create a small QR for testing (4 qubits = 16-dimensional state space)
    n_qubits = 4
    qr = QR(n_qubits=n_qubits, noise_level=0.001)
    logger.info(f"Created QR with {n_qubits} qubits")
    
    # Create a simple Hamiltonian (identity matrix)
    dim = 2**n_qubits
    H = np.eye(dim)
    
    # Perform a few steps
    for i in range(3):
        qr.step(H, dt=0.1)
        logger.info(f"Step {i+1} completed")
    
    # Get expectation values
    z_expectations = qr.get_expectation_values()
    logger.info(f"Z expectations: {z_expectations}")
    
    # Sample from the state
    spins = qr.sample()
    logger.info(f"Sampled spins: {spins}")
    
    # Get state summary
    summary = qr.get_state_summary()
    logger.info(f"State entropy: {summary['state_entropy']}")
    
    return qr

def test_phase_encoding():
    """Test phase encoding in quantum reservoir."""
    logger.info("\n--- Testing Phase Encoding ---")
    
    # Generate random phases
    n_oscillators = 6
    phases = np.random.random(n_oscillators) * 2 * np.pi
    logger.info(f"Generated {n_oscillators} random phases")
    
    # Create Hamiltonian from phases
    try:
        H = phases_to_hamiltonian(phases)
        logger.info(f"Created Hamiltonian with shape {H.shape}")
        
        # Check if Hamiltonian is Hermitian
        is_hermitian = np.allclose(H, H.conj().T)
        logger.info(f"Hamiltonian is Hermitian: {is_hermitian}")
        
        # Success
        logger.info("Phase encoding test successful")
        return H
    except Exception as e:
        logger.error(f"Error in phase encoding test: {e}")
        raise

def test_entangled_attention():
    """Test entangled attention mechanism."""
    logger.info("\n--- Testing Entangled Attention ---")
    
    # Create entangled attention with small number of qubits
    n_qubits = 4
    depth = 2
    attention = EntangledAttention(n_qubits=n_qubits, depth=depth)
    logger.info(f"Created EntangledAttention with {n_qubits} qubits and depth {depth}")
    
    # Generate random phases and spins
    phases = np.random.random(n_qubits) * 2 * np.pi
    spins = np.random.choice([-1, 1], size=n_qubits)
    
    # Map to qubit representation
    qubit_phases = attention.phase_to_qubit(phases)
    qubit_spins = attention.spin_to_qubit(spins)
    
    logger.info(f"Mapped phases to qubit representation: {qubit_phases}")
    logger.info(f"Mapped spins to qubit representation: {qubit_spins}")
    
    # Get attention weights for a feature vector
    features = np.random.random(n_qubits)
    weights = attention.get_attention_weights(features)
    
    logger.info(f"Attention weights: {weights}")
    logger.info(f"Sum of weights: {np.sum(weights)}")  # Should be close to 1.0
    
    return attention

def performance_test():
    """Test performance of quantum reservoir for different qubit counts."""
    logger.info("\n--- Performance Test ---")
    
    qubit_counts = [2, 4, 6]  # Small counts for testing
    times = []
    
    for n_qubits in qubit_counts:
        logger.info(f"Testing with {n_qubits} qubits...")
        
        # Create QR
        start_time = time.time()
        qr = QR(n_qubits=n_qubits)
        
        # Create random phases and convert to Hamiltonian
        phases = np.random.random(n_qubits) * 2 * np.pi
        H = phases_to_hamiltonian(phases)
        
        # Perform steps
        for _ in range(3):
            qr.step(H, dt=0.1)
        
        # Get expectations
        _ = qr.get_expectation_values()
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        logger.info(f"Completed in {elapsed:.4f} seconds")
    
    # Log scaling behavior
    logger.info(f"Qubit counts: {qubit_counts}")
    logger.info(f"Times (s): {[f'{t:.4f}' for t in times]}")
    
    return qubit_counts, times

def main():
    """Main test function."""
    logger.info("Starting quantum reservoir tests")
    
    try:
        # Run basic tests
        qr = test_quantum_reservoir_basic()
        H = test_phase_encoding()
        attention = test_entangled_attention()
        
        # Optional performance test (can be slow for larger qubit counts)
        if os.environ.get("RUN_PERFORMANCE_TEST", "false").lower() == "true":
            performance_test()
        
        logger.info("\n--- All tests completed successfully ---")
        return 0
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
