#!/usr/bin/env python3
"""
Demo of the ParallelVerifier and DepGraph components.

This script demonstrates how to use the parallel verifier with the dependency
graph to efficiently verify multiple systems in parallel, skipping proofs that
are already verified.
"""

import os
import logging
import pathlib
import time
from typing import Dict, List, Any

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
if str(project_root) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    import sys
    sys.path.insert(0, str(project_root))

# Import ELFIN components
from alan_backend.elfin.stability.agents import StabilityAgent
from alan_backend.elfin.stability.verify import DepGraph, ParallelVerifier
from alan_backend.elfin.stability.core import Interaction


class SimpleSystem:
    """
    A simple system for demonstration.
    
    This class represents a system with a simple ID and a hash value.
    """
    
    def __init__(self, id: str, hash_value: str):
        """
        Initialize a simple system.
        
        Args:
            id: ID of the system
            hash_value: Hash value for the proof
        """
        self.id = id
        self.hash = hash_value


def generate_test_systems(count: int) -> List[Dict[str, Any]]:
    """
    Generate test systems for demonstration.
    
    Args:
        count: Number of systems to generate
        
    Returns:
        List of verification jobs
    """
    systems = []
    
    for i in range(count):
        system_id = f"system_{i}"
        hash_value = f"proof_{i}"
        system = SimpleSystem(system_id, hash_value)
        
        # Create a verification job
        job = {
            "system": system,
            "domain": (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
            "hash": hash_value,
            "system_id": system_id,
            "entities": [system_id]
        }
        
        systems.append(job)
    
    return systems


def demonstrate_parallel_verifier():
    """
    Demonstrate the parallel verifier and dependency graph.
    
    This function shows:
    1. Creating a dependency graph
    2. Creating a parallel verifier
    3. Verifying systems in parallel
    4. Skipping already verified systems
    """
    # Create a temporary directory for cache
    cache_dir = pathlib.Path("temp_cache")
    cache_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print(" ELFIN ParallelVerifier Demo ".center(80, "="))
    print("="*80 + "\n")
    
    # Create a StabilityAgent
    print("Creating StabilityAgent...")
    agent = StabilityAgent("parallel_demo", cache_dir)
    
    # Generate test systems
    num_systems = 10
    print(f"Generating {num_systems} test systems...")
    systems = generate_test_systems(num_systems)
    
    # ==== Scenario 1: Verify all systems ====
    print("\n[Scenario 1] Verifying all systems")
    print("-" * 50)
    
    # Mark all systems as dirty
    for job in systems:
        agent.mark_entity_dirty(job["system_id"])
    
    # Time the verification
    start_time = time.time()
    
    # Verify all systems
    results = agent.verify_many(systems)
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    
    print(f"Verified {len(results)} systems in {elapsed:.2f} seconds")
    print(f"Systems per second: {len(results) / elapsed:.2f}")
    
    # ==== Scenario 2: Verify again, should skip all ====
    print("\n[Scenario 2] Verifying again, should skip all")
    print("-" * 50)
    
    # Time the verification
    start_time = time.time()
    
    # Verify all systems again
    results = agent.verify_many(systems)
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    
    print(f"Processed {len(results)} systems in {elapsed:.2f} seconds")
    print(f"All systems were skipped, already verified")
    
    # ==== Scenario 3: Mark some as dirty ====
    print("\n[Scenario 3] Mark some systems as dirty")
    print("-" * 50)
    
    # Mark some systems as dirty
    for i in range(0, num_systems, 2):
        agent.mark_entity_dirty(f"system_{i}")
    
    # Time the verification
    start_time = time.time()
    
    # Verify all systems again
    results = agent.verify_many(systems)
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    
    print(f"Processed {len(results)} systems in {elapsed:.2f} seconds")
    print(f"Half of the systems were re-verified, half were skipped")
    
    # ==== Scenario 4: View interaction log ====
    print("\n[Scenario 4] View interaction log")
    print("-" * 50)
    
    # Get log summary
    summary = agent.get_summary(tail=10)
    print(summary)
    
    # ==== Cleanup ====
    print("\nCleaning up...")
    import shutil
    shutil.rmtree(cache_dir)
    
    print("\n" + "="*80)
    print(" Demo Complete ".center(80, "="))
    print("="*80 + "\n")


if __name__ == "__main__":
    demonstrate_parallel_verifier()
