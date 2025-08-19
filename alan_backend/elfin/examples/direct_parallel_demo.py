#!/usr/bin/env python3
"""
Standalone demo of the ParallelVerifier and DepGraph components.

This script demonstrates how to use the parallel verifier with the dependency
graph to efficiently verify multiple systems in parallel, skipping proofs that
are already verified.
"""

import os
import logging
import pathlib
import sys
import time
from typing import Dict, List, Any

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("elfin.demo")

# Import local modules
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

# Direct imports without going through the main __init__.py
from alan_backend.elfin.stability.verify.dep_graph import DepGraph
from alan_backend.elfin.stability.verify.parallel_verifier import ParallelVerifier
from alan_backend.elfin.stability.verify.milp_verifier import VerificationResult, VerificationStatus
from alan_backend.elfin.stability.core.interactions import Interaction, InteractionLog


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


class SimpleAgent:
    """A simplified agent for demonstration."""
    
    def __init__(self, name: str, cache_dir: pathlib.Path):
        """Initialize a simple agent."""
        self.name = name
        self.cache_dir = cache_dir
        self.dep_graph = DepGraph()
        self.parallel_verifier = ParallelVerifier(self.dep_graph)
        self.log = []
        
    def mark_entity_dirty(self, entity_id: str) -> None:
        """Mark an entity as dirty."""
        self.dep_graph.mark_dirty(entity_id)
        logger.debug(f"Marked entity {entity_id} as dirty")
        
    def verify_many(self, systems: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Verify many systems in parallel."""
        results = {}
        
        def on_done(result: Dict[str, Any]) -> None:
            results[result["hash"]] = result
            self.log.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "action": "verify",
                "hash": result["hash"],
                "result": result
            })
        
        # Run verification in parallel
        processed = self.parallel_verifier.verify_many(systems, on_done)
        
        # Log batch result
        self.log.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "action": "verify_many",
            "processed_count": len(processed),
            "total_count": len(systems),
            "skipped_count": len(systems) - len(processed)
        })
        
        return results
    
    def get_summary(self, tail: int = None) -> str:
        """Get a summary of the log."""
        log = self.log
        
        if tail is not None:
            log = log[-tail:]
        
        if not log:
            return f"No entries in log for agent '{self.name}'"
        
        lines = [f"Log for agent '{self.name}' ({len(log)} entries):"]
        
        for entry in log:
            timestamp = entry["timestamp"]
            action = entry["action"].ljust(15)
            
            if action.startswith("verify_many"):
                processed = entry.get("processed_count", 0)
                total = entry.get("total_count", 0)
                skipped = entry.get("skipped_count", 0)
                line = f"[{timestamp}] {action} processed={processed}/{total} skipped={skipped}"
            elif action.startswith("verify"):
                hash_value = entry.get("hash", "")
                result = entry.get("result", {})
                status = result.get("status", "UNKNOWN")
                line = f"[{timestamp}] {action} hash={hash_value} status={status}"
            else:
                line = f"[{timestamp}] {action}"
                
            lines.append(line)
            
        return "\n".join(lines)


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
    
    # Create a SimpleAgent
    print("Creating SimpleAgent...")
    agent = SimpleAgent("parallel_demo", cache_dir)
    
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
    
    # ==== Scenario 4: View log ====
    print("\n[Scenario 4] View log")
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
    demonstrate_parallel_verifier()
