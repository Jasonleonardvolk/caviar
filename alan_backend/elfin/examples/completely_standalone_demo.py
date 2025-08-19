#!/usr/bin/env python3
"""
Completely standalone demo of dependency tracking and parallel verification.

This script demonstrates a dependency tracking system and parallel verification
without relying on any imports from the ELFIN framework.
"""

import os
import logging
import pathlib
import time
from typing import Dict, List, Set, Any, Optional, Iterable, Callable
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("elfin.demo")


# ==========================================================================
# DepGraph component (simplified version)
# ==========================================================================
class DepGraph:
    """
    Maps proof hashes ⇆ entity IDs.
    
    The dependency graph allows for quick determination of which proofs
    are *dirty* (need to be reverified) after an entity changes.
    """

    def __init__(self) -> None:
        """Initialize an empty dependency graph."""
        self.parents = defaultdict(set)   # proof → {entity}
        self.children = defaultdict(set)  # entity → {proof}
        self._dirty = set()

    def add_edge(self, proof_hash: str, entities: Iterable[str]) -> None:
        """Register a dependency between a proof and one or more entities."""
        for e in entities:
            self.parents[proof_hash].add(e)
            self.children[e].add(proof_hash)

    def mark_dirty(self, entity_id: str) -> None:
        """Mark all proofs that depend on an entity as dirty."""
        for proof in self.children.get(entity_id, ()):
            self._dirty.add(proof)

    def mark_fresh(self, proof_hash: str) -> None:
        """Mark a proof as fresh (not dirty)."""
        self._dirty.discard(proof_hash)

    def is_dirty(self, proof_hash: str) -> bool:
        """Check if a proof is dirty."""
        return proof_hash in self._dirty

    def dirty_proofs(self) -> Set[str]:
        """Get all dirty proofs."""
        return set(self._dirty)


# ==========================================================================
# ParallelVerifier component (simplified version)
# ==========================================================================
class ParallelVerifier:
    """Fan-out verification jobs across CPU cores, skip clean proofs."""

    def __init__(self, dep_graph: DepGraph, max_workers: Optional[int] = None):
        """Initialize a parallel verifier."""
        self.dep = dep_graph
        self.pool = ProcessPoolExecutor(max_workers or os.cpu_count())

    def verify_many(self,
                    jobs: List[Dict[str, Any]],
                    on_done: Callable[[Dict[str, Any]], None]) -> Set[str]:
        """Submit dirty jobs and process results as they complete."""
        futures = {}
        processed_hashes = set()
        
        # Only submit dirty jobs
        for job in jobs:
            # Force submission for demo purposes if no dirty jobs are found
            if not self.dep.is_dirty(job["hash"]):
                print(f"Skipping clean job: {job['hash']}")
                continue
            
            print(f"Submitting dirty job: {job['hash']}")    
            processed_hashes.add(job["hash"])
            future = self.pool.submit(_worker, job)
            futures[future] = job["hash"]
        
        # Process results as they complete
        for future in as_completed(futures):
            result = future.result()
            on_done(result)
        
        return processed_hashes
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the process pool."""
        self.pool.shutdown(wait=wait)


def _worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function to run in a separate process."""
    try:
        start = time.perf_counter()
        
        # Simulate verification work
        time.sleep(0.05)  # Simulate some processing time
        
        # Mock verification result (always successful for demo)
        return {
            "hash": job["hash"],
            "status": "VERIFIED",
            "solve_time": time.perf_counter() - start,
            "counterexample": None
        }
    except Exception as e:
        # Handle exceptions
        import traceback
        return {
            "hash": job["hash"],
            "status": "ERROR",
            "error": str(e),
            "tb": traceback.format_exc()
        }


# ==========================================================================
# Demo Components
# ==========================================================================
class SimpleSystem:
    """A simple system for demonstration."""
    
    def __init__(self, id: str, hash_value: str):
        """Initialize a simple system."""
        self.id = id
        self.hash = hash_value


def generate_test_systems(count: int) -> List[Dict[str, Any]]:
    """Generate test systems for demonstration."""
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
        
        # Register dependencies for all systems
        for job in systems:
            self.dep_graph.add_edge(job["hash"], job["entities"])
        
        def on_done(result: Dict[str, Any]) -> None:
            results[result["hash"]] = result
            # Mark the proof as fresh when verified
            self.dep_graph.mark_fresh(result["hash"])
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


# ==========================================================================
# Main Demo Function
# ==========================================================================
def demonstrate_parallel_verifier():
    """Demonstrate the parallel verifier and dependency graph."""
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
    if elapsed > 0:
        print(f"Systems per second: {len(results) / elapsed:.2f}")
    else:
        print("Systems per second: N/A (elapsed time too small)")
    
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
    try:
        import shutil
        shutil.rmtree(cache_dir)
    except Exception as e:
        print(f"Warning: Could not clean up directory {cache_dir}: {e}")
    
    print("\n" + "="*80)
    print(" Demo Complete ".center(80, "="))
    print("="*80 + "\n")


if __name__ == "__main__":
    demonstrate_parallel_verifier()
