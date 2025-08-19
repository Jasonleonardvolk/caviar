"""
Parallel verification engine for stability properties.

This module provides a parallel verification engine that uses multiple
processes to verify stability properties across multiple CPU cores. It
integrates with the dependency graph to skip proofs that haven't been
affected by changes.
"""

import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Callable, Any, Optional, Set

# Import the dependency graph
from .dep_graph import DepGraph

# We'll import these later when we create them
# from .milp_verifier import MILPVerifier


class ParallelVerifier:
    """
    Distribute verification jobs across CPU cores, skip clean proofs.
    
    This class uses a process pool to distribute verification jobs across
    multiple CPU cores. It integrates with the dependency graph to skip
    proofs that haven't been affected by changes.
    
    Attributes:
        dep: Dependency graph for tracking proof dependencies
        pool: Process pool for distributing jobs
    """

    def __init__(self, dep_graph: DepGraph, max_workers: Optional[int] = None):
        """
        Initialize a parallel verifier.
        
        Args:
            dep_graph: Dependency graph for tracking proof dependencies
            max_workers: Maximum number of worker processes (default: CPU count)
        """
        self.dep = dep_graph
        self.pool = ProcessPoolExecutor(max_workers or os.cpu_count())

    # -------- public ----------------------------------------------------
    def verify_many(self,
                    jobs: List[Dict[str, Any]],             # [{lyap, system, domain, hash, entities}]
                    on_done: Callable[[Dict[str, Any]], None]) -> Set[str]:   # callback from main process
        """
        Submit dirty jobs and return results as they complete.
        
        Args:
            jobs: List of verification jobs, each with system, domain, hash, and entities
            on_done: Callback function to handle each job result
            
        Returns:
            Set of proof hashes that were processed
        """
        futures = {}
        processed_hashes = set()
        
        # Only submit dirty jobs
        for job in jobs:
            if not self.dep.is_dirty(job["hash"]):
                continue
                
            processed_hashes.add(job["hash"])
            future = self.pool.submit(_worker, job)
            futures[future] = job["hash"]
        
        # Process results as they complete
        for future in as_completed(futures):
            result = future.result()
            on_done(result)
        
        return processed_hashes
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the process pool.
        
        This method should be called when the verifier is no longer needed
        to ensure graceful shutdown of worker processes.
        
        Args:
            wait: Whether to wait for pending jobs to complete (default: True)
        """
        self.pool.shutdown(wait=wait)


def _worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function to run in a separate process.
    
    This function performs the actual verification work in a separate process.
    
    Args:
        job: Verification job with system, domain, hash, and entities
        
    Returns:
        Dictionary with verification result information
    """
    try:
        start = time.perf_counter()
        
        # Import here to avoid circular imports
        from .milp_verifier import MILPVerifier
        
        # Run verification
        res = MILPVerifier(job["system"], job["domain"]).run()
        
        # Process result
        status = res.status.name
        counterexample = res.counterexample.tolist() if res.counterexample is not None else None
        
        return {
            "hash": job["hash"],
            "status": status,
            "solve_time": time.perf_counter() - start,
            "counterexample": counterexample
        }
    except Exception as e:
        # Handle exceptions
        return {
            "hash": job["hash"],
            "status": "ERROR",
            "error": str(e),
            "tb": traceback.format_exc()
        }


def collect_dirty_jobs(dep_graph: DepGraph, systems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collect dirty verification jobs.
    
    This helper function collects verification jobs for all dirty proofs
    in the dependency graph.
    
    Args:
        dep_graph: Dependency graph for tracking proof dependencies
        systems: List of systems to verify, each with system, domain, hash, and entities
        
    Returns:
        List of verification jobs for dirty proofs
    """
    dirty_proofs = dep_graph.dirty_proofs()
    return [s for s in systems if s["hash"] in dirty_proofs]
