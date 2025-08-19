"""
Incremental Verification Engine for ELFIN Stability Framework.

This module provides an incremental verification engine with proof caching,
dependency tracking, and parallel execution capabilities. When concepts or
dynamics change, only the affected proofs need to be re-verified.

The core components are:
1. DepGraph - Tracks dependencies between concepts and proof objects
2. ProofCache - Caches verification results with hash-based indexing
3. VerificationResult - Structured result of verification with proof certificates
4. ParallelVerifier - Executes verification tasks in parallel with caching
"""

import os
import logging
import time
import json
import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from enum import Enum, auto
from dataclasses import dataclass, field

try:
    from alan_backend.elfin.stability.lyapunov import LyapunovFunction
    from alan_backend.elfin.stability.verifier import LyapunovVerifier, ProofStatus
except ImportError:
    # Minimal implementation for standalone testing
    class LyapunovFunction:
        def __init__(self, name, domain_ids=None):
            self.name = name
            self.domain_ids = domain_ids or []
            
        def evaluate(self, x):
            return float(sum(x**2))
    
    class ProofStatus(Enum):
        UNKNOWN = auto()
        VERIFIED = auto()
        REFUTED = auto()
        TIMEOUT = auto()
        ERROR = auto()
    
    class LyapunovVerifier:
        def verify(self, lyap, dynamics_fn=None):
            return ProofStatus.UNKNOWN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DepGraph:
    """
    Dependency graph for concepts and proofs.
    
    This class tracks dependencies between concepts and proofs, enabling
    incremental verification by determining which proofs need to be
    re-verified when concepts change.
    """
    
    def __init__(self):
        """Initialize empty dependency graph."""
        # Maps: node_id -> set of parent_ids (dependencies)
        self.dependencies = {}
        
        # Maps: node_id -> set of child_ids (dependents)
        self.dependents = {}
        
        # Node type mapping
        self.node_types = {}
        
        # Last hash value for each concept
        self.concept_hashes = {}
    
    def add_dependency(
        self,
        node_id: str,
        parent_id: str,
        node_type: str = "proof",
        parent_type: str = "concept"
    ):
        """
        Add dependency between nodes.
        
        Args:
            node_id: ID of dependent node
            parent_id: ID of dependency node
            node_type: Type of dependent node
            parent_type: Type of dependency node
        """
        # Ensure nodes exist
        if node_id not in self.dependencies:
            self.dependencies[node_id] = set()
            self.node_types[node_id] = node_type
        
        if parent_id not in self.dependents:
            self.dependents[parent_id] = set()
            self.node_types[parent_id] = parent_type
        
        # Add dependency
        self.dependencies[node_id].add(parent_id)
        
        # Add dependent
        self.dependents[parent_id].add(node_id)
        
        logger.debug(f"Added dependency: {node_id} -> {parent_id}")
    
    def remove_dependency(self, node_id: str, parent_id: str):
        """
        Remove dependency between nodes.
        
        Args:
            node_id: ID of dependent node
            parent_id: ID of dependency node
        """
        if node_id in self.dependencies and parent_id in self.dependencies[node_id]:
            self.dependencies[node_id].remove(parent_id)
            logger.debug(f"Removed dependency: {node_id} -> {parent_id}")
        
        if parent_id in self.dependents and node_id in self.dependents[parent_id]:
            self.dependents[parent_id].remove(node_id)
    
    def get_dependencies(self, node_id: str) -> Set[str]:
        """
        Get all dependencies of a node.
        
        Args:
            node_id: ID of node
            
        Returns:
            Set of dependency node IDs
        """
        return self.dependencies.get(node_id, set())
    
    def get_dependents(self, node_id: str) -> Set[str]:
        """
        Get all dependents of a node.
        
        Args:
            node_id: ID of node
            
        Returns:
            Set of dependent node IDs
        """
        return self.dependents.get(node_id, set())
    
    def get_affected_proofs(self, concept_ids: List[str]) -> Set[str]:
        """
        Get all proofs affected by changes to concepts.
        
        Args:
            concept_ids: List of concept IDs that changed
            
        Returns:
            Set of affected proof IDs
        """
        affected = set()
        
        for concept_id in concept_ids:
            # Add all direct dependents
            if concept_id in self.dependents:
                for dependent in self.dependents[concept_id]:
                    if self.node_types.get(dependent) == "proof":
                        affected.add(dependent)
                        
                    # Add indirect dependents (proof -> proof dependencies)
                    # In a recursive manner
                    proof_dependents = self._get_proof_dependents(dependent)
                    affected.update(proof_dependents)
        
        return affected
    
    def _get_proof_dependents(self, node_id: str) -> Set[str]:
        """
        Get all proof dependents recursively.
        
        Args:
            node_id: ID of node
            
        Returns:
            Set of dependent proof IDs
        """
        proof_dependents = set()
        
        if node_id in self.dependents:
            for dependent in self.dependents[node_id]:
                if self.node_types.get(dependent) == "proof":
                    proof_dependents.add(dependent)
                    
                    # Recursive call for nested dependencies
                    nested = self._get_proof_dependents(dependent)
                    proof_dependents.update(nested)
        
        return proof_dependents
    
    def diff_update(
        self,
        old_hashes: Dict[str, str],
        new_hashes: Dict[str, str]
    ) -> Set[str]:
        """
        Update concept hashes and get affected proofs.
        
        Args:
            old_hashes: Old concept hash values
            new_hashes: New concept hash values
            
        Returns:
            Set of affected proof IDs
        """
        changed_concepts = []
        
        # Find changed concepts
        for concept_id, new_hash in new_hashes.items():
            old_hash = old_hashes.get(concept_id)
            
            if old_hash != new_hash:
                changed_concepts.append(concept_id)
                logger.debug(f"Concept changed: {concept_id}")
                
        # Update stored hashes
        self.concept_hashes.update(new_hashes)
        
        # Get affected proofs
        affected = self.get_affected_proofs(changed_concepts)
        
        return affected
    
    def to_dict(self) -> Dict:
        """
        Convert graph to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "dependencies": {k: list(v) for k, v in self.dependencies.items()},
            "dependents": {k: list(v) for k, v in self.dependents.items()},
            "node_types": self.node_types,
            "concept_hashes": self.concept_hashes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DepGraph':
        """
        Create graph from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            DepGraph instance
        """
        graph = cls()
        
        graph.dependencies = {k: set(v) for k, v in data.get("dependencies", {}).items()}
        graph.dependents = {k: set(v) for k, v in data.get("dependents", {}).items()}
        graph.node_types = data.get("node_types", {})
        graph.concept_hashes = data.get("concept_hashes", {})
        
        return graph
    
    def save(self, path: str):
        """
        Save graph to file.
        
        Args:
            path: File path
        """
        data = self.to_dict()
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'DepGraph':
        """
        Load graph from file.
        
        Args:
            path: File path
            
        Returns:
            DepGraph instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
            
        return cls.from_dict(data)
    
    def clear(self):
        """Clear all graph data."""
        self.dependencies.clear()
        self.dependents.clear()
        self.node_types.clear()
        self.concept_hashes.clear()


@dataclass
class VerificationResult:
    """Result of Lyapunov function verification."""
    
    status: ProofStatus
    lyapunov_name: Optional[str] = None
    certificate: Optional[Any] = None
    counterexample: Optional[Any] = None
    verification_time: float = 0.0
    error_message: Optional[str] = None
    
    def is_verified(self) -> bool:
        """Check if proof was verified."""
        return self.status == ProofStatus.VERIFIED
    
    def is_refuted(self) -> bool:
        """Check if proof was refuted."""
        return self.status == ProofStatus.REFUTED
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "status": self.status.name,
            "lyapunov_name": self.lyapunov_name,
            "has_certificate": self.certificate is not None,
            "has_counterexample": self.counterexample is not None,
            "verification_time": self.verification_time,
            "error_message": self.error_message
        }


@dataclass
class ProofCertificate:
    """Certificate of verification proof."""
    
    proof_hash: str
    status: ProofStatus
    lyapunov_name: str
    certificate_data: Any = None
    counterexample: Any = None
    verification_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    @classmethod
    def from_result(
        cls, 
        proof_hash: str, 
        result: VerificationResult
    ) -> 'ProofCertificate':
        """Create from verification result."""
        return cls(
            proof_hash=proof_hash,
            status=result.status,
            lyapunov_name=result.lyapunov_name or "unknown",
            certificate_data=result.certificate,
            counterexample=result.counterexample,
            verification_time=result.verification_time
        )


class ProofCache:
    """
    Cache for verification proofs.
    
    This class provides caching of verification results to avoid
    re-verifying unchanged Lyapunov functions.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize proof cache.
        
        Args:
            cache_dir: Directory for persistent cache (None for in-memory only)
        """
        self.cache_dir = cache_dir
        
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        
        # In-memory cache: proof_hash -> ProofCertificate
        self.proofs = {}
        
        # Dependency graph
        self.dep_graph = DepGraph()
    
    def _make_proof_hash(
        self,
        lyap: LyapunovFunction,
        dynamics_fn: Optional[Callable] = None,
        context: Optional[Dict] = None
    ) -> str:
        """
        Compute hash for a verification task.
        
        Args:
            lyap: Lyapunov function
            dynamics_fn: Dynamics function
            context: Additional context
            
        Returns:
            Hash string
        """
        # Start with Lyapunov function name and domain IDs
        hash_parts = [
            lyap.name,
            ",".join(lyap.domain_ids or [])
        ]
        
        # Add dynamics function if available
        if dynamics_fn is not None:
            if hasattr(dynamics_fn, "__name__"):
                hash_parts.append(dynamics_fn.__name__)
            elif hasattr(dynamics_fn, "__class__"):
                hash_parts.append(dynamics_fn.__class__.__name__)
        
        # Add context if available
        if context is not None:
            for k, v in sorted(context.items()):
                hash_parts.append(f"{k}:{v}")
        
        # Compute hash
        hasher = hashlib.sha256()
        hasher.update(":".join(hash_parts).encode())
        return hasher.hexdigest()
    
    def has_proof(
        self,
        lyap: LyapunovFunction,
        dynamics_fn: Optional[Callable] = None,
        context: Optional[Dict] = None
    ) -> bool:
        """
        Check if proof is cached.
        
        Args:
            lyap: Lyapunov function
            dynamics_fn: Dynamics function
            context: Additional context
            
        Returns:
            Whether proof is cached
        """
        proof_hash = self._make_proof_hash(lyap, dynamics_fn, context)
        
        # Check in-memory cache
        if proof_hash in self.proofs:
            return True
        
        # Check persistent cache
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f"{proof_hash}.pickle")
            return os.path.exists(cache_path)
        
        return False
    
    def get_proof(
        self,
        lyap: LyapunovFunction,
        dynamics_fn: Optional[Callable] = None,
        context: Optional[Dict] = None
    ) -> Optional[ProofCertificate]:
        """
        Get cached proof if available.
        
        Args:
            lyap: Lyapunov function
            dynamics_fn: Dynamics function
            context: Additional context
            
        Returns:
            ProofCertificate or None if not found
        """
        proof_hash = self._make_proof_hash(lyap, dynamics_fn, context)
        
        # Check in-memory cache
        if proof_hash in self.proofs:
            return self.proofs[proof_hash]
        
        # Check persistent cache
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f"{proof_hash}.pickle")
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        certificate = pickle.load(f)
                        
                    # Add to in-memory cache
                    self.proofs[proof_hash] = certificate
                    
                    return certificate
                except Exception as e:
                    logger.warning(f"Error loading cached proof: {e}")
        
        return None
    
    def get_result(
        self,
        lyap: LyapunovFunction,
        dynamics_fn: Optional[Callable] = None,
        context: Optional[Dict] = None
    ) -> Optional[VerificationResult]:
        """
        Get cached verification result if available.
        
        Args:
            lyap: Lyapunov function
            dynamics_fn: Dynamics function
            context: Additional context
            
        Returns:
            VerificationResult or None if not found
        """
        certificate = self.get_proof(lyap, dynamics_fn, context)
        
        if certificate is not None:
            return VerificationResult(
                status=certificate.status,
                lyapunov_name=certificate.lyapunov_name,
                certificate=certificate.certificate_data,
                counterexample=certificate.counterexample,
                verification_time=certificate.verification_time
            )
        
        return None
    
    def add_proof(
        self,
        lyap: LyapunovFunction,
        result: VerificationResult,
        dynamics_fn: Optional[Callable] = None,
        context: Optional[Dict] = None
    ) -> str:
        """
        Add proof to cache.
        
        Args:
            lyap: Lyapunov function
            result: Verification result
            dynamics_fn: Dynamics function
            context: Additional context
            
        Returns:
            Proof hash
        """
        proof_hash = self._make_proof_hash(lyap, dynamics_fn, context)
        
        # Create certificate
        certificate = ProofCertificate.from_result(proof_hash, result)
        
        # Add to in-memory cache
        self.proofs[proof_hash] = certificate
        
        # Add to persistent cache
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f"{proof_hash}.pickle")
            
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(certificate, f)
            except Exception as e:
                logger.warning(f"Error saving proof: {e}")
        
        # Add dependencies to graph
        for domain_id in lyap.domain_ids or []:
            self.dep_graph.add_dependency(
                proof_hash, domain_id,
                node_type="proof", parent_type="concept"
            )
        
        return proof_hash
    
    def invalidate_proof(
        self,
        lyap: LyapunovFunction,
        dynamics_fn: Optional[Callable] = None,
        context: Optional[Dict] = None
    ):
        """
        Invalidate cached proof.
        
        Args:
            lyap: Lyapunov function
            dynamics_fn: Dynamics function
            context: Additional context
        """
        proof_hash = self._make_proof_hash(lyap, dynamics_fn, context)
        
        # Remove from in-memory cache
        if proof_hash in self.proofs:
            del self.proofs[proof_hash]
        
        # Remove from persistent cache
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f"{proof_hash}.pickle")
            
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception as e:
                    logger.warning(f"Error removing proof: {e}")
    
    def clear(self):
        """Clear all cached proofs."""
        # Clear in-memory cache
        self.proofs.clear()
        
        # Clear persistent cache
        if self.cache_dir is not None:
            for file in os.listdir(self.cache_dir):
                if file.endswith(".pickle"):
                    try:
                        os.remove(os.path.join(self.cache_dir, file))
                    except Exception as e:
                        logger.warning(f"Error removing proof: {e}")
        
        # Clear dependency graph
        self.dep_graph.clear()


class ParallelVerifier:
    """
    Parallel verifier with proof caching.
    
    This class provides parallel execution of verification tasks
    with proof caching capabilities.
    """
    
    def __init__(
        self,
        verifier: LyapunovVerifier,
        cache: Optional[ProofCache] = None,
        max_workers: Optional[int] = None,
        timeout: float = 300.0
    ):
        """
        Initialize parallel verifier.
        
        Args:
            verifier: Lyapunov function verifier
            cache: Proof cache (None for no caching)
            max_workers: Maximum number of parallel workers
            timeout: Verification timeout
        """
        self.verifier = verifier
        self.cache = cache or ProofCache()
        self.max_workers = max_workers
        self.timeout = timeout
    
    def verify(
        self,
        lyap: LyapunovFunction,
        dynamics_fn: Optional[Callable] = None,
        context: Optional[Dict] = None,
        force: bool = False
    ) -> VerificationResult:
        """
        Verify Lyapunov function.
        
        Args:
            lyap: Lyapunov function
            dynamics_fn: Dynamics function
            context: Additional context
            force: Force re-verification
            
        Returns:
            Verification result
        """
        # Check cache if not forcing re-verification
        if not force and self.cache is not None:
            cached = self.cache.get_result(lyap, dynamics_fn, context)
            
            if cached is not None:
                logger.info(f"Using cached result for {lyap.name}")
                return cached
        
        # Perform verification
        logger.info(f"Verifying {lyap.name}")
        start_time = time.time()
        
        try:
            status = self.verifier.verify(lyap, dynamics_fn)
            
            # Get counterexample if available
            counterexample = None
            if hasattr(self.verifier, 'get_counterexample'):
                counterexample = self.verifier.get_counterexample()
            
            # Create result
            result = VerificationResult(
                status=status,
                lyapunov_name=lyap.name,
                certificate=None,  # TODO: Add proof certificate
                counterexample=counterexample,
                verification_time=time.time() - start_time
            )
            
            # Cache result
            if self.cache is not None:
                self.cache.add_proof(lyap, result, dynamics_fn, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            
            # Create error result
            result = VerificationResult(
                status=ProofStatus.ERROR,
                lyapunov_name=lyap.name,
                error_message=str(e),
                verification_time=time.time() - start_time
            )
            
            return result
    
    def verify_batch(
        self,
        tasks: List[Tuple[LyapunovFunction, Optional[Callable], Optional[Dict]]],
        force: bool = False,
        show_progress: bool = False
    ) -> Dict[str, VerificationResult]:
        """
        Verify batch of Lyapunov functions in parallel.
        
        Args:
            tasks: List of (lyap, dynamics_fn, context) tuples
            force: Force re-verification
            show_progress: Show progress
            
        Returns:
            Dictionary of proof_hash -> VerificationResult
        """
        results = {}
        
        if self.max_workers is None or self.max_workers <= 1:
            # Sequential verification
            for i, (lyap, dynamics_fn, context) in enumerate(tasks):
                if show_progress:
                    logger.info(f"Verifying {i+1}/{len(tasks)}: {lyap.name}")
                
                result = self.verify(lyap, dynamics_fn, context, force)
                proof_hash = self.cache._make_proof_hash(lyap, dynamics_fn, context)
                results[proof_hash] = result
                
                if show_progress:
                    logger.info(f"  Status: {result.status.name}")
        else:
            # Parallel verification
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {}
                
                # Submit tasks
                for lyap, dynamics_fn, context in tasks:
                    # Check cache first
                    if not force and self.cache is not None:
                        cached = self.cache.get_result(lyap, dynamics_fn, context)
                        
                        if cached is not None:
                            proof_hash = self.cache._make_proof_hash(
                                lyap, dynamics_fn, context
                            )
                            results[proof_hash] = cached
                            
                            if show_progress:
                                logger.info(
                                    f"Using cached result for {lyap.name}: "
                                    f"{cached.status.name}"
                                )
                            
                            continue
                    
                    # Submit for verification
                    future = executor.submit(
                        self._verify_task, lyap, dynamics_fn, context
                    )
                    future_to_task[future] = (lyap, dynamics_fn, context)
                
                # Process results
                total = len(future_to_task)
                completed = 0
                
                for future in as_completed(future_to_task):
                    lyap, dynamics_fn, context = future_to_task[future]
                    completed += 1
                    
                    if show_progress:
                        logger.info(f"Completed {completed}/{total}: {lyap.name}")
                    
                    try:
                        result = future.result()
                        
                        # Cache result
                        if self.cache is not None:
                            proof_hash = self.cache.add_proof(
                                lyap, result, dynamics_fn, context
                            )
                            results[proof_hash] = result
                        
                        if show_progress:
                            logger.info(f"  Status: {result.status.name}")
                    except Exception as e:
                        logger.error(f"Task error: {e}")
                        
                        # Create error result
                        result = VerificationResult(
                            status=ProofStatus.ERROR,
                            lyapunov_name=lyap.name,
                            error_message=str(e)
                        )
                        
                        proof_hash = self.cache._make_proof_hash(
                            lyap, dynamics_fn, context
                        )
                        results[proof_hash] = result
        
        return results
    
    def _verify_task(
        self,
        lyap: LyapunovFunction,
        dynamics_fn: Optional[Callable] = None,
        context: Optional[Dict] = None
    ) -> VerificationResult:
        """
        Verify a single task.
        
        Args:
            lyap: Lyapunov function
            dynamics_fn: Dynamics function
            context: Additional context
            
        Returns:
            Verification result
        """
        start_time = time.time()
        
        try:
            status = self.verifier.verify(lyap, dynamics_fn)
            
            # Create result
            result = VerificationResult(
                status=status,
                lyapunov_name=lyap.name,
                verification_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            
            # Create error result
            result = VerificationResult(
                status=ProofStatus.ERROR,
                lyapunov_name=lyap.name,
                error_message=str(e),
                verification_time=time.time() - start_time
            )
            
            return result


def run_demo():
    """Run a simple demonstration of incremental verification."""
    import numpy as np
    import time
    
    # Define a simple system
    A = np.array([
        [0.9, 0.2],
        [-0.1, 0.8]
    ])
    
    def linear_system(x):
        return A @ x
    
    # Define a simple Lyapunov function
    class QuadraticLyapunov(LyapunovFunction):
        def __init__(self, name, Q, domain_ids=None):
            super().__init__(name, domain_ids)
            self.Q = Q
            
        def evaluate(self, x):
            x = np.asarray(x).flatten()
            return float(x.T @ self.Q @ x)
    
    # Define a simple verifier
    class SimpleVerifier(LyapunovVerifier):
        def verify(self, lyap, dynamics_fn=None):
            time.sleep(0.1)  # Simulate verification time
            return ProofStatus.VERIFIED
    
    # Create cache
    cache = ProofCache(cache_dir="./proof_cache")
    
    # Create verifier
    verifier = SimpleVerifier()
    
    # Create parallel verifier
    parallel_verifier = ParallelVerifier(
        verifier=verifier,
        cache=cache,
        max_workers=4
    )
    
    # Create Lyapunov functions
    lyap_functions = []
    
    for i in range(5):
        Q = np.array([
            [1.0 + 0.1 * i, 0.0],
            [0.0, 1.0 + 0.2 * i]
        ])
        lyap = QuadraticLyapunov(
            f"V_{i}", Q, domain_ids=["x1", "x2"]
        )
        lyap_functions.append(lyap)
    
    # Verify batch
    tasks = [(lyap, linear_system, None) for lyap in lyap_functions]
    
    print("First verification (cold cache):")
    start_time = time.time()
    results = parallel_verifier.verify_batch(tasks, show_progress=True)
    print(f"  Time: {time.time() - start_time:.4f}s")
    
    # Verify again with cache
    print("\nSecond verification (warm cache):")
    start_time = time.time()
    results = parallel_verifier.verify_batch(tasks, show_progress=True)
    print(f"  Time: {time.time() - start_time:.4f}s")
    
    # Modify one function and verify again
    Q_new = np.array([
        [1.5, 0.3],
        [0.3, 1.8]
    ])
    lyap_new = QuadraticLyapunov(
        "V_2", Q_new, domain_ids=["x1", "x2"]
    )
    
    tasks[2] = (lyap_new, linear_system, None)
    
    print("\nThird verification (partial cache):")
    start_time = time.time()
    results = parallel_verifier.verify_batch(tasks, show_progress=True)
    print(f"  Time: {time.time() - start_time:.4f}s")
    
    # Demonstrate dependency tracking
    print("\nDemonstrating dependency tracking:")
    
    # Add dependencies to graph
    for i, lyap in enumerate(lyap_functions):
        proof_hash = cache._make_proof_hash(lyap, linear_system, None)
        
        for domain_id in lyap.domain_ids or []:
            cache.dep_graph.add_dependency(
                proof_hash, domain_id,
                node_type="proof", parent_type="concept"
            )
    
    # Simulate a change to a concept
    old_hashes = {"x1": "hash1", "x2": "hash2"}
    new_hashes = {"x1": "hash1_changed", "x2": "hash2"}
    
    affected = cache.dep_graph.diff_update(old_hashes, new_hashes)
    print(f"  After changing concept x1, {len(affected)} proofs need reverification")
    print(f"  Affected proofs: {', '.join(affected)}")
    
    print("\nDependency graph structure:")
    for node_id, deps in cache.dep_graph.dependencies.items():
        if deps:
            print(f"  {node_id} depends on: {', '.join(deps)}")


if __name__ == "__main__":
    run_demo()   
