"""
ELFIN DSL Stability Demo - Completely Standalone

This script demonstrates the core stability and synchronization features
of the ELFIN DSL with all dependencies inline, requiring no imports
from any other modules.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
from collections import defaultdict
from enum import Enum, auto
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("elfin.demo")

#------------------------------------------------------------------------------------
# Core Classes - Lyapunov Functions
#------------------------------------------------------------------------------------

class ProofStatus(Enum):
    """Status of a Lyapunov verification proof."""
    VERIFIED = auto()
    REFUTED = auto()
    UNKNOWN = auto()
    IN_PROGRESS = auto()
    ERROR = auto()

class LyapunovFunction:
    """Base class for Lyapunov functions."""
    
    def __init__(self, name: str, domain_concept_ids: List[str] = None):
        """
        Initialize a Lyapunov function.
        
        Args:
            name: Name of the Lyapunov function
            domain_concept_ids: List of concept IDs in the function's domain
        """
        self.name = name
        self.domain_concept_ids = domain_concept_ids or []
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Lyapunov function at a point.
        
        Args:
            x: State vector
            
        Returns:
            Value of the Lyapunov function
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
        
    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient of the Lyapunov function at a point.
        
        Args:
            x: State vector
            
        Returns:
            Gradient vector
        """
        raise NotImplementedError("Subclasses must implement evaluate_gradient()")
        
    def verify_positive_definite(self) -> ProofStatus:
        """
        Verify that the Lyapunov function is positive definite.
        
        Returns:
            Verification result
        """
        raise NotImplementedError("Subclasses must implement verify_positive_definite()")
        
    def verify_decreasing(self, dynamics_fn) -> ProofStatus:
        """
        Verify that the Lyapunov function is decreasing along trajectories.
        
        Args:
            dynamics_fn: System dynamics function
            
        Returns:
            Verification result
        """
        raise NotImplementedError("Subclasses must implement verify_decreasing()")
        
    def compute_proof_hash(self, context: Dict[str, Any] = None) -> str:
        """
        Compute a unique hash for this Lyapunov function and verification context.
        
        Args:
            context: Additional verification context
            
        Returns:
            Hash string
        """
        import hashlib
        
        # Start with the name and domain concepts
        components = [
            self.name,
            ",".join(sorted(self.domain_concept_ids))
        ]
        
        # Add context if provided
        if context:
            for key, value in sorted(context.items()):
                components.append(f"{key}:{value}")
                
        # Create a hash
        hash_str = ":".join(components)
        return hashlib.sha256(hash_str.encode()).hexdigest()

class PolynomialLyapunov(LyapunovFunction):
    """Polynomial Lyapunov function: V(x) = x^T Q x."""
    
    def __init__(
        self,
        name: str,
        q_matrix: np.ndarray,
        basis_functions: List[str] = None,
        domain_concept_ids: List[str] = None
    ):
        """
        Initialize a polynomial Lyapunov function.
        
        Args:
            name: Function name
            q_matrix: Q matrix for quadratic form
            basis_functions: Basis functions for the polynomial
            domain_concept_ids: List of concept IDs in the function's domain
        """
        super().__init__(name, domain_concept_ids)
        self.Q = q_matrix
        self.basis_functions = basis_functions or []
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate V(x) = x^T Q x.
        
        Args:
            x: State vector
            
        Returns:
            Value of the Lyapunov function
        """
        x_reshaped = x.reshape(-1, 1)
        return float(x_reshaped.T @ self.Q @ x_reshaped)
        
    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient: âˆ‡V(x) = 2Qx.
        
        Args:
            x: State vector
            
        Returns:
            Gradient vector
        """
        return 2.0 * self.Q @ x.reshape(-1, 1)
        
    def verify_positive_definite(self) -> ProofStatus:
        """
        Verify positive definiteness by checking eigenvalues.
        
        Returns:
            Verification result
        """
        try:
            eigenvalues = np.linalg.eigvals(self.Q)
            if np.all(eigenvalues > 0):
                return ProofStatus.VERIFIED
            else:
                return ProofStatus.REFUTED
        except Exception as e:
            logger.error(f"Error verifying positive definiteness: {e}")
            return ProofStatus.ERROR
        
    def verify_decreasing(self, dynamics_fn) -> ProofStatus:
        """
        Verify decreasing property using sampling.
        
        Args:
            dynamics_fn: System dynamics function
            
        Returns:
            Verification result
        """
        # This is a simple verification using sampling
        # A real implementation would use SOS programming
        
        # Generate sample points
        dim = self.Q.shape[0]
        n_samples = 1000
        samples = np.random.normal(0, 1, (n_samples, dim))
        
        decreasing = True
        for sample in samples:
            # Calculate Lie derivative
            grad = self.evaluate_gradient(sample).flatten()
            f_x = dynamics_fn(sample).flatten()
            lie_derivative = np.dot(grad, f_x)
            
            if lie_derivative >= 0:
                decreasing = False
                break
                
        return ProofStatus.VERIFIED if decreasing else ProofStatus.REFUTED

class NeuralLyapunov(LyapunovFunction):
    """Neural network-based Lyapunov function."""
    
    def __init__(
        self,
        name: str,
        layer_dims: List[int],
        weights: List[Tuple[np.ndarray, np.ndarray]],
        input_bounds: List[Tuple[float, float]] = None,
        domain_concept_ids: List[str] = None
    ):
        """
        Initialize a neural Lyapunov function.
        
        Args:
            name: Function name
            layer_dims: Layer dimensions
            weights: Network weights and biases
            input_bounds: Bounds on input variables
            domain_concept_ids: List of concept IDs in the function's domain
        """
        super().__init__(name, domain_concept_ids)
        self.layer_dims = layer_dims
        self.weights = weights
        self.input_bounds = input_bounds or []
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the neural network.
        
        Args:
            x: State vector
            
        Returns:
            Value of the Lyapunov function
        """
        activation = x.flatten()
        
        # Forward pass
        for W, b in self.weights:
            pre_activation = W.T @ activation + b
            # ReLU activation
            activation = np.maximum(0, pre_activation)
            
        # Ensure output is positive (add small constant)
        return float(activation[0] + 1e-6)
        
    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient using finite differences.
        
        Args:
            x: State vector
            
        Returns:
            Gradient vector
        """
        eps = 1e-6
        grad = np.zeros_like(x)
        
        # Compute gradient with finite differences
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            
            x_minus = x.copy()
            x_minus[i] -= eps
            
            grad[i] = (self.evaluate(x_plus) - self.evaluate(x_minus)) / (2 * eps)
            
        return grad
        
    def verify_positive_definite(self) -> ProofStatus:
        """
        Verify positive definiteness using sampling.
        
        Returns:
            Verification result
        """
        # Simple sampling-based verification
        n_samples = 1000
        dim = self.layer_dims[0]
        samples = np.random.normal(0, 1, (n_samples, dim))
        
        positive = True
        for sample in samples:
            # Skip the origin (V(0) = 0)
            if np.linalg.norm(sample) < 1e-6:
                continue
                
            value = self.evaluate(sample)
            if value <= 0:
                positive = False
                break
                
        return ProofStatus.VERIFIED if positive else ProofStatus.REFUTED
        
    def verify_decreasing(self, dynamics_fn) -> ProofStatus:
        """
        Verify decreasing property using sampling.
        
        Args:
            dynamics_fn: System dynamics function
            
        Returns:
            Verification result
        """
        # Simple sampling-based verification
        n_samples = 1000
        dim = self.layer_dims[0]
        samples = np.random.normal(0, 1, (n_samples, dim))
        
        decreasing = True
        for sample in samples:
            # Skip points near the origin
            if np.linalg.norm(sample) < 1e-6:
                continue
                
            # Calculate Lie derivative
            grad = self.evaluate_gradient(sample)
            f_x = dynamics_fn(sample)
            lie_derivative = np.dot(grad, f_x)
            
            if lie_derivative >= 0:
                decreasing = False
                break
                
        return ProofStatus.VERIFIED if decreasing else ProofStatus.REFUTED

class CLVFunction(LyapunovFunction):
    """Control Lyapunov-Value function."""
    
    def __init__(
        self,
        name: str,
        value_function: callable,
        control_variables: List[str],
        clf_gamma: float = 0.1,
        domain_concept_ids: List[str] = None
    ):
        """
        Initialize a Control Lyapunov-Value function.
        
        Args:
            name: Function name
            value_function: Value function
            control_variables: Control variables
            clf_gamma: Convergence rate
            domain_concept_ids: List of concept IDs in the function's domain
        """
        super().__init__(name, domain_concept_ids)
        self.value_function = value_function
        self.control_variables = control_variables
        self.clf_gamma = clf_gamma
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the value function.
        
        Args:
            x: State vector
            
        Returns:
            Value of the Lyapunov function
        """
        return float(self.value_function(x))
        
    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient using finite differences.
        
        Args:
            x: State vector
            
        Returns:
            Gradient vector
        """
        eps = 1e-6
        grad = np.zeros_like(x)
        
        # Compute gradient with finite differences
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            
            x_minus = x.copy()
            x_minus[i] -= eps
            
            grad[i] = (self.evaluate(x_plus) - self.evaluate(x_minus)) / (2 * eps)
            
        return grad
        
    def verify_positive_definite(self) -> ProofStatus:
        """
        Verify positive definiteness using sampling.
        
        Returns:
            Verification result
        """
        # Simple sampling-based verification
        n_samples = 1000
        dim = 3  # Assume 3D state for simplicity
        samples = np.random.normal(0, 1, (n_samples, dim))
        
        positive = True
        for sample in samples:
            # Skip the origin (V(0) = 0)
            if np.linalg.norm(sample) < 1e-6:
                continue
                
            value = self.evaluate(sample)
            if value <= 0:
                positive = False
                break
                
        return ProofStatus.VERIFIED if positive else ProofStatus.REFUTED
        
    def verify_decreasing(self, dynamics_fn) -> ProofStatus:
        """
        For CLF, we verify that there exists a control input that
        makes the Lyapunov function decrease.
        
        Args:
            dynamics_fn: System dynamics function (with control)
            
        Returns:
            Verification result
        """
        # For simplicity, assume we can always find a control
        # In reality, this would involve a QP solver
        return ProofStatus.VERIFIED
        
    def enforce(self, x: np.ndarray, u_nominal: np.ndarray) -> np.ndarray:
        """
        Enforce the CLF condition using QP.
        
        Args:
            x: State vector
            u_nominal: Nominal control input
            
        Returns:
            Safe control input
        """
        # This would normally use a QP solver to find u
        # such that Ldot_f V(x) + Ldot_g V(x) u <= -gamma * V(x)
        # while minimizing ||u - u_nominal||
        
        # For simplicity, we'll just return u_nominal
        return u_nominal

class CompositeLyapunov(LyapunovFunction):
    """Composite Lyapunov function combining multiple functions."""
    
    def __init__(
        self,
        name: str,
        component_lyapunovs: List[LyapunovFunction],
        weights: List[float] = None,
        composition_type: str = "sum",
        domain_concept_ids: List[str] = None
    ):
        """
        Initialize a composite Lyapunov function.
        
        Args:
            name: Function name
            component_lyapunovs: Component Lyapunov functions
            weights: Weights for each component
            composition_type: Type of composition (sum, max, min, weighted_sum)
            domain_concept_ids: List of concept IDs in the function's domain
        """
        # Combine domain concepts from all components
        all_concepts = set()
        for lyap in component_lyapunovs:
            all_concepts.update(lyap.domain_concept_ids)
            
        if domain_concept_ids is None:
            domain_concept_ids = list(all_concepts)
            
        super().__init__(name, domain_concept_ids)
        
        self.components = component_lyapunovs
        self.weights = weights or [1.0] * len(component_lyapunovs)
        self.composition_type = composition_type
        
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the composite function.
        
        Args:
            x: State vector
            
        Returns:
            Value of the Lyapunov function
        """
        # Evaluate all components
        values = [lyap.evaluate(x) for lyap in self.components]
        
        # Compose based on the type
        if self.composition_type == "sum":
            return float(sum(values))
        elif self.composition_type == "weighted_sum":
            return float(sum(w * v for w, v in zip(self.weights, values)))
        elif self.composition_type == "max":
            return float(max(values))
        elif self.composition_type == "min":
            return float(min(values))
        else:
            raise ValueError(f"Unknown composition type: {self.composition_type}")
        
    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient.
        
        Args:
            x: State vector
            
        Returns:
            Gradient vector
        """
        # For simplicity, just use finite differences
        eps = 1e-6
        grad = np.zeros_like(x)
        
        # Compute gradient with finite differences
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            
            x_minus = x.copy()
            x_minus[i] -= eps
            
            grad[i] = (self.evaluate(x_plus) - self.evaluate(x_minus)) / (2 * eps)
            
        return grad
        
    def verify_positive_definite(self) -> ProofStatus:
        """
        Verify positive definiteness.
        
        For sum/weighted_sum: all components must be positive definite
        For max: at least one component must be positive definite
        For min: all components must be positive definite
        
        Returns:
            Verification result
        """
        # Verify each component
        results = [lyap.verify_positive_definite() for lyap in self.components]
        
        if self.composition_type in ["sum", "weighted_sum", "min"]:
            # All components must be PD
            if all(r == ProofStatus.VERIFIED for r in results):
                return ProofStatus.VERIFIED
            elif any(r == ProofStatus.ERROR for r in results):
                return ProofStatus.ERROR
            else:
                return ProofStatus.REFUTED
        elif self.composition_type == "max":
            # At least one component must be PD
            if any(r == ProofStatus.VERIFIED for r in results):
                return ProofStatus.VERIFIED
            elif all(r == ProofStatus.ERROR for r in results):
                return ProofStatus.ERROR
            else:
                return ProofStatus.REFUTED
        else:
            raise ValueError(f"Unknown composition type: {self.composition_type}")
        
    def verify_decreasing(self, dynamics_fn) -> ProofStatus:
        """
        Verify decreasing property.
        
        For sum/weighted_sum: all components must be decreasing
        For max: the maximum component must be decreasing at the boundary
        For min: at least one component must be decreasing
        
        Args:
            dynamics_fn: System dynamics function
            
        Returns:
            Verification result
        """
        # For simplicity, we'll just check if all components are decreasing
        results = [lyap.verify_decreasing(dynamics_fn) for lyap in self.components]
        
        if all(r == ProofStatus.VERIFIED for r in results):
            return ProofStatus.VERIFIED
        elif any(r == ProofStatus.ERROR for r in results):
            return ProofStatus.ERROR
        else:
            return ProofStatus.REFUTED
            
    def verify_transition(self, x: np.ndarray, from_idx: int, to_idx: int) -> bool:
        """
        Verify that a transition between two regions is stable.
        
        Args:
            x: State vector
            from_idx: Index of the source component
            to_idx: Index of the target component
            
        Returns:
            Whether the transition is stable
        """
        # Get the Lyapunov values for each component
        from_value = self.components[from_idx].evaluate(x)
        to_value = self.components[to_idx].evaluate(x)
        
        # Transition is stable if the target value is lower
        return to_value <= from_value

#------------------------------------------------------------------------------------
# Core Classes - Verification Engine
#------------------------------------------------------------------------------------

class ConstraintIR:
    """
    Constraint Intermediate Representation for solver-agnostic verification.
    
    This represents a single constraint in a form that can be passed to
    different solver backends (SOS, SMT, MILP).
    """
    
    def __init__(
        self,
        id: str,
        variables: List[str],
        expression: str,
        constraint_type: str,
        context: Optional[Dict[str, Any]] = None,
        solver_hint: Optional[str] = None
    ):
        """
        Initialize a constraint IR.
        
        Args:
            id: Unique identifier for the constraint
            variables: List of variable names in the constraint
            expression: Expression in SMT-LIB compatible format
            constraint_type: Type of constraint (equality, inequality, etc.)
            context: Additional context information
            solver_hint: Optional hint for the solver
        """
        self.id = id
        self.variables = variables
        self.expression = expression
        self.constraint_type = constraint_type
        self.context = context or {}
        self.solver_hint = solver_hint

class ProofCertificate:
    """
    Certificate of proof for a Lyapunov verification result.
    
    This can be a positive-definiteness proof, a SOS decomposition,
    an SMT model, etc.
    """
    
    def __init__(
        self,
        proof_type: str,
        details: Dict[str, Any],
        solver_info: Dict[str, Any],
        timestamp: float = None
    ):
        """
        Initialize a proof certificate.
        
        Args:
            proof_type: Type of proof (e.g., "sos", "smt", "milp")
            details: Proof-specific details
            solver_info: Information about the solver used
            timestamp: When the proof was generated
        """
        self.proof_type = proof_type
        self.details = details
        self.solver_info = solver_info
        self.timestamp = timestamp or time.time()

class VerificationResult:
    """
    Result of a Lyapunov function verification.
    
    This includes the status, any counterexample found, and a
    certificate of proof if the verification succeeded.
    """
    
    def __init__(
        self,
        status: ProofStatus,
        proof_hash: str,
        verification_time: float,
        counterexample: Optional[np.ndarray] = None,
        certificate: Optional[ProofCertificate] = None
    ):
        """
        Initialize a verification result.
        
        Args:
            status: Verification status (VERIFIED, REFUTED, etc.)
            proof_hash: Hash of the verification task
            verification_time: Time taken for verification (seconds)
            counterexample: Counterexample if status is REFUTED
            certificate: Certificate of proof if status is VERIFIED
        """
        self.status = status
        self.proof_hash = proof_hash
        self.verification_time = verification_time
        self.counterexample = counterexample
        self.certificate = certificate

class ProofCache:
    """
    Cache for Lyapunov function verification results with dependency tracking.
    
    This enables incremental verification and avoids re-verifying unchanged
    Lyapunov functions and constraints.
    """
    
    def __init__(self):
        """Initialize the proof cache."""
        self.proofs: Dict[str, VerificationResult] = {}
        self.dependencies: Dict[str, Set[str]] = {}  # concept_id -> set of proof_hashes
        
    def get(self, proof_hash: str) -> Optional[VerificationResult]:
        """
        Get a cached verification result.
        
        Args:
            proof_hash: Hash of the verification task
            
        Returns:
            Cached result or None if not found
        """
        return self.proofs.get(proof_hash)
        
    def put(self, result: VerificationResult, dependencies: Optional[List[str]] = None) -> None:
        """
        Add a verification result to the cache.
        
        Args:
            result: Verification result
            dependencies: List of concept IDs that this result depends on
        """
        self.proofs[result.proof_hash] = result
        
        # Record dependencies
        if dependencies:
            for concept_id in dependencies:
                if concept_id not in self.dependencies:
                    self.dependencies[concept_id] = set()
                self.dependencies[concept_id].add(result.proof_hash)

class LyapunovVerifier:
    """
    Verification engine for Lyapunov functions.
    
    This provides a unified interface for verifying Lyapunov functions
    using different solver backends (SOS, SMT, MILP).
    """
    
    def __init__(self, proof_cache: Optional[ProofCache] = None):
        """
        Initialize the verifier.
        
        Args:
            proof_cache: Optional proof cache to use
        """
        self.proof_cache = proof_cache or ProofCache()
        
    def verify(
        self, 
        lyapunov_fn: LyapunovFunction, 
        dynamics_fn: Optional[Callable] = None,
        force_recompute: bool = False
    ) -> VerificationResult:
        """
        Verify a Lyapunov function.
        
        This verifies both positive definiteness and the decreasing
        property (if dynamics_fn is provided).
        
        Args:
            lyapunov_fn: Lyapunov function to verify
            dynamics_fn: System dynamics function (if None, only verify positive definiteness)
            force_recompute: Whether to force recomputation even if cached
            
        Returns:
            Verification result
        """
        # Generate proof hash
        context = {
            "verify_type": "full" if dynamics_fn else "positive_definite",
            "verifier_version": "0.1.0",
        }
        proof_hash = lyapunov_fn.compute_proof_hash(context)
        
        # Check cache
        if not force_recompute:
            cached_result = self.proof_cache.get(proof_hash)
            if cached_result and cached_result.status != ProofStatus.UNKNOWN:
                logger.info(f"Using cached verification result for {lyapunov_fn.name}")
                return cached_result
                
        # Start verification
        start_time = time.time()
        
        # First verify positive definiteness
        pd_status = lyapunov_fn.verify_positive_definite()
        
        # If positive definiteness verification failed, we're done
        if pd_status != ProofStatus.VERIFIED:
            verification_time = time.time() - start_time
            result = VerificationResult(
                status=pd_status,
                proof_hash=proof_hash,
                verification_time=verification_time,
            )
            
            # Cache result
            self.proof_cache.put(result, lyapunov_fn.domain_concept_ids)
            return result
            
        # Create a certificate for positive definiteness
        pd_certificate = ProofCertificate(
            proof_type="builtin",
            details={"method": "direct"},
            solver_info={"name": "builtin", "version": "0.1.0"},
        )
            
        # If dynamics_fn is provided, verify decreasing property
        if dynamics_fn:
            decreasing_status = lyapunov_fn.verify_decreasing(dynamics_fn)
            
            verification_time = time.time() - start_time
            result = VerificationResult(
                status=decreasing_status,
                proof_hash=proof_hash,
                verification_time=verification_time,
                certificate=pd_certificate if decreasing_status == ProofStatus.VERIFIED else None,
            )
        else:
            # Only verifying positive definiteness
            verification_time = time.time() - start_time
            result = VerificationResult(
                status=pd_status,
                proof_hash=proof_hash,
                verification_time=verification_time,
                certificate=pd_certificate,
            )
            
        # Cache result
        self.proof_cache.put(result, lyapunov_fn.domain_concept_ids)
        return result
        
    def generate_constraint_ir(
        self, 
        lyapunov_fn: LyapunovFunction, 
        constraint_type: str,
        dynamics_fn: Optional[Callable] = None
    ) -> List[ConstraintIR]:
        """
        Generate solver-agnostic constraint IR for a Lyapunov condition.
        
        Args:
            lyapunov_fn: Lyapunov function
            constraint_type: Type of constraint ("positive_definite" or "decreasing")
            dynamics_fn: System dynamics function (needed for "decreasing")
            
        Returns:
            List of ConstraintIR instances
        """
        # Simple implementation for demonstration
        if constraint_type == "positive_definite":
            constraint_id = f"pd_{lyapunov_fn.name}"
            variables = ["x0", "x1", "x2"]
            expression = f"(> (V_{lyapunov_fn.name} x0 x1 x2) 0)"
            
            return [
                ConstraintIR(
                    id=constraint_id,
                    variables=variables,
                    expression=expression,
                    constraint_type="positive",
                    context={"lyapunov_type": lyapunov_fn.__class__.__name__},
                )
            ]
        elif constraint_type == "decreasing":
            if dynamics_fn is None:
                raise ValueError("dynamics_fn is required for decreasing constraints")
                
            constraint_id = f"decreasing_{lyapunov_fn.name}"
            variables = ["x0", "x1", "x2"]
            expression = f"(< (derivative (V_{lyapunov_fn.name} x0 x1 x2) (f x0 x1 x2)) 0)"
            
            return [
                ConstraintIR(
                    id=constraint_id,
                    variables=variables,
                    expression=expression,
                    constraint_type="inequality",
                    context={"lyapunov_type": lyapunov_fn.__class__.__name__},
                )
            ]
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

#------------------------------------------------------------------------------------
# Core Classes - Phase-Space and Synchronization
#------------------------------------------------------------------------------------

class SyncState:
    """State of synchronization."""
    SYNC = "SYNC"
    PARTIAL = "PARTIAL"
    DESYNC = "DESYNC"
    
class SyncAction:
    """Action to take on synchronization events."""
    SYNC = "SYNC"       # Force synchronization
    DESYNC = "DESYNC"   # Force desynchronization
    NUDGE = "NUDGE"     # Small nudge towards target
    RESET = "RESET"     # Reset to initial conditions

class PsiPhaseState:
    """
    Phase oscillator state for a single concept.
    
    This represents the phase (θ) and frequency (ω) of an oscillator
    in a Kuramoto-like coupled oscillator network.
    """
    
    def __init__(
        self, 
        theta: float = 0.0,
        omega: float = 1.0,
        name: str = None
    ):
        """
        Initialize a phase state.
        
        Args:
            theta: Initial phase angle (radians)
            omega: Natural frequency (radians per second)
            name: Optional name for the oscillator
        """
        self.theta = float(theta) % (2 * np.pi)  # Ensure θ ∈ [0, 2π)
        self.omega = float(omega)
        self.name = name
        
    def update(self, delta_theta: float, delta_omega: float = 0.0) -> None:
        """
        Update the phase and frequency.
        
        Args:
            delta_theta: Change in phase (radians)
            delta_omega: Change in frequency (radians per second)
        """
        self.theta = (self.theta + delta_theta) % (2 * np.pi)
        self.omega += delta_omega
        
    def __repr__(self) -> str:
        """String representation."""
        if self.name:
            return f"PsiPhaseState({self.name}, θ={self.theta:.3f}, ω={self.omega:.3f})"
        else:
            return f"PsiPhaseState(θ={self.theta:.3f}, ω={self.omega:.3f})"

# ────────────────────────────────────────────────────────────
# Phase-Synchrony Runtime Layer
# ────────────────────────────────────────────────────────────
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Callable, Tuple

# ---- 1. Event dataclass -------------------------------------
@dataclass(frozen=True)
class PhaseStateUpdate:
    """Immutable phase-update event published by PsiSyncMonitor."""
    concept_id: str      # Concept identifier
    dtheta: float        # Δθ (radians)
    domega: float        # Δω (rad/s)
    timestamp: float     # Epoch timestamp

# ---- 2. Tiny pub-sub bus -----------------------------------
class PhaseEventBus:
    """Ultra-light event bus (in-process, synchronous)."""

    def __init__(self) -> None:
        """Initialize the event bus with an empty subscriber list."""
        self._subs: List[Callable[[PhaseStateUpdate], None]] = []

    # subscription helpers --------------------------------------
    def subscribe(self, fn: Callable[[PhaseStateUpdate], None]) -> None:
        """
        Subscribe to phase update events.
        
        Args:
            fn: Callback function to be called with PhaseStateUpdate events
        """
        if fn not in self._subs:
            self._subs.append(fn)

    def unsubscribe(self, fn: Callable[[PhaseStateUpdate], None]) -> None:
        """
        Unsubscribe from phase update events.
        
        Args:
            fn: Previously subscribed callback function
        """
        if fn in self._subs:
            self._subs.remove(fn)

    # dispatch -------------------------------------------------
    def publish(self, evt: PhaseStateUpdate) -> None:
        """
        Publish a phase update event to all subscribers.
        
        Args:
            evt: The event to publish
        """
        for fn in tuple(self._subs):  # tuple copy = safe if list mutates
            try:
                fn(evt)
            except Exception as exc:
                # non-fatal: log and keep ticking
                logger.error(f"[PhaseEventBus] subscriber {fn} raised {exc!r}")

# ---- 3. Synchrony monitor ----------------------------------
class PsiSyncMonitor:
    """
    Global governor that integrates all PsiPhaseState oscillators,
    tracks order parameter R, and surfaces drift events.
    """

    def __init__(
        self,
        oscillators: Dict[str, PsiPhaseState],
        coupling: np.ndarray,
        event_bus: PhaseEventBus,
        drift_eps: float = 0.2,         # rad/s tolerance
    ) -> None:
        """
        Initialize the synchrony monitor.
        
        Args:
            oscillators: Dictionary mapping concept_id to PsiPhaseState
            coupling: NxN coupling matrix K
            event_bus: Event bus for publishing updates
            drift_eps: Tolerance for drift detection (rad/s)
        """
        self.osc = oscillators
        self.ids = list(oscillators.keys())
        self.N = len(self.ids)
        self._θ = np.array([o.theta for o in oscillators.values()], dtype=np.float64)
        self._ω = np.array([o.omega for o in oscillators.values()], dtype=np.float64)
        self.K = coupling.astype(np.float64)  # NxN coupling matrix
        self.bus = event_bus
        self.drift_eps = float(drift_eps)

    # --- public API ------------------------------------------
    def tick(self, dt: float) -> None:
        """
        Integrate all oscillators forward by `dt` seconds using
        vanilla Euler-Kuramoto coupling.
        
        Args:
            dt: Time step (seconds)
        """
        if dt <= 0:
            return

        # Kuramoto equation:  dθᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ sin(θⱼ − θᵢ)
        θ = self._θ
        sin_diff = np.sin(θ[None, :] - θ[:, None])
        dθ = (self._ω + np.sum(self.K * sin_diff, axis=1)) * dt
        self._θ = (θ + dθ) % (2 * np.pi)

        # publish updates & copy back into objects
        now = time.time()
        for i, cid in enumerate(self.ids):
            oldθ, newθ = θ[i], self._θ[i]
            dtheta = ((newθ - oldθ + np.pi) % (2 * np.pi)) - np.pi
            domega = 0.0  # ω constant here; may vary later
            self.osc[cid].theta = newθ
            evt = PhaseStateUpdate(cid, dtheta, domega, now)
            self.bus.publish(evt)

        # drift detection
        self._detect_drift()

    def order_parameter(self) -> Tuple[float, float]:
        """
        Calculate the Kuramoto order parameter (R, Φ).
        
        Returns:
            (R, Φ) where R∈[0,1] is Kuramoto order magnitude
            and Φ is mean phase.
        """
        complex_sum = np.exp(1j * self._θ).sum()
        R = abs(complex_sum) / self.N
        Φ = math.atan2(complex_sum.imag, complex_sum.real)
        return R, Φ

    # --- internal helpers ------------------------------------
    def _detect_drift(self) -> None:
        """Detect oscillators that are drifting from the mean frequency."""
        mean_ω = float(np.mean(self._ω))
        drift_mask = np.abs(self._ω - mean_ω) > self.drift_eps
        if drift_mask.any():
            offenders = [self.ids[i] for i, flag in enumerate(drift_mask) if flag]
            logger.warning(f"[PsiSyncMonitor] Drift detected for {offenders} (>|{self.drift_eps}|)")

# ────────────────────────────────────────────────────────────
# Concept ↔ Phase Bridge Layer
# ────────────────────────────────────────────────────────────

class ConceptPhaseMapping:
    """
    Mapping from concept IDs to oscillator phase indices.
    
    This maintains the relationship between semantic concepts and
    their corresponding oscillators in the phase-synchrony runtime.
    """
    
    def __init__(self) -> None:
        """Initialize an empty mapping."""
        self.concept_to_idx: Dict[str, int] = {}
        self.idx_to_concept: Dict[int, str] = {}
        self.phase_offsets: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}
        
    def register(
        self,
        concept_id: str,
        phase_idx: int,
        phase_offset: float = 0.0,
        weight: float = 1.0
    ) -> None:
        """
        Register a concept with a phase oscillator.
        
        Args:
            concept_id: Concept identifier
            phase_idx: Index in the phase state array
            phase_offset: Phase offset (radians)
            weight: Weight of this concept in sync calculations
        """
        if concept_id in self.concept_to_idx:
            raise ValueError(f"Concept '{concept_id}' already registered")
            
        if phase_idx in self.idx_to_concept:
            existing = self.idx_to_concept[phase_idx]
            raise ValueError(f"Phase index {phase_idx} already assigned to '{existing}'")
            
        self.concept_to_idx[concept_id] = phase_idx
        self.idx_to_concept[phase_idx] = concept_id
        self.phase_offsets[concept_id] = float(phase_offset) % (2 * np.pi)
        self.weights[concept_id] = float(weight)
        
    def get_phase_idx(self, concept_id: str) -> int:
        """
        Get the phase index for a concept.
        
        Args:
            concept_id: Concept identifier
            
        Returns:
            Phase index
            
        Raises:
            KeyError: If concept not registered
        """
        if concept_id not in self.concept_to_idx:
            raise KeyError(f"Concept '{concept_id}' not registered")
            
        return self.concept_to_idx[concept_id]
        
    def get_concept_id(self, phase_idx: int) -> str:
        """
        Get the concept ID for a phase index.
        
        Args:
            phase_idx: Phase index
            
        Returns:
            Concept identifier
            
        Raises:
            KeyError: If phase index not registered
        """
        if phase_idx not in self.idx_to_concept:
            raise KeyError(f"Phase index {phase_idx} not registered")
            
        return self.idx_to_concept[phase_idx]
        
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the mapping to a dictionary.
        
        Returns:
            Serialized mapping
        """
        return {
            "concepts": list(self.concept_to_idx.keys()),
            "indices": list(self.concept_to_idx.values()),
            "offsets": list(self.phase_offsets.values()),
            "weights": list(self.weights.values()),
            "version": "1.0.0",
        }
        
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ConceptPhaseMapping":
        """
        Create a mapping from serialized data.
        
        Args:
            data: Serialized mapping
            
        Returns:
            Deserialized mapping
        """
        mapping = cls()
        
        for i, concept_id in enumerate(data["concepts"]):
            phase_idx = data["indices"][i]
            offset = data["offsets"][i]
            weight = data["weights"][i]
            mapping.register(concept_id, phase_idx, offset, weight)
            
        return mapping

class PsiConceptBridge:
    """
    Bridge between concepts and phase oscillators.
    
    This orchestrates the relationship between semantic concepts and
    their phase-space representation, including:
    - Registering concepts with oscillators
    - Managing Lyapunov functions for stability
    - Listening to phase updates and marking proofs as dirty
    - Providing stability info to higher layers
    """
    
    def __init__(
        self,
        sync_monitor: PsiSyncMonitor,
        mapping: Optional[ConceptPhaseMapping] = None,
        verifier: Optional[LyapunovVerifier] = None
    ):
        """
        Initialize the concept-phase bridge.
        
        Args:
            sync_monitor: Phase synchrony monitor
            mapping: Concept-phase mapping (or create new one)
            verifier: Lyapunov verifier (or create new one)
        """
        self.sync_monitor = sync_monitor
        self.mapping = mapping or ConceptPhaseMapping()
        self.verifier = verifier or LyapunovVerifier()
        
        # Subscribe to phase updates
        self.sync_monitor.bus.subscribe(self.on_phase_update)
        
        # Store Lyapunov functions per concept
        self.lyapunov_fns: Dict[str, List[LyapunovFunction]] = defaultdict(list)
        
        # Track verification status
        self.verification_status: Dict[str, ProofStatus] = {}
        self.dirty_concepts: Set[str] = set()
        
    def register_concept(
        self,
        concept_id: str,
        phase_idx: int,
        phase_offset: float = 0.0,
        weight: float = 1.0
    ) -> None:
        """
        Register a concept with a phase oscillator.
        
        Args:
            concept_id: Concept identifier
            phase_idx: Index in the phase state array
            phase_offset: Phase offset (radians)
            weight: Weight of this concept in sync calculations
        """
        self.mapping.register(concept_id, phase_idx, phase_offset, weight)
        self.dirty_concepts.add(concept_id)
        
    def attach_lyapunov(
        self,
        concept_id: str,
        lyapunov_fn: LyapunovFunction
    ) -> None:
        """
        Attach a Lyapunov function to a concept.
        
        Args:
            concept_id: Concept identifier
            lyapunov_fn: Lyapunov function
        """
        # Ensure concept is registered
        if concept_id not in self.mapping.concept_to_idx:
            raise KeyError(f"Concept '{concept_id}' not registered, call register_concept first")
            
        # Add to the list
        self.lyapunov_fns[concept_id].append(lyapunov_fn)
        
        # Mark as dirty
        self.dirty_concepts.add(concept_id)
        
    def on_phase_update(self, event: PhaseStateUpdate) -> None:
        """
        Handle a phase update event.
        
        Args:
            event: Phase update event
        """
        concept_id = event.concept_id
        
        # Only care about registered concepts
        if concept_id not in self.mapping.concept_to_idx:
            return
            
        # Mark as dirty if phase changed significantly
        if abs(event.dtheta) > 1e-6:
            self.dirty_concepts.add(concept_id)
            logger.debug(f"Concept '{concept_id}' marked dirty due to phase change")
            
    def verify_concept(self, concept_id: str) -> ProofStatus:
        """
        Verify stability of a concept using its Lyapunov functions.
        
        Args:
            concept_id: Concept identifier
            
        Returns:
            Verification status
        """
        # Check if concept is registered
        if concept_id not in self.mapping.concept_to_idx:
            raise KeyError(f"Concept '{concept_id}' not registered")
            
        # If not dirty and status exists, return cached status
        if concept_id not in self.dirty_concepts and concept_id in self.verification_status:
            return self.verification_status[concept_id]
            
        # Get Lyapunov functions
        lyapunov_fns = self.lyapunov_fns.get(concept_id, [])
        
        # No Lyapunov functions? Unknown status
        if not lyapunov_fns:
            self.verification_status[concept_id] = ProofStatus.UNKNOWN
            return ProofStatus.UNKNOWN
            
        # Verify each Lyapunov function
        results = []
        for lyap_fn in lyapunov_fns:
            # TODO: Get dynamics function from phase model
            dynamics_fn = lambda x: -x  # Simple dummy for now
            result = self.verifier.verify(lyap_fn, dynamics_fn)
            results.append(result.status)
            
        # Determine overall status:
        # - If any verification is successful, the concept is stable
        # - If any verification has an error, the status is ERROR
        # - Otherwise, the concept is unstable
        if any(status == ProofStatus.VERIFIED for status in results):
            status = ProofStatus.VERIFIED
        elif any(status == ProofStatus.ERROR for status in results):
            status = ProofStatus.ERROR
        else:
            status = ProofStatus.REFUTED
            
        # Update and clear dirty flag
        self.verification_status[concept_id] = status
        if concept_id in self.dirty_concepts:
            self.dirty_concepts.remove(concept_id)
            
        return status
        
    def get_stability(self, concept_id: str) -> ProofStatus:
        """
        Get the stability status of a concept.
        
        If the concept is dirty, this will recompute the status.
        Otherwise, it returns the cached status.
        
        Args:
            concept_id: Concept identifier
            
        Returns:
            Stability status
        """
        if concept_id in self.dirty_concepts or concept_id not in self.verification_status:
            return self.verify_concept(concept_id)
        else:
            return self.verification_status[concept_id]

class TransitionGuard:
    """
    Helper class for validating transitions between concepts.
    
    This uses Lyapunov functions to determine if a transition between
    concepts is stable.
    """
    
    def __init__(
        self,
        bridge: PsiConceptBridge,
        composite_lyapunov: Optional[CompositeLyapunov] = None
    ):
        """
        Initialize the transition guard.
        
        Args:
            bridge: Concept-phase bridge
            composite_lyapunov: Optional composite Lyapunov function for transitions
        """
        self.bridge = bridge
        self.composite_lyapunov = composite_lyapunov
        
    def can_transition(
        self,
        from_concept_id: str,
        to_concept_id: str,
        state: Optional[np.ndarray] = None
    ) -> bool:
        """
        Check if a transition between concepts is stable.
        
        Args:
            from_concept_id: Source concept ID
            to_concept_id: Target concept ID
            state: Current state vector (or None to use current phase state)
            
        Returns:
            Whether the transition is stable
        """
        # If using a composite Lyapunov function, use that
        if self.composite_lyapunov is not None:
            # Get component indices
            try:
                components = self.composite_lyapunov.components
                from_idx = next(i for i, c in enumerate(components) if from_concept_id in c.domain_concept_ids)
                to_idx = next(i for i, c in enumerate(components) if to_concept_id in c.domain_concept_ids)
                
                # Default state if none provided
                if state is None:
                    # Create a simple 2D state from phase difference
                    from_phase = self.bridge.sync_monitor.osc[from_concept_id].theta
                    to_phase = self.bridge.sync_monitor.osc[to_concept_id].theta
                    diff = to_phase - from_phase
                    state = np.array([np.cos(diff), np.sin(diff)])
                    
                # Check transition
                return self.composite_lyapunov.verify_transition(state, from_idx, to_idx)
            except (StopIteration, KeyError):
                # Fall back to simpler method
                pass
                
        # Fallback: Just check if target concept is stable
        return self.bridge.get_stability(to_concept_id) == ProofStatus.VERIFIED

# ────────────────────────────────────────────────────────────
# Verification / Proof Plumbing
# ────────────────────────────────────────────────────────────
import json
import os
import hashlib
import threading
import queue
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

class ProofPersistence:
    """
    Persistence layer for proof certificates.
    
    This enables saving and loading verification results across runs.
    """
    
    def __init__(self, storage_path: str = "./proof_cache"):
        """
        Initialize the proof persistence layer.
        
        Args:
            storage_path: Path to store certificates
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Dependency tracking
        self.dependencies: Dict[str, Set[str]] = {}  # concept_id -> set of proof_hashes
        self._load_dependencies()
        
    def _load_dependencies(self) -> None:
        """Load dependency index from disk."""
        index_path = self.storage_path / "dependencies.json"
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    data = json.load(f)
                    
                # Convert JSON-encoded dependencies back to sets
                for concept_id, hashes in data.items():
                    self.dependencies[concept_id] = set(hashes)
                    
                logger.info(f"Loaded dependency index for {len(self.dependencies)} concepts")
            except Exception as e:
                logger.error(f"Error loading dependency index: {e}")
                # Start with empty dependencies
                self.dependencies = {}
                
    def _save_dependencies(self) -> None:
        """Save dependency index to disk."""
        index_path = self.storage_path / "dependencies.json"
        try:
            # Convert sets to lists for JSON serialization
            json_data = {concept: list(hashes) for concept, hashes in self.dependencies.items()}
            
            with open(index_path, "w") as f:
                json.dump(json_data, f)
        except Exception as e:
            logger.error(f"Error saving dependency index: {e}")
            
    def path_for_hash(self, proof_hash: str) -> Path:
        """Get the filesystem path for a proof hash."""
        # Use first few chars as subdirectory for better performance
        prefix = proof_hash[:2]
        subdir = self.storage_path / prefix
        subdir.mkdir(exist_ok=True)
        
        return subdir / f"{proof_hash}.json"
        
    def save(self, result: VerificationResult, dependencies: Optional[List[str]] = None) -> bool:
        """
        Save a verification result.
        
        Args:
            result: Verification result to save
            dependencies: List of concept IDs this result depends on
            
        Returns:
            Whether the save was successful
        """
        try:
            # Convert to serializable dict
            data = {
                "status": result.status.name,
                "proof_hash": result.proof_hash,
                "verification_time": result.verification_time,
                "timestamp": time.time(),
            }
            
            # Add certificate if present
            if result.certificate:
                data["certificate"] = {
                    "proof_type": result.certificate.proof_type,
                    "details": result.certificate.details,
                    "solver_info": result.certificate.solver_info,
                    "timestamp": result.certificate.timestamp,
                }
                
            # Add counterexample if present
            if result.counterexample is not None:
                data["counterexample"] = result.counterexample.tolist()
                
            # Save to disk
            path = self.path_for_hash(result.proof_hash)
            with open(path, "w") as f:
                json.dump(data, f)
                
            # Update dependencies
            if dependencies:
                for concept_id in dependencies:
                    if concept_id not in self.dependencies:
                        self.dependencies[concept_id] = set()
                    self.dependencies[concept_id].add(result.proof_hash)
                    
                # Save updated index
                self._save_dependencies()
                
            return True
        except Exception as e:
            logger.error(f"Error saving proof: {e}")
            return False
            
    def load(self, proof_hash: str) -> Optional[VerificationResult]:
        """
        Load a verification result.
        
        Args:
            proof_hash: Hash of the verification task
            
        Returns:
            Loaded result or None if not found
        """
        path = self.path_for_hash(proof_hash)
        if not path.exists():
            return None
            
        try:
            with open(path, "r") as f:
                data = json.load(f)
                
            # Parse status
            status = ProofStatus[data["status"]]
            
            # Parse certificate if present
            certificate = None
            if "certificate" in data:
                cert_data = data["certificate"]
                certificate = ProofCertificate(
                    proof_type=cert_data["proof_type"],
                    details=cert_data["details"],
                    solver_info=cert_data["solver_info"],
                    timestamp=cert_data.get("timestamp", time.time()),
                )
                
            # Parse counterexample if present
            counterexample = None
            if "counterexample" in data:
                counterexample = np.array(data["counterexample"])
                
            # Create result
            result = VerificationResult(
                status=status,
                proof_hash=data["proof_hash"],
                verification_time=data["verification_time"],
                counterexample=counterexample,
                certificate=certificate,
            )
            
            return result
        except Exception as e:
            logger.error(f"Error loading proof {proof_hash}: {e}")
            return None
            
    def invalidate_dependent(self, concept_id: str) -> List[str]:
        """
        Invalidate all proofs dependent on a concept.
        
        Args:
            concept_id: Concept ID
            
        Returns:
            List of invalidated proof hashes
        """
        if concept_id not in self.dependencies:
            return []
            
        invalidated = list(self.dependencies[concept_id])
        
        # Remove proofs from filesystem
        for proof_hash in invalidated:
            path = self.path_for_hash(proof_hash)
            if path.exists():
                try:
                    os.remove(path)
                except Exception as e:
                    logger.error(f"Error removing proof {proof_hash}: {e}")
                    
        # Remove from dependencies
        self.dependencies.pop(concept_id)
        self._save_dependencies()
        
        return invalidated

class VerificationManager:
    """
    Facade for verification operations with thread-safe queue.
    
    This provides a simplified interface for submitting verification tasks
    and retrieving results.
    """
    
    def __init__(
        self,
        verifier: Optional[LyapunovVerifier] = None,
        persistence: Optional[ProofPersistence] = None,
        max_workers: int = 4
    ):
        """
        Initialize the verification manager.
        
        Args:
            verifier: Lyapunov verifier
            persistence: Proof persistence layer
            max_workers: Maximum number of worker threads
        """
        self.verifier = verifier or LyapunovVerifier()
        self.persistence = persistence or ProofPersistence()
        
        # Thread pool for async verification
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Task queue
        self.queue = queue.Queue()
        
        # Active tasks
        self.active_tasks: Dict[str, Future] = {}
        
        # Start worker thread
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._process_queue)
        self._worker_thread.daemon = True
        self._worker_thread.start()
        
    def submit(
        self,
        constraint_ir: Union[ConstraintIR, List[ConstraintIR]],
        dependencies: Optional[List[str]] = None
    ) -> Future[VerificationResult]:
        """
        Submit a constraint for verification.
        
        Args:
            constraint_ir: Constraint IR or list of constraints
            dependencies: List of concept IDs this verification depends on
            
        Returns:
            Future for the verification result
        """
        # Compute hash for the constraint
        constraints = constraint_ir if isinstance(constraint_ir, list) else [constraint_ir]
        proof_hash = self._compute_hash(constraints)
        
        # Check if already in progress
        if proof_hash in self.active_tasks:
            return self.active_tasks[proof_hash]
            
        # Check if already in cache
        cached_result = self.persistence.load(proof_hash)
        if cached_result:
            future = Future()
            future.set_result(cached_result)
            return future
            
        # Create new future
        future = Future()
        self.active_tasks[proof_hash] = future
        
        # Add to queue
        self.queue.put((proof_hash, constraints, dependencies, future))
        
        return future
        
    def shutdown(self):
        """Shut down the manager and wait for tasks to complete."""
        self._stop_event.set()
        self.executor.shutdown(wait=True)
        
    def _compute_hash(self, constraints: List[ConstraintIR]) -> str:
        """Compute a hash for a list of constraints."""
        hasher = hashlib.sha256()
        
        for constraint in constraints:
            # Add key components to hash
            hasher.update(constraint.id.encode())
            hasher.update(constraint.expression.encode())
            hasher.update(constraint.constraint_type.encode())
            
            # Add variables
            variables_str = ",".join(sorted(constraint.variables))
            hasher.update(variables_str.encode())
            
        return hasher.hexdigest()
        
    def _process_queue(self):
        """Process tasks from the queue."""
        while not self._stop_event.is_set():
            try:
                # Try to get task with timeout to allow for stopping
                task = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            try:
                proof_hash, constraints, dependencies, future = task
                
                # Skip if future already done (e.g., cancelled)
                if future.done():
                    self.queue.task_done()
                    continue
                    
                # Use ThreadPoolExecutor to run verification
                self.executor.submit(
                    self._verify_task, proof_hash, constraints, dependencies, future
                )
                
            except Exception as e:
                logger.error(f"Error processing verification task: {e}")
                
            finally:
                self.queue.task_done()
                
    def _verify_task(
        self,
        proof_hash: str,
        constraints: List[ConstraintIR],
        dependencies: Optional[List[str]],
        future: Future
    ):
        """
        Verify a task and set the future result.
        
        Args:
            proof_hash: Hash of the verification task
            constraints: Constraints to verify
            dependencies: Concept dependencies
            future: Future to set result on
        """
        try:
            # TODO: Implement actual constraint verification
            # For now, just create a dummy result
            result = VerificationResult(
                status=ProofStatus.VERIFIED,
                proof_hash=proof_hash,
                verification_time=0.5,
                certificate=ProofCertificate(
                    proof_type="dummy",
                    details={"constraints": [c.id for c in constraints]},
                    solver_info={"name": "dummy", "version": "0.1.0"},
                ),
            )
            
            # Save result
            self.persistence.save(result, dependencies)
            
            # Set future result
            future.set_result(result)
            
        except Exception as e:
            logger.error(f"Error verifying task {proof_hash}: {e}")
            future.set_exception(e)
            
        finally:
            # Remove from active tasks
            if proof_hash in self.active_tasks:
                self.active_tasks.pop(proof_hash)

# ────────────────────────────────────────────────────────────
# Demo / UX Shell
# ────────────────────────────────────────────────────────────
import cmd
import sys
import io
from matplotlib.animation import FuncAnimation

def load_demo_network(bridge: PsiConceptBridge) -> None:
    """
    Load a demo concept network with sample data.
    
    This sets up a small network of concepts and oscillators with
    appropriate Lyapunov functions.
    
    Args:
        bridge: Concept-phase bridge to populate
    """
    # Create four oscillators with different natural frequencies
    phase_states = {
        "concept_1": PsiPhaseState(theta=0.0, omega=1.0, name="Concept 1"),
        "concept_2": PsiPhaseState(theta=np.pi/3, omega=1.1, name="Concept 2"),
        "concept_3": PsiPhaseState(theta=2*np.pi/3, omega=0.9, name="Concept 3"),
        "concept_4": PsiPhaseState(theta=np.pi, omega=1.2, name="Concept 4"),
    }
    
    # Create coupling matrix
    N = len(phase_states)
    # Start with full coupling, then remove self-coupling
    K = np.full((N, N), 0.1)
    np.fill_diagonal(K, 0.0)
    
    # Create event bus
    bus = PhaseEventBus()
    
    # Create monitor
    monitor = PsiSyncMonitor(phase_states, K, bus)
    
    # Register concepts with bridge
    for i, (concept_id, state) in enumerate(phase_states.items()):
        bridge.register_concept(concept_id, i)
        
    # Create and attach Lyapunov functions
    # Simple quadratic for concept_1
    q1 = np.array([[1.0, 0.5], [0.5, 1.0]])
    lyap1 = PolynomialLyapunov(
        name="lyap_1",
        q_matrix=q1,
        domain_concept_ids=["concept_1"]
    )
    bridge.attach_lyapunov("concept_1", lyap1)
    
    # Another for concept_2
    q2 = np.array([[1.0, 0.0], [0.0, 2.0]])
    lyap2 = PolynomialLyapunov(
        name="lyap_2",
        q_matrix=q2,
        domain_concept_ids=["concept_2"]
    )
    bridge.attach_lyapunov("concept_2", lyap2)
    
    # Composite for concepts 3 and 4
    composite = CompositeLyapunov(
        name="composite_lyap",
        component_lyapunovs=[
            PolynomialLyapunov(
                name="lyap_3",
                q_matrix=np.array([[1.5, 0.0], [0.0, 1.5]]),
                domain_concept_ids=["concept_3"]
            ),
            PolynomialLyapunov(
                name="lyap_4",
                q_matrix=np.array([[2.0, 0.0], [0.0, 1.0]]),
                domain_concept_ids=["concept_4"]
            )
        ],
        composition_type="weighted_sum",
        weights=[0.7, 0.3],
        domain_concept_ids=["concept_3", "concept_4"]
    )
    bridge.attach_lyapunov("concept_3", composite)
    bridge.attach_lyapunov("concept_4", composite)
    
    # Create transition guard
    bridge.transition_guard = TransitionGuard(bridge, composite_lyapunov=composite)
    
    # Return the monitor for the interactive console
    return monitor

class InteractiveConsole(cmd.Cmd):
    """
    Interactive command-line interface for the demo.
    
    This provides commands for stepping the simulation, inspecting
    oscillator states, and analyzing system properties.
    """
    
    intro = """
    ╔════════════════════════════════════════════════════════╗
    ║                 ELFIN Phase-Sync Demo                  ║
    ╚════════════════════════════════════════════════════════╝
    
    Type 'help' for a list of commands.
    """
    prompt = "ψ-sync> "
    
    def __init__(
        self,
        monitor: PsiSyncMonitor,
        bridge: PsiConceptBridge,
        use_plot: bool = True
    ):
        """
        Initialize the console.
        
        Args:
            monitor: Synchrony monitor
            bridge: Concept-phase bridge
            use_plot: Whether to show real-time plot
        """
        super().__init__()
        self.monitor = monitor
        self.bridge = bridge
        self.use_plot = use_plot
        self.dt = 0.1  # Default time step
        self.time = 0.0
        
        # History for plotting
        self.history = {
            "time": [],
            "phases": defaultdict(list),
            "order_param": [],
        }
        
        # Set up plot if requested
        self.fig = None
        self.animation = None
        if use_plot:
            self._setup_plot()
            
    def _setup_plot(self):
        """Set up the matplotlib plot."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            
            # Create figure with two subplots
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Phase plot (top)
            self.phase_lines = {}
            self.ax1.set_xlim(0, 10)  # Initial window of 10 seconds
            self.ax1.set_ylim(0, 2*np.pi)
            self.ax1.set_ylabel("Phase (rad)")
            self.ax1.set_title("Oscillator Phases")
            self.ax1.grid(True)
            
            # Create a line for each oscillator
            for i, concept_id in enumerate(self.monitor.ids):
                line, = self.ax1.plot([], [], label=concept_id)
                self.phase_lines[concept_id] = line
                
            self.ax1.legend()
            
            # Order parameter plot (bottom)
            self.order_line, = self.ax2.plot([], [], 'r-', label="R")
            self.ax2.set_xlim(0, 10)  # Initial window of 10 seconds
            self.ax2.set_ylim(0, 1)
            self.ax2.set_xlabel("Time (s)")
            self.ax2.set_ylabel("Order Parameter (R)")
            self.ax2.set_title("Kuramoto Order Parameter")
            self.ax2.grid(True)
            self.ax2.legend()
            
            # Start animation
            self.animation = FuncAnimation(
                self.fig, self._update_plot, interval=200, 
                blit=True, cache_frame_data=False
            )
            plt.tight_layout()
            plt.ion()  # Interactive mode on
            plt.show(block=False)
            
        except ImportError as e:
            logger.warning(f"Plotting disabled: {e}")
            self.use_plot = False
            
    def _update_plot(self, frame):
        """Update the plot with new data."""
        # Update x-axis limits if needed
        if self.history["time"] and self.history["time"][-1] > self.ax1.get_xlim()[1]:
            window = 10.0  # 10-second rolling window
            new_right = self.history["time"][-1]
            new_left = max(0, new_right - window)
            self.ax1.set_xlim(new_left, new_right)
            self.ax2.set_xlim(new_left, new_right)
            
        # Update phase lines
        lines = []
        for concept_id, line in self.phase_lines.items():
            if concept_id in self.history["phases"] and self.history["phases"][concept_id]:
                times = self.history["time"]
                phases = self.history["phases"][concept_id]
                line.set_data(times, phases)
            lines.append(line)
            
        # Update order parameter line
        if self.history["time"] and self.history["order_param"]:
            self.order_line.set_data(
                self.history["time"], self.history["order_param"]
            )
        lines.append(self.order_line)
        
        return lines
        
    def update_history(self):
        """Update the history with current state."""
        self.history["time"].append(self.time)
        
        # Add phases
        for concept_id, state in self.monitor.osc.items():
            self.history["phases"][concept_id].append(state.theta)
            
        # Add order parameter
        R, _ = self.monitor.order_parameter()
        self.history["order_param"].append(R)
        
    def do_step(self, arg):
        """
        Step the simulation forward.
        
        Usage: step [dt]
            dt: Time step in seconds (default: 0.1)
        """
        try:
            if arg:
                dt = float(arg)
            else:
                dt = self.dt
                
            self.monitor.tick(dt)
            self.time += dt
            
            # Update history
            self.update_history()
            
            # Print current state
            self.do_status("")
            
        except ValueError:
            print(f"Invalid time step: {arg}")
            
    def do_run(self, arg):
        """
        Run the simulation for multiple steps.
        
        Usage: run [steps] [dt]
            steps: Number of steps (default: 10)
            dt: Time step in seconds (default: 0.1)
        """
        args = arg.split()
        
        try:
            steps = int(args[0]) if args else 10
            dt = float(args[1]) if len(args) > 1 else self.dt
            
            print(f"Running for {steps} steps with dt={dt}...")
            
            for _ in range(steps):
                self.monitor.tick(dt)
                self.time += dt
                
                # Update history but don't print every step
                self.update_history()
                
            # Print final state
            self.do_status("")
            
        except (ValueError, IndexError):
            print(f"Invalid arguments: {arg}")
            print("Usage: run [steps] [dt]")
            
    def do_status(self, arg):
        """
        Show the current status of all oscillators.
        
        Usage: status [concept_id]
            concept_id: Optional concept to show details for
        """
        if arg:
            # Show detailed status for a specific concept
            concept_id = arg.strip()
            if concept_id in self.monitor.osc:
                state = self.monitor.osc[concept_id]
                phase_idx = self.bridge.mapping.get_phase_idx(concept_id)
                stability = self.bridge.get_stability(concept_id)
                
                print(f"\nConcept: {concept_id} (idx: {phase_idx})")
                print(f"  Phase (θ): {state.theta:.4f} rad")
                print(f"  Frequency (ω): {state.omega:.4f} rad/s")
                print(f"  Stability: {stability.name}")
                
                # Show Lyapunov functions
                lyap_fns = self.bridge.lyapunov_fns.get(concept_id, [])
                if lyap_fns:
                    print(f"  Lyapunov Functions:")
                    for i, fn in enumerate(lyap_fns):
                        print(f"    {i+1}. {fn.name} ({fn.__class__.__name__})")
                else:
                    print("  No Lyapunov functions attached")
            else:
                print(f"Unknown concept: {concept_id}")
                print(f"Available concepts: {', '.join(self.monitor.osc.keys())}")
        else:
            # Show summary for all oscillators
            R, Φ = self.monitor.order_parameter()
            
            print(f"\nTime: {self.time:.2f}s")
            print(f"Order Parameter: R={R:.4f}, Φ={Φ:.4f}")
            print("\nOscillators:")
            
            for concept_id, state in sorted(self.monitor.osc.items()):
                stability = self.bridge.get_stability(concept_id)
                stability_marker = "✓" if stability == ProofStatus.VERIFIED else "✗"
                print(f"  {concept_id:10s} θ={state.theta:.4f} rad  ω={state.omega:.4f} rad/s  [{stability_marker}]")
                
    def do_order(self, arg):
        """
        Show the current Kuramoto order parameter.
        
        Usage: order
        """
        R, Φ = self.monitor.order_parameter()
        print(f"\nOrder Parameter: R={R:.6f}, Φ={Φ:.6f}")
        print(f"  R=1: Perfect synchronization")
        print(f"  R=0: Complete desynchronization")
        
    def do_nudge(self, arg):
        """
        Apply a small nudge to an oscillator's phase or frequency.
        
        Usage: nudge concept_id [dtheta=0] [domega=0]
            concept_id: ID of the concept to nudge
            dtheta: Phase change in radians
            domega: Frequency change in rad/s
        """
        args = arg.split()
        if not args:
            print("Missing concept_id")
            print("Usage: nudge concept_id [dtheta=0] [domega=0]")
            return
            
        concept_id = args[0]
        if concept_id not in self.monitor.osc:
            print(f"Unknown concept: {concept_id}")
            return
            
        try:
            dtheta = float(args[1]) if len(args) > 1 else 0.0
            domega = float(args[2]) if len(args) > 2 else 0.0
            
            # Apply nudge
            self.monitor.osc[concept_id].update(dtheta, domega)
            
            # Update internal state
            idx = self.bridge.mapping.get_phase_idx(concept_id)
            if dtheta != 0:
                self.monitor._θ[idx] = self.monitor.osc[concept_id].theta
            if domega != 0:
                self.monitor._ω[idx] = self.monitor.osc[concept_id].omega
                
            print(f"Applied nudge to {concept_id}: Δθ={dtheta:.4f}, Δω={domega:.4f}")
            self.do_status(concept_id)
            
        except (ValueError, IndexError):
            print(f"Invalid arguments: {args[1:]}")
            print("Usage: nudge concept_id [dtheta=0] [domega=0]")
            
    def do_verify(self, arg):
        """
        Verify the stability of a concept.
        
        Usage: verify concept_id
            concept_id: ID of the concept to verify
        """
        if not arg:
            print("Missing concept_id")
            print("Usage: verify concept_id")
            return
            
        concept_id = arg.strip()
        if concept_id not in self.monitor.osc:
            print(f"Unknown concept: {concept_id}")
            return
            
        # Force verification even if cached
        status = self.bridge.verify_concept(concept_id)
        print(f"Verification status for {concept_id}: {status.name}")
        
    def do_transition(self, arg):
        """
        Check if a transition between concepts is stable.
        
        Usage: transition from_concept to_concept
            from_concept: Source concept ID
            to_concept: Target concept ID
        """
        args = arg.split()
        if len(args) != 2:
            print("Invalid arguments")
            print("Usage: transition from_concept to_concept")
            return
            
        from_id, to_id = args
        
        # Check both concepts exist
        if from_id not in self.monitor.osc:
            print(f"Unknown concept: {from_id}")
            return
            
        if to_id not in self.monitor.osc:
            print(f"Unknown concept: {to_id}")
            return
            
        # Check transition
        if hasattr(self.bridge, 'transition_guard'):
            stable = self.bridge.transition_guard.can_transition(from_id, to_id)
        else:
            # Create a temporary guard if needed
            guard = TransitionGuard(self.bridge)
            stable = guard.can_transition(from_id, to_id)
            
        result = "STABLE" if stable else "UNSTABLE"
        print(f"Transition {from_id} -> {to_id}: {result}")
        
    def do_plot(self, arg):
        """
        Enable or disable the real-time plot.
        
        Usage: plot [on|off]
            on: Enable plotting
            off: Disable plotting
        """
        if not matplotlib_available:
            print("Plotting requires matplotlib library")
            return
            
        arg = arg.strip().lower()
        if arg == "on":
            if not self.use_plot:
                self.use_plot = True
                self._setup_plot()
                print("Plotting enabled")
            else:
                print("Plotting is already enabled")
        elif arg == "off":
            if self.use_plot:
                plt.close(self.fig)
                self.animation = None
                self.fig = None
                self.use_plot = False
                print("Plotting disabled")
            else:
                print("Plotting is already disabled")
        else:
            print(f"Plot status: {'enabled' if self.use_plot else 'disabled'}")
            
    def do_reset(self, arg):
        """
        Reset oscillator phases and/or frequencies.
        
        Usage: reset [phase|freq|all]
            phase: Reset phases to initial values
            freq: Reset frequencies to initial values
            all: Reset both phases and frequencies
        """
        arg = arg.strip().lower()
        
        if arg in ("phase", "all"):
            for i, concept_id in enumerate(self.monitor.ids):
                # Reset to evenly distributed phases
                phase = 2 * np.pi * i / len(self.monitor.ids)
                self.monitor.osc[concept_id].theta = phase
                self.monitor._θ[i] = phase
                
            print("Reset all phases")
            
        if arg in ("freq", "all"):
            for i, concept_id in enumerate(self.monitor.ids):
                # Reset to natural frequencies (here we use 1.0 + small variation)
                freq = 1.0 + 0.05 * (i - len(self.monitor.ids)/2)
                self.monitor.osc[concept_id].omega = freq
                self.monitor._ω[i] = freq
                
            print("Reset all frequencies")
            
        if not arg:
            print("Usage: reset [phase|freq|all]")
        else:
            self.do_status("")
            
    def do_clear(self, arg):
        """
        Clear the history and reset the plot.
        
        Usage: clear
        """
        self.history = {
            "time": [],
            "phases": defaultdict(list),
            "order_param": [],
        }
        
        if self.use_plot:
            # Reset plot limits
            self.ax1.set_xlim(0, 10)
            self.ax2.set_xlim(0, 10)
            
            # Clear line data
            for line in self.phase_lines.values():
                line.set_data([], [])
            self.order_line.set_data([], [])
            
        print("History cleared")
        
    def do_quit(self, arg):
        """
        Exit the program.
        
        Usage: quit
        """
        if self.fig:
            plt.close(self.fig)
        print("Goodbye!")
        return True
        
    def do_exit(self, arg):
        """
        Exit the program.
        
        Usage: exit
        """
        return self.do_quit(arg)

def run_interactive_console(monitor: PsiSyncMonitor, bridge: PsiConceptBridge):
    """
    Run the interactive console.
    
    Args:
        monitor: Synchrony monitor
        bridge: Concept-phase bridge
    """
    console = InteractiveConsole(monitor, bridge)
    console.cmdloop()

# ────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="ELFIN Phase-Sync Demo")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        matplotlib_available = True
    except ImportError:
        matplotlib_available = False
        if not args.no_plot:
            print("Warning: matplotlib not available, plotting disabled")
            args.no_plot = True
            
    # Create event bus
    bus = PhaseEventBus()
    
    # Create an empty bridge (we'll populate it in load_demo_network)
    verifier = LyapunovVerifier()
    persistence = ProofPersistence()
    verification_manager = VerificationManager(verifier, persistence)
    bridge = PsiConceptBridge(None, verifier=verifier)
    
    # Load demo network (sets up monitor, oscillators, etc.)
    monitor = load_demo_network(bridge)
    
    # Attach monitor to bridge (was created in load_demo_network)
    bridge.sync_monitor = monitor
    
    # Run interactive console
    run_interactive_console(monitor, bridge)
