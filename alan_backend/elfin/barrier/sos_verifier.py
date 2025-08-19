"""
Sum-of-Squares verification for barrier certificates.

This module provides functionality for formal verification of barrier
certificates using Sum-of-Squares (SOS) programming. It can verify that
a barrier function satisfies the required safety properties.
"""

import os
import sys
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass

from alan_backend.elfin.barrier.learner import BarrierFunction

# Configure logging
logger = logging.getLogger("elfin.barrier.sos_verifier")

# Try to import SOS programming libraries
try:
    import mosek
    MOSEK_AVAILABLE = True
except ImportError:
    MOSEK_AVAILABLE = False
    logger.warning("MOSEK not available, SOS verification with MOSEK disabled")

try:
    import SparsePOP
    SPARSE_POP_AVAILABLE = True
except ImportError:
    SPARSE_POP_AVAILABLE = False
    logger.warning("SparsePOP not available, SOS verification with SparsePOP disabled")


@dataclass
class VerificationResult:
    """
    Result of a barrier certificate verification.
    
    Attributes:
        success: Whether verification was successful
        status: Status string ("verified", "refuted", "unknown", "error")
        counterexample: Counterexample point if verification failed
        certificate: Verification certificate if available
        solver_output: Raw solver output for debugging
        solver_time: Time taken by the solver in seconds
        barrier_value: Barrier function value at counterexample if any
        boundary_decrease: Boundary decreasing value at counterexample if any
        violation_reason: Reason for violation if verification failed
    """
    success: bool
    status: str
    counterexample: Optional[np.ndarray] = None
    certificate: Optional[Any] = None
    solver_output: Optional[Dict[str, Any]] = None
    solver_time: float = 0.0
    barrier_value: Optional[float] = None
    boundary_decrease: Optional[float] = None
    violation_reason: Optional[str] = None
    
    def get_error_code(self) -> Optional[str]:
        """
        Get the error code for the verification result.
        
        Returns:
            Error code string or None if verification was successful
        """
        if self.success:
            return None
        
        if self.violation_reason == "positivity":
            return "E-BARR-001"
        elif self.violation_reason == "boundary_decreasing":
            return "E-BARR-002"
        else:
            return None
    
    def get_violation_message(self) -> Optional[str]:
        """
        Get a human-readable message describing the violation.
        
        Returns:
            Message string or None if verification was successful
        """
        if self.success:
            return None
        
        if self.violation_reason == "positivity" and self.counterexample is not None:
            return f"Barrier positivity violated at x={self.counterexample.tolist()}; B(x)={self.barrier_value:.3f}"
        elif self.violation_reason == "boundary_decreasing" and self.counterexample is not None:
            return f"Barrier not decreasing (∇B⋅f={self.boundary_decrease:.3f}) on ∂S at x={self.counterexample.tolist()}"
        else:
            return "Unknown violation"


class SOSVerifier:
    """
    Sum-of-Squares verifier for barrier certificates.
    
    This class implements SOS programming-based verification methods for
    checking that a barrier function satisfies the required safety properties:
    1. B(x) > 0 for all x in the unsafe set
    2. ∇B(x) · f(x) < 0 for all x on the boundary of the safe set
    """
    
    def __init__(
        self,
        domain: Tuple[np.ndarray, np.ndarray],
        unsafe_region: Callable[[np.ndarray], bool],
        dynamics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SOS verifier.
        
        Args:
            domain: Domain bounds (lower, upper)
            unsafe_region: Function that returns True if x is in the unsafe region
            dynamics_fn: System dynamics function (optional, for decreasing condition)
            options: Additional options for verification
        """
        self.domain = domain
        self.unsafe_region = unsafe_region
        self.dynamics_fn = dynamics_fn
        self.options = options or {}
        
        # Check if SOS programming libraries are available
        if not MOSEK_AVAILABLE and not SPARSE_POP_AVAILABLE:
            logger.warning("No SOS programming libraries available, verification will be sampling-based only")
    
    def verify(
        self,
        barrier_fn: BarrierFunction,
        method: str = 'mosek',
        n_samples: int = 1000,
        check_positivity: bool = True,
        check_boundary_decreasing: bool = True,
        options: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """
        Verify that a barrier function satisfies safety properties.
        
        Args:
            barrier_fn: Barrier function to verify
            method: Verification method ('mosek', 'sparsepop', 'sampling')
            n_samples: Number of samples for sampling-based verification
            check_positivity: Whether to check B(x) > 0 for unsafe region
            check_boundary_decreasing: Whether to check ∇B(x) · f(x) < 0 for boundary
            options: Additional options for verification
            
        Returns:
            Verification result
        """
        # Combine options with instance options
        combined_options = self.options.copy()
        if options:
            combined_options.update(options)
        
        # Check which methods are available
        if method == 'mosek':
            try:
                # Try to import the Mosek verifier
                from alan_backend.elfin.barrier.sos_mosek import verify_with_mosek, MOSEK_AVAILABLE
                
                if MOSEK_AVAILABLE:
                    # Use Mosek verifier
                    return verify_with_mosek(
                        barrier_fn=barrier_fn,
                        domain=self.domain,
                        unsafe_region=self.unsafe_region,
                        dynamics_fn=self.dynamics_fn,
                        check_positivity=check_positivity,
                        check_boundary_decreasing=check_boundary_decreasing,
                        options=combined_options
                    )
                else:
                    logger.warning("MOSEK not available, falling back to sampling-based verification")
                    method = 'sampling'
            except ImportError:
                logger.warning("Mosek verifier not available, falling back to sampling-based verification")
                method = 'sampling'
        elif method == 'sparsepop' and not SPARSE_POP_AVAILABLE:
            logger.warning("SparsePOP not available, falling back to sampling-based verification")
            method = 'sampling'
        
        # Choose verification method
        if method == 'sparsepop' and SPARSE_POP_AVAILABLE:
            return self._verify_with_sparsepop(barrier_fn, check_positivity, check_boundary_decreasing)
        else:
            # Update n_samples from options if provided
            if 'n_samples' in combined_options:
                n_samples = combined_options['n_samples']
                
            return self._verify_with_sampling(barrier_fn, n_samples, check_positivity, check_boundary_decreasing)
    
    
    def _verify_with_sparsepop(
        self,
        barrier_fn: BarrierFunction,
        check_positivity: bool = True,
        check_boundary_decreasing: bool = True
    ) -> VerificationResult:
        """
        Verify barrier function using SparsePOP solver.
        
        Args:
            barrier_fn: Barrier function to verify
            check_positivity: Whether to check B(x) > 0 for unsafe region
            check_boundary_decreasing: Whether to check ∇B(x) · f(x) < 0 for boundary
            
        Returns:
            Verification result
        """
        # TODO: Implement SOS verification using SparsePOP
        # This requires setting up a sparse SDP problem for verification
        # For now, fall back to sampling-based verification
        
        logger.warning("SparsePOP-based verification not implemented yet, falling back to sampling")
        return self._verify_with_sampling(barrier_fn, 10000, check_positivity, check_boundary_decreasing)
    
    def _verify_with_sampling(
        self,
        barrier_fn: BarrierFunction,
        n_samples: int = 1000,
        check_positivity: bool = True,
        check_boundary_decreasing: bool = True
    ) -> VerificationResult:
        """
        Verify barrier function using sampling-based approach.
        
        This method samples points from the domain and checks the barrier
        properties at those points. While not a formal verification, it can
        identify counterexamples to the barrier properties.
        
        Args:
            barrier_fn: Barrier function to verify
            n_samples: Number of samples for verification
            check_positivity: Whether to check B(x) > 0 for unsafe region
            check_boundary_decreasing: Whether to check ∇B(x) · f(x) < 0 for boundary
            
        Returns:
            Verification result
        """
        import time
        start_time = time.time()
        
        # Initialize result
        result = VerificationResult(
            success=True,
            status="verified"
        )
        
        # Get domain bounds
        lower, upper = self.domain
        dim = len(lower)
        
        # Generate random samples within the domain
        samples = []
        for _ in range(n_samples):
            sample = lower + np.random.random(dim) * (upper - lower)
            samples.append(sample)
        
        # Check positivity constraint for unsafe region
        if check_positivity:
            unsafe_samples = [s for s in samples if self.unsafe_region(s)]
            
            for sample in unsafe_samples:
                value = barrier_fn(sample)
                
                if value <= 0:
                    # Found a counterexample
                    result.success = False
                    result.status = "refuted"
                    result.counterexample = sample
                    result.barrier_value = value
                    result.violation_reason = "positivity"
                    
                    logger.info(f"Positivity constraint violated: B({sample}) = {value}")
                    
                    # Stop early
                    break
        
        # Check boundary decreasing constraint if not already refuted
        if result.success and check_boundary_decreasing and self.dynamics_fn is not None:
            # Find samples that are close to the boundary (|B(x)| < epsilon)
            epsilon = self.options.get('boundary_epsilon', 1e-3)
            boundary_samples = []
            
            for sample in samples:
                value = barrier_fn(sample)
                if abs(value) < epsilon:
                    boundary_samples.append(sample)
            
            # Check decreasing condition on boundary samples
            for sample in boundary_samples:
                f_x = self.dynamics_fn(sample)
                decreasing_value = barrier_fn.decreasing_condition(sample, f_x)
                
                if decreasing_value >= 0:
                    # Found a counterexample
                    result.success = False
                    result.status = "refuted"
                    result.counterexample = sample
                    result.boundary_decrease = decreasing_value
                    result.violation_reason = "boundary_decreasing"
                    
                    logger.info(f"Boundary decreasing constraint violated: ∇B({sample}) · f({sample}) = {decreasing_value}")
                    
                    # Stop early
                    break
        
        # Set solver time
        result.solver_time = time.time() - start_time
        
        return result
    
    def get_boundary_points(
        self,
        barrier_fn: BarrierFunction,
        n_points: int = 100,
        epsilon: float = 1e-3
    ) -> List[np.ndarray]:
        """
        Generate points on the boundary of the safe set.
        
        Args:
            barrier_fn: Barrier function
            n_points: Number of points to generate
            epsilon: Tolerance for boundary check
            
        Returns:
            List of boundary points
        """
        # Get domain bounds
        lower, upper = self.domain
        dim = len(lower)
        
        # Generate random samples within the domain
        samples = []
        for _ in range(n_points * 10):  # Oversample to get enough boundary points
            sample = lower + np.random.random(dim) * (upper - lower)
            samples.append(sample)
        
        # Find points that are close to the boundary (|B(x)| < epsilon)
        boundary_points = []
        
        for sample in samples:
            value = barrier_fn(sample)
            if abs(value) < epsilon:
                boundary_points.append(sample)
                
                if len(boundary_points) >= n_points:
                    break
        
        return boundary_points
    
    def get_unsafe_points(
        self,
        n_points: int = 100
    ) -> List[np.ndarray]:
        """
        Generate points in the unsafe region.
        
        Args:
            n_points: Number of points to generate
            
        Returns:
            List of unsafe points
        """
        # Get domain bounds
        lower, upper = self.domain
        dim = len(lower)
        
        # Generate random samples within the domain
        unsafe_points = []
        
        for _ in range(n_points * 10):  # Oversample to get enough unsafe points
            sample = lower + np.random.random(dim) * (upper - lower)
            
            if self.unsafe_region(sample):
                unsafe_points.append(sample)
                
                if len(unsafe_points) >= n_points:
                    break
        
        return unsafe_points
