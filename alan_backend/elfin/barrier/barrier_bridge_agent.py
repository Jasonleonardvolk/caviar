"""
Barrier Bridge Agent for ELFIN.

This module provides a bridge between the ELFIN framework and barrier certificates
for safety verification. It allows learning and verifying barrier certificates
that ensure safety properties of dynamical systems.
"""

import os
import sys
import logging
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from pathlib import Path

from alan_backend.elfin.barrier.learner import BarrierLearner, BarrierFunction
from alan_backend.elfin.barrier.sos_verifier import SOSVerifier, VerificationResult
from alan_backend.elfin.koopman.dictionaries import create_dictionary

# Configure logging
logger = logging.getLogger("elfin.barrier.barrier_bridge_agent")


class BarrierBridgeAgent:
    """
    Bridge agent for barrier certificates.
    
    This class provides a high-level interface for learning and verifying
    barrier certificates for safety properties. It integrates the barrier
    learner and verifier components.
    
    Attributes:
        name: Name of the agent
        cache_dir: Directory for caching results
        results: Dictionary of results for different systems
        interaction_log: Event emitter for dependency tracking
    """
    
    def __init__(
        self,
        name: str = "barrier_agent",
        cache_dir: Optional[Path] = None,
        auto_verify: bool = False,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize barrier bridge agent.
        
        Args:
            name: Name of the agent
            cache_dir: Directory for caching results
            auto_verify: Whether to automatically verify after learning
            options: Additional options for learning and verification
        """
        self.name = name
        self.cache_dir = cache_dir
        self.auto_verify = auto_verify
        self.options = options or {}
        self.results = {}
        self.interaction_log = None
        
        # Try to load results from cache
        if cache_dir is not None:
            self.load_cache()
        
        # Set up interaction log for dependency tracking
        try:
            from alan_backend.elfin.stability.core.interactions import InteractionLog
            self.interaction_log = InteractionLog()
            logger.info("Interaction log initialized for dependency tracking")
        except ImportError:
            logger.warning("InteractionLog not available, dependency tracking disabled")
    
    def learn_barrier(
        self,
        system_name: str,
        safe_samples: np.ndarray,
        unsafe_samples: np.ndarray,
        dictionary_type: str = "rbf",
        dictionary_size: int = 100,
        domain: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        dynamics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        safe_region: Optional[Callable[[np.ndarray], bool]] = None,
        unsafe_region: Optional[Callable[[np.ndarray], bool]] = None,
        boundary_samples: Optional[np.ndarray] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> BarrierFunction:
        """
        Learn a barrier certificate from data.
        
        Args:
            system_name: Name of the system
            safe_samples: Safe state samples
            unsafe_samples: Unsafe state samples
            dictionary_type: Type of dictionary to use ('rbf', 'fourier', 'poly')
            dictionary_size: Size of the dictionary
            domain: Domain bounds (lower, upper)
            dynamics_fn: System dynamics function
            safe_region: Function to determine if a state is in the safe region
            unsafe_region: Function to determine if a state is in the unsafe region
            boundary_samples: Samples on the boundary of the safe set
            options: Additional options for learning
            
        Returns:
            Learned barrier function
        """
        # Combine local options with global options
        local_options = self.options.copy()
        if options:
            local_options.update(options)
        
        # Get problem dimensions
        if len(safe_samples) > 0:
            state_dim = safe_samples.shape[1]
        elif len(unsafe_samples) > 0:
            state_dim = unsafe_samples.shape[1]
        else:
            raise ValueError("No samples provided")
        
        # Infer domain if not provided
        if domain is None:
            logger.info("Domain not provided, inferring from samples")
            
            # Combine all samples
            all_samples = np.vstack([safe_samples, unsafe_samples])
            if boundary_samples is not None and len(boundary_samples) > 0:
                all_samples = np.vstack([all_samples, boundary_samples])
            
            # Get min and max values for each dimension
            lower = np.min(all_samples, axis=0)
            upper = np.max(all_samples, axis=0)
            
            # Add margin
            margin = 0.1 * (upper - lower)
            lower -= margin
            upper += margin
            
            domain = (lower, upper)
        
        # Default safe and unsafe region functions if not provided
        if safe_region is None and unsafe_region is not None:
            logger.info("Safe region not provided, using complement of unsafe region")
            safe_region = lambda x: not unsafe_region(x)
        elif unsafe_region is None and safe_region is not None:
            logger.info("Unsafe region not provided, using complement of safe region")
            unsafe_region = lambda x: not safe_region(x)
        elif safe_region is None and unsafe_region is None:
            logger.warning("Neither safe nor unsafe region provided, using samples to infer")
            
            # Train a simple classifier to distinguish safe and unsafe regions
            from sklearn.neighbors import KNeighborsClassifier
            classifier = KNeighborsClassifier(n_neighbors=3)
            
            # Create training data
            X = np.vstack([safe_samples, unsafe_samples])
            y = np.zeros(len(X))
            y[len(safe_samples):] = 1  # 0 for safe, 1 for unsafe
            
            # Train classifier
            classifier.fit(X, y)
            
            # Define region functions
            safe_region = lambda x: classifier.predict(np.array([x]))[0] == 0
            unsafe_region = lambda x: classifier.predict(np.array([x]))[0] == 1
        
        # Create dictionary
        dictionary = create_dictionary(
            dict_type=dictionary_type,
            state_dim=state_dim,
            dict_size=dictionary_size,
            domain=domain,
            options=local_options
        )
        
        # Create barrier learner
        learner = BarrierLearner(
            dictionary=dictionary,
            safe_region=safe_region,
            dynamics_fn=dynamics_fn,
            options=local_options
        )
        
        # Learn barrier function
        start_time = time.time()
        barrier_fn = learner.fit(
            safe_samples=safe_samples,
            unsafe_samples=unsafe_samples,
            boundary_samples=boundary_samples
        )
        learning_time = time.time() - start_time
        
        # Create verifier
        verifier = SOSVerifier(
            domain=domain,
            unsafe_region=unsafe_region,
            dynamics_fn=dynamics_fn,
            options=local_options
        )
        
        # Store results
        self.results[system_name] = {
            'barrier': barrier_fn,
            'learner': learner,
            'verifier': verifier,
            'domain': domain,
            'dictionary': dictionary,
            'safe_region': safe_region,
            'unsafe_region': unsafe_region,
            'dynamics_fn': dynamics_fn,
            'learning_time': learning_time,
            'options': local_options,
            'data': {
                'safe_samples': safe_samples,
                'unsafe_samples': unsafe_samples,
                'boundary_samples': boundary_samples
            }
        }
        
        # Emit event if interaction log is available
        if self.interaction_log is not None:
            self.interaction_log.emit("barrier_learned", {
                "system_id": system_name,
                "dictionary_type": dictionary_type,
                "dictionary_size": dictionary_size,
                "n_safe_samples": len(safe_samples),
                "n_unsafe_samples": len(unsafe_samples),
                "n_boundary_samples": 0 if boundary_samples is None else len(boundary_samples),
                "learning_time": learning_time
            })
        
        # Verify if auto_verify is enabled
        if self.auto_verify:
            self.verify(system_name)
        
        # Save results to cache
        if self.cache_dir is not None:
            self.save_cache()
        
        return barrier_fn
    
    def verify(
        self,
        system_name: str,
        method: str = 'mosek',
        options: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """
        Verify barrier certificate for a system.
        
        Args:
            system_name: Name of the system
            method: Verification method ('mosek', 'sparsepop', 'sampling')
            options: Additional options for verification
            
        Returns:
            Verification result
        """
        # Check if system exists
        if system_name not in self.results:
            raise ValueError(f"System '{system_name}' not found")
        
        # Get system result
        result = self.results[system_name]
        
        # Combine local options with global options
        local_options = result['options'].copy()
        if options:
            local_options.update(options)
        
        # Get barrier function and verifier
        barrier_fn = result['barrier']
        verifier = result['verifier']
        
        # Verify barrier function
        start_time = time.time()
        verification_result = verifier.verify(
            barrier_fn=barrier_fn,
            method=method,
            options=local_options
        )
        verification_time = time.time() - start_time
        
        # Update result
        result['verification'] = {
            'result': verification_result,
            'method': method,
            'time': verification_time
        }
        
        # Emit event if interaction log is available
        if self.interaction_log is not None:
            self.interaction_log.emit("barrier_verified", {
                "system_id": system_name,
                "success": verification_result.success,
                "status": verification_result.status,
                "verification_time": verification_time,
                "error_code": verification_result.get_error_code()
            })
        
        # Save results to cache
        if self.cache_dir is not None:
            self.save_cache()
        
        return verification_result
    
    def refine_once(
        self,
        system_name: str,
        counterexample: np.ndarray,
        is_unsafe: bool = True,
        is_boundary: bool = False
    ) -> BarrierFunction:
        """
        Refine barrier function with a counterexample.
        
        Args:
            system_name: Name of the system
            counterexample: Counterexample point
            is_unsafe: Whether counterexample is from unsafe region
            is_boundary: Whether counterexample is a boundary point
            
        Returns:
            Refined barrier function
        """
        # Check if system exists
        if system_name not in self.results:
            raise ValueError(f"System '{system_name}' not found")
        
        # Get system result
        result = self.results[system_name]
        
        # Get barrier function and learner
        barrier_fn = result['barrier']
        learner = result['learner']
        
        # Refine barrier function
        start_time = time.time()
        refined_fn = learner.refine_with_counterexample(
            barrier_fn=barrier_fn,
            counterexample=counterexample,
            is_unsafe=is_unsafe,
            is_boundary=is_boundary
        )
        refinement_time = time.time() - start_time
        
        # Update result
        result['barrier'] = refined_fn
        result['refinement'] = {
            'counterexample': counterexample,
            'is_unsafe': is_unsafe,
            'is_boundary': is_boundary,
            'time': refinement_time,
            'n_refinements': learner.refinements
        }
        
        # Emit event if interaction log is available
        if self.interaction_log is not None:
            self.interaction_log.emit("barrier_refined", {
                "system_id": system_name,
                "counterexample": counterexample.tolist(),
                "is_unsafe": is_unsafe,
                "is_boundary": is_boundary,
                "refinement_time": refinement_time,
                "n_refinements": learner.refinements
            })
        
        # Save results to cache
        if self.cache_dir is not None:
            self.save_cache()
        
        return refined_fn
    
    def refine_auto(
        self,
        system_name: str,
        max_iterations: int = 10,
        stop_on_success: bool = True,
        method: str = 'sampling'
    ) -> VerificationResult:
        """
        Automatically refine barrier function until verification succeeds or max iterations.
        
        Args:
            system_name: Name of the system
            max_iterations: Maximum number of refinement iterations
            stop_on_success: Whether to stop when verification succeeds
            method: Verification method
            
        Returns:
            Final verification result
        """
        # Check if system exists
        if system_name not in self.results:
            raise ValueError(f"System '{system_name}' not found")
        
        # Get system result
        result = self.results[system_name]
        
        # Initialize verification result
        verification_result = None
        
        # Refinement loop
        for i in range(max_iterations):
            # Verify current barrier function
            verification_result = self.verify(
                system_name=system_name,
                method=method
            )
            
            # Check if verification succeeded
            if verification_result.success:
                logger.info(f"Verification succeeded after {i} refinements")
                if stop_on_success:
                    break
            
            # Get counterexample for refinement
            if verification_result.counterexample is not None:
                # Determine counterexample type
                is_unsafe = verification_result.violation_reason == "positivity"
                is_boundary = verification_result.violation_reason == "boundary_decreasing"
                
                # Refine with counterexample
                self.refine_once(
                    system_name=system_name,
                    counterexample=verification_result.counterexample,
                    is_unsafe=is_unsafe,
                    is_boundary=is_boundary
                )
                
                logger.info(f"Refined barrier function with counterexample from {verification_result.violation_reason}")
            else:
                logger.warning("Verification failed but no counterexample provided")
                break
        
        # Update auto-refinement statistics
        result['auto_refinement'] = {
            'iterations': i + 1,
            'success': verification_result.success if verification_result else False,
            'max_iterations': max_iterations
        }
        
        # Emit event if interaction log is available
        if self.interaction_log is not None:
            self.interaction_log.emit("barrier_auto_refined", {
                "system_id": system_name,
                "iterations": i + 1,
                "success": verification_result.success if verification_result else False,
                "max_iterations": max_iterations
            })
        
        # Save results to cache
        if self.cache_dir is not None:
            self.save_cache()
        
        return verification_result
    
    def is_safe(
        self,
        system_name: str,
        state: np.ndarray
    ) -> bool:
        """
        Check if a state is safe according to the barrier function.
        
        Args:
            system_name: Name of the system
            state: State to check
            
        Returns:
            True if state is safe (B(x) ≤ 0)
        """
        # Check if system exists
        if system_name not in self.results:
            raise ValueError(f"System '{system_name}' not found")
        
        # Get barrier function
        barrier_fn = self.results[system_name]['barrier']
        
        # Check if state is safe
        return barrier_fn.is_safe(state)
    
    def get_distance_to_boundary(
        self,
        system_name: str,
        state: np.ndarray
    ) -> float:
        """
        Get approximate distance to the boundary of the safe set.
        
        Args:
            system_name: Name of the system
            state: State to check
            
        Returns:
            Approximate distance to the boundary
        """
        # Check if system exists
        if system_name not in self.results:
            raise ValueError(f"System '{system_name}' not found")
        
        # Get barrier function
        barrier_fn = self.results[system_name]['barrier']
        
        # Get distance to boundary
        return barrier_fn.distance_to_boundary(state)
    
    def get_summary(
        self,
        system_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get summary of barrier certificate and verification results.
        
        Args:
            system_name: Name of the system (if None, summarize all systems)
            
        Returns:
            Summary dictionary
        """
        if system_name is not None:
            # Check if system exists
            if system_name not in self.results:
                raise ValueError(f"System '{system_name}' not found")
            
            # Get system result
            result = self.results[system_name]
            
            # Create summary
            summary = {
                'system_name': system_name,
                'has_barrier': 'barrier' in result,
                'has_verification': 'verification' in result,
                'has_refinement': 'refinement' in result,
                'has_auto_refinement': 'auto_refinement' in result
            }
            
            # Add barrier information
            if 'barrier' in result:
                barrier_fn = result['barrier']
                dictionary = result['dictionary']
                
                summary['barrier'] = {
                    'weights_shape': barrier_fn.weights.shape,
                    'dictionary_type': type(dictionary).__name__,
                    'learning_time': result.get('learning_time', 0.0)
                }
            
            # Add verification information
            if 'verification' in result:
                verification = result['verification']
                verification_result = verification['result']
                
                summary['verification'] = {
                    'success': verification_result.success,
                    'status': verification_result.status,
                    'method': verification['method'],
                    'time': verification['time'],
                    'has_counterexample': verification_result.counterexample is not None,
                    'violation_reason': verification_result.violation_reason
                }
            
            # Add refinement information
            if 'refinement' in result:
                refinement = result['refinement']
                
                summary['refinement'] = {
                    'n_refinements': refinement.get('n_refinements', 0),
                    'time': refinement.get('time', 0.0)
                }
            
            # Add auto-refinement information
            if 'auto_refinement' in result:
                auto_refinement = result['auto_refinement']
                
                summary['auto_refinement'] = {
                    'iterations': auto_refinement.get('iterations', 0),
                    'success': auto_refinement.get('success', False),
                    'max_iterations': auto_refinement.get('max_iterations', 0)
                }
            
            return summary
        else:
            # Summarize all systems
            return {
                'systems': list(self.results.keys()),
                'n_systems': len(self.results),
                'summaries': {
                    name: self.get_summary(name)
                    for name in self.results
                }
            }
    
    def save_cache(self) -> None:
        """Save agent results to cache directory."""
        if self.cache_dir is None:
            logger.warning("No cache directory specified, results not saved")
            return
        
        # Create cache directory if it doesn't exist
        cache_dir = self.cache_dir / f"{self.name}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a serializable version of the results
        cache_data = {
            'name': self.name,
            'timestamp': time.time(),
            'systems': list(self.results.keys())
        }
        
        # Save each system to a separate file
        for system_name, result in self.results.items():
            system_cache = {
                'system_name': system_name,
                'options': result['options'],
                'domain': [x.tolist() for x in result['domain']],
                'data': {
                    'safe_samples': result['data']['safe_samples'].tolist(),
                    'unsafe_samples': result['data']['unsafe_samples'].tolist(),
                    'boundary_samples': [] if result['data']['boundary_samples'] is None else result['data']['boundary_samples'].tolist()
                }
            }
            
            # Add barrier weights
            if 'barrier' in result:
                system_cache['barrier'] = {
                    'weights': result['barrier'].weights.tolist()
                }
            
            # Add dictionary information
            if 'dictionary' in result:
                dictionary = result['dictionary']
                system_cache['dictionary'] = {
                    'type': type(dictionary).__name__,
                    'state_dim': dictionary.state_dim if hasattr(dictionary, 'state_dim') else None,
                    'dict_size': dictionary.dict_size if hasattr(dictionary, 'dict_size') else None
                }
            
            # Add verification information
            if 'verification' in result:
                verification = result['verification']
                verification_result = verification['result']
                
                system_cache['verification'] = {
                    'success': verification_result.success,
                    'status': verification_result.status,
                    'method': verification['method'],
                    'time': verification['time'],
                    'counterexample': None if verification_result.counterexample is None else verification_result.counterexample.tolist(),
                    'violation_reason': verification_result.violation_reason
                }
            
            # Add refinement information
            if 'refinement' in result:
                refinement = result['refinement']
                
                system_cache['refinement'] = {
                    'counterexample': refinement['counterexample'].tolist(),
                    'is_unsafe': refinement['is_unsafe'],
                    'is_boundary': refinement['is_boundary'],
                    'time': refinement['time'],
                    'n_refinements': refinement['n_refinements']
                }
            
            # Add auto-refinement information
            if 'auto_refinement' in result:
                auto_refinement = result['auto_refinement']
                
                system_cache['auto_refinement'] = {
                    'iterations': auto_refinement['iterations'],
                    'success': auto_refinement['success'],
                    'max_iterations': auto_refinement['max_iterations']
                }
            
            # Save system cache
            system_cache_file = cache_dir / f"{system_name}.json"
            with open(system_cache_file, "w") as f:
                json.dump(system_cache, f, indent=2)
        
        # Save main cache file
        cache_file = cache_dir / "cache.json"
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Saved cache to {cache_file}")
    
    def load_cache(self) -> None:
        """Load agent results from cache directory."""
        if self.cache_dir is None:
            logger.warning("No cache directory specified, no results loaded")
            return
        
        # Check if cache file exists
        cache_dir = self.cache_dir / f"{self.name}"
        cache_file = cache_dir / "cache.json"
        
        if not cache_file.exists():
            logger.warning(f"Cache file {cache_file} not found")
            return
        
        # Load main cache file
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            
            # Load each system
            for system_name in cache_data.get('systems', []):
                system_cache_file = cache_dir / f"{system_name}.json"
                
                if system_cache_file.exists():
                    with open(system_cache_file, "r") as f:
                        system_cache = json.load(f)
                    
                    # Recreate system
                    self._recreate_system_from_cache(system_cache)
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    
    def _recreate_system_from_cache(self, system_cache: Dict[str, Any]) -> None:
        """
        Recreate a system from cache data.
        
        Args:
            system_cache: Cached system data
        """
        # Get system name
        system_name = system_cache['system_name']
        
        # Get domain
        domain = tuple(np.array(x) for x in system_cache['domain'])
        
        # Get options
        options = system_cache.get('options', {})
        
        # Get data
        data = system_cache.get('data', {})
        safe_samples = np.array(data.get('safe_samples', []))
        unsafe_samples = np.array(data.get('unsafe_samples', []))
        boundary_samples = None
        if 'boundary_samples' in data and data['boundary_samples']:
            boundary_samples = np.array(data['boundary_samples'])
        
        # Get dictionary information
        dictionary_info = system_cache.get('dictionary', {})
        dictionary_type = dictionary_info.get('type', 'rbf').lower().replace('dictionary', '')
        state_dim = dictionary_info.get('state_dim', safe_samples.shape[1] if len(safe_samples) > 0 else 2)
        dict_size = dictionary_info.get('dict_size', 100)
        
        # Create dictionary
        dictionary = create_dictionary(
            dict_type=dictionary_type,
            state_dim=state_dim,
            dict_size=dict_size,
            domain=domain,
            options=options
        )
        
        # Create safe and unsafe region functions
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=3)
        
        # Create training data
        X = np.vstack([safe_samples, unsafe_samples])
        y = np.zeros(len(X))
        y[len(safe_samples):] = 1  # 0 for safe, 1 for unsafe
        
        # Train classifier
        classifier.fit(X, y)
        
        # Define region functions
        safe_region = lambda x: classifier.predict(np.array([x]))[0] == 0
        unsafe_region = lambda x: classifier.predict(np.array([x]))[0] == 1
        
        # Create barrier learner
        learner = BarrierLearner(
            dictionary=dictionary,
            safe_region=safe_region,
            dynamics_fn=None,  # We can't recreate the dynamics function
            options=options
        )
        
        # Create barrier function
        barrier_weights = np.array(system_cache.get('barrier', {}).get('weights', []))
        if len(barrier_weights) > 0:
            barrier_fn = BarrierFunction(
                dictionary=dictionary,
                weights=barrier_weights,
                safe_region=safe_region
            )
        else:
            # Learn a new barrier function
            barrier_fn = learner.fit(
                safe_samples=safe_samples,
                unsafe_samples=unsafe_samples,
                boundary_samples=boundary_samples
            )
        
        # Create verifier
        verifier = SOSVerifier(
            domain=domain,
            unsafe_region=unsafe_region,
            dynamics_fn=None,  # We can't recreate the dynamics function
            options=options
        )
        
        # Store results
        self.results[system_name] = {
            'barrier': barrier_fn,
            'learner': learner,
            'verifier': verifier,
            'domain': domain,
            'dictionary': dictionary,
            'safe_region': safe_region,
            'unsafe_region': unsafe_region,
            'dynamics_fn': None,  # We can't recreate the dynamics function
            'options': options,
            'data': {
                'safe_samples': safe_samples,
                'unsafe_samples': unsafe_samples,
                'boundary_samples': boundary_samples
            }
        }
        
        logger.info(f"Recreated system {system_name} from cache")


# Helper function to create a double integrator agent
def create_double_integrator_agent(
    verify: bool = True,
    dict_type: str = "rbf",
    dict_size: int = 100,
    cache_dir: Optional[Path] = None
) -> Tuple[BarrierBridgeAgent, BarrierFunction]:
    """
    Create a barrier bridge agent for a double integrator system.
    
    Args:
        verify: Whether to verify the barrier certificate
        dict_type: Dictionary type ('rbf', 'fourier', 'poly')
        dict_size: Dictionary size
        cache_dir: Cache directory
        
    Returns:
        Barrier bridge agent and barrier function
    """
    import numpy as np
    
    # Create agent
    agent = BarrierBridgeAgent(
        name=f"barrier_double_integrator_{dict_type}_{dict_size}",
        cache_dir=cache_dir,
        auto_verify=verify
    )
    
    # Check if agent already has results
    if "double_integrator" in agent.results:
        logger.info("Using cached double integrator barrier certificate")
        return agent, agent.results["double_integrator"]["barrier"]
    
    # Define state space and domain
    state_dim = 4  # [x, y, vx, vy]
    domain = (
        np.array([-5.0, -5.0, -2.0, -2.0]),  # Lower bounds
        np.array([5.0, 5.0, 2.0, 2.0])       # Upper bounds
    )
    
    # Define obstacle (circular region around origin)
    obstacle_center = np.array([0.0, 0.0])
    obstacle_radius = 1.0
    
    # Define safe and unsafe regions
    def safe_region(state):
        # Check if outside obstacle
        x, y = state[0], state[1]
        dist = np.sqrt(x**2 + y**2)
        return dist >= obstacle_radius
    
    def unsafe_region(state):
        # Check if inside obstacle
        x, y = state[0], state[1]
        dist = np.sqrt(x**2 + y**2)
        return dist < obstacle_radius
    
    # Define double integrator dynamics
    def double_integrator_dynamics(state):
        """
        Double integrator dynamics.
        
        dx/dt = vx
        dy/dt = vy
        dvx/dt = 0
        dvy/dt = 0
        
        Args:
            state: [x, y, vx, vy]
            
        Returns:
            State derivative
        """
        # Extract state components
        x, y, vx, vy = state
        
        # Define dynamics (no acceleration)
        return np.array([vx, vy, 0.0, 0.0])
    
    # Generate safe and unsafe samples
    n_samples = 1000
    safe_samples = []
    unsafe_samples = []
    
    for _ in range(n_samples * 2):
        # Generate random point in domain
        x = np.random.uniform(domain[0][0], domain[1][0])
        y = np.random.uniform(domain[0][1], domain[1][1])
        vx = np.random.uniform(domain[0][2], domain[1][2])
        vy = np.random.uniform(domain[0][3], domain[1][3])
        
        state = np.array([x, y, vx, vy])
        
        # Determine if point is safe or unsafe
        if safe_region(state):
            safe_samples.append(state)
            if len(safe_samples) >= n_samples:
                break
        else:
            unsafe_samples.append(state)
    
    # Ensure we have enough samples of each type
    safe_samples = np.array(safe_samples[:n_samples])
    
    # Fill remaining unsafe samples if needed
    while len(unsafe_samples) < n_samples:
        # Generate random point inside obstacle
        r = np.random.uniform(0, obstacle_radius)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        vx = np.random.uniform(domain[0][2], domain[1][2])
        vy = np.random.uniform(domain[0][3], domain[1][3])
        
        state = np.array([x, y, vx, vy])
        unsafe_samples.append(state)
    
    unsafe_samples = np.array(unsafe_samples[:n_samples])
    
    # Learn barrier certificate
    barrier_fn = agent.learn_barrier(
        system_name="double_integrator",
        safe_samples=safe_samples,
        unsafe_samples=unsafe_samples,
        dictionary_type=dict_type,
        dictionary_size=dict_size,
        domain=domain,
        dynamics_fn=double_integrator_dynamics,
        safe_region=safe_region,
        unsafe_region=unsafe_region,
        options={
            'safe_margin': 0.1,
            'unsafe_margin': 0.1,
            'boundary_margin': 0.1
        }
    )
    
    return agent, barrier_fn
