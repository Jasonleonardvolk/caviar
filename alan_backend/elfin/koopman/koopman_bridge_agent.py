"""
Koopman Bridge Agent for Koopman-based Lyapunov function generation.

This module provides an agent that bridges the gap between trajectory data
and Lyapunov verification by automatically learning Koopman operators and
constructing Lyapunov functions from them.
"""

import os
import logging
import pathlib
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

from alan_backend.elfin.stability.agents import StabilityAgent
from alan_backend.elfin.stability.verify import VerificationResult, ProofStatus
from alan_backend.elfin.errors import VerificationError
from .dictionaries import StandardDictionary, create_dictionary
from .edmd import edmd_fit, load_trajectory_data, estimate_optimal_dict_size
from .koopman_lyap import create_koopman_lyapunov, KoopmanLyapunov

logger = logging.getLogger(__name__)


class KoopmanBridgeAgent:
    """
    Agent for learning Koopman operators and generating Lyapunov functions.
    
    This agent automates the process of learning Koopman operators from
    trajectory data and constructing Lyapunov functions that can be
    verified for stability.
    """
    
    def __init__(
        self,
        name: str,
        stability_agent: Optional[StabilityAgent] = None,
        cache_dir: Optional[pathlib.Path] = None,
        auto_verify: bool = True
    ):
        """
        Initialize a Koopman Bridge Agent.
        
        Args:
            name: Name of the agent
            stability_agent: StabilityAgent for verification and tracking
            cache_dir: Cache directory for storing results
            auto_verify: Whether to automatically verify Lyapunov functions after creation
        """
        self.name = name
        
        # Create or use the provided StabilityAgent
        if stability_agent is None:
            # Default cache location
            if cache_dir is None:
                cache_dir = pathlib.Path(os.environ.get(
                    "ELFIN_CACHE_DIR",
                    pathlib.Path.home() / ".elfin" / "cache"
                ))
            
            # Create StabilityAgent
            self.stab_agent = StabilityAgent(
                f"{name}_stability",
                cache_dir=cache_dir
            )
        else:
            self.stab_agent = stability_agent
        
        self.auto_verify = auto_verify
        self.cache_dir = self.stab_agent.cache_dir
        self.results = {}
    
    def learn_from_file(
        self,
        file_path: str,
        system_name: str,
        state_dim: int,
        dict_type: str = "rbf",
        dict_size: int = 100,
        continuous_time: bool = True,
        dt: Optional[float] = None,
        λ_cut: float = 0.99,
        skip_header: int = 0,
        delimiter: str = ",",
        verify_domain: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **dict_params
    ) -> KoopmanLyapunov:
        """
        Learn a Koopman operator and create a Lyapunov function from trajectory data file.
        
        Args:
            file_path: Path to the data file
            system_name: Name of the system
            state_dim: Dimension of the state space
            dict_type: Type of dictionary ('rbf', 'fourier', 'poly')
            dict_size: Size of the dictionary
            continuous_time: Whether the system is continuous-time
            dt: Time step (if None, inferred from data)
            λ_cut: Cutoff for selecting stable modes
            skip_header: Number of header rows to skip
            delimiter: Delimiter used in the file
            verify_domain: Domain for verification (if None, [-1,1]^dim)
            **dict_params: Additional parameters for the dictionary
            
        Returns:
            KoopmanLyapunov function
        """
        # Load trajectory data
        try:
            x, x_next = load_trajectory_data(
                file_path=file_path,
                state_dim=state_dim,
                dt=dt,
                skip_header=skip_header,
                delimiter=delimiter
            )
            
            interaction_id = self.stab_agent.interaction_log.add_interaction(
                topic="koopman_data_loaded",
                source=self.name,
                target=system_name,
                payload={
                    "file_path": file_path,
                    "state_dim": state_dim,
                    "n_samples": len(x),
                    "dt": dt
                }
            )
            
            logger.info(f"Loaded {len(x)} data points from {file_path}")
            
        except Exception as e:
            # Log error
            error_msg = f"Failed to load trajectory data: {e}"
            logger.error(error_msg)
            
            self.stab_agent.interaction_log.add_interaction(
                topic="koopman_error",
                source=self.name,
                target=system_name,
                payload={
                    "error": error_msg,
                    "file_path": file_path
                }
            )
            
            raise VerificationError(
                code="KOOP_001",
                detail=f"Failed to load trajectory data: {e}",
                system_id=system_name
            )
        
        # Create Koopman-based Lyapunov function
        return self.learn_from_data(
            x=x,
            x_next=x_next,
            system_name=system_name,
            dict_type=dict_type,
            dict_size=dict_size,
            continuous_time=continuous_time,
            dt=dt,
            λ_cut=λ_cut,
            verify_domain=verify_domain,
            **dict_params
        )
    
    def learn_from_data(
        self,
        x: np.ndarray,
        x_next: np.ndarray,
        system_name: str,
        dict_type: str = "rbf",
        dict_size: int = 100,
        continuous_time: bool = True,
        dt: float = 1.0,
        λ_cut: float = 0.99,
        verify_domain: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **dict_params
    ) -> KoopmanLyapunov:
        """
        Learn a Koopman operator and create a Lyapunov function from trajectory data.
        
        Args:
            x: State data
            x_next: Next state data
            system_name: Name of the system
            dict_type: Type of dictionary ('rbf', 'fourier', 'poly')
            dict_size: Size of the dictionary
            continuous_time: Whether the system is continuous-time
            dt: Time step
            λ_cut: Cutoff for selecting stable modes
            verify_domain: Domain for verification (if None, [-1,1]^dim)
            **dict_params: Additional parameters for the dictionary
            
        Returns:
            KoopmanLyapunov function
        """
        # Start timing
        start_time = time.time()
        
        # Infer state dimension
        state_dim = x.shape[1]
        
        # Create dictionary
        log_msg = f"Creating {dict_type} dictionary with size {dict_size} for {state_dim}D system"
        logger.info(log_msg)
        
        try:
            if dict_type == "rbf":
                dictionary = create_dictionary(
                    dict_type=dict_type,
                    dim=state_dim,
                    n_centers=dict_size,
                    **dict_params
                )
            elif dict_type == "fourier":
                dictionary = create_dictionary(
                    dict_type=dict_type,
                    dim=state_dim,
                    n_frequencies=dict_size,
                    **dict_params
                )
            elif dict_type == "poly":
                dictionary = create_dictionary(
                    dict_type=dict_type,
                    dim=state_dim,
                    degree=dict_size,
                    **dict_params
                )
            else:
                raise ValueError(f"Unknown dictionary type: {dict_type}")
        
        except Exception as e:
            # Log error
            error_msg = f"Failed to create dictionary: {e}"
            logger.error(error_msg)
            
            self.stab_agent.interaction_log.add_interaction(
                topic="koopman_error",
                source=self.name,
                target=system_name,
                payload={
                    "error": error_msg,
                    "dict_type": dict_type,
                    "dict_size": dict_size
                }
            )
            
            raise VerificationError(
                code="KOOP_002",
                detail=f"Failed to create dictionary: {e}",
                system_id=system_name
            )
        
        # Fit Koopman operator
        log_msg = f"Fitting Koopman operator using EDMD on {len(x)} data points"
        logger.info(log_msg)
        
        try:
            k_matrix, meta = edmd_fit(
                dictionary=dictionary,
                x=x,
                x_next=x_next
            )
            
            # Log the fitting
            self.stab_agent.interaction_log.add_interaction(
                topic="koopman_edmd_fitted",
                source=self.name,
                target=system_name,
                payload={
                    "dict_type": dict_type,
                    "dict_size": dict_size,
                    "k_matrix_shape": k_matrix.shape,
                    "mse": meta["mse"],
                    "rank": meta.get("rank", 0)
                }
            )
            
        except Exception as e:
            # Log error
            error_msg = f"Failed to fit Koopman operator: {e}"
            logger.error(error_msg)
            
            self.stab_agent.interaction_log.add_interaction(
                topic="koopman_error",
                source=self.name,
                target=system_name,
                payload={
                    "error": error_msg,
                    "n_samples": len(x)
                }
            )
            
            raise VerificationError(
                code="KOOP_003",
                detail=f"Failed to fit Koopman operator: {e}",
                system_id=system_name
            )
        
        # Create Lyapunov function
        lyap_name = f"koopman_{system_name}"
        log_msg = f"Creating Koopman-based Lyapunov function with λ_cut={λ_cut}"
        logger.info(log_msg)
        
        try:
            # Set default parameters for different dictionary types
            if not dict_params:
                if dict_type == "rbf":
                    # Scale sigma based on data range
                    x_range = np.max(x) - np.min(x)
                    dict_params = {"sigma": 0.1 * x_range}
                elif dict_type == "fourier":
                    dict_params = {"max_freq": 3.0}
                elif dict_type == "poly":
                    dict_params = {"include_cross_terms": True}
            
            # Create Lyapunov function
            lyap_fn = create_koopman_lyapunov(
                name=lyap_name,
                k_matrix=k_matrix,
                dictionary=dictionary,
                lambda_cut=λ_cut,
                continuous_time=continuous_time,
                dt=dt,
                metadata={
                    "fit_mse": meta["mse"],
                    "state_dim": state_dim,
                    "n_samples": len(x),
                    "dict_type": dict_type,
                    "dict_size": dict_size,
                    "dict_params": dict_params
                }
            )
            
            # Log the creation
            self.stab_agent.interaction_log.add_interaction(
                topic="koopman_lyapunov_created",
                source=self.name,
                target=system_name,
                payload={
                    "lyapunov_name": lyap_fn.name,
                    "n_modes": len(lyap_fn.stable_indices),
                    "eigenvalues": [complex(ev) for ev in lyap_fn.eigenvalues],
                    "stable_indices": lyap_fn.stable_indices,
                    "dict_type": dict_type,
                    "elapsed_time": time.time() - start_time
                }
            )
            
        except Exception as e:
            # Log error
            error_msg = f"Failed to create Lyapunov function: {e}"
            logger.error(error_msg)
            
            self.stab_agent.interaction_log.add_interaction(
                topic="koopman_error",
                source=self.name,
                target=system_name,
                payload={
                    "error": error_msg,
                    "lambda_cut": λ_cut
                }
            )
            
            raise VerificationError(
                code="KOOP_004",
                detail=f"Failed to create Lyapunov function: {e}",
                system_id=system_name
            )
        
        # Verify Lyapunov function if requested
        if self.auto_verify and verify_domain is not None:
            self.verify(lyap_fn, system_name, verify_domain)
        elif self.auto_verify:
            # Create default domain based on data bounds
            lower = np.min(x, axis=0)
            upper = np.max(x, axis=0)
            # Add some margin
            margin = 0.1 * (upper - lower)
            domain = (lower - margin, upper + margin)
            self.verify(lyap_fn, system_name, domain)
        
        # Store result
        self.results[system_name] = {
            "lyapunov": lyap_fn,
            "dictionary": dictionary,
            "k_matrix": k_matrix,
            "meta": meta,
            "data": {
                "n_samples": len(x),
                "state_dim": state_dim
            }
        }
        
        return lyap_fn
    
    def refine_once(
        self,
        lyap_fn: KoopmanLyapunov,
        system_name: str,
        counterexample: np.ndarray,
        dynamics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> KoopmanLyapunov:
        """
        Refine a Koopman Lyapunov function using a counterexample from MILP verification.
        
        This function takes a counterexample where verification failed, simulates a trajectory
        from that point, and adds it to the training data to refit the Koopman operator.
        
        Args:
            lyap_fn: Lyapunov function to refine
            system_name: Name of the system
            counterexample: State where verification failed
            dynamics_fn: Dynamics function to simulate a trajectory (if None, use learned dynamics)
            
        Returns:
            Refined Lyapunov function
        """
        if system_name not in self.results:
            raise ValueError(f"System {system_name} not found in results")
        
        # Get original data
        original_result = self.results[system_name]
        original_x = original_result.get("data", {}).get("x")
        original_x_next = original_result.get("data", {}).get("x_next")
        
        if original_x is None or original_x_next is None:
            # No data available, fall back to stored parameters
            params = original_result.get("params", {})
            dictionary = original_result["dictionary"]
            
            return lyap_fn
        
        # Check if dynamics_fn is provided
        if dynamics_fn is None:
            # Use a default dynamics function based on the dictionary
            msg = f"No dynamics function provided for {system_name}, using original Koopman model"
            logger.warning(msg)
            return lyap_fn
        
        # Generate a short trajectory from the counterexample
        n_steps = 10
        x_ce = [counterexample]
        
        # Time step - use from the Lyapunov function or default
        dt = lyap_fn.dt if hasattr(lyap_fn, "dt") else 0.1
        
        # Generate trajectory
        for i in range(n_steps):
            # Simple Euler integration
            x_prev = x_ce[-1]
            x_next = x_prev + dt * dynamics_fn(x_prev)
            x_ce.append(x_next)
        
        # Convert to numpy array
        x_ce = np.array(x_ce)
        
        # Extract x and x_next
        x_new = x_ce[:-1]
        x_next_new = x_ce[1:]
        
        # Combine with original data (but upweight the counterexample data)
        # We duplicate the counterexample trajectory to give it more weight
        n_duplicates = 5  # Duplication factor
        
        x_combined = np.vstack([original_x] + [x_new] * n_duplicates)
        x_next_combined = np.vstack([original_x_next] + [x_next_new] * n_duplicates)
        
        # Log refinement
        self.stab_agent.interaction_log.add_interaction(
            topic="koopman_refine",
            source=self.name,
            target=system_name,
            payload={
                "counterexample": counterexample.tolist(),
                "n_original_samples": len(original_x),
                "n_new_samples": len(x_new) * n_duplicates
            }
        )
        
        # Create new dictionary (same type and parameters as original)
        dictionary = original_result["dictionary"]
        
        # Refit Koopman operator
        k_matrix, meta = edmd_fit(
            dictionary=dictionary,
            x=x_combined,
            x_next=x_next_combined
        )
        
        # Update result
        self.results[system_name].update({
            "k_matrix": k_matrix,
            "meta": meta,
            "data": {
                "x": x_combined,
                "x_next": x_next_combined,
                "n_samples": len(x_combined)
            }
        })
        
        # Create new Lyapunov function
        new_lyap_fn = create_koopman_lyapunov(
            name=lyap_fn.name,
            k_matrix=k_matrix,
            dictionary=dictionary,
            lambda_cut=0.98,  # Use same cut as before
            continuous_time=lyap_fn.is_continuous,
            dt=lyap_fn.dt,
            weighting=lyap_fn.weighting if hasattr(lyap_fn, "weighting") else "uniform"
        )
        
        # Update result
        self.results[system_name]["lyapunov"] = new_lyap_fn
        
        return new_lyap_fn
    
    def verify(
        self,
        lyap_fn: KoopmanLyapunov,
        system_name: str,
        domain: Tuple[np.ndarray, np.ndarray],
        dynamics_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        refine_on_failure: bool = False,
        **verify_params
    ) -> Dict[str, Any]:
        """
        Verify a Koopman-based Lyapunov function.
        
        Args:
            lyap_fn: Lyapunov function to verify
            system_name: Name of the system
            domain: Domain for verification as (lower, upper)
            dynamics_fn: Dynamics function (if None, use learned dynamics)
            **verify_params: Additional verification parameters
            
        Returns:
            Verification result
        """
        # Log verification
        log_msg = f"Verifying Koopman-based Lyapunov function for {system_name}"
        logger.info(log_msg)
        
        try:
            # Run verification
            result = self.stab_agent.verify(
                system=lyap_fn,
                domain=domain,
                system_id=system_name,
                **verify_params
            )
            
            # Get the proof object
            proof = self.stab_agent.get_latest_proof(system_id=system_name)
            
            # Emit events for dependency graph integration
            if proof is not None:
                # Emit proof_added event with proof hash
                self.stab_agent.interaction_log.emit("proof_added", {
                    "proof_hash": proof.proof_hash,
                    "system_id": system_name,
                    "proof_type": "koopman"
                })
                
                # If verification failed, emit counterexample
                if proof.status != ProofStatus.VERIFIED and proof.counterexample is not None:
                    self.stab_agent.interaction_log.emit("counterexample", {
                        "system_id": system_name,
                        "counterexample": proof.counterexample.tolist() if isinstance(proof.counterexample, np.ndarray) else proof.counterexample,
                        "proof_hash": proof.proof_hash
                    })
            
            # Log result
            if result["status"] == "VERIFIED":
                self.stab_agent.interaction_log.add_interaction(
                    topic="koopman_verified",
                    source=self.name,
                    target=system_name,
                    payload={
                        "result": result,
                        "domain": {
                            "lower": domain[0].tolist(),
                            "upper": domain[1].tolist()
                        }
                    }
                )
                
                logger.info(f"Verification successful for {system_name}")
            else:
                # Log the failure
                self.stab_agent.interaction_log.add_interaction(
                    topic="koopman_verification_failed",
                    source=self.name,
                    target=system_name,
                    payload={
                        "result": result,
                        "domain": {
                            "lower": domain[0].tolist(),
                            "upper": domain[1].tolist()
                        }
                    }
                )
                
                if "counterexample" in result:
                    logger.warning(
                        f"Verification failed for {system_name} at "
                        f"x={result['counterexample']}"
                    )
                else:
                    logger.warning(f"Verification failed for {system_name}")
            
            return result
            
        except Exception as e:
            # Log error
            error_msg = f"Verification failed with error: {e}"
            logger.error(error_msg)
            
            self.stab_agent.interaction_log.add_interaction(
                topic="koopman_error",
                source=self.name,
                target=system_name,
                payload={
                    "error": error_msg,
                    "domain": {
                        "lower": domain[0].tolist(),
                        "upper": domain[1].tolist()
                    }
                }
            )
            
            # If it's not already a VerificationError, wrap it
            if not isinstance(e, VerificationError):
                e = VerificationError(
                    code="KOOP_005",
                    detail=f"Verification failed with error: {e}",
                    system_id=system_name
                )
            
            raise e
    
    def get_summary(self) -> str:
        """
        Get a summary of the Koopman Bridge Agent.
        
        Returns:
            Summary string
        """
        lines = [f"Koopman Bridge Agent: {self.name}"]
        lines.append("=" * 50)
        
        # Add summary of results
        lines.append(f"Results: {len(self.results)} systems")
        for system_name, result in self.results.items():
            lyap_fn = result["lyapunov"]
            lines.append(f"  {system_name}:")
            lines.append(f"    Lyapunov: {lyap_fn.name}")
            lines.append(f"    Dictionary: {result['dictionary'].name} (dim={result['dictionary'].dimension})")
            lines.append(f"    Modes: {len(lyap_fn.stable_indices)} stable out of {len(lyap_fn.eigenvalues)}")
            lines.append(f"    Data: {result['data']['n_samples']} samples, {result['data']['state_dim']} dimensions")
            lines.append(f"    Fit MSE: {result['meta']['mse']:.6f}")
        
        return "\n".join(lines)


def create_pendulum_agent(
    name: str = "pendulum",
    dict_type: str = "rbf",
    dict_size: int = 100,
    λ_cut: float = 0.98,
    n_points: int = 1000,
    noise_level: float = 0.01,
    verify: bool = True
) -> Tuple[KoopmanBridgeAgent, KoopmanLyapunov]:
    """
    Create a Koopman Bridge Agent for a pendulum system.
    
    Args:
        name: Name of the agent
        dict_type: Type of dictionary ('rbf', 'fourier', 'poly')
        dict_size: Size of the dictionary
        λ_cut: Cutoff for selecting stable modes
        n_points: Number of data points to generate
        noise_level: Level of noise to add to the data
        verify: Whether to verify the Lyapunov function
        
    Returns:
        Tuple of (agent, lyapunov_function)
    """
    # Create pendulum trajectory data
    t = np.linspace(0, 10, n_points)
    dt = t[1] - t[0]
    
    # Pendulum parameters
    alpha = 0.1  # Damping coefficient
    
    # Initial conditions (multiple trajectories)
    n_trajectories = 10
    x0_list = [
        np.array([np.pi/4, 0.0]),  # Small angle
        np.array([np.pi/2, 0.0]),  # Medium angle
        np.array([np.pi*0.8, 0.0]),  # Large angle
        np.array([0.0, 0.5]),  # Zero angle, nonzero velocity
        np.array([np.pi/4, 0.5]),  # Small angle, positive velocity
        np.array([np.pi/4, -0.5]),  # Small angle, negative velocity
        np.array([-np.pi/4, 0.0]),  # Negative angle
        np.array([-np.pi/4, -0.5]),  # Negative angle, negative velocity
        np.array([0.1, 0.1]),  # Small perturbation
        np.array([-0.1, -0.1])  # Small negative perturbation
    ]
    
    # Pendulum dynamics
    def pendulum_dynamics(x, alpha=0.1):
        """Pendulum dynamics: x' = [x[1], -sin(x[0]) - alpha*x[1]]"""
        theta, omega = x
        return np.array([omega, -np.sin(theta) - alpha*omega])
    
    # Generate trajectory data
    all_x = []
    all_x_next = []
    
    for x0 in x0_list:
        # Simulate trajectory
        x_traj = [x0]
        for i in range(1, len(t)):
            # Simple Euler integration
            x_prev = x_traj[-1]
            x_next = x_prev + dt * pendulum_dynamics(x_prev, alpha)
            x_traj.append(x_next)
        
        # Convert to numpy array
        x_traj = np.array(x_traj)
        
        # Add noise
        x_traj += noise_level * np.random.randn(*x_traj.shape)
        
        # Extract x and x_next
        x = x_traj[:-1]
        x_next = x_traj[1:]
        
        # Append to data
        all_x.append(x)
        all_x_next.append(x_next)
    
    # Concatenate data
    x = np.vstack(all_x)
    x_next = np.vstack(all_x_next)
    
    # Create agent
    agent = KoopmanBridgeAgent(name, auto_verify=verify)
    
    # Learn Koopman operator and create Lyapunov function
    lyap_fn = agent.learn_from_data(
        x=x,
        x_next=x_next,
        system_name="pendulum",
        dict_type=dict_type,
        dict_size=dict_size,
        continuous_time=True,
        dt=dt,
        λ_cut=λ_cut
    )
    
    return agent, lyap_fn


def create_vdp_agent(
    name: str = "vdp",
    dict_type: str = "rbf",
    dict_size: int = 100,
    λ_cut: float = 0.98,
    n_points: int = 1000,
    noise_level: float = 0.01,
    mu: float = 1.0,
    verify: bool = True
) -> Tuple[KoopmanBridgeAgent, KoopmanLyapunov]:
    """
    Create a Koopman Bridge Agent for a Van der Pol oscillator.
    
    Args:
        name: Name of the agent
        dict_type: Type of dictionary ('rbf', 'fourier', 'poly')
        dict_size: Size of the dictionary
        λ_cut: Cutoff for selecting stable modes
        n_points: Number of data points to generate
        noise_level: Level of noise to add to the data
        mu: Nonlinearity parameter
        verify: Whether to verify the Lyapunov function
        
    Returns:
        Tuple of (agent, lyapunov_function)
    """
    # Create Van der Pol trajectory data
    t = np.linspace(0, 10, n_points)
    dt = t[1] - t[0]
    
    # Initial conditions (multiple trajectories)
    n_trajectories = 10
    x0_list = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([0.5, 0.5]),
        np.array([-0.5, -0.5]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
        np.array([-1.0, -1.0]),
        np.array([0.2, 0.2]),
        np.array([-0.2, -0.2])
    ]
    
    # Van der Pol dynamics
    def vdp_dynamics(x, mu=1.0):
        """Van der Pol dynamics: x' = [x[1], mu*(1-x[0]^2)*x[1] - x[0]]"""
        return np.array([x[1], mu*(1-x[0]**2)*x[1] - x[0]])
    
    # Generate trajectory data
    all_x = []
    all_x_next = []
    
    for x0 in x0_list:
        # Simulate trajectory
        x_traj = [x0]
        for i in range(1, len(t)):
            # Simple Euler integration
            x_prev = x_traj[-1]
            x_next = x_prev + dt * vdp_dynamics(x_prev, mu)
            x_traj.append(x_next)
        
        # Convert to numpy array
        x_traj = np.array(x_traj)
        
        # Add noise
        x_traj += noise_level * np.random.randn(*x_traj.shape)
        
        # Extract x and x_next
        x = x_traj[:-1]
        x_next = x_traj[1:]
        
        # Append to data
        all_x.append(x)
        all_x_next.append(x_next)
    
    # Concatenate data
    x = np.vstack(all_x)
    x_next = np.vstack(all_x_next)
    
    # Create agent
    agent = KoopmanBridgeAgent(name, auto_verify=verify)
    
    # Learn Koopman operator and create Lyapunov function
    lyap_fn = agent.learn_from_data(
        x=x,
        x_next=x_next,
        system_name="vdp",
        dict_type=dict_type,
        dict_size=dict_size,
        continuous_time=True,
        dt=dt,
        λ_cut=λ_cut
    )
    
    return agent, lyap_fn
