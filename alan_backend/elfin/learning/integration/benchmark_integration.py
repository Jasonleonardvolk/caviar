"""
Benchmark Integration for Neural Barrier and Lyapunov Functions

This module provides utilities to integrate learned neural barrier and Lyapunov
functions with the ELFIN benchmark suite.
"""

import os
import numpy as np
import torch
import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

from ...benchmarks import run_benchmark
from ...benchmarks.benchmark import BenchmarkSystem
from ..models.neural_barrier import NeuralBarrierNetwork
from ..models.neural_lyapunov import NeuralLyapunovNetwork
from ..models.torch_models import TorchBarrierNetwork, TorchLyapunovNetwork
from ..models.jax_models import JAXBarrierNetwork, JAXLyapunovNetwork


class NeuralBarrierSystem(BenchmarkSystem):
    """
    Wrapper for integrating neural barrier functions with the benchmark suite.
    
    This class wraps a neural barrier function and provides the necessary interface
    for the benchmark suite to evaluate its performance.
    """
    
    def __init__(
        self,
        name: str,
        barrier_network: NeuralBarrierNetwork,
        dynamics_fn: Callable,
        state_dim: int,
        input_dim: int = 0,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a neural barrier system.
        
        Args:
            name: Name of the system
            barrier_network: Neural barrier network
            dynamics_fn: System dynamics function
            state_dim: Dimension of the state space
            input_dim: Dimension of the input space
            params: Additional parameters
        """
        super().__init__(
            name=name,
            state_dim=state_dim,
            input_dim=input_dim,
            params=params or {}
        )
        
        self.barrier_network = barrier_network
        self.dynamics_fn = dynamics_fn
    
    def dynamics(self, state: np.ndarray, input_vec: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute system dynamics.
        
        Args:
            state: System state, shape (state_dim,) or (batch_size, state_dim)
            input_vec: System input, shape (input_dim,) or (batch_size, input_dim)
            
        Returns:
            State derivative, shape (state_dim,) or (batch_size, state_dim)
        """
        # Handle single state case
        if state.ndim == 1:
            state = state.reshape(1, -1)
            if input_vec is not None:
                input_vec = input_vec.reshape(1, -1)
            
            # Compute dynamics
            derivative = self.dynamics_fn(state, input_vec)
            
            # Return result with the correct shape
            return derivative.squeeze()
        else:
            # Compute dynamics for batch
            return self.dynamics_fn(state, input_vec)
    
    def barrier_function(self, state: np.ndarray) -> float:
        """
        Compute the barrier function value.
        
        Args:
            state: System state, shape (state_dim,) or (batch_size, state_dim)
            
        Returns:
            Barrier function value, shape (1,) or (batch_size, 1)
        """
        # Handle single state case
        if state.ndim == 1:
            state = state.reshape(1, -1)
            result = self.barrier_network(state)
            return result.item() if hasattr(result, 'item') else result[0, 0]
        else:
            # Compute barrier function for batch
            return self.barrier_network(state)
    
    def is_safe(self, state: np.ndarray) -> bool:
        """
        Check if a state is safe.
        
        Args:
            state: System state, shape (state_dim,) or (batch_size, state_dim)
            
        Returns:
            Boolean indicating whether the state is safe
        """
        # Compute barrier function value
        barrier_value = self.barrier_function(state)
        
        # Handle both scalar and array cases
        if isinstance(barrier_value, (int, float)):
            return barrier_value > 0
        else:
            return barrier_value > 0
    
    def verify_barrier_condition(
        self,
        state: np.ndarray,
        input_vec: Optional[np.ndarray] = None
    ) -> bool:
        """
        Verify if the barrier condition is satisfied.
        
        Args:
            state: System state, shape (state_dim,) or (batch_size, state_dim)
            input_vec: System input, shape (input_dim,) or (batch_size, input_dim)
            
        Returns:
            Boolean indicating whether the barrier condition is satisfied
        """
        # Handle different network types
        if isinstance(self.barrier_network, TorchBarrierNetwork):
            # Convert to PyTorch tensors
            device = next(self.barrier_network.parameters()).device
            
            if state.ndim == 1:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                if input_vec is not None:
                    input_tensor = torch.tensor(input_vec, dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    input_tensor = None
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                if input_vec is not None:
                    input_tensor = torch.tensor(input_vec, dtype=torch.float32, device=device)
                else:
                    input_tensor = None
            
            # Make state require gradients
            state_tensor = state_tensor.requires_grad_(True)
            
            # Compute barrier value
            barrier_value = self.barrier_network(state_tensor)
            
            # Compute gradient
            gradient = torch.autograd.grad(
                barrier_value.sum(),
                state_tensor,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Compute dynamics
            dynamics_tensor = torch.tensor(
                self.dynamics(state, input_vec),
                dtype=torch.float32,
                device=device
            )
            
            # Compute Lie derivative
            lie_derivative = torch.sum(gradient * dynamics_tensor, dim=1, keepdim=True)
            
            # Compute α(B(x))
            alpha_fn = lambda b: 0.1 * torch.abs(b)
            alpha_value = alpha_fn(barrier_value)
            
            # Check condition: Lie derivative >= -α(B(x))
            condition = lie_derivative >= -alpha_value
            
            # Convert to NumPy and return
            return condition.detach().cpu().numpy()
        
        elif isinstance(self.barrier_network, JAXBarrierNetwork):
            # Convert to JAX arrays
            if state.ndim == 1:
                state_array = jnp.array(state).reshape(1, -1)
                if input_vec is not None:
                    input_array = jnp.array(input_vec).reshape(1, -1)
                else:
                    input_array = None
            else:
                state_array = jnp.array(state)
                if input_vec is not None:
                    input_array = jnp.array(input_vec)
                else:
                    input_array = None
            
            # Compute barrier value
            barrier_value = self.barrier_network(state_array)
            
            # Compute gradient
            barrier_grad_fn = jax.vmap(
                lambda x: jax.grad(lambda x_single: self.barrier_network(
                    self.barrier_network.params, x_single.reshape(1, -1))[0, 0]
                )(x)
            )
            gradient = barrier_grad_fn(state_array)
            
            # Compute dynamics
            dynamics_array = jnp.array(self.dynamics(state, input_vec))
            
            # Compute Lie derivative
            lie_derivative = jnp.sum(gradient * dynamics_array, axis=1, keepdims=True)
            
            # Compute α(B(x))
            alpha_fn = lambda b: 0.1 * jnp.abs(b)
            alpha_value = alpha_fn(barrier_value)
            
            # Check condition: Lie derivative >= -α(B(x))
            condition = lie_derivative >= -alpha_value
            
            # Convert to NumPy and return
            return np.array(condition)
        
        else:
            # Generic implementation for other barrier network types
            # This may not work for all types and should be extended as needed
            if state.ndim == 1:
                state_reshaped = state.reshape(1, -1)
                
                # Compute barrier value
                barrier_value = self.barrier_network(state_reshaped)
                
                # Compute numerical gradient
                eps = 1e-4
                gradient = np.zeros_like(state_reshaped)
                
                for i in range(state.shape[0]):
                    plus_state = state_reshaped.copy()
                    plus_state[0, i] += eps
                    
                    minus_state = state_reshaped.copy()
                    minus_state[0, i] -= eps
                    
                    plus_value = self.barrier_network(plus_state)
                    minus_value = self.barrier_network(minus_state)
                    
                    gradient[0, i] = (plus_value - minus_value) / (2 * eps)
                
                # Compute dynamics
                dynamics_value = self.dynamics(state, input_vec).reshape(1, -1)
                
                # Compute Lie derivative
                lie_derivative = np.sum(gradient * dynamics_value)
                
                # Compute α(B(x))
                alpha_fn = lambda b: 0.1 * np.abs(b)
                alpha_value = alpha_fn(barrier_value)
                
                # Check condition: Lie derivative >= -α(B(x))
                return lie_derivative >= -alpha_value
            else:
                # Compute result for each state individually
                results = []
                for i in range(state.shape[0]):
                    results.append(
                        self.verify_barrier_condition(
                            state[i],
                            None if input_vec is None else input_vec[i]
                        )
                    )
                
                return np.array(results)
    
    def get_safe_set_samples(self, num_samples: int) -> np.ndarray:
        """
        Generate samples from the safe set.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Array of safe states, shape (num_safe, state_dim)
        """
        # Generate random samples from the state space
        x_bounds = self.get_state_bounds()
        samples = np.random.uniform(
            x_bounds[:, 0],
            x_bounds[:, 1],
            size=(num_samples * 10, self.state_dim)
        )
        
        # Filter safe samples
        safe_mask = np.array([self.is_safe(x) for x in samples])
        safe_samples = samples[safe_mask]
        
        # If we have enough safe samples, return them
        if len(safe_samples) >= num_samples:
            return safe_samples[:num_samples]
        else:
            # Not enough safe samples, print warning and return what we have
            print(f"Warning: Could only generate {len(safe_samples)} safe samples")
            return safe_samples


class NeuralLyapunovSystem(BenchmarkSystem):
    """
    Wrapper for integrating neural Lyapunov functions with the benchmark suite.
    
    This class wraps a neural Lyapunov function and provides the necessary interface
    for the benchmark suite to evaluate its performance.
    """
    
    def __init__(
        self,
        name: str,
        lyapunov_network: NeuralLyapunovNetwork,
        dynamics_fn: Callable,
        state_dim: int,
        input_dim: int = 0,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a neural Lyapunov system.
        
        Args:
            name: Name of the system
            lyapunov_network: Neural Lyapunov network
            dynamics_fn: System dynamics function
            state_dim: Dimension of the state space
            input_dim: Dimension of the input space
            params: Additional parameters
        """
        super().__init__(
            name=name,
            state_dim=state_dim,
            input_dim=input_dim,
            params=params or {}
        )
        
        self.lyapunov_network = lyapunov_network
        self.dynamics_fn = dynamics_fn
    
    def dynamics(self, state: np.ndarray, input_vec: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute system dynamics.
        
        Args:
            state: System state, shape (state_dim,) or (batch_size, state_dim)
            input_vec: System input, shape (input_dim,) or (batch_size, input_dim)
            
        Returns:
            State derivative, shape (state_dim,) or (batch_size, state_dim)
        """
        # Handle single state case
        if state.ndim == 1:
            state = state.reshape(1, -1)
            if input_vec is not None:
                input_vec = input_vec.reshape(1, -1)
            
            # Compute dynamics
            derivative = self.dynamics_fn(state, input_vec)
            
            # Return result with the correct shape
            return derivative.squeeze()
        else:
            # Compute dynamics for batch
            return self.dynamics_fn(state, input_vec)
    
    def lyapunov_function(self, state: np.ndarray) -> float:
        """
        Compute the Lyapunov function value.
        
        Args:
            state: System state, shape (state_dim,) or (batch_size, state_dim)
            
        Returns:
            Lyapunov function value, shape (1,) or (batch_size, 1)
        """
        # Handle single state case
        if state.ndim == 1:
            state = state.reshape(1, -1)
            result = self.lyapunov_network(state)
            return result.item() if hasattr(result, 'item') else result[0, 0]
        else:
            # Compute Lyapunov function for batch
            return self.lyapunov_network(state)
    
    def is_stable(self, state: np.ndarray, input_vec: Optional[np.ndarray] = None) -> bool:
        """
        Check if a state is stable.
        
        Args:
            state: System state, shape (state_dim,) or (batch_size, state_dim)
            input_vec: System input, shape (input_dim,) or (batch_size, input_dim)
            
        Returns:
            Boolean indicating whether the state is stable
        """
        # Verify Lyapunov condition
        return self.verify_lyapunov_condition(state, input_vec)
    
    def verify_lyapunov_condition(
        self,
        state: np.ndarray,
        input_vec: Optional[np.ndarray] = None
    ) -> bool:
        """
        Verify if the Lyapunov condition is satisfied.
        
        Args:
            state: System state, shape (state_dim,) or (batch_size, state_dim)
            input_vec: System input, shape (input_dim,) or (batch_size, input_dim)
            
        Returns:
            Boolean indicating whether the Lyapunov condition is satisfied
        """
        # Handle different network types
        if isinstance(self.lyapunov_network, TorchLyapunovNetwork):
            # Convert to PyTorch tensors
            device = next(self.lyapunov_network.parameters()).device
            
            if state.ndim == 1:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                if input_vec is not None:
                    input_tensor = torch.tensor(input_vec, dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    input_tensor = None
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                if input_vec is not None:
                    input_tensor = torch.tensor(input_vec, dtype=torch.float32, device=device)
                else:
                    input_tensor = None
            
            # Make state require gradients
            state_tensor = state_tensor.requires_grad_(True)
            
            # Compute Lyapunov value
            lyapunov_value = self.lyapunov_network(state_tensor)
            
            # Compute gradient
            gradient = torch.autograd.grad(
                lyapunov_value.sum(),
                state_tensor,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Compute dynamics
            dynamics_tensor = torch.tensor(
                self.dynamics(state, input_vec),
                dtype=torch.float32,
                device=device
            )
            
            # Compute Lie derivative
            lie_derivative = torch.sum(gradient * dynamics_tensor, dim=1, keepdim=True)
            
            # Check condition: Lie derivative < 0
            condition = lie_derivative < 0
            
            # Convert to NumPy and return
            return condition.detach().cpu().numpy()
        
        elif isinstance(self.lyapunov_network, JAXLyapunovNetwork):
            # Convert to JAX arrays
            if state.ndim == 1:
                state_array = jnp.array(state).reshape(1, -1)
                if input_vec is not None:
                    input_array = jnp.array(input_vec).reshape(1, -1)
                else:
                    input_array = None
            else:
                state_array = jnp.array(state)
                if input_vec is not None:
                    input_array = jnp.array(input_vec)
                else:
                    input_array = None
            
            # Compute Lyapunov value
            lyapunov_value = self.lyapunov_network(state_array)
            
            # Compute gradient
            lyapunov_grad_fn = jax.vmap(
                lambda x: jax.grad(lambda x_single: self.lyapunov_network(
                    self.lyapunov_network.params, x_single.reshape(1, -1))[0, 0]
                )(x)
            )
            gradient = lyapunov_grad_fn(state_array)
            
            # Compute dynamics
            dynamics_array = jnp.array(self.dynamics(state, input_vec))
            
            # Compute Lie derivative
            lie_derivative = jnp.sum(gradient * dynamics_array, axis=1, keepdims=True)
            
            # Check condition: Lie derivative < 0
            condition = lie_derivative < 0
            
            # Convert to NumPy and return
            return np.array(condition)
        
        else:
            # Generic implementation for other Lyapunov network types
            # This may not work for all types and should be extended as needed
            if state.ndim == 1:
                state_reshaped = state.reshape(1, -1)
                
                # Compute Lyapunov value
                lyapunov_value = self.lyapunov_network(state_reshaped)
                
                # Compute numerical gradient
                eps = 1e-4
                gradient = np.zeros_like(state_reshaped)
                
                for i in range(state.shape[0]):
                    plus_state = state_reshaped.copy()
                    plus_state[0, i] += eps
                    
                    minus_state = state_reshaped.copy()
                    minus_state[0, i] -= eps
                    
                    plus_value = self.lyapunov_network(plus_state)
                    minus_value = self.lyapunov_network(minus_state)
                    
                    gradient[0, i] = (plus_value - minus_value) / (2 * eps)
                
                # Compute dynamics
                dynamics_value = self.dynamics(state, input_vec).reshape(1, -1)
                
                # Compute Lie derivative
                lie_derivative = np.sum(gradient * dynamics_value)
                
                # Check condition: Lie derivative < 0
                return lie_derivative < 0
            else:
                # Compute result for each state individually
                results = []
                for i in range(state.shape[0]):
                    results.append(
                        self.verify_lyapunov_condition(
                            state[i],
                            None if input_vec is None else input_vec[i]
                        )
                    )
                
                return np.array(results)
    
    def get_region_of_attraction_samples(self, num_samples: int, level_set: Optional[float] = None) -> np.ndarray:
        """
        Generate samples from the region of attraction.
        
        Args:
            num_samples: Number of samples to generate
            level_set: Level set value (optional)
            
        Returns:
            Array of states in the region of attraction, shape (num_stable, state_dim)
        """
        # Generate random samples from the state space
        x_bounds = self.get_state_bounds()
        samples = np.random.uniform(
            x_bounds[:, 0],
            x_bounds[:, 1],
            size=(num_samples * 10, self.state_dim)
        )
        
        # Filter stable samples
        if level_set is not None:
            # Filter by level set first
            lyapunov_values = np.array([self.lyapunov_function(x) for x in samples])
            level_set_mask = lyapunov_values <= level_set
            level_set_samples = samples[level_set_mask]
            
            # Check stability for samples in level set
            stable_mask = np.array([self.is_stable(x) for x in level_set_samples])
            stable_samples = level_set_samples[stable_mask]
        else:
            # Check stability for all samples
            stable_mask = np.array([self.is_stable(x) for x in samples])
            stable_samples = samples[stable_mask]
        
        # If we have enough stable samples, return them
        if len(stable_samples) >= num_samples:
            return stable_samples[:num_samples]
        else:
            # Not enough stable samples, print warning and return what we have
            print(f"Warning: Could only generate {len(stable_samples)} stable samples")
            return stable_samples


class BenchmarkIntegration:
    """
    Integration of neural barrier and Lyapunov functions with the benchmark suite.
    
    This class provides utilities to benchmark the performance of learned neural
    barrier and Lyapunov functions using the ELFIN benchmark suite.
    """
    
    @staticmethod
    def benchmark_barrier_network(
        barrier_system: NeuralBarrierSystem,
        metrics: List[str] = ["validation_success_rate", "computation_time", "conservativeness"],
        output_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Benchmark a neural barrier system.
        
        Args:
            barrier_system: Neural barrier system to benchmark
            metrics: List of metrics to compute
            output_dir: Directory to save results
            
        Returns:
            Dictionary of benchmark results
        """
        # Run benchmark
        result = run_benchmark(
            barrier_system,
            metrics=metrics,
            output_dir=output_dir
        )
        
        return result.metrics
    
    @staticmethod
    def benchmark_lyapunov_network(
        lyapunov_system: NeuralLyapunovSystem,
        metrics: List[str] = ["validation_success_rate", "computation_time", "conservativeness"],
        output_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Benchmark a neural Lyapunov system.
        
        Args:
            lyapunov_system: Neural Lyapunov system to benchmark
            metrics: List of metrics to compute
            output_dir: Directory to save results
            
        Returns:
            Dictionary of benchmark results
        """
        # Run benchmark
        result = run_benchmark(
            lyapunov_system,
            metrics=metrics,
            output_dir=output_dir
        )
        
        return result.metrics
    
    @staticmethod
    def compare_barrier_networks(
        barrier_systems: List[NeuralBarrierSystem],
        metrics: List[str] = ["validation_success_rate", "computation_time", "conservativeness"],
        output_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple neural barrier systems.
        
        Args:
            barrier_systems: List of neural barrier systems to compare
            metrics: List of metrics to compute
            output_dir: Directory to save results
            
        Returns:
            Dictionary of benchmark results for each system
        """
        # Run benchmarks
        results = {}
        
        for system in barrier_systems:
            system_results = BenchmarkIntegration.benchmark_barrier_network(
                system,
                metrics=metrics,
                output_dir=output_dir
            )
            
            results[system.name] = system_results
        
        return results
    
    @staticmethod
    def compare_lyapunov_networks(
        lyapunov_systems: List[NeuralLyapunovSystem],
        metrics: List[str] = ["validation_success_rate", "computation_time", "conservativeness"],
        output_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple neural Lyapunov systems.
        
        Args:
            lyapunov_systems: List of neural Lyapunov systems to compare
            metrics: List of metrics to compute
            output_dir: Directory to save results
            
        Returns:
            Dictionary of benchmark results for each system
        """
        # Run benchmarks
        results = {}
        
        for system in lyapunov_systems:
            system_results = BenchmarkIntegration.benchmark_lyapunov_network(
                system,
                metrics=metrics,
                output_dir=output_dir
            )
            
            results[system.name] = system_results
        
        return results
