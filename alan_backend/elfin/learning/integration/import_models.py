"""
Import ELFIN Models to Neural Barrier and Lyapunov Functions

This module provides utilities to import ELFIN barrier and Lyapunov functions 
for neural network training or verification.
"""

import os
import re
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

from ..models.torch_models import TorchBarrierNetwork, TorchLyapunovNetwork


def import_from_elfin(
    filepath: str,
    system_name: Optional[str] = None,
    state_dim: int = 0,
    input_dim: int = 0,
    state_names: Optional[List[str]] = None,
    input_names: Optional[List[str]] = None,
    framework: str = "torch",
    hidden_layers: List[int] = [64, 64],
    activation: str = "tanh",
    device: Optional[Union[str, torch.device]] = None
) -> Union[TorchBarrierNetwork, TorchLyapunovNetwork]:
    """
    Import an ELFIN barrier or Lyapunov function and convert it to a neural network.
    
    This function analyzes the structure of an ELFIN file and creates a neural network
    that approximates the barrier or Lyapunov function defined in the file.
    
    Args:
        filepath: Path to the ELFIN file
        system_name: Name of the system (if None, extract from file)
        state_dim: Dimension of the state space (if 0, extract from file)
        input_dim: Dimension of the input space (if 0, extract from file)
        state_names: Names of state variables (if None, extract from file)
        input_names: Names of input variables (if None, extract from file)
        framework: Neural network framework ('torch' or 'jax')
        hidden_layers: List of hidden layer sizes for the neural network
        activation: Activation function for the neural network
        device: Device to use for computations (PyTorch only)
        
    Returns:
        Neural network approximating the ELFIN barrier or Lyapunov function
    """
    # Read ELFIN file
    with open(filepath, "r") as f:
        elfin_code = f.read()
    
    # Extract model type (barrier or lyapunov)
    model_type = None
    if re.search(r"barrier\s+\w+", elfin_code):
        model_type = "barrier"
    elif re.search(r"lyapunov\s+\w+", elfin_code):
        model_type = "lyapunov"
    else:
        raise ValueError("Could not determine model type (barrier or lyapunov) from file")
    
    # Extract system name if not provided
    if system_name is None:
        if model_type == "barrier":
            match = re.search(r"barrier\s+(\w+)", elfin_code)
            if match:
                system_name = match.group(1)
        else:
            match = re.search(r"lyapunov\s+(\w+)", elfin_code)
            if match:
                system_name = match.group(1)
    
    # Extract system reference
    system_ref_match = re.search(r"system:\s+(\w+);", elfin_code)
    if system_ref_match:
        system_ref = system_ref_match.group(1)
    else:
        raise ValueError("Could not find system reference in ELFIN file")
    
    # Extract function definition
    function_match = None
    if model_type == "barrier":
        function_match = re.search(r"B:\s+(.*?);", elfin_code, re.DOTALL)
    else:
        function_match = re.search(r"V:\s+(.*?);", elfin_code, re.DOTALL)
    
    if not function_match:
        raise ValueError(f"Could not find {'barrier' if model_type == 'barrier' else 'Lyapunov'} function definition in ELFIN file")
    
    function_definition = function_match.group(1).strip()
    
    # Extract state and input variables if not provided
    if state_names is None or state_dim == 0:
        # Look for continuous_state in system definition
        # This would require parsing the system definition from another file
        # For simplicity, we'll use a heuristic approach
        
        # Assume state variables are x1, x2, ..., or named variables
        var_pattern = r"[a-zA-Z_][a-zA-Z0-9_]*"
        vars_found = set(re.findall(var_pattern, function_definition))
        
        # Filter out common functions, operators, and constants
        common_terms = {'if', 'then', 'else', 'and', 'or', 'not', 'true', 'false',
                       'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'pi', 'e'}
        vars_found = vars_found - common_terms
        
        if not state_names:
            state_names = sorted(list(vars_found))
        
        if state_dim == 0:
            state_dim = len(state_names)
    
    # Create a function to evaluate the ELFIN expression
    def create_evaluator(expression: str, var_names: List[str]) -> Callable:
        """Create a function that evaluates the ELFIN expression."""
        # Replace ELFIN-specific syntax with Python syntax
        python_expr = expression
        python_expr = re.sub(r"(\w+)\s*\*\*\s*(\w+)", r"\1**\2", python_expr)  # Fix ** spacing
        python_expr = re.sub(r"if\s+(.*?)\s+then\s+(.*?)\s+else\s+(.*?)(\s|$)", r"(\2 if \1 else \3)", python_expr)
        
        # Create function code
        arg_list = ", ".join(var_names)
        func_code = f"def evaluator({arg_list}):\n    import numpy as np\n    import math\n"
        func_code += f"    pi = math.pi\n    e = math.e\n"
        func_code += f"    return {python_expr}\n"
        
        # Compile function
        local_vars = {}
        exec(func_code, {"np": np, "math": __import__("math")}, local_vars)
        
        return local_vars["evaluator"]
    
    # Create evaluator function
    evaluator = create_evaluator(function_definition, state_names)
    
    # Create neural network approximation
    if framework == "torch":
        # Initialize PyTorch model
        if model_type == "barrier":
            model = TorchBarrierNetwork(
                state_dim=state_dim,
                hidden_layers=hidden_layers,
                activation=activation,
                device=device
            )
        else:
            model = TorchLyapunovNetwork(
                state_dim=state_dim,
                hidden_layers=hidden_layers,
                activation=activation,
                device=device
            )
        
        # Train the model to approximate the ELFIN function
        # This is a simplified training procedure
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Generate training data
        num_samples = 1000
        bounds = np.array([[-5.0, 5.0]] * state_dim)
        samples = np.random.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(num_samples, state_dim)
        )
        
        # Evaluate ELFIN function on samples
        values = np.zeros((num_samples, 1))
        for i, sample in enumerate(samples):
            values[i, 0] = evaluator(*sample)
        
        # Convert to PyTorch tensors
        samples_tensor = torch.tensor(samples, dtype=torch.float32, device=device)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
        
        # Train model
        epochs = 1000
        batch_size = 64
        
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(num_samples)
            samples_shuffled = samples_tensor[indices]
            values_shuffled = values_tensor[indices]
            
            # Train in batches
            for i in range(0, num_samples, batch_size):
                # Get batch
                batch_samples = samples_shuffled[i:i+batch_size]
                batch_values = values_shuffled[i:i+batch_size]
                
                # Forward pass
                outputs = model(batch_samples)
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(outputs, batch_values)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                with torch.no_grad():
                    outputs = model(samples_tensor)
                    loss = torch.nn.functional.mse_loss(outputs, values_tensor)
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        return model
    
    elif framework == "jax":
        # TODO: Implement JAX model training
        raise NotImplementedError("JAX framework import not yet implemented")
    
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def import_barrier_function(
    filepath: str,
    system_name: Optional[str] = None,
    state_dim: int = 0,
    input_dim: int = 0,
    state_names: Optional[List[str]] = None,
    input_names: Optional[List[str]] = None,
    framework: str = "torch",
    hidden_layers: List[int] = [64, 64],
    activation: str = "tanh",
    device: Optional[Union[str, torch.device]] = None
) -> TorchBarrierNetwork:
    """
    Import an ELFIN barrier function and convert it to a neural network.
    
    Args:
        filepath: Path to the ELFIN file
        system_name: Name of the system (if None, extract from file)
        state_dim: Dimension of the state space (if 0, extract from file)
        input_dim: Dimension of the input space (if 0, extract from file)
        state_names: Names of state variables (if None, extract from file)
        input_names: Names of input variables (if None, extract from file)
        framework: Neural network framework ('torch' or 'jax')
        hidden_layers: List of hidden layer sizes for the neural network
        activation: Activation function for the neural network
        device: Device to use for computations (PyTorch only)
        
    Returns:
        Neural network approximating the ELFIN barrier function
    """
    return import_from_elfin(
        filepath=filepath,
        system_name=system_name,
        state_dim=state_dim,
        input_dim=input_dim,
        state_names=state_names,
        input_names=input_names,
        framework=framework,
        hidden_layers=hidden_layers,
        activation=activation,
        device=device
    )


def import_lyapunov_function(
    filepath: str,
    system_name: Optional[str] = None,
    state_dim: int = 0,
    input_dim: int = 0,
    state_names: Optional[List[str]] = None,
    input_names: Optional[List[str]] = None,
    framework: str = "torch",
    hidden_layers: List[int] = [64, 64],
    activation: str = "tanh",
    device: Optional[Union[str, torch.device]] = None
) -> TorchLyapunovNetwork:
    """
    Import an ELFIN Lyapunov function and convert it to a neural network.
    
    Args:
        filepath: Path to the ELFIN file
        system_name: Name of the system (if None, extract from file)
        state_dim: Dimension of the state space (if 0, extract from file)
        input_dim: Dimension of the input space (if 0, extract from file)
        state_names: Names of state variables (if None, extract from file)
        input_names: Names of input variables (if None, extract from file)
        framework: Neural network framework ('torch' or 'jax')
        hidden_layers: List of hidden layer sizes for the neural network
        activation: Activation function for the neural network
        device: Device to use for computations (PyTorch only)
        
    Returns:
        Neural network approximating the ELFIN Lyapunov function
    """
    return import_from_elfin(
        filepath=filepath,
        system_name=system_name,
        state_dim=state_dim,
        input_dim=input_dim,
        state_names=state_names,
        input_names=input_names,
        framework=framework,
        hidden_layers=hidden_layers,
        activation=activation,
        device=device
    )
