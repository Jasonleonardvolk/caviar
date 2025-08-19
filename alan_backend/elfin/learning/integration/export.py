"""
Export Neural Barrier and Lyapunov Functions to ELFIN

This module provides utilities to export learned neural barrier and Lyapunov
functions to the ELFIN format.
"""

import os
import json
import numpy as np
import torch
import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

from ..models.neural_barrier import NeuralBarrierNetwork
from ..models.neural_lyapunov import NeuralLyapunovNetwork
from ..models.torch_models import TorchBarrierNetwork, TorchLyapunovNetwork
from ..models.jax_models import JAXBarrierNetwork, JAXLyapunovNetwork


def export_to_elfin(
    model: Union[NeuralBarrierNetwork, NeuralLyapunovNetwork],
    model_type: str,
    system_name: str,
    state_dim: int,
    input_dim: int = 0,
    state_names: Optional[List[str]] = None,
    input_names: Optional[List[str]] = None,
    filepath: Optional[str] = None,
    approximation_method: str = "explicit",
    approximation_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export a neural barrier or Lyapunov function to the ELFIN format.
    
    Args:
        model: Neural barrier or Lyapunov model
        model_type: Type of model ('barrier' or 'lyapunov')
        system_name: Name of the system
        state_dim: Dimension of the state space
        input_dim: Dimension of the input space
        state_names: Names of state variables (optional)
        input_names: Names of input variables (optional)
        filepath: Path to save the exported model (optional)
        approximation_method: Method to approximate the neural network
            ('explicit', 'taylor', 'polynomial', 'piecewise')
        approximation_params: Parameters for the approximation method
            
    Returns:
        ELFIN code for the exported model
    """
    # Validate model type
    if model_type not in ["barrier", "lyapunov"]:
        raise ValueError(f"Invalid model type: {model_type}, must be 'barrier' or 'lyapunov'")
    
    # Validate approximation method
    if approximation_method not in ["explicit", "taylor", "polynomial", "piecewise"]:
        raise ValueError(f"Invalid approximation method: {approximation_method}")
    
    # Generate default state and input names if not provided
    if state_names is None:
        state_names = [f"x{i+1}" for i in range(state_dim)]
    
    if input_names is None and input_dim > 0:
        input_names = [f"u{i+1}" for i in range(input_dim)]
    
    # Create approximation of the neural network
    if approximation_method == "explicit":
        elfin_code = _export_explicit(
            model=model,
            model_type=model_type,
            system_name=system_name,
            state_dim=state_dim,
            input_dim=input_dim,
            state_names=state_names,
            input_names=input_names,
            params=approximation_params or {}
        )
    elif approximation_method == "taylor":
        elfin_code = _export_taylor(
            model=model,
            model_type=model_type,
            system_name=system_name,
            state_dim=state_dim,
            input_dim=input_dim,
            state_names=state_names,
            input_names=input_names,
            params=approximation_params or {}
        )
    elif approximation_method == "polynomial":
        elfin_code = _export_polynomial(
            model=model,
            model_type=model_type,
            system_name=system_name,
            state_dim=state_dim,
            input_dim=input_dim,
            state_names=state_names,
            input_names=input_names,
            params=approximation_params or {}
        )
    elif approximation_method == "piecewise":
        elfin_code = _export_piecewise(
            model=model,
            model_type=model_type,
            system_name=system_name,
            state_dim=state_dim,
            input_dim=input_dim,
            state_names=state_names,
            input_names=input_names,
            params=approximation_params or {}
        )
    
    # Save to file if filepath is provided
    if filepath is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "w") as f:
            f.write(elfin_code)
    
    return elfin_code


def _export_explicit(
    model: Union[NeuralBarrierNetwork, NeuralLyapunovNetwork],
    model_type: str,
    system_name: str,
    state_dim: int,
    input_dim: int,
    state_names: List[str],
    input_names: Optional[List[str]],
    params: Dict[str, Any]
) -> str:
    """
    Export a neural network using explicit layer-by-layer computation.
    
    This method exports the exact structure of the neural network as a
    sequence of matrix multiplications and nonlinear activation functions.
    """
    # Define helper functions for the ELFIN file
    helpers = [
        "# Helper functions for neural network computation",
        "helpers: {",
        "  # ReLU activation function",
        "  relu(x) = if x > 0 then x else 0;",
        "  ",
        "  # Sigmoid activation function",
        "  sigmoid(x) = 1 / (1 + exp(-x));",
        "  ",
        "  # Hyperbolic tangent activation function",
        "  # Note: Using built-in tanh if available",
        "  tanh_custom(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x));",
        "  ",
        "  # Softplus activation function",
        "  softplus(x) = log(1 + exp(x));",
        "  ",
        "  # Leaky ReLU activation function",
        "  leaky_relu(x, alpha) = if x > 0 then x else alpha * x;",
        "  ",
        "  # Swish activation function",
        "  swish(x) = x * sigmoid(x);",
        "}"
    ]
    
    # Extract network parameters
    if isinstance(model, TorchBarrierNetwork) or isinstance(model, TorchLyapunovNetwork):
        # Extract weights and biases from PyTorch model
        weights = []
        biases = []
        activation = None
        
        for name, param in model.state_dict().items():
            if 'weight' in name:
                weights.append(param.detach().cpu().numpy())
            elif 'bias' in name:
                biases.append(param.detach().cpu().numpy())
        
        # Determine activation function
        if hasattr(model, 'activation_name'):
            activation = model.activation_name
        else:
            # Guess activation function from model structure
            for module in model.modules():
                if isinstance(module, torch.nn.ReLU):
                    activation = 'relu'
                    break
                elif isinstance(module, torch.nn.Tanh):
                    activation = 'tanh'
                    break
                elif isinstance(module, torch.nn.Sigmoid):
                    activation = 'sigmoid'
                    break
                elif isinstance(module, torch.nn.LeakyReLU):
                    activation = 'leaky_relu'
                    break
                elif isinstance(module, torch.nn.SiLU):  # Swish activation
                    activation = 'swish'
                    break
    
    elif isinstance(model, JAXBarrierNetwork) or isinstance(model, JAXLyapunovNetwork):
        # Extract weights and biases from JAX model
        weights = []
        biases = []
        activation = model.activation
        
        # Extract parameters from FlatDict
        from flax.traverse_util import flatten_dict
        flat_params = flatten_dict(model.params)
        
        # Sort param keys to ensure correct order
        sorted_keys = sorted(flat_params.keys())
        
        for key in sorted_keys:
            param = flat_params[key]
            if 'kernel' in key:
                weights.append(np.array(param))
            elif 'bias' in key:
                biases.append(np.array(param))
    
    else:
        raise ValueError(f"Unsupported model type for explicit export: {type(model)}")
    
    # Map activation function to ELFIN helper function
    activation_map = {
        'relu': 'relu',
        'tanh': 'tanh_custom',
        'sigmoid': 'sigmoid',
        'leaky_relu': 'leaky_relu',
        'swish': 'swish'
    }
    
    elfin_activation = activation_map.get(activation.lower(), 'tanh_custom')
    
    # Generate code for the barrier or Lyapunov function
    function_lines = []
    
    if model_type == "barrier":
        function_lines.append(f"barrier Neural{system_name}Barrier {{")
    else:
        function_lines.append(f"lyapunov Neural{system_name}Lyapunov {{")
    
    # Add system reference
    function_lines.append(f"  system: {system_name};")
    function_lines.append("")
    
    # Add function definition
    if model_type == "barrier":
        function_lines.append("  # Neural network barrier function")
        function_lines.append("  B: neural_output;")
        function_lines.append("  alpha_fun: alpha * B;  # Class-K function for barrier certificate")
    else:
        function_lines.append("  # Neural network Lyapunov function")
        function_lines.append("  V: neural_output;")
    
    function_lines.append("")
    
    # Add computation for each layer
    function_lines.append("  # Neural network computation")
    
    # Input layer
    function_lines.append("  # Input layer")
    for i in range(state_dim):
        function_lines.append(f"  layer0_{i} = {state_names[i]};")
    
    # Hidden layers
    for layer_idx in range(len(weights)):
        layer_in_size = weights[layer_idx].shape[1]
        layer_out_size = weights[layer_idx].shape[0]
        
        function_lines.append("")
        function_lines.append(f"  # Layer {layer_idx + 1}")
        
        # Compute linear combination for each neuron
        for neuron_idx in range(layer_out_size):
            # Initialize with bias
            expr = f"  layer{layer_idx + 1}_{neuron_idx}_pre = {biases[layer_idx][neuron_idx]}"
            
            # Add weighted inputs
            for prev_idx in range(layer_in_size):
                weight = weights[layer_idx][neuron_idx, prev_idx]
                if weight != 0:
                    expr += f" + {weight} * layer{layer_idx}_{prev_idx}"
            
            function_lines.append(expr + ";")
        
        # Apply activation function (except for the output layer)
        is_output_layer = (layer_idx == len(weights) - 1)
        
        if not is_output_layer:
            for neuron_idx in range(layer_out_size):
                function_lines.append(f"  layer{layer_idx + 1}_{neuron_idx} = "
                                     f"{elfin_activation}(layer{layer_idx + 1}_{neuron_idx}_pre);")
        else:
            # Output layer (typically no activation, or specific to barrier/Lyapunov)
            if model_type == "barrier":
                # For barrier functions, we might not have any output activation
                function_lines.append(f"  neural_output = layer{layer_idx + 1}_0_pre;")
            else:
                # For Lyapunov functions, ensure positive definiteness
                # Usually by squaring or using softplus
                function_lines.append(f"  neural_output = layer{layer_idx + 1}_0_pre ** 2;")
    
    # Add parameters
    function_lines.append("")
    function_lines.append("  params: {")
    function_lines.append("    # Class-K function parameter for barrier certificate")
    function_lines.append("    alpha: 1.0;")
    function_lines.append("  };")
    
    # Close the function definition
    function_lines.append("}")
    
    # Combine everything
    elfin_code = "\n".join(helpers + ["", ""] + function_lines)
    
    return elfin_code


def _export_taylor(
    model: Union[NeuralBarrierNetwork, NeuralLyapunovNetwork],
    model_type: str,
    system_name: str,
    state_dim: int,
    input_dim: int,
    state_names: List[str],
    input_names: Optional[List[str]],
    params: Dict[str, Any]
) -> str:
    """
    Export a neural network using Taylor series approximation.
    
    This method approximates the neural network using a Taylor series
    expansion around a reference point.
    """
    # Get expansion parameters
    expansion_order = params.get('order', 3)
    reference_point = params.get('reference_point', np.zeros(state_dim))
    
    # Sample points for function evaluation
    delta = params.get('delta', 0.1)
    
    # Generate sampling points for computing finite differences
    grid_points = []
    for i in range(state_dim):
        point = reference_point.copy()
        point[i] += delta
        grid_points.append(point)
        
        point = reference_point.copy()
        point[i] -= delta
        grid_points.append(point)
    
    # Add reference point itself
    grid_points = [reference_point] + grid_points
    grid_points = np.array(grid_points)
    
    # Evaluate model at sampling points
    if isinstance(model, (TorchBarrierNetwork, TorchLyapunovNetwork)):
        # For PyTorch models
        device = next(model.parameters()).device
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            values = model(grid_tensor).cpu().numpy()
    
    elif isinstance(model, (JAXBarrierNetwork, JAXLyapunovNetwork)):
        # For JAX models
        grid_array = jnp.array(grid_points)
        values = model(grid_array)
    
    else:
        # For generic models
        values = np.array([model(point.reshape(1, -1))[0, 0] for point in grid_points])
    
    # Compute numerical derivatives using finite differences
    # Base value at reference point
    f0 = values[0]
    
    # First derivatives
    gradients = np.zeros(state_dim)
    for i in range(state_dim):
        # Central difference
        gradients[i] = (values[2*i+1] - values[2*i+2]) / (2 * delta)
    
    # Generate Taylor series terms
    taylor_terms = []
    
    # Constant term
    taylor_terms.append(f"{f0}")
    
    # First-order terms
    for i in range(state_dim):
        if gradients[i] != 0:
            taylor_terms.append(f"{gradients[i]} * ({state_names[i]} - {reference_point[i]})")
    
    # Higher-order terms (simplified - we only compute mixed terms at reference point)
    if expansion_order >= 2:
        # TODO: Implement higher-order terms using more sophisticated
        # finite difference schemes or automatic differentiation
        pass
    
    # Combine terms
    function_expr = " + ".join(taylor_terms)
    
    # Generate code for the barrier or Lyapunov function
    function_lines = []
    
    if model_type == "barrier":
        function_lines.append(f"barrier Taylor{system_name}Barrier {{")
    else:
        function_lines.append(f"lyapunov Taylor{system_name}Lyapunov {{")
    
    # Add system reference
    function_lines.append(f"  system: {system_name};")
    function_lines.append("")
    
    # Add function definition
    if model_type == "barrier":
        function_lines.append("  # Taylor series approximation of barrier function")
        function_lines.append(f"  B: {function_expr};")
        function_lines.append("  alpha_fun: alpha * B;  # Class-K function for barrier certificate")
    else:
        function_lines.append("  # Taylor series approximation of Lyapunov function")
        function_lines.append(f"  V: {function_expr};")
    
    # Add parameters
    function_lines.append("")
    function_lines.append("  params: {")
    function_lines.append("    # Class-K function parameter for barrier certificate")
    function_lines.append("    alpha: 1.0;")
    function_lines.append("  };")
    
    # Close the function definition
    function_lines.append("}")
    
    # Combine everything
    elfin_code = "\n".join(function_lines)
    
    return elfin_code


def _export_polynomial(
    model: Union[NeuralBarrierNetwork, NeuralLyapunovNetwork],
    model_type: str,
    system_name: str,
    state_dim: int,
    input_dim: int,
    state_names: List[str],
    input_names: Optional[List[str]],
    params: Dict[str, Any]
) -> str:
    """
    Export a neural network using polynomial approximation.
    
    This method approximates the neural network using a polynomial fit.
    """
    # Get approximation parameters
    degree = params.get('degree', 3)
    num_samples = params.get('num_samples', 1000)
    bounds = params.get('bounds', None)
    
    # Default bounds if not provided
    if bounds is None:
        bounds = np.array([[-1, 1]] * state_dim)
    
    # Generate random samples for polynomial fitting
    samples = np.random.uniform(
        bounds[:, 0],
        bounds[:, 1],
        size=(num_samples, state_dim)
    )
    
    # Evaluate model at samples
    if isinstance(model, (TorchBarrierNetwork, TorchLyapunovNetwork)):
        # For PyTorch models
        device = next(model.parameters()).device
        samples_tensor = torch.tensor(samples, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            values = model(samples_tensor).cpu().numpy()
    
    elif isinstance(model, (JAXBarrierNetwork, JAXLyapunovNetwork)):
        # For JAX models
        samples_array = jnp.array(samples)
        values = model(samples_array)
    
    else:
        # For generic models
        values = np.array([model(sample.reshape(1, -1))[0, 0] for sample in samples])
    
    # Fit polynomial using least squares
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    # Generate polynomial features
    poly = PolynomialFeatures(degree=degree)
    poly_features = poly.fit_transform(samples)
    
    # Fit model
    reg = LinearRegression()
    reg.fit(poly_features, values)
    
    # Get the feature names
    feature_names = poly.get_feature_names_out(input_features=state_names)
    
    # Generate polynomial expression
    poly_terms = []
    
    for i, coef in enumerate(reg.coef_[0]):
        if abs(coef) > 1e-8:  # Skip near-zero coefficients
            feature = feature_names[i]
            
            # Replace ^ with ** for ELFIN syntax
            feature = feature.replace("^", "**")
            
            poly_terms.append(f"{coef} * {feature}")
    
    # Add intercept term
    if abs(reg.intercept_[0]) > 1e-8:
        poly_terms.append(f"{reg.intercept_[0]}")
    
    # Combine terms
    function_expr = " + ".join(poly_terms)
    
    # Generate code for the barrier or Lyapunov function
    function_lines = []
    
    if model_type == "barrier":
        function_lines.append(f"barrier Poly{system_name}Barrier {{")
    else:
        function_lines.append(f"lyapunov Poly{system_name}Lyapunov {{")
    
    # Add system reference
    function_lines.append(f"  system: {system_name};")
    function_lines.append("")
    
    # Add function definition
    if model_type == "barrier":
        function_lines.append("  # Polynomial approximation of barrier function")
        function_lines.append(f"  B: {function_expr};")
        function_lines.append("  alpha_fun: alpha * B;  # Class-K function for barrier certificate")
    else:
        function_lines.append("  # Polynomial approximation of Lyapunov function")
        function_lines.append(f"  V: {function_expr};")
    
    # Add parameters
    function_lines.append("")
    function_lines.append("  params: {")
    function_lines.append("    # Class-K function parameter for barrier certificate")
    function_lines.append("    alpha: 1.0;")
    function_lines.append("  };")
    
    # Close the function definition
    function_lines.append("}")
    
    # Combine everything
    elfin_code = "\n".join(function_lines)
    
    return elfin_code


def _export_piecewise(
    model: Union[NeuralBarrierNetwork, NeuralLyapunovNetwork],
    model_type: str,
    system_name: str,
    state_dim: int,
    input_dim: int,
    state_names: List[str],
    input_names: Optional[List[str]],
    params: Dict[str, Any]
) -> str:
    """
    Export a neural network using piecewise approximation.
    
    This method approximates the neural network using piecewise linear or
    polynomial functions.
    """
    # Get approximation parameters
    num_regions = params.get('num_regions', 5)
    bounds = params.get('bounds', None)
    method = params.get('method', 'linear')  # 'linear' or 'polynomial'
    
    # Default bounds if not provided
    if bounds is None:
        bounds = np.array([[-1, 1]] * state_dim)
    
    # Create grid of regions
    grid_points = []
    for dim in range(state_dim):
        points = np.linspace(bounds[dim, 0], bounds[dim, 1], num_regions + 1)
        grid_points.append(points)
    
    # Generate piecewise approximation
    if state_dim == 1:
        # 1D case - simple piecewise function
        regions = []
        
        for i in range(num_regions):
            x_min = grid_points[0][i]
            x_max = grid_points[0][i + 1]
            
            # Sample points in this region
            x_samples = np.linspace(x_min, x_max, 5)
            x_samples = x_samples.reshape(-1, 1)
            
            # Evaluate model at samples
            if isinstance(model, (TorchBarrierNetwork, TorchLyapunovNetwork)):
                # For PyTorch models
                device = next(model.parameters()).device
                samples_tensor = torch.tensor(x_samples, dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    values = model(samples_tensor).cpu().numpy()
            
            elif isinstance(model, (JAXBarrierNetwork, JAXLyapunovNetwork)):
                # For JAX models
                samples_array = jnp.array(x_samples)
                values = model(samples_array)
            
            else:
                # For generic models
                values = np.array([model(sample.reshape(1, -1))[0, 0] for sample in x_samples])
            
            # Fit function in this region
            if method == 'linear':
                # Linear fit (ax + b)
                a, b = np.polyfit(x_samples.flatten(), values.flatten(), 1)
                
                region_expr = f"if {state_names[0]} >= {x_min} and {state_names[0]} < {x_max} then {a} * {state_names[0]} + {b}"
            else:
                # Polynomial fit
                coeffs = np.polyfit(x_samples.flatten(), values.flatten(), 2)
                
                a, b, c = coeffs
                region_expr = f"if {state_names[0]} >= {x_min} and {state_names[0]} < {x_max} then {a} * {state_names[0]}**2 + {b} * {state_names[0]} + {c}"
            
            regions.append(region_expr)
        
        # Add last region (to include upper bound)
        regions[-1] = regions[-1].replace(f"and {state_names[0]} < {grid_points[0][-1]}", "")
        
        # Combine regions
        piecewise_expr = " else ".join(regions) + f" else {regions[-1].split('then ')[1]}"
    
    elif state_dim == 2:
        # 2D case - piecewise on a grid
        regions = []
        
        for i in range(num_regions):
            x_min = grid_points[0][i]
            x_max = grid_points[0][i + 1]
            
            for j in range(num_regions):
                y_min = grid_points[1][j]
                y_max = grid_points[1][j + 1]
                
                # Sample points in this region
                x_samples = np.linspace(x_min, x_max, 3)
                y_samples = np.linspace(y_min, y_max, 3)
                
                xx, yy = np.meshgrid(x_samples, y_samples)
                samples = np.column_stack((xx.flatten(), yy.flatten()))
                
                # Evaluate model at samples
                if isinstance(model, (TorchBarrierNetwork, TorchLyapunovNetwork)):
                    # For PyTorch models
                    device = next(model.parameters()).device
                    samples_tensor = torch.tensor(samples, dtype=torch.float32, device=device)
                    
                    with torch.no_grad():
                        values = model(samples_tensor).cpu().numpy().flatten()
                
                elif isinstance(model, (JAXBarrierNetwork, JAXLyapunovNetwork)):
                    # For JAX models
                    samples_array = jnp.array(samples)
                    values = model(samples_array).flatten()
                
                else:
                    # For generic models
                    values = np.array([model(sample.reshape(1, -1))[0, 0] for sample in samples])
                
                # Fit function in this region
                if method == 'linear':
                    # Linear fit (ax + by + c)
                    # Use least squares to fit a plane
                    A = np.column_stack((samples[:, 0], samples[:, 1], np.ones_like(samples[:, 0])))
                    coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
                    
                    a, b, c = coeffs
                    region_expr = (f"if {state_names[0]} >= {x_min} and {state_names[0]} < {x_max} and "
                                  f"{state_names[1]} >= {y_min} and {state_names[1]} < {y_max} then "
                                  f"{a} * {state_names[0]} + {b} * {state_names[1]} + {c}")
                else:
                    # Simple quadratic fit - this is a very simplified approach
                    # For more accurate results, use polynomial regression with cross-terms
                    x_mean = np.mean(samples[:, 0])
                    y_mean = np.mean(samples[:, 1])
                    z_mean = np.mean(values)
                    
                    region_expr = (f"if {state_names[0]} >= {x_min} and {state_names[0]} < {x_max} and "
                                  f"{state_names[1]} >= {y_min} and {state_names[1]} < {y_max} then "
                                  f"{z_mean} + 0.1 * (({state_names[0]} - {x_mean})**2 + ({state_names[1]} - {y_mean})**2)")
                
                regions.append(region_expr)
        
        # Combine regions
        piecewise_expr = " else ".join(regions) + f" else {regions[-1].split('then ')[1]}"
    
    else:
        # Higher dimensions - use simplified approach with regions around reference points
        num_references = min(10, num_regions**state_dim)
        
        # Generate reference points
        reference_points = np.random.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(num_references, state_dim)
        )
        
        # Evaluate model at reference points
        if isinstance(model, (TorchBarrierNetwork, TorchLyapunovNetwork)):
            # For PyTorch models
            device = next(model.parameters()).device
            ref_tensor = torch.tensor(reference_points, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                ref_values = model(ref_tensor).cpu().numpy().flatten()
        
        elif isinstance(model, (JAXBarrierNetwork, JAXLyapunovNetwork)):
            # For JAX models
            ref_array = jnp.array(reference_points)
            ref_values = model(ref_array).flatten()
        
        else:
            # For generic models
            ref_values = np.array([model(point.reshape(1, -1))[0, 0] for point in reference_points])
        
        # Generate piecewise expression based on distance to reference points
        ref_exprs = []
        
        for i in range(num_references):
            # Distance expression
            dist_terms = []
            for j in range(state_dim):
                dist_terms.append(f"({state_names[j]} - {reference_points[i, j]})**2")
            
            dist_expr = " + ".join(dist_terms)
            
            # Add reference point contribution
            ref_exprs.append(f"{ref_values[i]} * exp(-{dist_expr})")
        
        # Combine reference point expressions
        piecewise_expr = " + ".join(ref_exprs)
    
    # Generate code for the barrier or Lyapunov function
    function_lines = []
