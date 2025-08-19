"""
Command-line interface for ELFIN barrier certificates.

This module provides command-line tools for learning and verifying
barrier certificates for safety properties of dynamical systems.
"""

import os
import sys
import argparse
import logging
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from alan_backend.elfin.barrier.barrier_bridge_agent import BarrierBridgeAgent
from alan_backend.elfin.cli import ELFINCli, get_cache_dir

# Configure logging
logger = logging.getLogger("elfin.barrier.cli")


def add_barrier_commands(cli: ELFINCli) -> None:
    """
    Add barrier-related commands to the ELFIN CLI.
    
    Args:
        cli: ELFIN CLI instance
    """
    # Get the subparsers object
    parser = cli.parser
    subparsers = None
    
    # Find the subparsers object
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparsers = action
            break
    
    if not subparsers:
        logger.error("Failed to find subparsers in ELFIN CLI")
        return
    
    # Add barrier command
    barrier_parser = subparsers.add_parser(
        "barrier",
        help="Barrier certificate tools for safety verification"
    )
    barrier_subparsers = barrier_parser.add_subparsers(
        dest="barrier_command",
        help="Barrier command to execute"
    )
    
    # Add barrier learn command
    learn_parser = barrier_subparsers.add_parser(
        "learn",
        help="Learn a barrier certificate from data"
    )
    learn_parser.add_argument(
        "--system",
        type=str,
        required=True,
        help="Name of the system"
    )
    learn_parser.add_argument(
        "--safe-samples",
        type=str,
        required=True,
        help="Path to safe samples (CSV or NumPy file)"
    )
    learn_parser.add_argument(
        "--unsafe-samples",
        type=str,
        required=True,
        help="Path to unsafe samples (CSV or NumPy file)"
    )
    learn_parser.add_argument(
        "--boundary-samples",
        type=str,
        help="Path to boundary samples (CSV or NumPy file)"
    )
    learn_parser.add_argument(
        "--dict-type",
        type=str,
        choices=["rbf", "fourier", "poly"],
        default="rbf",
        help="Dictionary type (default: rbf)"
    )
    learn_parser.add_argument(
        "--dict-size",
        type=int,
        default=100,
        help="Dictionary size (default: 100)"
    )
    learn_parser.add_argument(
        "--domain",
        type=str,
        help="Domain bounds as min1,min2,...:max1,max2,... (default: inferred from samples)"
    )
    learn_parser.add_argument(
        "--safe-margin",
        type=float,
        default=0.1,
        help="Margin for safe region constraint (default: 0.1)"
    )
    learn_parser.add_argument(
        "--unsafe-margin",
        type=float,
        default=0.1,
        help="Margin for unsafe region constraint (default: 0.1)"
    )
    learn_parser.add_argument(
        "--boundary-margin",
        type=float,
        default=0.1,
        help="Margin for boundary decreasing constraint (default: 0.1)"
    )
    learn_parser.add_argument(
        "--auto-verify",
        action="store_true",
        help="Automatically verify after learning"
    )
    learn_parser.add_argument(
        "--output",
        type=str,
        help="Output file for barrier weights (JSON)"
    )
    
    # Add barrier verify command
    verify_parser = barrier_subparsers.add_parser(
        "verify",
        help="Verify a barrier certificate"
    )
    verify_parser.add_argument(
        "--system",
        type=str,
        required=True,
        help="Name of the system"
    )
    verify_parser.add_argument(
        "--method",
        type=str,
        choices=["mosek", "sparsepop", "sampling"],
        default="sampling",
        help="Verification method (default: sampling)"
    )
    verify_parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of samples for sampling-based verification (default: 10000)"
    )
    verify_parser.add_argument(
        "--output",
        type=str,
        help="Output file for verification result (JSON)"
    )
    
    # Add barrier refine command
    refine_parser = barrier_subparsers.add_parser(
        "refine",
        help="Refine a barrier certificate"
    )
    refine_parser.add_argument(
        "--system",
        type=str,
        required=True,
        help="Name of the system"
    )
    refine_parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Maximum number of refinement iterations (default: 5)"
    )
    refine_parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Stop refinement when verification succeeds"
    )
    refine_parser.add_argument(
        "--output",
        type=str,
        help="Output file for refinement result (JSON)"
    )
    
    # Add barrier visualize command
    visualize_parser = barrier_subparsers.add_parser(
        "visualize",
        help="Visualize a barrier certificate"
    )
    visualize_parser.add_argument(
        "--system",
        type=str,
        required=True,
        help="Name of the system"
    )
    visualize_parser.add_argument(
        "--dimensions",
        type=str,
        default="0,1",
        help="Dimensions to visualize as comma-separated indices (default: 0,1)"
    )
    visualize_parser.add_argument(
        "--resolution",
        type=int,
        default=100,
        help="Resolution of the grid for visualization (default: 100)"
    )
    visualize_parser.add_argument(
        "--fixed-values",
        type=str,
        help="Fixed values for non-visualized dimensions as comma-separated values"
    )
    visualize_parser.add_argument(
        "--output",
        type=str,
        default="barrier_visualization.png",
        help="Output image file (default: barrier_visualization.png)"
    )
    
    # Add barrier demo command
    demo_parser = barrier_subparsers.add_parser(
        "demo",
        help="Run a barrier certificate demo"
    )
    demo_parser.add_argument(
        "--scenario",
        type=str,
        choices=["double_integrator", "pendulum", "custom"],
        default="double_integrator",
        help="Demo scenario to run (default: double_integrator)"
    )
    demo_parser.add_argument(
        "--dict-type",
        type=str,
        choices=["rbf", "fourier", "poly"],
        default="rbf",
        help="Dictionary type (default: rbf)"
    )
    demo_parser.add_argument(
        "--dict-size",
        type=int,
        default=100,
        help="Dictionary size (default: 100)"
    )
    demo_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification"
    )
    demo_parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for visualization images (default: current directory)"
    )
    
    # Update cli._handle_command to handle barrier commands
    original_handle_command = cli._handle_command
    
    def new_handle_command(self, args: argparse.Namespace) -> int:
        """Handle commands."""
        if args.command == "barrier":
            return _handle_barrier_command(self, args)
        else:
            return original_handle_command(self, args)
    
    # Replace method
    cli._handle_command = new_handle_command.__get__(cli, type(cli))


def _handle_barrier_command(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Handle barrier commands.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    if not args.barrier_command:
        logger.error("No barrier command specified")
        return 1
        
    # Dispatch to barrier command handler
    if args.barrier_command == "learn":
        return _handle_barrier_learn(cli, args)
    elif args.barrier_command == "verify":
        return _handle_barrier_verify(cli, args)
    elif args.barrier_command == "refine":
        return _handle_barrier_refine(cli, args)
    elif args.barrier_command == "visualize":
        return _handle_barrier_visualize(cli, args)
    elif args.barrier_command == "demo":
        return _handle_barrier_demo(cli, args)
    else:
        logger.error(f"Unknown barrier command: {args.barrier_command}")
        return 1


def _load_samples(file_path: str) -> np.ndarray:
    """
    Load samples from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Samples as numpy array
    """
    try:
        # Check file extension
        if file_path.endswith('.csv'):
            # Load CSV file
            return np.loadtxt(file_path, delimiter=',')
        elif file_path.endswith('.npy'):
            # Load NumPy file
            return np.load(file_path)
        else:
            # Try to load as text file
            return np.loadtxt(file_path)
    except Exception as e:
        logger.error(f"Failed to load samples from {file_path}: {e}")
        raise ValueError(f"Failed to load samples from {file_path}: {e}")


def _handle_barrier_learn(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Handle barrier learn command.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    try:
        # Load samples
        safe_samples = _load_samples(args.safe_samples)
        unsafe_samples = _load_samples(args.unsafe_samples)
        boundary_samples = None
        if args.boundary_samples:
            boundary_samples = _load_samples(args.boundary_samples)
        
        # Parse domain if provided
        domain = None
        if args.domain:
            parts = args.domain.split(':')
            if len(parts) != 2:
                logger.error("Invalid domain format, expected min1,min2,...:max1,max2,...")
                return 1
            
            # Parse lower and upper bounds
            lower = np.array([float(x) for x in parts[0].split(',')])
            upper = np.array([float(x) for x in parts[1].split(',')])
            
            # Check dimensions
            if len(lower) != len(upper):
                logger.error("Lower and upper bounds must have the same dimension")
                return 1
            
            # Check dimensions with samples
            state_dim = safe_samples.shape[1] if len(safe_samples) > 0 else unsafe_samples.shape[1]
            if len(lower) != state_dim:
                logger.error(f"Domain dimension ({len(lower)}) does not match sample dimension ({state_dim})")
                return 1
            
            domain = (lower, upper)
        
        # Create options
        options = {
            'safe_margin': args.safe_margin,
            'unsafe_margin': args.unsafe_margin,
            'boundary_margin': args.boundary_margin
        }
        
        # Create agent
        agent = BarrierBridgeAgent(
            name=f"barrier_cli_{args.system}",
            cache_dir=cli.cache_dir,
            auto_verify=args.auto_verify,
            options=options
        )
        
        # Learn barrier function
        print(f"Learning barrier certificate for {args.system}...")
        print(f"Dictionary type: {args.dict_type}, size: {args.dict_size}")
        print(f"Safe samples: {len(safe_samples)}, unsafe samples: {len(unsafe_samples)}")
        if boundary_samples is not None:
            print(f"Boundary samples: {len(boundary_samples)}")
        
        start_time = time.time()
        barrier_fn = agent.learn_barrier(
            system_name=args.system,
            safe_samples=safe_samples,
            unsafe_samples=unsafe_samples,
            boundary_samples=boundary_samples,
            dictionary_type=args.dict_type,
            dictionary_size=args.dict_size,
            domain=domain,
            options=options
        )
        learning_time = time.time() - start_time
        
        print(f"Barrier certificate learned in {learning_time:.3f} seconds")
        
        # Print weights shape
        print(f"Barrier weights shape: {barrier_fn.weights.shape}")
        
        # Save barrier weights if output specified
        if args.output:
            # Create output dictionary
            output_data = {
                'system': args.system,
                'dict_type': args.dict_type,
                'dict_size': args.dict_size,
                'weights': barrier_fn.weights.tolist(),
                'learning_time': learning_time,
                'n_safe_samples': len(safe_samples),
                'n_unsafe_samples': len(unsafe_samples),
                'n_boundary_samples': 0 if boundary_samples is None else len(boundary_samples)
            }
            
            # Save to file
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Barrier weights saved to {args.output}")
        
        # Print verification results if auto-verify enabled
        if args.auto_verify:
            print("\nAutomatic verification results:")
            result = agent.results[args.system].get('verification', {}).get('result')
            if result:
                print(f"Success: {result.success}")
                print(f"Status: {result.status}")
                if not result.success and result.counterexample is not None:
                    print(f"Counterexample: {result.counterexample}")
                    print(f"Violation reason: {result.violation_reason}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error learning barrier certificate: {e}", exc_info=True)
        return 1


def _handle_barrier_verify(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Handle barrier verify command.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    try:
        # Create agent
        agent = BarrierBridgeAgent(
            name=f"barrier_cli_{args.system}",
            cache_dir=cli.cache_dir
        )
        
        # Check if system exists
        if args.system not in agent.results:
            logger.error(f"System '{args.system}' not found. Learn a barrier certificate first.")
            return 1
        
        # Verify barrier function
        print(f"Verifying barrier certificate for {args.system}...")
        print(f"Method: {args.method}")
        if args.method == 'sampling':
            print(f"Number of samples: {args.samples}")
            options = {'n_samples': args.samples}
        else:
            options = {}
        
        start_time = time.time()
        verification_result = agent.verify(
            system_name=args.system,
            method=args.method,
            options=options
        )
        verification_time = time.time() - start_time
        
        # Print verification results
        print(f"\nVerification completed in {verification_time:.3f} seconds")
        print(f"Success: {verification_result.success}")
        print(f"Status: {verification_result.status}")
        
        if not verification_result.success:
            if verification_result.counterexample is not None:
                print(f"Counterexample: {verification_result.counterexample}")
            
            print(f"Violation reason: {verification_result.violation_reason}")
            error_code = verification_result.get_error_code()
            if error_code:
                print(f"Error code: {error_code}")
                print(f"Error message: {verification_result.get_violation_message()}")
        
        # Save verification results if output specified
        if args.output:
            # Create output dictionary
            output_data = {
                'system': args.system,
                'method': args.method,
                'success': verification_result.success,
                'status': verification_result.status,
                'verification_time': verification_time
            }
            
            # Add counterexample if available
            if verification_result.counterexample is not None:
                output_data['counterexample'] = verification_result.counterexample.tolist()
                output_data['violation_reason'] = verification_result.violation_reason
                output_data['error_code'] = verification_result.get_error_code()
            
            # Save to file
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Verification results saved to {args.output}")
        
        return 0 if verification_result.success else 2
    
    except Exception as e:
        logger.error(f"Error verifying barrier certificate: {e}", exc_info=True)
        return 1


def _handle_barrier_refine(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Handle barrier refine command.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    try:
        # Create agent
        agent = BarrierBridgeAgent(
            name=f"barrier_cli_{args.system}",
            cache_dir=cli.cache_dir
        )
        
        # Check if system exists
        if args.system not in agent.results:
            logger.error(f"System '{args.system}' not found. Learn a barrier certificate first.")
            return 1
        
        # Refine barrier function
        print(f"Refining barrier certificate for {args.system}...")
        print(f"Max iterations: {args.iterations}")
        print(f"Stop on success: {args.stop_on_success}")
        
        start_time = time.time()
        verification_result = agent.refine_auto(
            system_name=args.system,
            max_iterations=args.iterations,
            stop_on_success=args.stop_on_success
        )
        refinement_time = time.time() - start_time
        
        # Print refinement results
        print(f"\nRefinement completed in {refinement_time:.3f} seconds")
        
        # Get refinement statistics
        auto_refinement = agent.results[args.system].get('auto_refinement', {})
        iterations = auto_refinement.get('iterations', 0)
        
        print(f"Iterations: {iterations}")
        print(f"Success: {verification_result.success}")
        print(f"Status: {verification_result.status}")
        
        if not verification_result.success:
            if verification_result.counterexample is not None:
                print(f"Counterexample: {verification_result.counterexample}")
            
            print(f"Violation reason: {verification_result.violation_reason}")
            error_code = verification_result.get_error_code()
            if error_code:
                print(f"Error code: {error_code}")
                print(f"Error message: {verification_result.get_violation_message()}")
        
        # Save refinement results if output specified
        if args.output:
            # Create output dictionary
            output_data = {
                'system': args.system,
                'iterations': iterations,
                'success': verification_result.success,
                'status': verification_result.status,
                'refinement_time': refinement_time
            }
            
            # Add counterexample if available
            if verification_result.counterexample is not None:
                output_data['counterexample'] = verification_result.counterexample.tolist()
                output_data['violation_reason'] = verification_result.violation_reason
                output_data['error_code'] = verification_result.get_error_code()
            
            # Save to file
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Refinement results saved to {args.output}")
        
        return 0 if verification_result.success else 2
    
    except Exception as e:
        logger.error(f"Error refining barrier certificate: {e}", exc_info=True)
        return 1


def _handle_barrier_visualize(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Handle barrier visualize command.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    try:
        # Import matplotlib
        import matplotlib.pyplot as plt
        
        # Create agent
        agent = BarrierBridgeAgent(
            name=f"barrier_cli_{args.system}",
            cache_dir=cli.cache_dir
        )
        
        # Check if system exists
        if args.system not in agent.results:
            logger.error(f"System '{args.system}' not found. Learn a barrier certificate first.")
            return 1
        
        # Get barrier function and domain
        barrier_fn = agent.results[args.system]['barrier']
        domain = agent.results[args.system]['domain']
        
        # Parse dimensions to visualize
        dimensions = [int(d) for d in args.dimensions.split(',')]
        if len(dimensions) != 2:
            logger.error("Dimensions must be two comma-separated indices")
            return 1
        
        # Get state dimensionality
        state_dim = safe_samples = agent.results[args.system]['data']['safe_samples'].shape[1]
        
        # Check dimensions validity
        for d in dimensions:
            if d < 0 or d >= state_dim:
                logger.error(f"Invalid dimension {d}, must be between 0 and {state_dim-1}")
                return 1
        
        # Parse fixed values for non-visualized dimensions
        fixed_values = np.zeros(state_dim)
        if args.fixed_values:
            values = [float(v) for v in args.fixed_values.split(',')]
            if len(values) != state_dim - 2:
                logger.error(f"Expected {state_dim-2} fixed values for non-visualized dimensions")
                return 1
            
            # Assign fixed values to non-visualized dimensions
            j = 0
            for i in range(state_dim):
                if i not in dimensions:
                    fixed_values[i] = values[j]
                    j += 1
        
        # Extract domain bounds for visualized dimensions
        x_dim, y_dim = dimensions
        x_min, y_min = domain[0][x_dim], domain[0][y_dim]
        x_max, y_max = domain[1][x_dim], domain[1][y_dim]
        
        # Create grid for visualization
        resolution = args.resolution
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate barrier function at each grid point
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                # Create full state vector with fixed values
                state = fixed_values.copy()
                state[x_dim] = X[i, j]
                state[y_dim] = Y[i, j]
                
                # Evaluate barrier function
                Z[i, j] = barrier_fn(state)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot barrier function as filled contour
        cmap = plt.cm.RdBu_r  # Red for unsafe (> 0), blue for safe (< 0)
        contour = plt.contourf(X, Y, Z, levels=50, cmap=cmap, alpha=0.8)
        plt.colorbar(contour, label="Barrier Function Value")
        
        # Plot zero level set (boundary between safe and unsafe regions)
        boundary = plt.contour(X, Y, Z, levels=[0], colors='k', linewidths=2)
        plt.clabel(boundary, fmt='B(x) = 0', fontsize=10)
        
        # Set labels and title
        plt.xlabel(f"Dimension {x_dim}")
        plt.ylabel(f"Dimension {y_dim}")
        plt.title(f"Barrier Function for {args.system}")
        plt.grid(True)
        
        # Save visualization
        plt.tight_layout()
        plt.savefig(args.output)
        
        print(f"Visualization saved to {args.output}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error visualizing barrier certificate: {e}", exc_info=True)
        return 1


def _handle_barrier_demo(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Handle barrier demo command.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    try:
        # Change directory to output directory
        original_dir = os.getcwd()
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
        
        # Run appropriate demo
        if args.scenario == "double_integrator":
            from alan_backend.elfin.examples.demo_barrier import demo_barrier_certificate
            
            # Run demo
            demo_barrier_certificate()
        elif args.scenario == "pendulum":
            logger.error("Pendulum demo not implemented yet")
            return 1
        elif args.scenario == "custom":
            logger.error("Custom demo not implemented yet")
            return 1
        else:
            logger.error(f"Unknown demo scenario: {args.scenario}")
            return 1
        
        # Change back to original directory
        os.chdir(original_dir)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error running barrier demo: {e}", exc_info=True)
        return 1


def init_module():
    """Initialize the Barrier CLI module."""
    # Register with ELFIN CLI
    try:
        from alan_backend.elfin.cli import ELFINCli
        
        # Create a dummy CLI instance for registration
        dummy_cli = ELFINCli()
        
        # Add barrier commands
        add_barrier_commands(dummy_cli)
        
        logger.info("Barrier commands registered with ELFIN CLI")
        
    except ImportError:
        logger.warning("ELFIN CLI not found, barrier commands not registered")


# Initialize when imported
init_module()
