"""
Command-line interface for Koopman operator analysis.

This module provides command-line tools for Koopman operator analysis
and Lyapunov function generation, along with dashboard API endpoints 
for interactive parameter adjustment.
"""

import os
import sys
import pathlib
import argparse
import logging
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# Import ELFIN components
from .koopman_bridge_agent import (
    KoopmanBridgeAgent,
    create_pendulum_agent,
    create_vdp_agent
)
from .edmd import load_trajectory_data
from ..cli import ELFINCli

# Configure logging
logger = logging.getLogger("elfin.koopman.cli")


def add_koopman_commands(cli: ELFINCli) -> None:
    """
    Add Koopman-related commands to the ELFIN CLI.
    
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
    
    # Add koopman command
    koopman_parser = subparsers.add_parser(
        "koopman",
        help="Koopman operator analysis and Lyapunov function generation"
    )
    koopman_subparsers = koopman_parser.add_subparsers(
        dest="koopman_command",
        help="Koopman command to execute"
    )
    
    # Add koopman learn command
    learn_parser = koopman_subparsers.add_parser(
        "learn",
        help="Learn a Koopman operator from trajectory data"
    )
    learn_parser.add_argument(
        "file",
        type=str,
        help="Path to trajectory data file"
    )
    learn_parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System name (default: derived from filename)"
    )
    learn_parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="State space dimension (default: 2)"
    )
    learn_parser.add_argument(
        "--dict",
        type=str,
        default="rbf",
        choices=["rbf", "fourier", "poly"],
        help="Dictionary type (default: rbf)"
    )
    learn_parser.add_argument(
        "--modes",
        type=int,
        default=100,
        help="Number of dictionary modes (default: 100)"
    )
    learn_parser.add_argument(
        "--lambda-cut",
        type=float,
        default=0.99,
        help="Cutoff for stable modes (default: 0.99)"
    )
    learn_parser.add_argument(
        "--continuous",
        action="store_true",
        help="Treat system as continuous-time"
    )
    learn_parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step (if None, inferred from data)"
    )
    learn_parser.add_argument(
        "--skip-header",
        type=int,
        default=0,
        help="Number of header rows to skip (default: 0)"
    )
    learn_parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="Delimiter used in data file (default: ,)"
    )
    learn_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification after learning"
    )
    learn_parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot Lyapunov function"
    )
    
    # Add koopman demo command
    demo_parser = koopman_subparsers.add_parser(
        "demo",
        help="Run a Koopman demo"
    )
    demo_parser.add_argument(
        "system",
        type=str,
        choices=["pendulum", "vdp", "both"],
        help="System to demonstrate"
    )
    demo_parser.add_argument(
        "--dict",
        type=str,
        default="rbf",
        choices=["rbf", "fourier", "poly"],
        help="Dictionary type (default: rbf)"
    )
    demo_parser.add_argument(
        "--modes",
        type=int,
        default=100,
        help="Number of dictionary modes (default: 100)"
    )
    demo_parser.add_argument(
        "--lambda-cut",
        type=float,
        default=0.98,
        help="Cutoff for stable modes (default: 0.98)"
    )
    demo_parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting"
    )
    
    # Add koopman info command
    info_parser = koopman_subparsers.add_parser(
        "info",
        help="Show information about a Koopman model"
    )
    info_parser.add_argument(
        "system",
        type=str,
        help="System name"
    )
    
    # Update cli._handle_command to handle koopman commands
    original_handle_command = cli._handle_command
    
    def new_handle_command(self, args: argparse.Namespace) -> int:
        """Handle commands."""
        if args.command == "koopman":
            return _handle_koopman_command(self, args)
        else:
            return original_handle_command(self, args)
    
    # Replace method
    cli._handle_command = new_handle_command.__get__(cli, type(cli))


def _handle_koopman_command(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Handle koopman commands.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    if not args.koopman_command:
        logger.error("No koopman command specified")
        return 1
        
    # Dispatch to koopman command handler
    if args.koopman_command == "learn":
        return _handle_koopman_learn(cli, args)
    elif args.koopman_command == "demo":
        return _handle_koopman_demo(cli, args)
    elif args.koopman_command == "info":
        return _handle_koopman_info(cli, args)
    else:
        logger.error(f"Unknown koopman command: {args.koopman_command}")
        return 1


def _handle_koopman_learn(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Handle koopman learn command.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    # Import visualization module only if needed
    if args.plot:
        try:
            from alan_backend.elfin.examples.koopman_bridge_demo import (
                plot_lyapunov_function,
                plot_verifier_results
            )
        except ImportError:
            logger.warning("Visualization module not available, plotting disabled")
            args.plot = False
    
    # Check if file exists
    file_path = pathlib.Path(args.file)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return 1
    
    # Derive system name from filename if not provided
    system_name = args.system
    if system_name is None:
        system_name = file_path.stem
    
    # Print info
    print(f"Learning Koopman operator for system: {system_name}")
    print(f"Data file: {file_path}")
    print(f"Dictionary: {args.dict} with {args.modes} modes")
    print(f"State dimension: {args.dim}")
    
    # Create Koopman Bridge Agent
    try:
        # Create agent
        agent = KoopmanBridgeAgent(
            name=f"koopman_{system_name}",
            cache_dir=cli.cache_dir,
            auto_verify=not args.no_verify
        )
        
        # Learn from file
        start_time = time.time()
        lyap_fn = agent.learn_from_file(
            file_path=str(file_path),
            system_name=system_name,
            state_dim=args.dim,
            dict_type=args.dict,
            dict_size=args.modes,
            continuous_time=args.continuous,
            dt=args.dt,
            λ_cut=args.lambda_cut,
            skip_header=args.skip_header,
            delimiter=args.delimiter
        )
        
        # Print elapsed time
        elapsed = time.time() - start_time
        print(f"\nLearning completed in {elapsed:.2f} seconds")
        
        # Print summary
        print("\nSummary:")
        print(agent.get_summary())
        
        # Plot if requested
        if args.plot:
            # Get domain bounds from data
            try:
                # Load trajectory data
                x, _ = load_trajectory_data(
                    file_path=str(file_path),
                    state_dim=args.dim,
                    dt=args.dt,
                    skip_header=args.skip_header,
                    delimiter=args.delimiter
                )
                
                # Define domain based on data bounds
                lower = np.min(x, axis=0)
                upper = np.max(x, axis=0)
                # Add some margin
                margin = 0.2 * (upper - lower)
                domain = (lower - margin, upper + margin)
                
                # Plot Lyapunov function
                print("\nPlotting Lyapunov function...")
                plot_lyapunov_function(
                    lyap_fn=lyap_fn,
                    domain=domain,
                    title=f"Koopman Lyapunov Function: {system_name}",
                    show_modes=True
                )
                
            except Exception as e:
                logger.error(f"Error plotting Lyapunov function: {e}")
        
        print(f"\nSuccess! Koopman-based Lyapunov function created for {system_name}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error learning Koopman operator: {e}", exc_info=True)
        return 1


def _handle_koopman_demo(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Handle koopman demo command.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    try:
        # Import demo module
        from alan_backend.elfin.examples.koopman_bridge_demo import (
            demo_pendulum,
            demo_vdp
        )
        
        # Run demo(s)
        if args.system == "pendulum" or args.system == "both":
            demo_pendulum(
                dict_type=args.dict,
                dict_size=args.modes,
                plot_results=not args.no_plot
            )
        
        if args.system == "vdp" or args.system == "both":
            demo_vdp(
                dict_type=args.dict,
                dict_size=args.modes,
                plot_results=not args.no_plot
            )
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running Koopman demo: {e}", exc_info=True)
        return 1


def _handle_koopman_info(cli: ELFINCli, args: argparse.Namespace) -> int:
    """
    Handle koopman info command.
    
    Args:
        cli: ELFIN CLI instance
        args: Parsed arguments
        
    Returns:
        Exit code
    """
    try:
        # Find the Koopman agent cache file
        agent_cache = cli.cache_dir / f"koopman_{args.system}_stability" / "cache.json"
        if not agent_cache.exists():
            logger.error(f"No Koopman model found for system: {args.system}")
            return 1
        
        # Create agent from cache
        agent = KoopmanBridgeAgent(
            name=f"koopman_{args.system}",
            cache_dir=cli.cache_dir
        )
        
        # Print info
        print(f"Koopman model info for system: {args.system}")
        print(agent.get_summary())
        
        return 0
        
    except Exception as e:
        logger.error(f"Error getting Koopman model info: {e}", exc_info=True)
        return 1


# Dashboard API endpoints
def add_koopman_api_endpoints(app):
    """
    Add Koopman API endpoints to the dashboard.
    
    Args:
        app: Flask app
    """
    try:
        from flask import request, jsonify
        
# API endpoints version 1 (previously unversioned)
@app.route('/api/v1/koopman/weighting', methods=['GET', 'POST'])
# Legacy support for unversioned endpoint
@app.route('/api/koopman/weighting', methods=['GET', 'POST'])
        def koopman_weighting():
            """API endpoint for adjusting Koopman Lyapunov function parameters."""
            # Get parameters
            if request.method == 'POST':
                data = request.json
                system_name = data.get('system', '')
                lambda_cut = float(data.get('cut', 0.98))
                weighting = data.get('weighting', 'uniform')
            else:
                system_name = request.args.get('system', '')
                lambda_cut = float(request.args.get('cut', 0.98))
                weighting = request.args.get('weighting', 'uniform')
            
            # Check parameters
            if not system_name:
                return jsonify({
                    'status': 'error',
                    'message': 'System name not provided'
                }), 400
            
            # Create agent from cache
            try:
                from alan_backend.elfin.cli import get_cache_dir
                
                cache_dir = get_cache_dir()
                agent = KoopmanBridgeAgent(
                    name=f"koopman_{system_name}",
                    cache_dir=cache_dir
                )
                
                # Check if system exists
                if system_name not in agent.results:
                    return jsonify({
                        'status': 'error',
                        'message': f'System {system_name} not found'
                    }), 404
                
                # Get original result
                original_result = agent.results[system_name]
                k_matrix = original_result['k_matrix']
                dictionary = original_result['dictionary']
                
                # Create new Lyapunov function with updated parameters
                lyap_fn = create_koopman_lyapunov(
                    name=f"{system_name}_adjusted",
                    k_matrix=k_matrix,
                    dictionary=dictionary,
                    lambda_cut=lambda_cut,
                    continuous_time=True,
                    weighting=weighting
                )
                
                # Update result
                agent.results[system_name]['lyapunov'] = lyap_fn
                
                # Return info for plotting
                eigenvalues = [complex(ev) for ev in lyap_fn.eigenvalues]
                stable_indices = lyap_fn.stable_indices
                
                return jsonify({
                    'status': 'success',
                    'message': 'Lyapunov function updated',
                    'system': system_name,
                    'lambda_cut': lambda_cut,
                    'weighting': weighting,
                    'n_stable_modes': len(stable_indices),
                    'n_total_modes': len(eigenvalues),
                    'stable_indices': stable_indices,
                    'eigenvalues': {
                        'real': [float(ev.real) for ev in eigenvalues],
                        'imag': [float(ev.imag) for ev in eigenvalues]
                    }
                })
                
            except Exception as e:
                logger.error(f"Error adjusting Koopman parameters: {e}", exc_info=True)
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
# API endpoints version 1 (previously unversioned)
@app.route('/api/v1/koopman/cv', methods=['POST'])
# Legacy support for unversioned endpoint
@app.route('/api/koopman/cv', methods=['POST'])
        def koopman_cross_validation():
            """API endpoint for cross-validation of Koopman models."""
            # Get parameters
            data = request.json
            system_name = data.get('system', '')
            n_splits = int(data.get('splits', 5))
            
            # Check parameters
            if not system_name:
                return jsonify({
                    'status': 'error',
                    'message': 'System name not provided'
                }), 400
            
            # Create agent from cache
            try:
                from alan_backend.elfin.cli import get_cache_dir
                from .edmd import kfold_validation
                
                cache_dir = get_cache_dir()
                agent = KoopmanBridgeAgent(
                    name=f"koopman_{system_name}",
                    cache_dir=cache_dir
                )
                
                # Check if system exists
                if system_name not in agent.results:
                    return jsonify({
                        'status': 'error',
                        'message': f'System {system_name} not found'
                    }), 404
                
                # Get original data
                original_result = agent.results[system_name]
                
                if 'data' not in original_result or 'x' not in original_result['data']:
                    return jsonify({
                        'status': 'error',
                        'message': f'No data found for system {system_name}'
                    }), 404
                
                # Get data and dictionary
                x = original_result['data']['x']
                x_next = original_result['data']['x_next']
                dictionary = original_result['dictionary']
                
                # Run cross-validation
                cv_results = kfold_validation(
                    dictionary=dictionary,
                    x=x,
                    x_next=x_next,
                    n_splits=n_splits
                )
                
                # Return results
                return jsonify({
                    'status': 'success',
                    'message': 'Cross-validation completed',
                    'system': system_name,
                    'n_splits': n_splits,
                    'train_mse': {
                        'mean': float(cv_results['train_mse_mean']),
                        'std': float(cv_results['train_mse_std'])
                    },
                    'val_mse': {
                        'mean': float(cv_results['val_mse_mean']),
                        'std': float(cv_results['val_mse_std'])
                    },
                    'all_mse': float(cv_results['all_mse']),
                    'eigenvalues_drift': {
                        'mean': float(np.mean([
                            np.abs(ev_fold - ev_all) 
                            for ev_fold, ev_all in zip(
                                cv_results['eigenvalues_folds'][0], 
                                cv_results['eigenvalues']
                            )
                        ])),
                        'max': float(np.max([
                            np.abs(ev_fold - ev_all)
                            for ev_fold, ev_all in zip(
                                cv_results['eigenvalues_folds'][0],
                                cv_results['eigenvalues']
                            )
                        ]))
                    }
                })
                
            except Exception as e:
                logger.error(f"Error running cross-validation: {e}", exc_info=True)
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500

def init_module():
    """Initialize the Koopman CLI module."""
    # Register with ELFIN CLI
    try:
        from alan_backend.elfin.cli import ELFINCli
        
        # Create a dummy CLI instance for registration
        dummy_cli = ELFINCli()
        
        # Add Koopman commands
        add_koopman_commands(dummy_cli)
        
        # Register dashboard API endpoints
        try:
            from alan_backend.elfin.api import get_flask_app
            app = get_flask_app()
            if app:
                add_koopman_api_endpoints(app)
                logger.info("Koopman API endpoints registered with dashboard")
        except ImportError:
            logger.warning("Dashboard API not found, endpoints not registered")
        
        logger.info("Koopman commands registered with ELFIN CLI")
        
    except ImportError:
        logger.warning("ELFIN CLI not found, Koopman commands not registered")


# Initialize when imported
init_module()
