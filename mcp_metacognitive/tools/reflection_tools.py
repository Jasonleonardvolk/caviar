"""
Reflection and self-modification tools
"""

import json
import numpy as np
from typing import Optional, List


def register_reflection_tools(mcp, state_manager):
    """Register reflection-related tools."""
    
    @mcp.tool()
    async def reflect(
        steps: int = 1,
        use_line_search: bool = False,
        momentum: Optional[float] = None
    ) -> str:
        """
        Apply reflective operator to current cognitive state.
        
        The reflective operator updates beliefs using natural gradient ascent
        on a log posterior function.
        
        Args:
            steps: Number of reflection steps to perform
            use_line_search: Use adaptive step size with line search
            momentum: Override default momentum (0-1, higher = more momentum)
        
        Returns:
            JSON with new state and metrics
        """
        current = await state_manager.get_current_state()
        state = np.array(current['state'])
        
        # Override momentum if specified
        if momentum is not None:
            old_momentum = state_manager.reflective_op.momentum
            state_manager.reflective_op.momentum = momentum
        
        try:
            # Apply reflection
            for _ in range(steps):
                if use_line_search:
                    state = state_manager.reflective_op.apply_with_line_search(state)
                else:
                    state = state_manager.reflective_op.apply(state)
            
            # Update state
            await state_manager.update_state(
                state,
                source='reflect',
                metadata={'steps': steps, 'line_search': use_line_search}
            )
            
            new_state_info = await state_manager.get_current_state()
            
            return json.dumps({
                'success': True,
                'steps_applied': steps,
                'new_state': new_state_info['state'],
                'phi': new_state_info['phi'],
                'free_energy': new_state_info['free_energy'],
                'state_change': float(np.linalg.norm(
                    np.array(new_state_info['state']) - np.array(current['state'])
                ))
            }, indent=2)
            
        finally:
            # Restore momentum if it was changed
            if momentum is not None:
                state_manager.reflective_op.momentum = old_momentum
    
    @mcp.tool()
    async def self_modify(
        lambda_d: float = 1.0,
        adaptive: bool = False,
        max_iterations: Optional[int] = None
    ) -> str:
        """
        Apply self-modification to optimize cognitive state.
        
        Minimizes free energy while preserving consciousness (IIT).
        
        Args:
            lambda_d: Regularization weight for distance from current state
            adaptive: Use adaptive optimization with decreasing regularization
            max_iterations: Override default maximum iterations
        
        Returns:
            JSON with optimization results and new state
        """
        current = await state_manager.get_current_state()
        state = np.array(current['state'])
        
        # Override max iterations if specified
        if max_iterations is not None:
            old_max_iter = state_manager.self_mod_op.max_iter
            state_manager.self_mod_op.max_iter = max_iterations
        
        try:
            # Apply self-modification
            if adaptive:
                new_state = state_manager.self_mod_op.adaptive_optimize(state, lambda_d)
            else:
                new_state = state_manager.self_mod_op.optimize(state, lambda_d)
            
            # Get optimization summary
            summary = state_manager.self_mod_op.get_optimization_summary()
            
            # Update state
            await state_manager.update_state(
                new_state,
                source='self_modify',
                metadata={
                    'lambda_d': lambda_d,
                    'adaptive': adaptive,
                    'optimization_summary': summary
                }
            )
            
            new_state_info = await state_manager.get_current_state()
            
            return json.dumps({
                'success': True,
                'optimization_summary': summary,
                'new_state': new_state_info['state'],
                'phi': new_state_info['phi'],
                'free_energy': new_state_info['free_energy'],
                'improvement': {
                    'free_energy_reduction': summary.get('free_energy_reduction', 0),
                    'iit_change': summary.get('iit_change', 0),
                    'converged': summary.get('converged', False)
                }
            }, indent=2)
            
        finally:
            # Restore max iterations if changed
            if max_iterations is not None:
                state_manager.self_mod_op.max_iter = old_max_iter
    
    @mcp.tool()
    async def find_fixed_point(
        method: str = "iteration",
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        use_current_state: bool = True
    ) -> str:
        """
        Find fixed point of cognitive dynamics.
        
        Computes state x* such that R(x*) = x* where R is the reflective operator.
        
        Args:
            method: Algorithm - 'iteration', 'newton', or 'anderson'
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            use_current_state: Use current state as initial guess
        
        Returns:
            JSON with fixed point and stability analysis
        """
        from cog.fixed_point import find_fixed_point, analyze_fixed_point_stability
        
        # Get initial guess
        if use_current_state:
            current = await state_manager.get_current_state()
            initial_guess = np.array(current['state'])
        else:
            initial_guess = np.random.randn(state_manager.dimension) * 0.1
        
        try:
            # Find fixed point
            fixed_point = find_fixed_point(
                state_manager.reflective_op,
                initial_guess,
                tol=tolerance,
                max_iter=max_iterations,
                method=method,
                verbose=False
            )
            
            # Analyze stability
            stability = analyze_fixed_point_stability(
                state_manager.reflective_op,
                fixed_point
            )
            
            # Format eigenvalues for JSON
            eigenvalues = stability['eigenvalues']
            eigenvalues_list = [
                {'real': float(np.real(ev)), 'imag': float(np.imag(ev))}
                for ev in eigenvalues
            ]
            
            return json.dumps({
                'success': True,
                'fixed_point': fixed_point.tolist(),
                'stability_analysis': {
                    'is_stable': bool(stability['is_stable']),
                    'is_attracting': bool(stability['is_attracting']),
                    'spectral_radius': float(stability['spectral_radius']),
                    'basin_radius_estimate': float(stability['basin_radius_estimate']),
                    'eigenvalues': eigenvalues_list,
                    'fixed_point_error': float(stability['fixed_point_error'])
                },
                'metrics': {
                    'phi': float(state_manager.compute_iit_phi(fixed_point)),
                    'free_energy': float(state_manager.compute_free_energy(fixed_point))
                }
            }, indent=2)
            
        except RuntimeError as e:
            return json.dumps({
                'success': False,
                'error': str(e),
                'method': method,
                'initial_guess_norm': float(np.linalg.norm(initial_guess))
            }, indent=2)