"""
Cognitive dynamics and evolution tools
"""

import json
import numpy as np
from typing import Optional, List, Dict, Any


def register_dynamics_tools(mcp, state_manager):
    """Register dynamics-related tools."""
    
    @mcp.tool()
    async def evolve(
        time_span: float = 1.0,
        dt: float = 0.01,
        adaptive: bool = False,
        noise_scale: Optional[float] = None
    ) -> str:
        """
        Evolve cognitive state using stochastic dynamics.
        
        Implements SDE: ds = (R(s) - s)dt + σ dW
        
        Args:
            time_span: Total time to evolve
            dt: Time step (ignored if adaptive=True)
            adaptive: Use adaptive time stepping
            noise_scale: Override noise scale (default uses configured value)
        
        Returns:
            JSON with trajectory information and final state
        """
        current = await state_manager.get_current_state()
        initial_state = np.array(current['state'])
        
        # Override noise if specified
        if noise_scale is not None:
            old_sigma = state_manager.dynamics.sigma
            state_manager.dynamics.sigma = noise_scale
        
        try:
            if adaptive:
                trajectory, times = state_manager.dynamics.evolve_adaptive(
                    initial_state,
                    time_span,
                    dt_min=dt/10,
                    dt_max=dt*10,
                    tol=1e-3
                )
            else:
                trajectory = state_manager.dynamics.evolve(
                    initial_state,
                    time_span,
                    dt,
                    return_times=False
                )
                times = np.linspace(0, time_span, len(trajectory))
            
            # Update to final state
            final_state = trajectory[-1]
            await state_manager.update_state(
                final_state,
                source='evolve',
                metadata={
                    'time_span': time_span,
                    'trajectory_length': len(trajectory),
                    'adaptive': adaptive
                }
            )
            
            # Analyze trajectory
            stability = state_manager.dynamics.analyze_stability(trajectory)
            
            # Compute trajectory metrics
            phi_values = [float(state_manager.compute_iit_phi(s)) for s in trajectory[::10]]
            fe_values = [float(state_manager.compute_free_energy(s)) for s in trajectory[::10]]
            
            return json.dumps({
                'success': True,
                'trajectory_length': len(trajectory),
                'time_span': time_span,
                'final_state': final_state.tolist(),
                'initial_state': initial_state.tolist(),
                'total_distance': float(np.sum([
                    state_manager.manifold.distance(trajectory[i], trajectory[i+1])
                    for i in range(len(trajectory)-1)
                ])),
                'stability_analysis': stability,
                'trajectory_metrics': {
                    'phi_mean': float(np.mean(phi_values)),
                    'phi_std': float(np.std(phi_values)),
                    'free_energy_mean': float(np.mean(fe_values)),
                    'free_energy_std': float(np.std(fe_values))
                },
                'times': times.tolist() if adaptive else None
            }, indent=2)
            
        finally:
            if noise_scale is not None:
                state_manager.dynamics.sigma = old_sigma
    
    @mcp.tool()
    async def compute_lyapunov_exponents(
        time_span: float = 10.0,
        n_exponents: Optional[int] = None,
        dt: float = 0.01
    ) -> str:
        """
        Compute Lyapunov exponents to analyze chaos/stability.
        
        Positive exponents indicate chaotic dynamics.
        
        Args:
            time_span: Time span for estimation
            n_exponents: Number of exponents to compute (default: all)
            dt: Time step for integration
        
        Returns:
            JSON with Lyapunov exponents and interpretation
        """
        current = await state_manager.get_current_state()
        initial_state = np.array(current['state'])
        
        try:
            exponents = state_manager.dynamics.compute_lyapunov_exponents(
                initial_state,
                time_span,
                n_exponents=n_exponents,
                dt=dt
            )
            
            # Interpret results
            max_exponent = float(np.max(exponents))
            is_chaotic = max_exponent > 0.01
            is_stable = max_exponent < -0.01
            
            return json.dumps({
                'success': True,
                'lyapunov_exponents': exponents.tolist(),
                'max_exponent': max_exponent,
                'interpretation': {
                    'is_chaotic': is_chaotic,
                    'is_stable': is_stable,
                    'is_neutral': not is_chaotic and not is_stable,
                    'predictability_horizon': 1.0 / max_exponent if max_exponent > 0 else float('inf')
                },
                'parameters': {
                    'time_span': time_span,
                    'n_exponents': len(exponents),
                    'dt': dt
                }
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            }, indent=2)
    
    @mcp.tool()
    async def stabilize(
        target_state: Optional[List[float]] = None,
        control_strength: float = 0.1,
        n_steps: int = 10
    ) -> str:
        """
        Apply stabilizing control to cognitive dynamics.
        
        Uses Control Lyapunov Function approach.
        
        Args:
            target_state: Target state to stabilize towards (default: origin)
            control_strength: Strength of control input (0-1)
            n_steps: Number of control steps to apply
        
        Returns:
            JSON with control sequence and stabilized trajectory
        """
        current = await state_manager.get_current_state()
        state = np.array(current['state'])
        
        if target_state is not None:
            target = np.array(target_state)
            if len(target) != state_manager.dimension:
                return json.dumps({
                    'success': False,
                    'error': f'Target state dimension {len(target)} does not match system dimension {state_manager.dimension}'
                }, indent=2)
        else:
            target = None
        
        # Apply control steps
        trajectory = [state.copy()]
        controls = []
        
        for _ in range(n_steps):
            # Compute control
            control = state_manager.stabilizer.compute_control(state, target)
            controls.append(control.tolist())
            
            # Apply control with specified strength
            state = state + control_strength * control
            trajectory.append(state.copy())
        
        # Update to final state
        await state_manager.update_state(
            state,
            source='stabilize',
            metadata={
                'n_steps': n_steps,
                'control_strength': control_strength,
                'has_target': target is not None
            }
        )
        
        # Analyze stabilization
        initial_distance = float(np.linalg.norm(trajectory[0] - (target if target is not None else np.zeros_like(state))))
        final_distance = float(np.linalg.norm(trajectory[-1] - (target if target is not None else np.zeros_like(state))))
        
        return json.dumps({
            'success': True,
            'n_steps': n_steps,
            'control_strength': control_strength,
            'final_state': state.tolist(),
            'trajectory_length': len(trajectory),
            'stabilization_metrics': {
                'initial_distance': initial_distance,
                'final_distance': final_distance,
                'improvement_ratio': (initial_distance - final_distance) / (initial_distance + 1e-10),
                'control_effort': float(np.sum([np.linalg.norm(c) for c in controls]))
            },
            'final_control_gain': float(state_manager.stabilizer.control_gain)
        }, indent=2)