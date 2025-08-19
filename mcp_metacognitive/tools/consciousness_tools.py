"""
Consciousness monitoring and IIT tools
"""

import json
import numpy as np
from typing import Optional, List, Dict
from cog import compute_iit_phi, create_connectivity_matrix


def register_consciousness_tools(mcp, state_manager):
    """Register consciousness-related tools."""
    
    @mcp.tool()
    async def measure_consciousness(
        state: Optional[List[float]] = None,
        connectivity_type: str = "full"
    ) -> str:
        """
        Measure integrated information (Φ) for consciousness assessment.
        
        Based on Integrated Information Theory (IIT).
        
        Args:
            state: State to measure (default: current state)
            connectivity_type: Type of connectivity - 'full', 'sparse', 'small_world', 'modular'
        
        Returns:
            JSON with Φ value and consciousness metrics
        """
        if state is None:
            current = await state_manager.get_current_state()
            state_array = np.array(current['state'])
        else:
            state_array = np.array(state)
        
        # Create connectivity matrix
        connectivity = create_connectivity_matrix(
            len(state_array),
            connection_type=connectivity_type
        )
        
        # Compute IIT
        phi = compute_iit_phi(state_array, connectivity)
        
        # Get consciousness monitor statistics
        stats = state_manager.consciousness_monitor.get_statistics()
        trend = state_manager.consciousness_monitor.get_trend()
        
        # Interpret consciousness level
        threshold = state_manager.config.consciousness_threshold
        if phi < threshold * 0.5:
            level = "minimal"
        elif phi < threshold:
            level = "low"
        elif phi < threshold * 1.5:
            level = "moderate"
        elif phi < threshold * 2:
            level = "high"
        else:
            level = "very_high"
        
        return json.dumps({
            'phi': float(phi),
            'consciousness_level': level,
            'threshold': float(threshold),
            'above_threshold': phi >= threshold,
            'relative_to_threshold': float(phi / threshold),
            'connectivity_type': connectivity_type,
            'monitor_statistics': stats,
            'trend': float(trend),
            'trend_interpretation': 'increasing' if trend > 0.01 else 'decreasing' if trend < -0.01 else 'stable'
        }, indent=2)
    
    @mcp.tool()
    async def consciousness_intervention(
        intervention_type: str = "auto"
    ) -> str:
        """
        Apply intervention to preserve or enhance consciousness.
        
        Args:
            intervention_type: Type of intervention - 'auto', 'increase_iit', 'stabilize', 'emergency'
        
        Returns:
            JSON with intervention results
        """
        current = await state_manager.get_current_state()
        state = np.array(current['state'])
        
        # Get suggested intervention if auto
        if intervention_type == "auto":
            suggestion = state_manager.consciousness_monitor.suggest_intervention()
            if suggestion is None:
                return json.dumps({
                    'success': True,
                    'intervention_type': 'none',
                    'message': 'No intervention needed',
                    'current_phi': current['phi']
                }, indent=2)
            intervention_type = suggestion['action']
        
        # Apply intervention based on type
        if intervention_type == "increase_iit_weight":
            # Temporarily increase IIT weight in self-modification
            old_weight = state_manager.self_mod_op.iit_weight
            state_manager.self_mod_op.iit_weight = 2.0
            
            try:
                new_state = state_manager.self_mod_op.optimize(state, lambda_d=0.5)
                await state_manager.update_state(
                    new_state,
                    source='consciousness_intervention',
                    metadata={'type': 'increase_iit_weight'}
                )
                
                new_info = await state_manager.get_current_state()
                
                return json.dumps({
                    'success': True,
                    'intervention_type': 'increase_iit_weight',
                    'phi_before': current['phi'],
                    'phi_after': new_info['phi'],
                    'improvement': float(new_info['phi'] - current['phi'])
                }, indent=2)
                
            finally:
                state_manager.self_mod_op.iit_weight = old_weight
        
        elif intervention_type == "stabilize":
            # Use Lyapunov stabilization
            control = state_manager.stabilizer.compute_control(state)
            new_state = state + 0.1 * control
            
            await state_manager.update_state(
                new_state,
                source='consciousness_intervention',
                metadata={'type': 'stabilize'}
            )
            
            new_info = await state_manager.get_current_state()
            
            return json.dumps({
                'success': True,
                'intervention_type': 'stabilize',
                'phi_before': current['phi'],
                'phi_after': new_info['phi'],
                'control_magnitude': float(np.linalg.norm(control))
            }, indent=2)
        
        elif intervention_type == "emergency":
            # Emergency reset to high-consciousness state
            # Create state with high integration
            dim = state_manager.dimension
            new_state = np.ones(dim) * 0.5
            new_state += np.random.randn(dim) * 0.1
            
            await state_manager.update_state(
                new_state,
                source='consciousness_intervention',
                metadata={'type': 'emergency'}
            )
            
            new_info = await state_manager.get_current_state()
            
            return json.dumps({
                'success': True,
                'intervention_type': 'emergency',
                'message': 'Emergency consciousness restoration',
                'phi_before': current['phi'],
                'phi_after': new_info['phi']
            }, indent=2)
        
        else:
            return json.dumps({
                'success': False,
                'error': f'Unknown intervention type: {intervention_type}'
            }, indent=2)
    
    @mcp.tool()
    async def analyze_consciousness_history(
        window_size: int = 50,
        compute_spectrum: bool = False
    ) -> str:
        """
        Analyze consciousness patterns over time.
        
        Args:
            window_size: Number of recent states to analyze
            compute_spectrum: Whether to compute frequency spectrum
        
        Returns:
            JSON with consciousness analysis
        """
        # Get recent history
        history = await state_manager.get_recent_history(window_size)
        
        if len(history) < 2:
            return json.dumps({
                'success': False,
                'error': 'Insufficient history for analysis'
            }, indent=2)
        
        # Extract phi values
        phi_values = [h['phi'] for h in history]
        
        # Basic statistics
        analysis = {
            'window_size': len(phi_values),
            'current_phi': phi_values[-1],
            'mean_phi': float(np.mean(phi_values)),
            'std_phi': float(np.std(phi_values)),
            'min_phi': float(np.min(phi_values)),
            'max_phi': float(np.max(phi_values)),
            'range_phi': float(np.max(phi_values) - np.min(phi_values))
        }
        
        # Compute trend
        if len(phi_values) > 5:
            x = np.arange(len(phi_values))
            trend_slope = float(np.polyfit(x, phi_values, 1)[0])
            analysis['trend_slope'] = trend_slope
            analysis['trend_direction'] = 'increasing' if trend_slope > 0 else 'decreasing'
        
        # Stability analysis
        if len(phi_values) > 10:
            recent = phi_values[-10:]
            analysis['recent_stability'] = float(np.std(recent))
            analysis['is_stable'] = analysis['recent_stability'] < 0.1
        
        # Frequency spectrum if requested
        if compute_spectrum and len(phi_values) > 20:
            # Remove mean and apply window
            phi_centered = np.array(phi_values) - np.mean(phi_values)
            phi_windowed = phi_centered * np.hanning(len(phi_centered))
            
            # Compute FFT
            fft = np.fft.fft(phi_windowed)
            freqs = np.fft.fftfreq(len(phi_windowed))
            
            # Get dominant frequencies
            power = np.abs(fft)**2
            dominant_idx = np.argsort(power)[-5:]  # Top 5 frequencies
            
            analysis['frequency_spectrum'] = {
                'dominant_frequencies': freqs[dominant_idx].tolist(),
                'dominant_powers': power[dominant_idx].tolist(),
                'has_oscillation': float(np.max(power[1:])) > 0.1 * power[0]
            }
        
        # Violation analysis
        threshold = state_manager.config.consciousness_threshold
        violations = [i for i, phi in enumerate(phi_values) if phi < threshold]
        analysis['violations'] = {
            'count': len(violations),
            'rate': len(violations) / len(phi_values),
            'indices': violations[-5:] if violations else []  # Last 5 violations
        }
        
        return json.dumps({
            'success': True,
            'analysis': analysis
        }, indent=2)