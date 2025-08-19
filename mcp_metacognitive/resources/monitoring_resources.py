"""
Monitoring and analysis resources
"""

import json
import numpy as np
from mcp.types import TextContent


def register_monitoring_resources(mcp, state_manager):
    """Register monitoring-related resources."""
    
    @mcp.resource("tori://consciousness/monitor")
    async def get_consciousness_monitor() -> TextContent:
        """Get comprehensive consciousness monitoring data."""
        monitor = state_manager.consciousness_monitor
        
        # Get all statistics
        stats = monitor.get_statistics()
        trend = monitor.get_trend()
        suggestion = monitor.suggest_intervention()
        
        # Analyze alert history
        alert_summary = {
            'total_alerts': len(monitor.alert_history),
            'recent_alerts': len([a for a in monitor.alert_history[-10:]])
        }
        
        if monitor.alert_history:
            recent_alert = monitor.alert_history[-1]
            alert_summary['last_alert'] = {
                'phi_drop': float(recent_alert['phi_before'] - recent_alert['phi_after']),
                'violation_number': recent_alert['violation_number']
            }
        
        return TextContent(
            type="text",
            text=json.dumps({
                'statistics': stats,
                'trend': {
                    'value': float(trend),
                    'interpretation': 'increasing' if trend > 0.01 else 'decreasing' if trend < -0.01 else 'stable'
                },
                'suggested_intervention': suggestion,
                'alert_summary': alert_summary,
                'thresholds': {
                    'phi_threshold': float(monitor.threshold),
                    'critical_level': float(monitor.threshold * 0.8),
                    'warning_level': float(monitor.threshold * 1.2)
                }
            }, indent=2)
        )
    
    @mcp.resource("tori://stability/analysis")
    async def get_stability_analysis() -> TextContent:
        """Get Lyapunov stability analysis."""
        stabilizer = state_manager.stabilizer
        
        # Get recent trajectory for analysis
        trajectory = await state_manager.get_trajectory(50)
        
        if len(trajectory) < 2:
            return TextContent(
                type="text",
                text=json.dumps({
                    'error': 'Insufficient trajectory data for stability analysis'
                }, indent=2)
            )
        
        # Compute Lyapunov function values
        lyapunov_values = stabilizer.compute_lyapunov(trajectory)
        
        # Check stability
        stability_check = stabilizer.check_stability(trajectory)
        
        # Compute phase portrait data if 2D
        phase_portrait = None
        if state_manager.dimension == 2:
            portrait_data = stabilizer.get_phase_portrait(bounds=(-2, 2), resolution=10)
            if portrait_data:
                phase_portrait = {
                    'has_data': True,
                    'grid_shape': portrait_data['X'].shape,
                    'vector_field_magnitude': float(np.mean(
                        np.sqrt(portrait_data['U']**2 + portrait_data['V']**2)
                    ))
                }
        
        return TextContent(
            type="text",
            text=json.dumps({
                'stability_check': stability_check,
                'lyapunov_analysis': {
                    'current_value': float(lyapunov_values[-1]),
                    'initial_value': float(lyapunov_values[0]),
                    'mean_value': float(np.mean(lyapunov_values)),
                    'is_decreasing': float(np.mean(np.diff(lyapunov_values))) < 0,
                    'max_increase': float(np.max(np.diff(lyapunov_values))) if len(lyapunov_values) > 1 else 0
                },
                'control_parameters': {
                    'current_gain': float(stabilizer.control_gain),
                    'adaptive_gain_enabled': stabilizer.adaptive_gain,
                    'stability_margin': float(stabilizer.margin),
                    'intervention_threshold': float(stabilizer.intervention_threshold)
                },
                'phase_portrait': phase_portrait
            }, indent=2)
        )
    
    @mcp.resource("tori://events/recent/{n}")
    async def get_recent_events(n: str) -> TextContent:
        """
        Get recent system events.
        
        Args:
            n: Number of recent events (max 50)
        """
        n_events = min(int(n), 50)
        events = list(state_manager.event_history)[-n_events:]
        
        # Summarize event types
        event_types = {}
        for event in events:
            event_type = event['type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return TextContent(
            type="text",
            text=json.dumps({
                'count': len(events),
                'event_types': event_types,
                'events': events
            }, indent=2)
        )
    
    @mcp.resource("tori://performance/metrics")
    async def get_performance_metrics() -> TextContent:
        """Get overall system performance metrics."""
        current = await state_manager.get_current_state()
        
        # Session metrics
        session_duration = current['session_duration']
        total_updates = len(state_manager.state_history)
        
        # Consciousness metrics
        cons_stats = state_manager.consciousness_monitor.get_statistics()
        
        # Memory usage
        memory_stats = {
            'state_history_size': len(state_manager.state_history),
            'event_history_size': len(state_manager.event_history),
            'curiosity_memory_size': len(state_manager.curiosity.memory_buffer),
            'visit_count_size': len(state_manager.curiosity.visit_counts)
        }
        
        # Component health
        component_health = {
            'manifold': 'healthy',
            'reflective_operator': 'healthy',
            'self_modification': 'healthy',
            'dynamics': 'healthy',
            'consciousness_monitor': 'warning' if cons_stats.get('violation_rate', 0) > 0.1 else 'healthy',
            'tower': 'healthy',
            'sheaf': 'healthy'
        }
        
        return TextContent(
            type="text",
            text=json.dumps({
                'session': {
                    'duration_seconds': session_duration,
                    'total_state_updates': total_updates,
                    'updates_per_minute': total_updates / (session_duration / 60) if session_duration > 0 else 0
                },
                'consciousness': {
                    'current_phi': current['phi'],
                    'violation_rate': cons_stats.get('violation_rate', 0),
                    'average_phi': cons_stats.get('average_phi', 0)
                },
                'memory': memory_stats,
                'component_health': component_health,
                'overall_status': 'healthy' if all(h == 'healthy' for h in component_health.values()) else 'warning'
            }, indent=2)
        )