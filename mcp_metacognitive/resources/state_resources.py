"""
Cognitive state resources
"""

import json
import numpy as np
from mcp.types import TextContent


def register_state_resources(mcp, state_manager):
    """Register state-related resources."""
    
    @mcp.resource("tori://state/current")
    async def get_current_state() -> TextContent:
        """Get the current cognitive state with full metadata."""
        state_info = await state_manager.get_current_state()
        
        # Add additional computed metrics
        state_array = np.array(state_info['state'])
        state_info['additional_metrics'] = {
            'l2_norm': float(np.linalg.norm(state_array)),
            'l1_norm': float(np.sum(np.abs(state_array))),
            'max_component': float(np.max(np.abs(state_array))),
            'sparsity': float(np.sum(np.abs(state_array) < 0.01) / len(state_array)),
            'entropy': float(-np.sum(
                np.abs(state_array) / np.sum(np.abs(state_array)) * 
                np.log(np.abs(state_array) / np.sum(np.abs(state_array)) + 1e-10)
            ))
        }
        
        return TextContent(
            type="text",
            text=json.dumps(state_info, indent=2)
        )
    
    @mcp.resource("tori://state/history/{n}")
    async def get_state_history(n: str) -> TextContent:
        """
        Get recent cognitive state history.
        
        Args:
            n: Number of recent states to retrieve (max 100)
        """
        n_states = min(int(n), 100)
        history = await state_manager.get_recent_history(n_states)
        
        # Compute summary statistics
        if history:
            phi_values = [h['phi'] for h in history]
            fe_values = [h['free_energy'] for h in history]
            
            summary = {
                'count': len(history),
                'time_span': (history[-1]['timestamp'], history[0]['timestamp']),
                'phi_stats': {
                    'mean': float(np.mean(phi_values)),
                    'std': float(np.std(phi_values)),
                    'min': float(np.min(phi_values)),
                    'max': float(np.max(phi_values))
                },
                'free_energy_stats': {
                    'mean': float(np.mean(fe_values)),
                    'std': float(np.std(fe_values)),
                    'min': float(np.min(fe_values)),
                    'max': float(np.max(fe_values))
                }
            }
        else:
            summary = {'count': 0}
        
        return TextContent(
            type="text",
            text=json.dumps({
                'summary': summary,
                'history': history
            }, indent=2)
        )
    
    @mcp.resource("tori://state/trajectory/{n}")
    async def get_trajectory(n: str) -> TextContent:
        """
        Get recent trajectory as coordinate data.
        
        Args:
            n: Number of recent states (max 100)
        """
        n_states = min(int(n), 100)
        trajectory = await state_manager.get_trajectory(n_states)
        
        # Compute trajectory metrics
        if len(trajectory) > 1:
            # Path length
            path_length = 0.0
            for i in range(len(trajectory) - 1):
                path_length += state_manager.manifold.distance(
                    trajectory[i], trajectory[i+1]
                )
            
            # Curvature estimate
            if len(trajectory) > 2:
                curvatures = []
                for i in range(1, len(trajectory) - 1):
                    v1 = trajectory[i] - trajectory[i-1]
                    v2 = trajectory[i+1] - trajectory[i]
                    
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    
                    if norm1 > 1e-10 and norm2 > 1e-10:
                        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        curvatures.append(np.arccos(cos_angle))
                
                avg_curvature = float(np.mean(curvatures)) if curvatures else 0.0
            else:
                avg_curvature = 0.0
            
            metrics = {
                'length': len(trajectory),
                'path_length': float(path_length),
                'average_step_size': float(path_length / (len(trajectory) - 1)),
                'average_curvature': avg_curvature,
                'start_point': trajectory[0].tolist(),
                'end_point': trajectory[-1].tolist(),
                'displacement': float(state_manager.manifold.distance(
                    trajectory[0], trajectory[-1]
                ))
            }
        else:
            metrics = {'length': len(trajectory)}
        
        return TextContent(
            type="text",
            text=json.dumps({
                'metrics': metrics,
                'trajectory': trajectory.tolist()
            }, indent=2)
        )
    
    @mcp.resource("tori://state/manifold")
    async def get_manifold_info() -> TextContent:
        """Get information about the cognitive manifold."""
        manifold = state_manager.manifold
        current = await state_manager.get_current_state()
        state = np.array(current['state'])
        
        # Compute local geometry
        if manifold.metric == "fisher_rao":
            fim = manifold.fisher_information_matrix(state)
            eigenvalues = np.linalg.eigvalsh(fim)
            
            geometry = {
                'metric_tensor_eigenvalues': eigenvalues.tolist(),
                'condition_number': float(np.max(eigenvalues) / np.min(eigenvalues)),
                'is_well_conditioned': float(np.max(eigenvalues) / np.min(eigenvalues)) < 100,
                'local_volume_element': float(np.sqrt(np.linalg.det(fim)))
            }
        else:
            geometry = {
                'metric_type': 'euclidean',
                'is_flat': True
            }
        
        return TextContent(
            type="text",
            text=json.dumps({
                'dimension': manifold.dimension,
                'metric': manifold.metric,
                'local_geometry': geometry,
                'tangent_space_dim': manifold.tangent_space_dim(state)
            }, indent=2)
        )