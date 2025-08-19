"""
Metacognitive tower and knowledge sheaf tools
"""

import json
import numpy as np
from typing import Optional, List, Dict, Any


def register_metacognitive_tools(mcp, state_manager):
    """Register metacognitive tools."""
    
    @mcp.tool()
    async def lift_to_metacognitive_level(
        target_level: int,
        validate: bool = True
    ) -> str:
        """
        Lift current state to higher metacognitive level.
        
        Higher levels represent increasingly abstract metacognitive representations.
        
        Args:
            target_level: Target level (1, 2, or 3)
            validate: Whether to validate functorial properties
        
        Returns:
            JSON with lifted state and level information
        """
        current = await state_manager.get_current_state()
        state = np.array(current['state'])
        
        tower = state_manager.tower
        
        if target_level < 0 or target_level >= len(tower.levels):
            return json.dumps({
                'success': False,
                'error': f'Invalid target level: {target_level}. Must be between 0 and {len(tower.levels)-1}'
            }, indent=2)
        
        try:
            # Lift state
            lifted = tower.lift(state, 0, target_level)
            
            # Get level information
            level_info = tower.get_level_summary(target_level)
            
            # Validate if requested
            validation_passed = True
            if validate:
                validation_passed = tower.validate_functoriality(
                    state,
                    max_level=target_level,
                    tolerance=1e-6
                )
            
            # Extract features
            features = {}
            for feat_name, (start, end) in level_info['features'].items():
                features[feat_name] = {
                    'values': lifted[start:end].tolist(),
                    'magnitude': float(np.linalg.norm(lifted[start:end]))
                }
            
            return json.dumps({
                'success': True,
                'target_level': target_level,
                'lifted_dimension': len(lifted),
                'level_info': level_info,
                'features': features,
                'validation_passed': validation_passed,
                'lifted_state_sample': lifted[:10].tolist()  # First 10 components
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            }, indent=2)
    
    @mcp.tool()
    async def project_from_metacognitive_level(
        from_level: int,
        lifted_state: List[float]
    ) -> str:
        """
        Project state from higher metacognitive level back to base.
        
        Args:
            from_level: Source metacognitive level
            lifted_state: State at the higher level
        
        Returns:
            JSON with projected state
        """
        tower = state_manager.tower
        
        if from_level <= 0 or from_level >= len(tower.levels):
            return json.dumps({
                'success': False,
                'error': f'Invalid source level: {from_level}'
            }, indent=2)
        
        lifted = np.array(lifted_state)
        expected_dim = tower.levels[from_level]['dim']
        
        if len(lifted) != expected_dim:
            return json.dumps({
                'success': False,
                'error': f'State dimension {len(lifted)} does not match level {from_level} dimension {expected_dim}'
            }, indent=2)
        
        try:
            # Project back to base
            projected = tower.project(lifted, from_level, 0)
            
            # Update state
            await state_manager.update_state(
                projected,
                source='metacognitive_projection',
                metadata={'from_level': from_level}
            )
            
            new_info = await state_manager.get_current_state()
            
            return json.dumps({
                'success': True,
                'from_level': from_level,
                'projected_state': projected.tolist(),
                'phi': new_info['phi'],
                'free_energy': new_info['free_energy']
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            }, indent=2)
    
    @mcp.tool()
    async def compute_holonomy(
        level: int = 0,
        loop_type: str = "circle",
        n_points: int = 20
    ) -> str:
        """
        Compute holonomy around a loop to measure curvature.
        
        Args:
            level: Metacognitive level
            loop_type: Type of loop - 'circle', 'square', 'random'
            n_points: Number of points in loop
        
        Returns:
            JSON with holonomy matrix and curvature measure
        """
        tower = state_manager.tower
        dim = tower.levels[level]['dim']
        
        # Generate loop
        if loop_type == "circle":
            t = np.linspace(0, 2*np.pi, n_points + 1)
            center = np.zeros(dim)
            radius = 0.5
            loop = [center + radius * np.array([np.cos(ti) if i == 0 else 
                                               np.sin(ti) if i == 1 else 
                                               0.1 * np.cos(2*ti) 
                                               for i in range(dim)]) 
                   for ti in t]
        
        elif loop_type == "square":
            # Square loop in first two dimensions
            side = 0.5
            corners = [
                np.array([side, side] + [0]*(dim-2)),
                np.array([-side, side] + [0]*(dim-2)),
                np.array([-side, -side] + [0]*(dim-2)),
                np.array([side, -side] + [0]*(dim-2)),
                np.array([side, side] + [0]*(dim-2))
            ]
            
            loop = []
            for i in range(len(corners)-1):
                for j in range(n_points//4):
                    alpha = j / (n_points//4)
                    loop.append((1-alpha) * corners[i] + alpha * corners[i+1])
        
        else:  # random
            # Random smooth loop
            key_points = [np.random.randn(dim) * 0.3 for _ in range(5)]
            key_points.append(key_points[0])  # Close loop
            
            loop = []
            for i in range(len(key_points)-1):
                for j in range(n_points//5):
                    alpha = j / (n_points//5)
                    loop.append((1-alpha) * key_points[i] + alpha * key_points[i+1])
        
        # Compute holonomy
        holonomy = tower.holonomy(loop, level)
        
        # Analyze holonomy
        eigenvalues = np.linalg.eigvals(holonomy)
        trace = np.trace(holonomy)
        det = np.linalg.det(holonomy)
        
        # Measure deviation from identity (curvature indicator)
        deviation = np.linalg.norm(holonomy - np.eye(dim), 'fro')
        
        return json.dumps({
            'success': True,
            'level': level,
            'loop_type': loop_type,
            'n_points': n_points,
            'holonomy_trace': float(np.real(trace)),
            'holonomy_determinant': float(np.real(det)),
            'deviation_from_identity': float(deviation),
            'curvature_present': deviation > 0.01,
            'eigenvalue_magnitudes': [float(np.abs(ev)) for ev in eigenvalues[:5]],  # First 5
            'is_unitary': bool(np.allclose(holonomy @ holonomy.T, np.eye(dim), atol=1e-6))
        }, indent=2)
    
    @mcp.tool()
    async def query_knowledge_sheaf(
        query_type: str = "global",
        region: Optional[str] = None
    ) -> str:
        """
        Query the knowledge sheaf for distributed knowledge.
        
        Args:
            query_type: Type of query - 'global', 'local', 'cohomology'
            region: Specific region for local queries
        
        Returns:
            JSON with knowledge information
        """
        sheaf = state_manager.sheaf
        current = await state_manager.get_current_state()
        state = np.array(current['state'])
        
        if query_type == "global":
            # Try to glue sections
            try:
                global_knowledge = sheaf.glue_sections(state, method='average')
                
                return json.dumps({
                    'success': True,
                    'query_type': 'global',
                    'has_global_section': global_knowledge is not None,
                    'global_knowledge': str(global_knowledge) if global_knowledge else None,
                    'regions': list(sheaf.topology.keys()),
                    'active_sections': list(sheaf.sections.keys())
                }, indent=2)
                
            except Exception as e:
                return json.dumps({
                    'success': False,
                    'query_type': 'global',
                    'error': str(e)
                }, indent=2)
        
        elif query_type == "local":
            if region is None:
                return json.dumps({
                    'success': False,
                    'error': 'Region must be specified for local queries'
                }, indent=2)
            
            if region not in sheaf.sections:
                return json.dumps({
                    'success': False,
                    'error': f'No section defined for region: {region}'
                }, indent=2)
            
            try:
                local_knowledge = sheaf.get_section_value(region, state)
                
                return json.dumps({
                    'success': True,
                    'query_type': 'local',
                    'region': region,
                    'knowledge': local_knowledge,
                    'neighbors': sheaf.topology.get(region, [])
                }, indent=2)
                
            except Exception as e:
                return json.dumps({
                    'success': False,
                    'query_type': 'local',
                    'region': region,
                    'error': str(e)
                }, indent=2)
        
        elif query_type == "cohomology":
            # Compute sheaf cohomology
            h0 = sheaf.compute_cohomology(degree=0)
            h1 = sheaf.compute_cohomology(degree=1)
            
            return json.dumps({
                'success': True,
                'query_type': 'cohomology',
                'H0': h0,
                'H1': h1,
                'interpretation': {
                    'global_consistency': h0['consistency_rate'] if 'consistency_rate' in h0 else 0,
                    'obstruction_level': h1['violation_rate'] if 'violation_rate' in h1 else 0,
                    'can_glue_globally': h1.get('dimension', 0) == 0
                }
            }, indent=2)
        
        else:
            return json.dumps({
                'success': False,
                'error': f'Unknown query type: {query_type}'
            }, indent=2)
    
    @mcp.tool()
    async def compute_curiosity(
        target_state: Optional[List[float]] = None,
        include_environment: bool = True
    ) -> str:
        """
        Compute curiosity score for exploration.
        
        Args:
            target_state: Target state to evaluate curiosity towards
            include_environment: Use advanced curiosity with environment
        
        Returns:
            JSON with curiosity metrics
        """
        current = await state_manager.get_current_state()
        state = np.array(current['state'])
        
        curiosity_func = state_manager.curiosity
        
        if include_environment:
            # Generate mock environment
            environment = np.random.randn(state_manager.dimension)
            
            # Compute advanced curiosity
            curiosity_score = curiosity_func.compute_advanced(
                state,
                environment,
                memory_buffer=None  # Use internal buffer
            )
            
            # Get exploration heatmap if 2D
            heatmap_data = None
            if state_manager.dimension == 2:
                try:
                    heatmap = curiosity_func.get_exploration_heatmap(
                        bounds=(-2, 2),
                        resolution=20
                    )
                    heatmap_data = {
                        'shape': heatmap.shape,
                        'max_visits': int(np.max(heatmap)),
                        'min_visits': int(np.min(heatmap)),
                        'total_visits': int(np.sum(heatmap))
                    }
                except:
                    heatmap_data = None
            
            return json.dumps({
                'success': True,
                'curiosity_score': float(curiosity_score),
                'curiosity_type': 'advanced',
                'memory_buffer_size': len(curiosity_func.memory_buffer),
                'total_visits': len(curiosity_func.visit_counts),
                'exploration_heatmap': heatmap_data
            }, indent=2)
        
        else:
            # Simple curiosity between states
            if target_state is None:
                target = np.random.randn(state_manager.dimension) * 0.5
            else:
                target = np.array(target_state)
            
            curiosity_score = curiosity_func.compute(state, target)
            
            return json.dumps({
                'success': True,
                'curiosity_score': float(curiosity_score),
                'curiosity_type': 'simple',
                'current_state_norm': float(np.linalg.norm(state)),
                'target_state_norm': float(np.linalg.norm(target)),
                'state_distance': float(state_manager.manifold.distance(state, target))
            }, indent=2)