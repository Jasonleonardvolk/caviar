"""
Knowledge and metacognitive resources
"""

import json
import numpy as np
from mcp.types import TextContent


def register_knowledge_resources(mcp, state_manager):
    """Register knowledge-related resources."""
    
    @mcp.resource("tori://tower/structure")
    async def get_tower_structure() -> TextContent:
        """Get complete metacognitive tower structure."""
        tower = state_manager.tower
        
        levels_info = []
        for i in range(len(tower.levels)):
            level_summary = tower.get_level_summary(i)
            
            # Add connectivity information
            blocks = tower._get_block_structure(i)
            level_summary['block_structure'] = {
                'n_blocks': len(blocks),
                'block_sizes': [b.shape[0] for b in blocks]
            }
            
            levels_info.append(level_summary)
        
        # Test functoriality
        current = await state_manager.get_current_state()
        state = np.array(current['state'])
        
        functoriality_tests = []
        for level in range(1, min(3, len(tower.levels))):
            test_passed = tower.validate_functoriality(
                state,
                max_level=level,
                tolerance=1e-6
            )
            functoriality_tests.append({
                'level': level,
                'passed': test_passed
            })
        
        return TextContent(
            type="text",
            text=json.dumps({
                'n_levels': len(tower.levels),
                'base_dimension': tower.base_dim,
                'max_dimension_ratio': tower.max_dim_ratio,
                'metric': tower.metric,
                'levels': levels_info,
                'functoriality_tests': functoriality_tests,
                'sparsity_threshold': tower.sparsity_threshold
            }, indent=2)
        )
    
    @mcp.resource("tori://sheaf/topology")
    async def get_sheaf_topology() -> TextContent:
        """Get knowledge sheaf topology and structure."""
        sheaf = state_manager.sheaf
        
        # Visualize structure
        viz = sheaf.visualize_structure()
        
        # Check consistency at current state
        current = await state_manager.get_current_state()
        state = np.array(current['state'])
        
        consistency_checks = {}
        for region in sheaf.topology:
            if region in sheaf.sections:
                try:
                    value = sheaf.get_section_value(region, state)
                    consistency_checks[region] = {
                        'has_value': True,
                        'value_type': type(value).__name__
                    }
                except:
                    consistency_checks[region] = {
                        'has_value': False,
                        'error': 'Failed to compute'
                    }
        
        # Check global consistency
        global_consistent = sheaf.check_consistency(state, list(sheaf.sections.keys()))
        
        return TextContent(
            type="text",
            text=json.dumps({
                'topology': sheaf.topology,
                'visualization': viz,
                'consistency_checks': consistency_checks,
                'global_consistency': global_consistent,
                'n_regions': len(sheaf.topology),
                'n_sections': len(sheaf.sections),
                'n_overlaps': len(sheaf.pairwise_overlaps),
                'n_triple_overlaps': len(sheaf.triple_overlaps)
            }, indent=2)
        )
    
    @mcp.resource("tori://cognitive/components")
    async def get_component_info() -> TextContent:
        """Get detailed information about all cognitive components."""
        components = {
            'manifold': {
                'class': type(state_manager.manifold).__name__,
                'dimension': state_manager.manifold.dimension,
                'metric': state_manager.manifold.metric
            },
            'reflective_operator': {
                'class': type(state_manager.reflective_op).__name__,
                'step_size': state_manager.reflective_op.step_size,
                'momentum': state_manager.reflective_op.momentum
            },
            'self_modification': {
                'class': type(state_manager.self_mod_op).__name__,
                'iit_weight': state_manager.self_mod_op.iit_weight,
                'max_iter': state_manager.self_mod_op.max_iter,
                'convergence_tol': state_manager.self_mod_op.convergence_tol
            },
            'curiosity': {
                'class': type(state_manager.curiosity).__name__,
                'decay_const': state_manager.curiosity.decay,
                'exploration_bonus': state_manager.curiosity.exploration_bonus,
                'memory_capacity': state_manager.curiosity.memory_capacity,
                'current_memory_size': len(state_manager.curiosity.memory_buffer)
            },
            'dynamics': {
                'class': type(state_manager.dynamics).__name__,
                'noise_sigma': state_manager.dynamics.sigma,
                'dt': state_manager.dynamics.dt
            },
            'consciousness_monitor': {
                'class': type(state_manager.consciousness_monitor).__name__,
                'phi_threshold': state_manager.consciousness_monitor.threshold,
                'history_size': state_manager.consciousness_monitor.history_size
            },
            'tower': {
                'class': type(state_manager.tower).__name__,
                'n_levels': len(state_manager.tower.levels),
                'base_dim': state_manager.tower.base_dim
            },
            'sheaf': {
                'class': type(state_manager.sheaf).__name__,
                'n_regions': len(state_manager.sheaf.topology),
                'n_sections': len(state_manager.sheaf.sections)
            }
        }
        
        # Add transfer morphism if available
        if state_manager.transfer is not None:
            components['transfer_morphism'] = {
                'class': type(state_manager.transfer).__name__,
                'homology_dim': state_manager.transfer.dim,
                'max_edge': state_manager.transfer.max_edge
            }
        
        return TextContent(
            type="text",
            text=json.dumps(components, indent=2)
        )
    
    @mcp.resource("tori://config/current")
    async def get_current_config() -> TextContent:
        """Get current server configuration."""
        config = state_manager.config
        
        return TextContent(
            type="text",
            text=json.dumps({
                'server': {
                    'name': config.server_name,
                    'version': config.server_version,
                    'transport': config.transport_type
                },
                'cognitive': {
                    'dimension': config.cognitive_dimension,
                    'manifold_metric': config.manifold_metric,
                    'consciousness_threshold': config.consciousness_threshold,
                    'max_metacognitive_levels': config.max_metacognitive_levels
                },
                'paths': {
                    'tori_base': str(config.tori_base_path),
                    'state_persistence': str(config.state_persistence_path)
                },
                'logging': {
                    'level': config.log_level,
                    'file': str(config.log_file) if config.log_file else None
                }
            }, indent=2)
        )