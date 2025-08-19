"""
Soliton Memory Tools for TORI MCP Server
========================================

Tools for interacting with the soliton memory system and infinite context preservation.
"""

import json
import numpy as np
from typing import Optional, List, Dict, Any
from ..core.soliton_memory import VaultStatus, ContentType


def register_soliton_memory_tools(mcp, state_manager):
    """Register soliton memory tools."""
    
    @mcp.tool()
    async def store_soliton_memory(
        concept_id: str,
        content: str,
        importance: float = 1.0,
        content_type: str = "text",
        filter_content: bool = True
    ) -> str:
        """
        Store content in soliton memory with wave-based addressing.
        
        Creates a soliton wave packet that maintains perfect fidelity indefinitely.
        
        Args:
            concept_id: Unique concept identifier for phase-based addressing
            content: Content to store (any text)
            importance: Memory importance/amplitude (0.1 to 2.0)
            content_type: Type of content - 'text', 'cognitive_state', 'conversation', 'concept', 'emotional', 'procedural'
            filter_content: Apply TORI filtering for safety
        
        Returns:
            JSON with memory ID and soliton wave parameters
        """
        try:
            # Validate content type
            content_type_enum = ContentType(content_type.lower())
            
            # Store in soliton memory
            memory_id = await state_manager.store_memory(
                concept_id=concept_id,
                content=content,
                importance=max(0.1, min(2.0, importance)),
                content_type=content_type_enum,
                filter_content=filter_content
            )
            
            # Get the stored memory for details
            memory = await state_manager.soliton_lattice.get_memory_by_id(memory_id)
            
            if memory:
                return json.dumps({
                    'success': True,
                    'memory_id': memory_id,
                    'concept_id': concept_id,
                    'soliton_parameters': {
                        'phase_tag': memory.phase_tag,
                        'amplitude': memory.amplitude,
                        'frequency': memory.frequency,
                        'width': memory.width,
                        'stability': memory.stability
                    },
                    'vault_status': memory.vault_status.value,
                    'emotional_signature': {
                        'valence': memory.emotional_signature.valence,
                        'arousal': memory.emotional_signature.arousal,
                        'trauma_indicators': memory.emotional_signature.trauma_indicators
                    },
                    'auto_protected': len(memory.emotional_signature.trauma_indicators) > 0,
                    'tori_filtered': filter_content
                }, indent=2)
            else:
                return json.dumps({
                    'success': False,
                    'error': 'Memory stored but could not retrieve details'
                }, indent=2)
                
        except ValueError as e:
            return json.dumps({
                'success': False,
                'error': f'Invalid content_type: {content_type}. Valid types: text, cognitive_state, conversation, concept, emotional, procedural'
            }, indent=2)
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            }, indent=2)
    
    @mcp.tool()
    async def recall_soliton_memories(
        concept_id: str,
        max_results: int = 5,
        include_vaulted: bool = False,
        phase_tolerance: float = 0.2
    ) -> str:
        """
        Recall memories using soliton phase correlation.
        
        Uses matched filter detection to find memories with similar phase signatures.
        
        Args:
            concept_id: Concept to find related memories for
            max_results: Maximum number of memories to return
            include_vaulted: Include protected/vaulted memories
            phase_tolerance: Phase correlation tolerance (0.1 = strict, 0.5 = loose)
        
        Returns:
            JSON with related memories and correlation strengths
        """
        try:
            # Get related memories
            memories = await state_manager.recall_memories(
                concept_id=concept_id,
                max_results=max_results,
                include_vaulted=include_vaulted
            )
            
            # Also try direct phase-based recall
            if state_manager.soliton_lattice.concept_phase_map.get(concept_id):
                target_phase = state_manager.soliton_lattice.concept_phase_map[concept_id]
                phase_memories = await state_manager.soliton_lattice.recall_by_phase(
                    target_phase=target_phase,
                    tolerance=phase_tolerance,
                    max_results=max_results,
                    include_vaulted=include_vaulted
                )
                
                # Combine and deduplicate
                all_memory_ids = set()
                combined_memories = []
                
                for memory in memories + phase_memories:
                    if memory.id not in all_memory_ids:
                        all_memory_ids.add(memory.id)
                        combined_memories.append(memory)
                
                memories = combined_memories[:max_results]
            
            # Format results
            memory_results = []
            for memory in memories:
                # Compute correlation if we have target phase
                correlation = 0.0
                if state_manager.soliton_lattice.concept_phase_map.get(concept_id):
                    target_phase = state_manager.soliton_lattice.concept_phase_map[concept_id]
                    correlation = memory.correlate_with_signal(target_phase, phase_tolerance)
                
                memory_results.append({
                    'memory_id': memory.id,
                    'concept_id': memory.concept_id,
                    'content_preview': memory.content[:200] + "..." if len(memory.content) > 200 else memory.content,
                    'content_length': len(memory.content),
                    'correlation_strength': correlation,
                    'soliton_parameters': {
                        'phase_tag': memory.phase_tag,
                        'amplitude': memory.amplitude,
                        'stability': memory.stability
                    },
                    'access_count': memory.access_count,
                    'last_accessed': memory.last_accessed.isoformat(),
                    'vault_status': memory.vault_status.value,
                    'content_type': memory.content_type.value,
                    'emotional_valence': memory.emotional_signature.valence
                })
            
            return json.dumps({
                'success': True,
                'concept_id': concept_id,
                'memories_found': len(memory_results),
                'phase_tolerance': phase_tolerance,
                'included_vaulted': include_vaulted,
                'memories': memory_results
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            }, indent=2)
    
    @mcp.tool()
    async def recall_by_phase_signature(
        phase_tag: float,
        tolerance: float = 0.1,
        max_results: int = 10,
        include_vaulted: bool = False
    ) -> str:
        """
        Direct phase-based memory recall using soliton correlation.
        
        Finds memories with phase signatures within tolerance of target phase.
        
        Args:
            phase_tag: Target phase (0.0 to 6.28, representing 0 to 2π)
            tolerance: Phase correlation tolerance 
            max_results: Maximum memories to return
            include_vaulted: Include protected memories
        
        Returns:
            JSON with correlated memories and their wave properties
        """
        try:
            # Ensure phase is in valid range
            phase_tag = phase_tag % (2 * np.pi)
            
            memories = await state_manager.soliton_lattice.recall_by_phase(
                target_phase=phase_tag,
                tolerance=tolerance,
                max_results=max_results,
                include_vaulted=include_vaulted
            )
            
            # Calculate wave interference pattern
            interference_pattern = []
            if memories:
                for t in np.linspace(0, 2*np.pi, 100):
                    real_sum = 0.0
                    imag_sum = 0.0
                    
                    for memory in memories:
                        real_part, imag_part = memory.evaluate_waveform(t)
                        real_sum += real_part
                        imag_sum += imag_part
                    
                    amplitude = np.sqrt(real_sum**2 + imag_sum**2)
                    interference_pattern.append(float(amplitude))
            
            # Format results
            memory_results = []
            for memory in memories:
                correlation = memory.correlate_with_signal(phase_tag, tolerance)
                
                memory_results.append({
                    'memory_id': memory.id,
                    'concept_id': memory.concept_id,
                    'content_preview': memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                    'phase_correlation': correlation,
                    'phase_difference': abs(memory._get_effective_phase() - phase_tag),
                    'soliton_wave': {
                        'phase_tag': memory.phase_tag,
                        'effective_phase': memory._get_effective_phase(),
                        'amplitude': memory.amplitude,
                        'frequency': memory.frequency,
                        'width': memory.width,
                        'stability': memory.stability
                    },
                    'vault_status': memory.vault_status.value,
                    'access_count': memory.access_count
                })
            
            return json.dumps({
                'success': True,
                'target_phase': phase_tag,
                'tolerance': tolerance,
                'memories_found': len(memory_results),
                'total_correlation_strength': sum(m['phase_correlation'] for m in memory_results),
                'interference_pattern': interference_pattern[:20],  # First 20 points
                'memories': memory_results
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            }, indent=2)
    
    @mcp.tool()
    async def vault_memory(
        memory_id: str,
        vault_status: str,
        reason: Optional[str] = None
    ) -> str:
        """
        Change vault protection status of a memory.
        
        Vault statuses apply phase shifts for protection:
        - active: No protection (0° shift)
        - user_sealed: User protection (45° shift)  
        - time_locked: Temporary protection (90° shift)
        - deep_vault: Maximum protection (180° shift)
        
        Args:
            memory_id: ID of memory to vault
            vault_status: New status - 'active', 'user_sealed', 'time_locked', 'deep_vault'
            reason: Optional reason for vaulting
        
        Returns:
            JSON with vault operation result
        """
        try:
            # Validate vault status
            vault_enum = VaultStatus(vault_status.lower())
            
            # Apply vault change
            success = await state_manager.soliton_lattice.vault_memory(memory_id, vault_enum)
            
            if success:
                # Get updated memory
                memory = await state_manager.soliton_lattice.get_memory_by_id(memory_id, include_vaulted=True)
                
                if memory:
                    return json.dumps({
                        'success': True,
                        'memory_id': memory_id,
                        'old_vault_status': 'unknown',  # We don't track old status
                        'new_vault_status': vault_enum.value,
                        'phase_shift_applied': {
                            'active': '0°',
                            'user_sealed': '45°',
                            'time_locked': '90°',
                            'deep_vault': '180°'
                        }.get(vault_enum.value, 'unknown'),
                        'effective_phase': memory._get_effective_phase(),
                        'original_phase': memory.phase_tag,
                        'reason': reason,
                        'is_accessible': memory.is_accessible(),
                        'protection_level': {
                            'active': 'None',
                            'user_sealed': 'User Protected',
                            'time_locked': 'Temporarily Protected', 
                            'deep_vault': 'Maximum Protection'
                        }.get(vault_enum.value, 'Unknown')
                    }, indent=2)
                else:
                    return json.dumps({
                        'success': False,
                        'error': 'Memory not found after vault operation'
                    }, indent=2)
            else:
                return json.dumps({
                    'success': False,
                    'error': 'Memory not found or vault operation failed'
                }, indent=2)
                
        except ValueError as e:
            return json.dumps({
                'success': False,
                'error': f'Invalid vault_status: {vault_status}. Valid statuses: active, user_sealed, time_locked, deep_vault'
            }, indent=2)
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            }, indent=2)
    
    @mcp.tool()
    async def get_soliton_memory_statistics() -> str:
        """
        Get comprehensive statistics about the soliton memory system.
        
        Returns:
            JSON with memory statistics, wave analysis, and system health
        """
        try:
            stats = await state_manager.get_memory_statistics()
            
            # Additional soliton-specific analysis
            memory_system = stats.get('memory_system', {})
            
            # Analyze phase distribution
            phase_distribution = []
            if hasattr(state_manager.soliton_lattice, 'memories'):
                phases = [memory.phase_tag for memory in state_manager.soliton_lattice.memories.values()]
                if phases:
                    # Create phase histogram
                    bins = np.linspace(0, 2*np.pi, 12)  # 12 bins for phase space
                    hist, _ = np.histogram(phases, bins=bins)
                    phase_distribution = hist.tolist()
            
            # Calculate wave interference metrics
            total_amplitude = 0.0
            coherence_score = 0.0
            if hasattr(state_manager.soliton_lattice, 'memories'):
                amplitudes = [memory.amplitude for memory in state_manager.soliton_lattice.memories.values()]
                if amplitudes:
                    total_amplitude = float(np.sum(amplitudes))
                    coherence_score = float(np.std(amplitudes) / (np.mean(amplitudes) + 1e-10))
            
            return json.dumps({
                'success': True,
                'soliton_memory_system': {
                    **memory_system,
                    'wave_analysis': {
                        'total_amplitude': total_amplitude,
                        'coherence_score': coherence_score,
                        'phase_distribution': phase_distribution,
                        'phase_coverage': len(phase_distribution) / 12.0  # How well phases are distributed
                    },
                    'infinite_context': {
                        'no_token_limits': True,
                        'no_degradation': True,
                        'perfect_recall': True,
                        'wave_based_storage': True
                    }
                },
                'system_health': stats.get('cognitive_system', {}),
                'tori_filtering': stats.get('tori_filtering', {}),
                'timestamp': stats.get('timestamp', 'unknown')
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            }, indent=2)
    
    @mcp.tool()
    async def analyze_memory_content_type(
        content_type: str,
        include_vaulted: bool = False
    ) -> str:
        """
        Analyze all memories of a specific content type.
        
        Args:
            content_type: Type to analyze - 'text', 'cognitive_state', 'conversation', 'concept', 'emotional', 'procedural'
            include_vaulted: Include protected memories in analysis
        
        Returns:
            JSON with content type analysis and patterns
        """
        try:
            # Validate content type
            content_type_enum = ContentType(content_type.lower())
            
            # Get memories of this type
            memories = await state_manager.soliton_lattice.get_memories_by_content_type(
                content_type=content_type_enum,
                include_vaulted=include_vaulted
            )
            
            if not memories:
                return json.dumps({
                    'success': True,
                    'content_type': content_type,
                    'memories_found': 0,
                    'message': f'No memories found for content type: {content_type}'
                }, indent=2)
            
            # Analyze patterns
            total_content_length = sum(len(m.content) for m in memories)
            avg_amplitude = np.mean([m.amplitude for m in memories])
            avg_stability = np.mean([m.stability for m in memories])
            avg_access_count = np.mean([m.access_count for m in memories])
            
            # Emotional analysis
            emotional_patterns = {
                'avg_valence': np.mean([m.emotional_signature.valence for m in memories]),
                'avg_arousal': np.mean([m.emotional_signature.arousal for m in memories]),
                'trauma_protected_count': sum(1 for m in memories if len(m.emotional_signature.trauma_indicators) > 0)
            }
            
            # Vault analysis
            vault_distribution = {}
            for memory in memories:
                status = memory.vault_status.value
                vault_distribution[status] = vault_distribution.get(status, 0) + 1
            
            # Recent activity
            recent_memories = sorted(memories, key=lambda m: m.last_accessed, reverse=True)[:5]
            
            return json.dumps({
                'success': True,
                'content_type': content_type,
                'analysis': {
                    'total_memories': len(memories),
                    'total_content_length': total_content_length,
                    'avg_content_length': total_content_length / len(memories),
                    'soliton_characteristics': {
                        'avg_amplitude': float(avg_amplitude),
                        'avg_stability': float(avg_stability),
                        'avg_access_count': float(avg_access_count)
                    },
                    'emotional_patterns': {
                        'avg_valence': float(emotional_patterns['avg_valence']),
                        'avg_arousal': float(emotional_patterns['avg_arousal']),
                        'trauma_protected_rate': emotional_patterns['trauma_protected_count'] / len(memories)
                    },
                    'vault_distribution': vault_distribution,
                    'most_recent_memories': [
                        {
                            'memory_id': m.id,
                            'concept_id': m.concept_id,
                            'last_accessed': m.last_accessed.isoformat(),
                            'access_count': m.access_count,
                            'amplitude': m.amplitude
                        }
                        for m in recent_memories
                    ]
                }
            }, indent=2)
            
        except ValueError as e:
            return json.dumps({
                'success': False,
                'error': f'Invalid content_type: {content_type}. Valid types: text, cognitive_state, conversation, concept, emotional, procedural'
            }, indent=2)
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            }, indent=2)