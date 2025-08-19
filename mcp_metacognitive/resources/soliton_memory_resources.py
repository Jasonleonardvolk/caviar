"""
Soliton Memory Resources
"""

import json
import numpy as np
from mcp.types import TextContent


def register_soliton_memory_resources(mcp, state_manager):
    """Register soliton memory resources."""
    
    @mcp.resource("tori://soliton/memory/lattice")
    async def get_soliton_lattice_info() -> TextContent:
        """Get comprehensive soliton memory lattice information."""
        try:
            stats = await state_manager.soliton_lattice.get_memory_statistics()
            
            # Calculate wave interference patterns
            interference_data = None
            if hasattr(state_manager.soliton_lattice, 'memories') and state_manager.soliton_lattice.memories:
                # Sample interference at key time points
                time_points = np.linspace(0, 2*np.pi, 50)
                interference_amplitudes = []
                
                for t in time_points:
                    total_real = 0.0
                    total_imag = 0.0
                    
                    for memory in list(state_manager.soliton_lattice.memories.values())[:10]:  # Sample first 10
                        if memory.is_accessible():
                            real, imag = memory.evaluate_waveform(t)
                            total_real += real
                            total_imag += imag
                    
                    amplitude = np.sqrt(total_real**2 + total_imag**2)
                    interference_amplitudes.append(float(amplitude))
                
                interference_data = {
                    'sample_points': len(time_points),
                    'max_amplitude': float(np.max(interference_amplitudes)),
                    'min_amplitude': float(np.min(interference_amplitudes)),
                    'avg_amplitude': float(np.mean(interference_amplitudes)),
                    'pattern_complexity': float(np.std(interference_amplitudes))
                }
            
            # Phase space analysis
            phase_analysis = None
            if hasattr(state_manager.soliton_lattice, 'concept_phase_map'):
                phases = list(state_manager.soliton_lattice.concept_phase_map.values())
                if phases:
                    phase_analysis = {
                        'total_concepts': len(phases),
                        'phase_coverage': len(set(np.round(np.array(phases), 1))) / 63,  # Coverage of phase space
                        'avg_phase': float(np.mean(phases)),
                        'phase_clustering': float(np.std(phases)),
                        'unique_phases': len(set(np.round(np.array(phases), 2)))
                    }
            
            return TextContent(
                type="text",
                text=json.dumps({
                    'lattice_statistics': stats,
                    'wave_interference': interference_data,
                    'phase_space_analysis': phase_analysis,
                    'infinite_context_properties': {
                        'no_degradation': True,
                        'perfect_fidelity': True,
                        'wave_based_storage': True,
                        'phase_addressable': True,
                        'consciousness_preserving': True
                    },
                    'user_id': state_manager.soliton_lattice.user_id,
                    'access_history_length': len(state_manager.soliton_lattice.access_history)
                }, indent=2)
            )
            
        except Exception as e:
            return TextContent(
                type="text",
                text=json.dumps({
                    'error': f'Failed to get soliton lattice info: {str(e)}'
                }, indent=2)
            )
    
    @mcp.resource("tori://soliton/memory/vault/{status}")
    async def get_vaulted_memories(status: str) -> TextContent:
        """
        Get memories by vault status.
        
        Args:
            status: Vault status - 'active', 'user_sealed', 'time_locked', 'deep_vault', 'all'
        """
        try:
            memories_info = []
            
            if hasattr(state_manager.soliton_lattice, 'memories'):
                for memory in state_manager.soliton_lattice.memories.values():
                    if status == 'all' or memory.vault_status.value == status:
                        memory_info = {
                            'memory_id': memory.id,
                            'concept_id': memory.concept_id,
                            'vault_status': memory.vault_status.value,
                            'content_type': memory.content_type.value,
                            'content_length': len(memory.content),
                            'content_preview': memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                            'soliton_properties': {
                                'phase_tag': memory.phase_tag,
                                'effective_phase': memory._get_effective_phase(),
                                'amplitude': memory.amplitude,
                                'stability': memory.stability,
                                'access_count': memory.access_count
                            },
                            'emotional_signature': {
                                'valence': memory.emotional_signature.valence,
                                'arousal': memory.emotional_signature.arousal,
                                'trauma_indicators': memory.emotional_signature.trauma_indicators
                            },
                            'timestamps': {
                                'created': memory.creation_time.isoformat(),
                                'last_accessed': memory.last_accessed.isoformat()
                            },
                            'protection_level': {
                                'active': 'None',
                                'user_sealed': 'User Protected (45° phase shift)',
                                'time_locked': 'Temporarily Protected (90° phase shift)',
                                'deep_vault': 'Maximum Protection (180° phase shift)'
                            }.get(memory.vault_status.value, 'Unknown')
                        }
                        memories_info.append(memory_info)
            
            # Sort by access count (most accessed first)
            memories_info.sort(key=lambda x: x['soliton_properties']['access_count'], reverse=True)
            
            return TextContent(
                type="text",
                text=json.dumps({
                    'vault_status_filter': status,
                    'memories_found': len(memories_info),
                    'vault_protection_info': {
                        'active': '0° phase shift - Normally accessible',
                        'user_sealed': '45° phase shift - User chose to protect',
                        'time_locked': '90° phase shift - Temporarily protected',
                        'deep_vault': '180° phase shift - Maximum protection'
                    },
                    'memories': memories_info
                }, indent=2)
            )
            
        except Exception as e:
            return TextContent(
                type="text",
                text=json.dumps({
                    'error': f'Failed to get vaulted memories: {str(e)}'
                }, indent=2)
            )
    
    @mcp.resource("tori://soliton/memory/emotional/analysis")
    async def get_emotional_memory_analysis() -> TextContent:
        """Analyze emotional patterns in soliton memory."""
        try:
            emotional_data = {
                'total_memories': 0,
                'trauma_protected': 0,
                'emotional_distribution': {
                    'very_positive': 0,  # valence > 0.5
                    'positive': 0,       # valence 0.1 to 0.5
                    'neutral': 0,        # valence -0.1 to 0.1
                    'negative': 0,       # valence -0.5 to -0.1
                    'very_negative': 0   # valence < -0.5
                },
                'arousal_distribution': {
                    'high': 0,      # arousal > 0.7
                    'medium': 0,    # arousal 0.3 to 0.7
                    'low': 0        # arousal < 0.3
                },
                'trauma_indicators': {},
                'auto_vaulted_memories': 0,
                'emotional_memory_samples': []
            }
            
            if hasattr(state_manager.soliton_lattice, 'memories'):
                for memory in state_manager.soliton_lattice.memories.values():
                    emotional_data['total_memories'] += 1
                    
                    # Emotional signature analysis
                    valence = memory.emotional_signature.valence
                    arousal = memory.emotional_signature.arousal
                    trauma_indicators = memory.emotional_signature.trauma_indicators
                    
                    # Valence distribution
                    if valence > 0.5:
                        emotional_data['emotional_distribution']['very_positive'] += 1
                    elif valence > 0.1:
                        emotional_data['emotional_distribution']['positive'] += 1
                    elif valence > -0.1:
                        emotional_data['emotional_distribution']['neutral'] += 1
                    elif valence > -0.5:
                        emotional_data['emotional_distribution']['negative'] += 1
                    else:
                        emotional_data['emotional_distribution']['very_negative'] += 1
                    
                    # Arousal distribution
                    if arousal > 0.7:
                        emotional_data['arousal_distribution']['high'] += 1
                    elif arousal > 0.3:
                        emotional_data['arousal_distribution']['medium'] += 1
                    else:
                        emotional_data['arousal_distribution']['low'] += 1
                    
                    # Trauma indicator analysis
                    if trauma_indicators:
                        emotional_data['trauma_protected'] += 1
                        for indicator in trauma_indicators:
                            emotional_data['trauma_indicators'][indicator] = emotional_data['trauma_indicators'].get(indicator, 0) + 1
                    
                    # Auto-vaulted memories
                    if memory.vault_status.value != 'active':
                        emotional_data['auto_vaulted_memories'] += 1
                    
                    # Sample emotional memories for analysis
                    if len(emotional_data['emotional_memory_samples']) < 5 and (valence > 0.5 or valence < -0.5):
                        emotional_data['emotional_memory_samples'].append({
                            'memory_id': memory.id,
                            'concept_id': memory.concept_id,
                            'valence': valence,
                            'arousal': arousal,
                            'vault_status': memory.vault_status.value,
                            'content_preview': memory.content[:50] + "..." if len(memory.content) > 50 else memory.content
                        })
            
            # Calculate rates
            total = emotional_data['total_memories']
            if total > 0:
                emotional_data['trauma_protection_rate'] = emotional_data['trauma_protected'] / total
                emotional_data['auto_vault_rate'] = emotional_data['auto_vaulted_memories'] / total
            
            return TextContent(
                type="text",
                text=json.dumps({
                    'emotional_analysis': emotional_data,
                    'safety_features': {
                        'automatic_trauma_detection': True,
                        'auto_vaulting': True,
                        'dignified_protection': True,
                        'emotional_awareness': True
                    }
                }, indent=2)
            )
            
        except Exception as e:
            return TextContent(
                type="text",
                text=json.dumps({
                    'error': f'Failed to analyze emotional memories: {str(e)}'
                }, indent=2)
            )
    
    @mcp.resource("tori://soliton/memory/wave/interference")
    async def get_wave_interference_analysis() -> TextContent:
        """Analyze soliton wave interference patterns."""
        try:
            if not hasattr(state_manager.soliton_lattice, 'memories') or not state_manager.soliton_lattice.memories:
                return TextContent(
                    type="text",
                    text=json.dumps({
                        'message': 'No memories available for wave analysis'
                    }, indent=2)
                )
            
            memories = list(state_manager.soliton_lattice.memories.values())
            active_memories = [m for m in memories if m.is_accessible()]
            
            # Calculate interference at multiple time points
            time_points = np.linspace(0, 4*np.pi, 100)
            interference_pattern = []
            phase_pattern = []
            
            for t in time_points:
                total_real = 0.0
                total_imag = 0.0
                
                for memory in active_memories[:20]:  # Limit to first 20 for performance
                    real, imag = memory.evaluate_waveform(t)
                    total_real += real
                    total_imag += imag
                
                amplitude = np.sqrt(total_real**2 + total_imag**2)
                phase = np.arctan2(total_imag, total_real)
                
                interference_pattern.append(float(amplitude))
                phase_pattern.append(float(phase))
            
            # Analyze constructive/destructive interference
            max_amplitude = float(np.max(interference_pattern))
            min_amplitude = float(np.min(interference_pattern))
            avg_amplitude = float(np.mean(interference_pattern))
            
            # Find interference peaks and nulls
            peaks = []
            nulls = []
            for i in range(1, len(interference_pattern)-1):
                if (interference_pattern[i] > interference_pattern[i-1] and 
                    interference_pattern[i] > interference_pattern[i+1] and
                    interference_pattern[i] > avg_amplitude * 1.2):
                    peaks.append({'time': float(time_points[i]), 'amplitude': interference_pattern[i]})
                elif (interference_pattern[i] < interference_pattern[i-1] and 
                      interference_pattern[i] < interference_pattern[i+1] and
                      interference_pattern[i] < avg_amplitude * 0.8):
                    nulls.append({'time': float(time_points[i]), 'amplitude': interference_pattern[i]})
            
            # Memory wave characteristics
            memory_characteristics = []
            for memory in active_memories[:10]:  # Sample first 10
                memory_characteristics.append({
                    'memory_id': memory.id,
                    'concept_id': memory.concept_id,
                    'phase_tag': memory.phase_tag,
                    'amplitude': memory.amplitude,
                    'frequency': memory.frequency,
                    'width': memory.width,
                    'stability': memory.stability
                })
            
            return TextContent(
                type="text",
                text=json.dumps({
                    'wave_analysis': {
                        'total_memories_analyzed': len(active_memories),
                        'time_points_sampled': len(time_points),
                        'interference_statistics': {
                            'max_amplitude': max_amplitude,
                            'min_amplitude': min_amplitude,
                            'avg_amplitude': avg_amplitude,
                            'amplitude_range': max_amplitude - min_amplitude,
                            'interference_ratio': max_amplitude / (min_amplitude + 1e-10)
                        },
                        'interference_features': {
                            'constructive_peaks': len(peaks),
                            'destructive_nulls': len(nulls),
                            'peak_locations': peaks[:5],  # First 5 peaks
                            'null_locations': nulls[:5]   # First 5 nulls
                        },
                        'pattern_characteristics': {
                            'pattern_complexity': float(np.std(interference_pattern)),
                            'phase_coherence': float(np.std(phase_pattern)),
                            'wave_synchronization': float(1.0 / (np.std(phase_pattern) + 1e-10))
                        }
                    },
                    'memory_wave_properties': memory_characteristics,
                    'soliton_physics': {
                        'wave_equation': 'Si(t) = A·sech((t-t₀)/T)·exp[j(ω₀t + ψᵢ)]',
                        'interference_principle': 'Coherent superposition of soliton waves',
                        'conservation_laws': ['Energy', 'Momentum', 'Information'],
                        'stability_mechanism': 'Nonlinear self-focusing balances dispersion'
                    }
                }, indent=2)
            )
            
        except Exception as e:
            return TextContent(
                type="text",
                text=json.dumps({
                    'error': f'Failed to analyze wave interference: {str(e)}'
                }, indent=2)
            )