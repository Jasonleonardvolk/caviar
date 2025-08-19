#!/usr/bin/env python3
"""
Patch to wire up comfort analysis feedback into nightly consolidation
"""

import numpy as np
from typing import Dict, List, Tuple


def integrate_comfort_feedback_into_nightly():
    """
    Patch for 23_nightly_consolidation_enhanced.py
    Replaces the _optimize_lattice_comfort method to use ComfortAnalyzer
    """
    
    # New implementation that uses ComfortAnalyzer
    async def _optimize_lattice_comfort(self) -> Dict[str, int]:
        """
        Optimize lattice based on soliton comfort metrics using ComfortAnalyzer
        Adjust local couplings to reduce stress and improve stability
        """
        results = {'adjustments': 0, 'comfort_actions': 0}
        lattice = get_global_lattice()
        
        # Import ComfortAnalyzer
        try:
            from comfort_analysis import ComfortAnalyzer
        except ImportError:
            logger.warning("ComfortAnalyzer not available, falling back to simple optimization")
            return await self._simple_optimize_lattice_comfort()
        
        # Create analyzer with config
        config = {
            'high_stress_action_threshold': 0.8,
            'low_energy_action_threshold': 0.2,
            'high_flux_action_threshold': 0.9,
            'high_perturbation_action_threshold': 0.8
        }
        auto_actions = {
            'reduce_coupling_on_high_stress': True,
            'boost_amplitude_on_low_energy': True,
            'migrate_on_high_flux': True,
            'pause_morphing_on_high_perturbation': False  # We control morphing
        }
        
        analyzer = ComfortAnalyzer.with_config(config, auto_actions)
        
        # Run comfort analysis
        feedback_list = lattice.run_comfort_analysis(analyzer)
        
        # Get coupling adjustment suggestions
        coupling_adjustments = []
        
        for mem_id, memory in lattice.memories.items():
            if hasattr(memory, 'comfort_metrics'):
                # Use the suggest_coupling_adjustments method
                adjustments = analyzer.suggest_coupling_adjustments(memory, lattice)
                coupling_adjustments.extend(adjustments)
        
        # Apply coupling adjustments
        for (i, j, adjustment_factor) in coupling_adjustments:
            if hasattr(lattice, 'K') and i < len(lattice.K) and j < len(lattice.K[0]):
                # Apply the suggested adjustment
                lattice.K[i, j] *= adjustment_factor
                lattice.K[j, i] *= adjustment_factor  # Keep symmetric
                results['adjustments'] += 1
                
                logger.debug(f"Applied coupling adjustment: ({i},{j}) *= {adjustment_factor:.2f}")
        
        # Apply medium and high severity feedback actions
        for feedback in feedback_list:
            if feedback.severity in [FeedbackSeverity.Medium, FeedbackSeverity.High]:
                if mem_id in lattice.memories:
                    memory = lattice.memories[mem_id]
                    applied = analyzer.apply_feedback(feedback, memory, lattice)
                    results['comfort_actions'] += len(applied)
                    
                    if applied:
                        logger.info(f"Applied comfort feedback for {mem_id}: {applied}")
        
        # Generate system report
        report = analyzer.get_system_comfort_report(lattice)
        logger.info(
            f"System comfort report: Health={report.system_health.name}, "
            f"Avg stress={report.avg_stress:.2f}, "
            f"Stressed memories={report.stressed_count}/{report.total_memories}"
        )
        
        return results


def add_suggest_coupling_adjustments_to_comfort_analyzer():
    """
    Add the missing suggest_coupling_adjustments method to ComfortAnalyzer
    """
    
    def suggest_coupling_adjustments(
        self, 
        memory: 'SolitonMemory', 
        lattice: 'SolitonLattice'
    ) -> List[Tuple[int, int, float]]:
        """
        Suggest coupling adjustments based on comfort metrics
        Returns list of (i, j, adjustment_factor) tuples
        """
        adjustments = []
        
        if not hasattr(memory, 'comfort_metrics'):
            return adjustments
        
        comfort = memory.comfort_metrics
        
        # Get oscillator index
        if 'oscillator_idx' not in memory.metadata:
            return adjustments
        
        idx = memory.metadata['oscillator_idx']
        
        # High stress: reduce all couplings
        if comfort.stress > self.high_stress_threshold:
            reduction_factor = 1.0 - (comfort.stress - self.high_stress_threshold) * 0.3
            reduction_factor = max(0.5, reduction_factor)  # Don't reduce by more than 50%
            
            # Find all connections to this oscillator
            if hasattr(lattice, 'K'):
                for j in range(len(lattice.oscillators)):
                    if j != idx and lattice.K[idx, j] > 0:
                        adjustments.append((idx, j, reduction_factor))
        
        # High flux: selectively reduce strong couplings
        if comfort.flux > self.high_flux_threshold:
            # Identify overly strong couplings
            if hasattr(lattice, 'K'):
                couplings = []
                for j in range(len(lattice.oscillators)):
                    if j != idx and lattice.K[idx, j] > 0:
                        couplings.append((j, lattice.K[idx, j]))
                
                # Sort by coupling strength
                couplings.sort(key=lambda x: x[1], reverse=True)
                
                # Reduce top 25% strongest couplings
                num_to_reduce = max(1, len(couplings) // 4)
                for j, strength in couplings[:num_to_reduce]:
                    # Reduce proportionally to how much over threshold
                    excess = comfort.flux - self.high_flux_threshold
                    reduction_factor = 1.0 - excess * 0.4
                    reduction_factor = max(0.6, reduction_factor)
                    adjustments.append((idx, j, reduction_factor))
        
        # Low energy: strengthen connections to high-energy neighbors
        if comfort.energy < self.low_energy_threshold:
            if hasattr(lattice, 'K') and hasattr(lattice, 'oscillators'):
                for j in range(len(lattice.oscillators)):
                    if j != idx and lattice.K[idx, j] > 0:
                        # Check neighbor's amplitude (proxy for energy)
                        neighbor_amp = lattice.oscillators[j].get('amplitude', 0.5)
                        if neighbor_amp > 0.8:  # High energy neighbor
                            # Strengthen connection
                            boost_factor = 1.0 + (self.low_energy_threshold - comfort.energy) * 0.2
                            boost_factor = min(1.3, boost_factor)  # Max 30% boost
                            adjustments.append((idx, j, boost_factor))
        
        # Remove duplicates and conflicting adjustments
        final_adjustments = {}
        for i, j, factor in adjustments:
            key = (min(i, j), max(i, j))
            if key not in final_adjustments:
                final_adjustments[key] = factor
            else:
                # If multiple adjustments, take the more conservative one
                final_adjustments[key] = min(final_adjustments[key], factor)
        
        return [(i, j, f) for (i, j), f in final_adjustments.items()]
    
    # This method would be added to the ComfortAnalyzer class
    return suggest_coupling_adjustments


# Also add missing imports to nightly consolidation
REQUIRED_IMPORTS = """
from comfort_analysis import ComfortAnalyzer, FeedbackSeverity, SystemHealth
from soliton_memory import VaultStatus
import numpy as np
"""
