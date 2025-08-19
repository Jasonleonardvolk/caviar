#!/usr/bin/env python3
"""
BPS Energy Harvesting and Topology Transition Wrappers
═══════════════════════════════════════════════════════

BPS (Bogomolny-Prasad-Sommerfield) integration for the hot-swappable Laplacian system.
Provides topological charge conservation and energy harvesting with BPS bounds.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# BPS Constants (imported from config)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .hot_swap_config import (
        BPS_CHARGE_CONSERVATION_TOLERANCE,
        BPS_ENERGY_HARVEST_EFFICIENCY,
        BPS_TOPOLOGY_TRANSITION_DAMPING,
        BPS_SOLITON_PHASE_LOCK_STRENGTH
    )
except ImportError:
    # Fallback BPS constants
    BPS_CHARGE_CONSERVATION_TOLERANCE = 1e-12
    BPS_ENERGY_HARVEST_EFFICIENCY = 0.95
    BPS_TOPOLOGY_TRANSITION_DAMPING = 0.1
    BPS_SOLITON_PHASE_LOCK_STRENGTH = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# BPS Energy Harvesting
# ═══════════════════════════════════════════════════════════════════════════════

def bps_energy_harvest(
    solitons: List[Dict[str, Any]], 
    chern_number: int,
    efficiency: float = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Harvest energy from solitons while preserving BPS topological charge.
    
    Args:
        solitons: List of active solitons with amplitude, phase, topological_charge
        chern_number: Current topology's Chern number (Q = topological charge)
        efficiency: Energy harvest efficiency (default from config)
        
    Returns:
        Tuple of (harvested_energy_array, bps_metrics)
    """
    if efficiency is None:
        efficiency = BPS_ENERGY_HARVEST_EFFICIENCY
    
    logger.debug(f"BPS energy harvest: {len(solitons)} solitons, Q={chern_number}")
    
    # Calculate total topological charge before harvest
    total_charge_before = sum(s.get('topological_charge', 0.0) for s in solitons)
    total_energy_before = sum(s.get('amplitude', 1.0)**2 for s in solitons)
    
    # BPS bound: E >= |Q| (energy bounded by absolute topological charge)
    bps_bound = abs(total_charge_before)
    
    if total_energy_before < bps_bound - BPS_CHARGE_CONSERVATION_TOLERANCE:
        logger.warning(f"System violates BPS bound: E={total_energy_before:.6f} < |Q|={bps_bound:.6f}")
    
    # Harvest energy while preserving topological charge structure
    harvested_energy = []
    charge_preserved = 0.0
    
    for soliton in solitons:
        amplitude = soliton.get('amplitude', 1.0)
        phase = soliton.get('phase', 0.0)
        charge = soliton.get('topological_charge', 0.0)
        
        # BPS-preserving energy extraction
        energy_density = amplitude**2
        harvestable_energy = energy_density * efficiency
        
        # Preserve charge-to-energy ratio (BPS saturation condition)
        if abs(charge) > BPS_CHARGE_CONSERVATION_TOLERANCE:
            charge_preserving_factor = min(efficiency, harvestable_energy / abs(charge))
        else:
            charge_preserving_factor = efficiency
        
        # Extract energy as complex amplitude preserving phase relationships
        harvested_amplitude = amplitude * np.sqrt(charge_preserving_factor)
        harvested_complex = harvested_amplitude * np.exp(1j * phase)
        
        harvested_energy.append(harvested_complex)
        charge_preserved += charge * charge_preserving_factor
        
        # Update soliton with remaining energy
        remaining_factor = np.sqrt(1.0 - charge_preserving_factor)
        soliton['amplitude'] *= remaining_factor
    
    harvested_array = np.array(harvested_energy, dtype=complex)
    
    # Verify charge conservation
    charge_error = abs(charge_preserved - total_charge_before * efficiency)
    if charge_error > BPS_CHARGE_CONSERVATION_TOLERANCE:
        logger.warning(f"BPS charge conservation violation: error={charge_error:.2e}")
    
    bps_metrics = {
        'total_charge_before': total_charge_before,
        'charge_preserved': charge_preserved,
        'charge_conservation_error': charge_error,
        'bps_bound': bps_bound,
        'energy_before': total_energy_before,
        'energy_harvested': np.sum(np.abs(harvested_array)**2),
        'efficiency_actual': efficiency,
        'bps_bound_satisfied': total_energy_before >= bps_bound - BPS_CHARGE_CONSERVATION_TOLERANCE
    }
    
    logger.info(f"BPS harvest: E_harvested={bps_metrics['energy_harvested']:.3f}, Q_preserved={charge_preserved:.6f}")
    
    return harvested_array, bps_metrics

# ═══════════════════════════════════════════════════════════════════════════════
# BPS Topology Transition
# ═══════════════════════════════════════════════════════════════════════════════

def bps_topology_transition(
    old_chern: int,
    new_chern: int,
    solitons: List[Dict[str, Any]],
    old_laplacian: sp.csr_matrix,
    new_laplacian: sp.csr_matrix
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Perform BPS-aware topology transition preserving topological invariants.
    
    Args:
        old_chern: Previous topology's Chern number
        new_chern: New topology's Chern number  
        solitons: Active solitons to transition
        old_laplacian: Previous Laplacian matrix
        new_laplacian: New Laplacian matrix
        
    Returns:
        Tuple of (transitioned_solitons, transition_metrics)
    """
    logger.debug(f"BPS topology transition: Q {old_chern} → {new_chern}")
    
    # Calculate topological charge difference
    charge_delta = new_chern - old_chern
    
    # Preserve total topological charge (BPS invariant)
    total_charge_before = sum(s.get('topological_charge', 0.0) for s in solitons)
    
    transitioned_solitons = []
    
    for i, soliton in enumerate(solitons):
        old_amplitude = soliton.get('amplitude', 1.0)
        old_phase = soliton.get('phase', 0.0)
        old_charge = soliton.get('topological_charge', 0.0)
        
        # BPS transition preserving energy-charge relationship
        # The transition creates adiabatic evolution preserving BPS bounds
        
        # Phase evolution due to topology change (Berry phase)
        berry_phase_correction = charge_delta * np.pi / len(solitons)
        new_phase = (old_phase + berry_phase_correction) % (2 * np.pi)
        
        # Amplitude adjustment to maintain BPS bound under new topology
        if abs(new_chern) > 0:
            # Scale amplitude to maintain E >= |Q| under new Chern number
            bps_scaling_factor = np.sqrt(abs(new_chern) / max(abs(old_chern), 1e-10))
            # Apply damping to prevent instabilities
            damped_scaling = 1.0 + BPS_TOPOLOGY_TRANSITION_DAMPING * (bps_scaling_factor - 1.0)
        else:
            damped_scaling = 1.0
        
        new_amplitude = old_amplitude * damped_scaling
        
        # Update topological charge to new topology
        # Distribute charge change proportionally
        if len(solitons) > 0:
            charge_redistribution = charge_delta / len(solitons)
            new_charge = old_charge + charge_redistribution
        else:
            new_charge = new_chern
        
        # Apply phase locking for stability
        if i > 0 and BPS_SOLITON_PHASE_LOCK_STRENGTH > 0:
            # Lock phase to previous soliton for coherence
            prev_phase = transitioned_solitons[i-1]['phase']
            phase_lock_correction = BPS_SOLITON_PHASE_LOCK_STRENGTH * (prev_phase - new_phase)
            new_phase += phase_lock_correction
        
        transitioned_soliton = {
            'amplitude': new_amplitude,
            'phase': new_phase,
            'topological_charge': new_charge,
            'width': soliton.get('width', 2.0),
            'velocity': soliton.get('velocity', 1.0),
            'position': soliton.get('position', i),
            'index': i
        }
        
        transitioned_solitons.append(transitioned_soliton)
    
    # Verify charge conservation
    total_charge_after = sum(s['topological_charge'] for s in transitioned_solitons)
    expected_charge = total_charge_before + charge_delta
    charge_error = abs(total_charge_after - expected_charge)
    
    # Verify energy conservation with BPS bounds
    total_energy_after = sum(s['amplitude']**2 for s in transitioned_solitons)
    bps_bound_after = abs(total_charge_after)
    
    transition_metrics = {
        'charge_delta': charge_delta,
        'charge_before': total_charge_before,
        'charge_after': total_charge_after,
        'charge_expected': expected_charge,
        'charge_conservation_error': charge_error,
        'energy_after': total_energy_after,
        'bps_bound_after': bps_bound_after,
        'bps_bound_satisfied': total_energy_after >= bps_bound_after - BPS_CHARGE_CONSERVATION_TOLERANCE,
        'spectral_gap_ratio': _compute_spectral_gap_ratio(old_laplacian, new_laplacian)
    }
    
    if charge_error > BPS_CHARGE_CONSERVATION_TOLERANCE:
        logger.warning(f"BPS transition charge error: {charge_error:.2e}")
    
    if not transition_metrics['bps_bound_satisfied']:
        logger.warning(f"BPS bound violation after transition: E={total_energy_after:.6f} < |Q|={bps_bound_after:.6f}")
    
    logger.info(f"BPS transition complete: Q={total_charge_after:.6f}, E={total_energy_after:.3f}")
    
    return transitioned_solitons, transition_metrics

# ═══════════════════════════════════════════════════════════════════════════════
# BPS Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_spectral_gap_ratio(old_laplacian: sp.csr_matrix, new_laplacian: sp.csr_matrix) -> float:
    """Compute ratio of spectral gaps between old and new Laplacians"""
    try:
        from scipy.sparse.linalg import eigsh
        
        # Get smallest non-zero eigenvalues
        old_eigs = eigsh(old_laplacian, k=2, which='SM', return_eigenvectors=False)
        new_eigs = eigsh(new_laplacian, k=2, which='SM', return_eigenvectors=False)
        
        old_gap = old_eigs[1] - old_eigs[0] if len(old_eigs) > 1 else 0.0
        new_gap = new_eigs[1] - new_eigs[0] if len(new_eigs) > 1 else 0.0
        
        if old_gap > 1e-10:
            return new_gap / old_gap
        else:
            return 1.0
            
    except Exception as e:
        logger.warning(f"Spectral gap computation failed: {e}")
        return 1.0

def verify_bps_bounds(solitons: List[Dict[str, Any]], chern_number: int) -> Dict[str, Any]:
    """
    Verify that soliton configuration satisfies BPS bounds.
    
    Args:
        solitons: List of soliton configurations
        chern_number: Topology's Chern number
        
    Returns:
        Dictionary with BPS verification results
    """
    total_energy = sum(s.get('amplitude', 1.0)**2 for s in solitons)
    total_charge = sum(s.get('topological_charge', 0.0) for s in solitons)
    bps_bound = abs(total_charge)
    
    bound_satisfied = total_energy >= bps_bound - BPS_CHARGE_CONSERVATION_TOLERANCE
    bound_saturation = abs(total_energy - bps_bound) / max(bps_bound, 1e-10)
    
    return {
        'total_energy': total_energy,
        'total_charge': total_charge,
        'bps_bound': bps_bound,
        'bound_satisfied': bound_satisfied,
        'bound_saturation': bound_saturation,
        'chern_number': chern_number,
        'soliton_count': len(solitons)
    }
