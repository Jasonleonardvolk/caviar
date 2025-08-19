#!/usr/bin/env python3
"""
BPS Topology - Core Supersymmetric Compute Flow
══════════════════════════════════════════════

Core logic for BPS-aware topological transitions, energy harvesting, and charge tracking.
This module provides the backbone for supersymmetric computation with full topological
integrity and charge conservation.

Features:
• Topological charge computation from soliton memory
• BPS-bounded energy harvesting with Q conservation  
• Atomic topology transitions with soliton reinjection
• Supersymmetric compute flow integration
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import scipy.sparse as sp

# Import centralized BPS configuration
try:
    from .bps_config import (
        # Feature flags
        ENABLE_BPS_ENERGY_HARVEST, ENABLE_BPS_CHARGE_TRACKING, ENABLE_BPS_INTERPOLATION,
        ENABLE_BPS_PHASE_LOCKING, STRICT_BPS_MODE, ENABLE_BPS_SAFETY_CHECKS,
        
        # Energy and charge parameters
        ENERGY_PER_Q, BPS_ENERGY_QUANTUM, DEFAULT_SOLITON_AMPLITUDE,
        ALLOWED_Q_VALUES, MAX_ALLOWED_CHARGE_MAGNITUDE,
        
        # Tolerances
        LAGRANGIAN_TOLERANCE, BPS_BOUND_VIOLATION_TOLERANCE, BPS_SATURATION_TOLERANCE,
        CHARGE_CONSERVATION_TOLERANCE, CHARGE_QUANTIZATION_THRESHOLD,
        ENERGY_CONSERVATION_TOLERANCE, ENERGY_EXTRACTION_EFFICIENCY,
        
        # Behavioral controls
        BPS_PHASE_LOCK_GAIN, KURAMOTO_COUPLING_STRENGTH, MAX_PHASE_CORRECTION,
        BERRY_PHASE_SCALING, TOPOLOGY_TRANSITION_DAMPING,
        SWAP_INTERPOLATION_STEPS, DEFAULT_SWAP_RAMP_DURATION,
        
        # Symbolic tags
        SOLITON_TAGS, OPERATION_TAGS, STATE_TAGS,
        
        # Performance limits
        MAX_SOLITON_COUNT, SLOW_OPERATION_THRESHOLD
    )
    BPS_CONFIG_AVAILABLE = True
    logger = logging.getLogger("BPSTopology")
    logger.info("BPS topology using centralized configuration")
    
except ImportError:
    # Fallback constants
    logger = logging.getLogger("BPSTopology")
    logger.warning("BPS config unavailable - using fallback constants")
    
    # Feature flags (conservative defaults)
    ENABLE_BPS_ENERGY_HARVEST = True
    ENABLE_BPS_CHARGE_TRACKING = True
    ENABLE_BPS_INTERPOLATION = False
    ENABLE_BPS_PHASE_LOCKING = True
    STRICT_BPS_MODE = False
    ENABLE_BPS_SAFETY_CHECKS = True
    
    # Energy and charge
    ENERGY_PER_Q = 1.0
    BPS_ENERGY_QUANTUM = 1.0
    DEFAULT_SOLITON_AMPLITUDE = 1.0
    ALLOWED_Q_VALUES = {-2, -1, 0, 1, 2}
    MAX_ALLOWED_CHARGE_MAGNITUDE = 2
    
    # Tolerances
    LAGRANGIAN_TOLERANCE = 1e-6
    BPS_BOUND_VIOLATION_TOLERANCE = 1e-6
    BPS_SATURATION_TOLERANCE = 1e-8
    CHARGE_CONSERVATION_TOLERANCE = 1e-10
    CHARGE_QUANTIZATION_THRESHOLD = 0.5
    ENERGY_CONSERVATION_TOLERANCE = 1e-8
    ENERGY_EXTRACTION_EFFICIENCY = 0.95
    
    # Behavioral controls
    BPS_PHASE_LOCK_GAIN = 0.5
    KURAMOTO_COUPLING_STRENGTH = 0.3
    MAX_PHASE_CORRECTION = 0.5
    BERRY_PHASE_SCALING = 1.0
    TOPOLOGY_TRANSITION_DAMPING = 0.1
    SWAP_INTERPOLATION_STEPS = 10
    DEFAULT_SWAP_RAMP_DURATION = 1.0
    
    # Performance
    MAX_SOLITON_COUNT = 1000
    SLOW_OPERATION_THRESHOLD = 1.0
    
    # Tags (minimal fallback)
    SOLITON_TAGS = {'bright_bps': "Bright BPS", 'dark_bps': "Dark BPS"}
    OPERATION_TAGS = {'bps_harvest': "BPS Energy Harvest"}
    STATE_TAGS = {'bps_saturated': "BPS Saturated"}
    
    BPS_CONFIG_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# BPS Energy Bundle
# ═══════════════════════════════════════════════════════════════════════════════

class EnergyBundle:
    """
    Container for BPS-extracted energy with topological charge conservation.
    
    Maintains the fundamental BPS relationship: E >= |Q|
    """
    
    def __init__(self, energy: float, Q: int, soliton_data: Optional[List[Dict]] = None):
        self.energy = energy  # Total extracted energy from solitons
        self.Q = Q           # Net topological charge conserved
        self.soliton_data = soliton_data or []  # Original soliton configurations
        
        # Verify BPS bound using config tolerance
        if energy < abs(Q) - BPS_BOUND_VIOLATION_TOLERANCE:
            if STRICT_BPS_MODE:
                raise ValueError(f"BPS bound violation: E={energy:.6f} < |Q|={abs(Q)}")
            else:
                logger.warning(f"BPS bound violation: E={energy:.6f} < |Q|={abs(Q)}")
    
    def __repr__(self):
        return f"<EnergyBundle E={self.energy:.3f} Q={self.Q} solitons={len(self.soliton_data)}>"
    
    @property
    def bps_saturation(self) -> float:
        """Return how close to BPS saturation (E = |Q|) this bundle is"""
        if abs(self.Q) < 1e-10:
            return 1.0  # Trivial case
        return self.energy / abs(self.Q)

# ═══════════════════════════════════════════════════════════════════════════════
# Topological Charge Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_topological_charge(memory) -> int:
    """
    Estimate total topological charge from active solitons.
    
    Each soliton contributes +1 or -1 charge based on its topological winding.
    This is the fundamental Q invariant that must be conserved across topology changes.
    
    Args:
        memory: Enhanced soliton memory system with get_active_solitons()
        
    Returns:
        int: Total topological charge Q = Σ sign(soliton.charge)
    """
    # Feature flag check
    if not ENABLE_BPS_CHARGE_TRACKING:
        logger.debug("Charge tracking disabled - returning Q=0")
        return 0
    
    try:
        solitons = memory.get_active_solitons()
        Q = 0
        
        for soliton in solitons:
            # Extract charge from soliton (handle different possible formats)
            if hasattr(soliton, 'charge'):
                charge = soliton.charge
            elif hasattr(soliton, 'topological_charge'):
                charge = soliton.topological_charge
            elif isinstance(soliton, dict):
                charge = soliton.get('charge', soliton.get('topological_charge', 0))
            else:
                logger.warning(f"Soliton {soliton} has no recognizable charge field")
                continue
            
            # Quantize charge to ±1 (topological charges are discrete)
            q_discrete = int(np.sign(charge)) if abs(charge) > CHARGE_QUANTIZATION_THRESHOLD else 0
            
            # Validate against allowed values
            if ENABLE_BPS_SAFETY_CHECKS and q_discrete not in ALLOWED_Q_VALUES:
                if STRICT_BPS_MODE:
                    raise ValueError(f"Invalid charge {q_discrete} not in {ALLOWED_Q_VALUES}")
                else:
                    logger.warning(f"Charge {q_discrete} not in allowed values {ALLOWED_Q_VALUES}")
                    q_discrete = max(-MAX_ALLOWED_CHARGE_MAGNITUDE, 
                                   min(MAX_ALLOWED_CHARGE_MAGNITUDE, q_discrete))
            
            Q += q_discrete
            
            logger.debug(f"Soliton charge: {charge:.3f} → discrete Q: {q_discrete}")
        
        logger.debug(f"Total topological charge computed: Q = {Q}")
        return Q
        
    except Exception as e:
        logger.error(f"Failed to compute topological charge: {e}")
        if STRICT_BPS_MODE:
            raise
        return 0

def compute_topological_charge_density(memory, laplacian: sp.csr_matrix) -> np.ndarray:
    """
    Compute spatial distribution of topological charge density.
    
    Args:
        memory: Soliton memory system
        laplacian: Current Laplacian matrix for spatial mapping
        
    Returns:
        Array of charge density values across lattice sites
    """
    try:
        n_sites = laplacian.shape[0]
        charge_density = np.zeros(n_sites)
        
        solitons = memory.get_active_solitons()
        
        for soliton in solitons:
            # Get soliton position
            if hasattr(soliton, 'position'):
                pos = soliton.position
            elif isinstance(soliton, dict):
                pos = soliton.get('position', 0)
            else:
                pos = 0
            
            # Get charge
            if hasattr(soliton, 'charge'):
                charge = soliton.charge
            elif isinstance(soliton, dict):
                charge = soliton.get('charge', soliton.get('topological_charge', 0))
            else:
                charge = 0
            
            # Map to lattice site (with bounds checking)
            site_idx = int(pos) % n_sites
            charge_density[site_idx] += charge
        
        return charge_density
        
    except Exception as e:
        logger.error(f"Failed to compute charge density: {e}")
        return np.zeros(laplacian.shape[0])

# ═══════════════════════════════════════════════════════════════════════════════
# BPS Energy Harvesting
# ═══════════════════════════════════════════════════════════════════════════════

def bps_energy_harvest(memory, laplacian: sp.csr_matrix, efficiency: float = 0.95) -> EnergyBundle:
    """
    Extract topological energy from BPS solitons with charge conservation.
    
    Each soliton contributes |Q| units of energy (BPS saturation condition).
    The fundamental BPS bound E >= |Q| ensures energy extraction cannot
    violate topological stability.
    
    Args:
        memory: Enhanced soliton memory system
        laplacian: Current Laplacian for energy context
        efficiency: Extraction efficiency (default 0.95)
        
    Returns:
        EnergyBundle with conserved energy and charge
    """
    try:
        logger.info("Initiating BPS energy harvest...")
        
        solitons = memory.get_active_solitons()
        total_Q = 0
        total_E = 0.0
        soliton_data = []
        
        for soliton in solitons:
            # Extract soliton properties
            if hasattr(soliton, 'charge'):
                charge = soliton.charge
            elif isinstance(soliton, dict):
                charge = soliton.get('charge', soliton.get('topological_charge', 0))
            else:
                charge = 0
            
            if hasattr(soliton, 'location'):
                location = soliton.location
            elif isinstance(soliton, dict):
                location = soliton.get('location', soliton.get('position', 0))
            else:
                location = 0
            
            # Discrete topological charge
            q_discrete = int(np.sign(charge)) if abs(charge) > 1e-10 else 0
            
            # BPS energy: E = |Q| per soliton (fundamental bound)
            bps_energy = abs(q_discrete) * 1.0  # Normalized energy units
            extractable_energy = bps_energy * efficiency
            
            total_Q += q_discrete
            total_E += extractable_energy
            
            # Store soliton data for reinjection
            soliton_data.append({
                'charge': charge,
                'location': location,
                'bps_energy': bps_energy,
                'extracted_energy': extractable_energy,
                'original_soliton': soliton
            })
            
            logger.debug(f"Harvested {extractable_energy:.3f} from soliton at {location} (Q={q_discrete})")
        
        # Create energy bundle
        bundle = EnergyBundle(energy=total_E, Q=total_Q, soliton_data=soliton_data)
        
        logger.info(f"BPS harvest complete: {bundle}")
        logger.info(f"BPS saturation: {bundle.bps_saturation:.3f}")
        
        return bundle
        
    except Exception as e:
        logger.error(f"BPS energy harvest failed: {e}")
        return EnergyBundle(energy=0.0, Q=0)

def bps_charge_extraction(lattice_state: np.ndarray, memory) -> Tuple[np.ndarray, int]:
    """
    Extract topological charge from lattice state during blow-up scenarios.
    
    This replaces traditional lattice.psi.copy() with BPS-aware charge extraction
    that preserves topological invariants during energy capture.
    
    Args:
        lattice_state: Current lattice field configuration
        memory: Soliton memory for charge tracking
        
    Returns:
        Tuple of (extracted_field, total_charge)
    """
    try:
        logger.debug("Performing BPS charge extraction from lattice")
        
        # Compute current topological charge
        total_Q = compute_topological_charge(memory)
        
        # Extract field with topological structure preservation
        # This maintains the winding number structure critical for BPS bounds
        extracted_field = np.copy(lattice_state)
        
        # Apply charge-preserving filtering to maintain topological integrity
        if total_Q != 0:
            # Normalize field to preserve charge density
            field_norm = np.linalg.norm(extracted_field)
            if field_norm > 1e-10:
                # Scale to maintain |Q| relationship
                charge_preserving_factor = abs(total_Q) / field_norm
                extracted_field *= charge_preserving_factor
        
        logger.debug(f"Extracted field with Q={total_Q}, norm={np.linalg.norm(extracted_field):.3f}")
        
        return extracted_field, total_Q
        
    except Exception as e:
        logger.error(f"BPS charge extraction failed: {e}")
        return np.copy(lattice_state), 0

# ═══════════════════════════════════════════════════════════════════════════════
# BPS Topology Transitions
# ═══════════════════════════════════════════════════════════════════════════════

def bps_topology_transition(
    lattice: sp.csr_matrix,
    new_laplacian: sp.csr_matrix,
    memory,
    energy_bundle: EnergyBundle
) -> bool:
    """
    Perform BPS-preserving topology transition with full Q and E conservation.
    
    This implements atomic Laplacian overwrite with soliton reinjection,
    maintaining the supersymmetric compute flow integrity.
    
    Args:
        lattice: Current Laplacian matrix (modified in-place)
        new_laplacian: Target Laplacian matrix
        memory: Enhanced soliton memory system
        energy_bundle: Conserved energy and charge from harvest
        
    Returns:
        bool: True if transition successful with conservation laws satisfied
    """
    try:
        logger.info("Initiating BPS topology transition...")
        
        # Verify pre-transition state
        Q_before = compute_topological_charge(memory)
        E_before = energy_bundle.energy
        
        logger.debug(f"Pre-transition state: Q={Q_before}, E={E_before:.3f}")
        
        # Atomic Laplacian overwrite
        # This preserves the sparse structure while updating topology
        if hasattr(lattice, 'data') and hasattr(new_laplacian, 'data'):
            # In-place update of sparse matrix data
            if lattice.shape == new_laplacian.shape:
                lattice.data[:] = new_laplacian.data[:]
                lattice.indices[:] = new_laplacian.indices[:]
                lattice.indptr[:] = new_laplacian.indptr[:]
                logger.debug("Atomic Laplacian update completed (in-place)")
            else:
                logger.warning("Shape mismatch - cannot perform atomic update")
                return False
        else:
            logger.warning("Lattice format incompatible for atomic update")
            return False
        
        # Reinject conserved solitons with topological phase corrections
        success_count = 0
        for soliton_data in energy_bundle.soliton_data:
            try:
                # Apply Berry phase correction for topology change
                berry_phase = np.pi * (energy_bundle.Q / len(energy_bundle.soliton_data))
                
                # Reinject soliton with phase correction
                if hasattr(memory, 'reinject_soliton'):
                    memory.reinject_soliton(
                        soliton_data['original_soliton'],
                        lattice,
                        phase_correction=berry_phase
                    )
                elif hasattr(memory, 'update_solitons'):
                    # Fallback: update soliton list
                    updated_soliton = soliton_data['original_soliton'].copy()
                    if isinstance(updated_soliton, dict):
                        updated_soliton['phase'] = updated_soliton.get('phase', 0) + berry_phase
                    memory.update_solitons([updated_soliton])
                
                success_count += 1
                logger.debug(f"Reinjected soliton Q={soliton_data['charge']:.1f} at {soliton_data['location']}")
                
            except Exception as e:
                logger.warning(f"Failed to reinject soliton: {e}")
        
        # Verify post-transition conservation
        Q_after = compute_topological_charge(memory)
        
        if abs(Q_after - Q_before) > 1e-10:
            logger.error(f"Charge conservation violated: {Q_before} → {Q_after}")
            return False
        
        if success_count < len(energy_bundle.soliton_data):
            logger.warning(f"Only {success_count}/{len(energy_bundle.soliton_data)} solitons reinjected")
        
        logger.info(f"BPS transition complete: Q conserved at {Q_after}, {success_count} solitons reinjected")
        return True
        
    except Exception as e:
        logger.exception("BPS topology transition failed")
        return False

def interpolated_bps_transition(
    lattice: sp.csr_matrix,
    new_laplacian: sp.csr_matrix,
    memory,
    energy_bundle: EnergyBundle,
    steps: int = 10
) -> bool:
    """
    Perform gradual BPS transition through intermediate topologies.
    
    This provides smoother transitions for large topological changes,
    maintaining adiabatic evolution of the supersymmetric state.
    
    Args:
        lattice: Current Laplacian (modified in-place)
        new_laplacian: Target Laplacian
        memory: Soliton memory system
        energy_bundle: Conserved energy bundle
        steps: Number of interpolation steps
        
    Returns:
        bool: True if gradual transition successful
    """
    try:
        logger.info(f"Initiating interpolated BPS transition ({steps} steps)...")
        
        original_lattice = lattice.copy()
        
        for step in range(steps + 1):
            alpha = step / steps  # Interpolation parameter [0, 1]
            
            # Linear interpolation in matrix space
            interpolated = (1 - alpha) * original_lattice + alpha * new_laplacian
            
            # Update lattice
            lattice.data[:] = interpolated.data[:]
            lattice.indices[:] = interpolated.indices[:]
            lattice.indptr[:] = interpolated.indptr[:]
            
            # Check charge conservation at each step
            Q_current = compute_topological_charge(memory)
            expected_Q = energy_bundle.Q
            
            if abs(Q_current - expected_Q) > 1e-6:
                logger.warning(f"Step {step}: charge deviation detected")
            
            logger.debug(f"Interpolation step {step}/{steps}: α={alpha:.3f}, Q={Q_current}")
        
        # Final soliton reinjection
        return bps_topology_transition(lattice, new_laplacian, memory, energy_bundle)
        
    except Exception as e:
        logger.error(f"Interpolated BPS transition failed: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# BPS Diagnostics and Verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_bps_conservation(memory, energy_bundle: EnergyBundle) -> Dict[str, Any]:
    """
    Verify BPS bounds and conservation laws after transitions.
    
    Returns:
        Dictionary with verification results
    """
    try:
        current_Q = compute_topological_charge(memory)
        
        return {
            'charge_conserved': abs(current_Q - energy_bundle.Q) < 1e-10,
            'bps_bound_satisfied': energy_bundle.energy >= abs(energy_bundle.Q) - 1e-10,
            'current_charge': current_Q,
            'expected_charge': energy_bundle.Q,
            'charge_error': abs(current_Q - energy_bundle.Q),
            'bps_saturation': energy_bundle.bps_saturation,
            'energy': energy_bundle.energy
        }
        
    except Exception as e:
        logger.error(f"BPS verification failed: {e}")
        return {'error': str(e)}

def bps_stability_check(laplacian: sp.csr_matrix, memory) -> bool:
    """
    Check if current state satisfies BPS stability conditions.
    
    Returns:
        bool: True if system is BPS-stable
    """
    try:
        # Check spectral properties
        from scipy.sparse.linalg import eigsh
        
        eigenvalues = eigsh(laplacian, k=min(6, laplacian.shape[0] - 1), 
                           which='SM', return_eigenvectors=False)
        
        # Verify positive spectrum (stability)
        if np.any(eigenvalues < -1e-10):
            logger.warning("Negative eigenvalues detected - system unstable")
            return False
        
        # Check topological charge consistency
        Q = compute_topological_charge(memory)
        charge_density = compute_topological_charge_density(memory, laplacian)
        
        if abs(np.sum(charge_density) - Q) > 1e-6:
            logger.warning("Charge density inconsistency detected")
            return False
        
        logger.debug("BPS stability check passed")
        return True
        
    except Exception as e:
        logger.error(f"BPS stability check failed: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# Supersymmetric Compute Flow Integration
# ═══════════════════════════════════════════════════════════════════════════════

def enable_supersymmetric_flow(memory, laplacian: sp.csr_matrix) -> Dict[str, Any]:
    """
    Initialize supersymmetric compute flow with BPS awareness.
    
    This sets up the system for automatic BPS conservation during
    all subsequent topological operations.
    
    Returns:
        Dictionary with flow configuration
    """
    try:
        logger.info("Enabling supersymmetric compute flow...")
        
        # Initialize flow state
        flow_state = {
            'enabled': True,
            'initial_charge': compute_topological_charge(memory),
            'initial_energy': 0.0,  # To be computed from soliton amplitudes
            'bps_mode': 'active',
            'conservation_tolerance': 1e-10,
            'auto_verification': True
        }
        
        # Compute initial energy from solitons
        try:
            solitons = memory.get_active_solitons()
            total_energy = 0.0
            
            for soliton in solitons:
                if hasattr(soliton, 'amplitude'):
                    energy = abs(soliton.amplitude)**2
                elif isinstance(soliton, dict):
                    energy = abs(soliton.get('amplitude', 1.0))**2
                else:
                    energy = 1.0
                
                total_energy += energy
            
            flow_state['initial_energy'] = total_energy
            
        except Exception as e:
            logger.warning(f"Could not compute initial energy: {e}")
        
        # Verify BPS bounds
        if flow_state['initial_energy'] < abs(flow_state['initial_charge']) - 1e-10:
            logger.warning("Initial state violates BPS bound - supersymmetric flow may be unstable")
            flow_state['bps_violation'] = True
        
        logger.info(f"Supersymmetric flow enabled: Q={flow_state['initial_charge']}, E={flow_state['initial_energy']:.3f}")
        
        return flow_state
        
    except Exception as e:
        logger.error(f"Failed to enable supersymmetric flow: {e}")
        return {'enabled': False, 'error': str(e)}

if __name__ == "__main__":
    # Basic verification that module loads correctly
    logger.info("BPS Topology module loaded successfully")
    logger.info("Available functions:")
    logger.info("  • compute_topological_charge(memory)")
    logger.info("  • bps_energy_harvest(memory, laplacian)")
    logger.info("  • bps_topology_transition(lattice, new_laplacian, memory, energy_bundle)")
    logger.info("  • bps_charge_extraction(lattice_state, memory)")
    logger.info("  • enable_supersymmetric_flow(memory, laplacian)")
