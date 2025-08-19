#!/usr/bin/env python3
"""
BPS-Aware Blowup Harness
═══════════════════════

Enhanced energy harvesting system with BPS topological charge conservation.
Replaces traditional lattice.psi.copy() with bps_charge_extraction() to maintain
topological integrity during blow-up capture scenarios.

Features:
• BPS-preserving energy amplification and harvesting
• Topological charge conservation during blow-up
• Adaptive step scaling based on charge density
• Safe energy extraction with automatic bounds checking
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

# Import centralized BPS configuration
try:
    from .bps_config import (
        # Feature flags
        ENABLE_BPS_ENERGY_HARVEST, ENABLE_BPS_CHARGE_TRACKING, ENABLE_BPS_SAFETY_CHECKS,
        STRICT_BPS_MODE, ENABLE_BPS_ADAPTIVE_SCALING,
        
        # Energy parameters
        ENERGY_PER_Q, MAX_BPS_ENERGY_MULTIPLIER, ENERGY_EXTRACTION_EFFICIENCY,
        
        # Harvest parameters
        DEFAULT_HARVEST_EPSILON, DEFAULT_HARVEST_STEPS, MAX_AMPLIFICATION_FACTOR,
        HARVEST_SAFETY_MARGIN, ADAPTIVE_EPSILON_MIN, ADAPTIVE_EPSILON_MAX,
        
        # Tolerances
        CHARGE_CONSERVATION_TOLERANCE, BPS_BOUND_VIOLATION_TOLERANCE,
        ENERGY_CONSERVATION_TOLERANCE,
        
        # Performance limits
        MAX_SOLITON_COUNT, MAX_HARVEST_STEPS, SLOW_OPERATION_THRESHOLD,
        
        # Symbolic tags
        OPERATION_TAGS, STATE_TAGS
    )
    
    # Map BPS config to legacy constants for compatibility
    DEFAULT_EPSILON = DEFAULT_HARVEST_EPSILON
    DEFAULT_STEPS = DEFAULT_HARVEST_STEPS
    BPS_ENERGY_SAFETY_MARGIN = HARVEST_SAFETY_MARGIN
    
    BPS_CONFIG_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Blowup harness using centralized BPS configuration")
    
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("BPS config unavailable - using fallback constants")
    
    # Feature flags (conservative defaults)
    ENABLE_BPS_ENERGY_HARVEST = True
    ENABLE_BPS_CHARGE_TRACKING = True
    ENABLE_BPS_SAFETY_CHECKS = True
    STRICT_BPS_MODE = False
    ENABLE_BPS_ADAPTIVE_SCALING = False
    
    # Energy parameters
    ENERGY_PER_Q = 1.0
    MAX_BPS_ENERGY_MULTIPLIER = 10.0
    ENERGY_EXTRACTION_EFFICIENCY = 0.95
    
    # Harvest parameters
    DEFAULT_HARVEST_EPSILON = 0.5
    DEFAULT_HARVEST_STEPS = 10
    MAX_AMPLIFICATION_FACTOR = 10.0
    HARVEST_SAFETY_MARGIN = 0.1
    ADAPTIVE_EPSILON_MIN = 0.1
    ADAPTIVE_EPSILON_MAX = 1.0
    
    # Tolerances
    CHARGE_CONSERVATION_TOLERANCE = 1e-10
    BPS_BOUND_VIOLATION_TOLERANCE = 1e-6
    ENERGY_CONSERVATION_TOLERANCE = 1e-8
    
    # Performance
    MAX_SOLITON_COUNT = 1000
    MAX_HARVEST_STEPS = 50
    SLOW_OPERATION_THRESHOLD = 1.0
    
    # Legacy constants
    DEFAULT_EPSILON = DEFAULT_HARVEST_EPSILON
    DEFAULT_STEPS = DEFAULT_HARVEST_STEPS
    BPS_ENERGY_SAFETY_MARGIN = HARVEST_SAFETY_MARGIN
    
    # Tags
    OPERATION_TAGS = {'bps_harvest': "BPS Energy Harvest"}
    STATE_TAGS = {'bps_saturated': "BPS Saturated"}
    
    BPS_CONFIG_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# BPS Integration
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from .bps_topology import bps_charge_extraction, compute_topological_charge
    BPS_AVAILABLE = True
except ImportError:
    def bps_charge_extraction(lattice_state, memory):
        """Fallback BPS charge extraction"""
        return np.copy(lattice_state), 0
    
    def compute_topological_charge(memory):
        """Fallback charge computation"""
        return 0
    
    BPS_AVAILABLE = False

def extract_energy_from_lattice(
    lattice,
    memory=None, 
    bps_aware: bool = True,
    preserve_phase: bool = True
) -> np.ndarray:
    """
    Direct energy extraction from lattice with BPS preservation.
    
    Args:
        lattice: Lattice system with psi field
        memory: Soliton memory for BPS integration
        bps_aware: Enable BPS-preserving extraction
        preserve_phase: Maintain phase relationships
        
    Returns:
        Extracted energy field
    """
    # Feature flag check
    if not ENABLE_BPS_ENERGY_HARVEST and BPS_CONFIG_AVAILABLE:
        logger.warning("Energy extraction disabled by feature flag")
        if STRICT_BPS_MODE:
            raise RuntimeError("Energy extraction required in strict BPS mode but disabled")
        bps_aware = False
    
    try:
        if not hasattr(lattice, 'psi'):
            logger.warning("Lattice has no psi field - returning zero array")
            return np.array([0.0])
        
        initial_Q = 0
        if bps_aware and BPS_AVAILABLE and memory and ENABLE_BPS_CHARGE_TRACKING:
            initial_Q = compute_topological_charge(memory)
            logger.debug(f"Pre-extraction charge: Q = {initial_Q}")
        
        # BPS-aware extraction
        if bps_aware and BPS_AVAILABLE and memory:
            extracted_field, final_Q = bps_charge_extraction(lattice.psi, memory)
            
            # Verify charge conservation
            charge_error = abs(final_Q - initial_Q)
            if charge_error > CHARGE_CONSERVATION_TOLERANCE:
                message = f"Charge conservation violated during extraction: ΔQ = {charge_error:.2e}"
                if STRICT_BPS_MODE:
                    raise RuntimeError(message)
                else:
                    logger.warning(message)
            
            logger.debug(f"BPS-aware extraction: Q conserved = {final_Q}")
        else:
            # Fallback extraction
            if preserve_phase:
                extracted_field = lattice.psi.copy()
            else:
                extracted_field = np.abs(lattice.psi)**2
            
            logger.debug("Standard energy extraction (no BPS awareness)")
        
        # Apply extraction efficiency
        extracted_field *= ENERGY_EXTRACTION_EFFICIENCY
        
        extracted_energy = np.sum(np.abs(extracted_field)**2)
        logger.info(f"Energy extracted: {extracted_energy:.3f}")
        
        return extracted_field
        
    except Exception as e:
        logger.error(f"Energy extraction failed: {e}")
        if STRICT_BPS_MODE:
            raise
        return np.array([0.0])

# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced Blowup Functions
# ═══════════════════════════════════════════════════════════════════════════════

def induce_blowup(
    lattice, 
    epsilon: float = None, 
    steps: int = None,
    memory=None,
    bps_aware: bool = True
) -> np.ndarray:
    """
    BPS-aware energy amplification and harvesting with topological charge conservation.
    
    This enhanced version replaces lattice.psi.copy() with bps_charge_extraction()
    to maintain topological integrity during energy capture.
    
    Args:
        lattice: Lattice system with psi field and step() method
        epsilon: Energy amplification factor per step (default from config)
        steps: Number of amplification steps (default from config)
        memory: Enhanced soliton memory for BPS integration (optional)
        bps_aware: Enable BPS-preserving extraction (default True)
        
    Returns:
        np.ndarray: Harvested energy field with preserved topological structure
    """
    # Use config defaults if not specified
    if epsilon is None:
        epsilon = DEFAULT_EPSILON
    if steps is None:
        steps = DEFAULT_STEPS
        
    # Feature flag checks
    if not ENABLE_BPS_ENERGY_HARVEST and BPS_CONFIG_AVAILABLE:
        logger.warning("BPS energy harvest disabled by feature flag")
        if STRICT_BPS_MODE:
            raise RuntimeError("Energy harvest required in strict BPS mode but disabled")
        bps_aware = False
    
    try:
        logger.info(f"Initiating {'BPS-aware' if bps_aware and BPS_AVAILABLE else 'standard'} blowup induction")
        logger.debug(f"Parameters: ε={epsilon}, steps={steps}")
        
        # Pre-blowup state assessment
        initial_norm = np.linalg.norm(lattice.psi) if hasattr(lattice, 'psi') else 0.0
        initial_Q = 0
        
        if bps_aware and BPS_AVAILABLE and memory and ENABLE_BPS_CHARGE_TRACKING:
            initial_Q = compute_topological_charge(memory)
            logger.debug(f"Initial topological charge: Q = {initial_Q}")
            
            # Adjust amplification to respect BPS bounds
            if initial_Q != 0:
                # Limit amplification to avoid excessive BPS bound violation
                max_safe_epsilon = min(epsilon, MAX_AMPLIFICATION_FACTOR / abs(initial_Q))
                if max_safe_epsilon < epsilon:
                    logger.info(f"Reducing ε to {max_safe_epsilon:.3f} for BPS safety")
                    epsilon = max_safe_epsilon
        
        # Controlled energy amplification with monitoring
        for step in range(steps):
            try:
                # Apply amplification
                amplification_factor = 1 + epsilon
                lattice.psi *= amplification_factor
                
                # Advance integration step
                lattice.step()
                
                # Monitor energy growth
                current_norm = np.linalg.norm(lattice.psi)
                growth_ratio = current_norm / max(initial_norm, 1e-10)
                
                logger.debug(f"Step {step+1}/{steps}: norm={current_norm:.3f}, growth={growth_ratio:.2f}x")
                
                # Safety check for runaway amplification
                if ENABLE_BPS_SAFETY_CHECKS and growth_ratio > MAX_AMPLIFICATION_FACTOR:
                    logger.warning(f"Excessive amplification detected at step {step+1}, stopping early")
                    if STRICT_BPS_MODE:
                        raise RuntimeError(f"Amplification exceeded safety limit: {growth_ratio:.2f}x")
                    break
                    
            except Exception as e:
                logger.warning(f"Amplification step {step+1} failed: {e}")
                if STRICT_BPS_MODE:
                    raise
                break
        
        # BPS-aware energy harvesting
        if bps_aware and BPS_AVAILABLE and memory:
            logger.info("Performing BPS charge extraction...")
            harvested, final_Q = bps_charge_extraction(lattice.psi, memory)
            
            # Verify charge conservation
            charge_error = abs(final_Q - initial_Q)
            if charge_error > CHARGE_CONSERVATION_TOLERANCE:
                message = f"Topological charge deviation: ΔQ = {charge_error:.2e}"
                if STRICT_BPS_MODE:
                    raise RuntimeError(message)
                else:
                    logger.warning(message)
            else:
                logger.debug(f"Topological charge conserved: Q = {final_Q}")
                
        else:
            # Fallback: traditional field copy
            logger.debug("Using traditional field extraction")
            harvested = lattice.psi.copy()
            final_Q = initial_Q
        
        # Safe lattice reset with charge awareness
        reset_lattice_safely(lattice, memory, preserve_topology=bps_aware and BPS_AVAILABLE)
        
        # Log harvest summary
        harvested_energy = np.sum(np.abs(harvested)**2)
        logger.info(f"Blowup harvest complete: E={harvested_energy:.3f}, Q={final_Q}")
        
        return harvested
        
    except Exception as e:
        logger.error(f"Blowup induction failed: {e}")
        if STRICT_BPS_MODE:
            raise
        # Emergency fallback
        if hasattr(lattice, 'psi'):
            return lattice.psi.copy()
        else:
            return np.array([0.0])

def extract_energy_from_lattice(
    lattice, 
    memory=None,
    extraction_efficiency: float = 0.95,
    preserve_charge: bool = True
) -> np.ndarray:
    """
    Direct energy extraction from lattice with BPS awareness.
    
    This function provides controlled energy extraction without amplification,
    suitable for steady-state energy harvesting scenarios.
    
    Args:
        lattice: Lattice system
        memory: Soliton memory for charge tracking
        extraction_efficiency: Fraction of energy to extract (0-1)
        preserve_charge: Maintain topological charge conservation
        
    Returns:
        np.ndarray: Extracted energy field
    """
    try:
        logger.debug(f"Extracting energy (efficiency={extraction_efficiency:.2f})")
        
        if not hasattr(lattice, 'psi'):
            logger.warning("Lattice has no psi field")
            return np.array([0.0])
        
        initial_energy = np.sum(np.abs(lattice.psi)**2)
        
        if preserve_charge and BPS_AVAILABLE and memory:
            # BPS-preserving extraction
            extracted_field, charge = bps_charge_extraction(lattice.psi, memory)
            
            # Scale by extraction efficiency
            extracted_field *= np.sqrt(extraction_efficiency)
            
            # Update lattice with remaining energy
            remaining_factor = np.sqrt(1 - extraction_efficiency)
            lattice.psi *= remaining_factor
            
            logger.debug(f"BPS extraction: Q={charge}, E_extracted={np.sum(np.abs(extracted_field)**2):.3f}")
            
        else:
            # Standard extraction
            extracted_field = lattice.psi.copy()
            extracted_field *= np.sqrt(extraction_efficiency)
            
            # Reduce lattice energy
            lattice.psi *= np.sqrt(1 - extraction_efficiency)
        
        final_energy = np.sum(np.abs(lattice.psi)**2)
        extracted_energy = np.sum(np.abs(extracted_field)**2)
        
        logger.debug(f"Energy extraction: {initial_energy:.3f} → {final_energy:.3f} (extracted: {extracted_energy:.3f})")
        
        return extracted_field
        
    except Exception as e:
        logger.error(f"Energy extraction failed: {e}")
        return np.array([0.0])

def reset_lattice_safely(
    lattice, 
    memory=None, 
    preserve_topology: bool = True,
    reset_factor: float = 0.0
) -> bool:
    """
    Safely reset lattice state while preserving topological information.
    
    Args:
        lattice: Lattice system to reset
        memory: Soliton memory for topology preservation
        preserve_topology: Maintain topological charge structure
        reset_factor: Multiplicative reset factor (0.0 = full reset)
        
    Returns:
        bool: True if reset successful
    """
    try:
        logger.debug(f"Resetting lattice (factor={reset_factor}, preserve_topology={preserve_topology})")
        
        if not hasattr(lattice, 'psi'):
            logger.warning("Lattice has no psi field to reset")
            return False
        
        if preserve_topology and BPS_AVAILABLE and memory:
            # Preserve topological structure during reset
            initial_Q = compute_topological_charge(memory)
            
            # Apply reset
            lattice.psi *= reset_factor
            
            # Verify topology preservation (charge should remain in memory)
            if memory and hasattr(memory, 'get_active_solitons'):
                # Update soliton amplitudes to reflect lattice reset
                solitons = memory.get_active_solitons()
                for soliton in solitons:
                    if hasattr(soliton, 'amplitude'):
                        soliton.amplitude *= reset_factor
                    elif isinstance(soliton, dict):
                        soliton['amplitude'] = soliton.get('amplitude', 1.0) * reset_factor
            
            logger.debug(f"Topology-preserving reset complete (Q={initial_Q} preserved)")
            
        else:
            # Standard reset
            lattice.psi *= reset_factor
            logger.debug("Standard lattice reset complete")
        
        return True
        
    except Exception as e:
        logger.error(f"Lattice reset failed: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Blowup Strategies
# ═══════════════════════════════════════════════════════════════════════════════

def adaptive_blowup_induction(
    lattice,
    memory=None,
    target_energy: float = None,
    max_steps: int = None,
    bps_aware: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Adaptive blowup that adjusts parameters based on system response.
    
    Args:
        lattice: Lattice system
        memory: Soliton memory
        target_energy: Desired energy level (default from config)
        max_steps: Maximum amplification steps (default from config)
        bps_aware: Enable BPS conservation
        
    Returns:
        Tuple of (harvested_field, metrics)
    """
    # Use config defaults if not specified
    if target_energy is None:
        target_energy = ENERGY_PER_Q * MAX_BPS_ENERGY_MULTIPLIER * 10  # Config-derived target
    if max_steps is None:
        max_steps = MAX_HARVEST_STEPS
    
    # Feature flag checks
    if not ENABLE_BPS_ADAPTIVE_SCALING and BPS_CONFIG_AVAILABLE:
        logger.warning("Adaptive scaling disabled - using fixed parameters")
        return induce_blowup(lattice, memory=memory, bps_aware=bps_aware), {'adaptive_disabled': True}
    
    try:
        logger.info(f"Starting adaptive blowup (target E={target_energy})")
        
        # Initialize metrics
        metrics = {
            'initial_energy': np.sum(np.abs(lattice.psi)**2) if hasattr(lattice, 'psi') else 0.0,
            'steps_taken': 0,
            'epsilon_used': DEFAULT_EPSILON,
            'target_achieved': False,
            'bps_conserved': True,
            'config_driven': BPS_CONFIG_AVAILABLE
        }
        
        initial_Q = 0
        if bps_aware and BPS_AVAILABLE and memory and ENABLE_BPS_CHARGE_TRACKING:
            initial_Q = compute_topological_charge(memory)
            metrics['initial_charge'] = initial_Q
        
        # Adaptive amplification with config-driven bounds
        epsilon = DEFAULT_EPSILON
        current_energy = metrics['initial_energy']
        
        for step in range(max_steps):
            if current_energy >= target_energy:
                metrics['target_achieved'] = True
                break
            
            # Adaptive epsilon adjustment using config bounds
            energy_ratio = current_energy / max(target_energy, 1e-10)
            if energy_ratio < 0.1:
                epsilon = min(ADAPTIVE_EPSILON_MAX, DEFAULT_EPSILON * 2)  # Boost for low energy
            elif energy_ratio > 0.8:
                epsilon = max(ADAPTIVE_EPSILON_MIN, DEFAULT_EPSILON * 0.5)  # Reduce near target
            
            # BPS safety check - respect charge-based limits
            if initial_Q != 0 and ENABLE_BPS_SAFETY_CHECKS:
                max_safe_epsilon = min(epsilon, MAX_AMPLIFICATION_FACTOR / abs(initial_Q))
                epsilon = max_safe_epsilon
            
            # Single amplification step
            try:
                lattice.psi *= (1 + epsilon)
                lattice.step()
                current_energy = np.sum(np.abs(lattice.psi)**2)
                metrics['steps_taken'] += 1
                
                logger.debug(f"Adaptive step {step+1}: E={current_energy:.3f}, ε={epsilon:.3f}")
                
                # Performance tracking
                if PERFORMANCE_PROFILING_ENABLED:
                    metrics[f'step_{step+1}_energy'] = current_energy
                    metrics[f'step_{step+1}_epsilon'] = epsilon
                
            except Exception as e:
                logger.warning(f"Amplification failed at step {step+1}: {e}")
                if STRICT_BPS_MODE:
                    raise
                break
        
        # Final harvest with BPS awareness
        if bps_aware and BPS_AVAILABLE and memory:
            harvested, final_Q = bps_charge_extraction(lattice.psi, memory)
            metrics['final_charge'] = final_Q
            metrics['charge_error'] = abs(final_Q - initial_Q)
            metrics['bps_conserved'] = metrics['charge_error'] < CHARGE_CONSERVATION_TOLERANCE
            
            if not metrics['bps_conserved'] and STRICT_BPS_MODE:
                raise RuntimeError(f"Charge conservation violated: ΔQ = {metrics['charge_error']:.2e}")
        else:
            harvested = lattice.psi.copy()
        
        # Update metrics
        metrics['final_energy'] = current_energy
        metrics['energy_gain'] = current_energy / max(metrics['initial_energy'], 1e-10)
        metrics['harvested_energy'] = np.sum(np.abs(harvested)**2)
        metrics['efficiency'] = min(1.0, metrics['harvested_energy'] / max(target_energy, 1e-10))
        
        # Safe reset
        reset_lattice_safely(lattice, memory, preserve_topology=bps_aware)
        
        logger.info(f"Adaptive blowup complete: {metrics['steps_taken']} steps, "
                   f"E={metrics['harvested_energy']:.3f}, efficiency={metrics['efficiency']:.1%}")
        
        return harvested, metrics
        
    except Exception as e:
        logger.error(f"Adaptive blowup failed: {e}")
        if STRICT_BPS_MODE:
            raise
        return np.array([0.0]), {'error': str(e)}

def multi_stage_harvest(
    lattice,
    memory=None,
    stages: int = None,
    stage_steps: int = None,
    bps_aware: bool = True
) -> List[np.ndarray]:
    """
    Perform multi-stage energy harvesting for gradual accumulation.
    
    Args:
        lattice: Lattice system
        memory: Soliton memory
        stages: Number of harvest stages (default from config)
        stage_steps: Steps per stage (default from config)
        bps_aware: Enable BPS conservation
        
    Returns:
        List of harvested energy arrays from each stage
    """
    # Use config defaults if not specified
    if stages is None:
        stages = 3  # Reasonable default
    if stage_steps is None:
        stage_steps = DEFAULT_STEPS // 2  # Half the standard steps per stage
        
    # Feature flag check
    if not ENABLE_BPS_ENERGY_HARVEST and BPS_CONFIG_AVAILABLE:
        logger.warning("Multi-stage harvest disabled by feature flag")
        if STRICT_BPS_MODE:
            raise RuntimeError("Energy harvest required in strict BPS mode but disabled")
        return [np.array([0.0])]
    
    try:
        logger.info(f"Starting multi-stage harvest ({stages} stages, {stage_steps} steps/stage)")
        
        harvested_stages = []
        total_harvested_energy = 0.0
        
        # Track charge conservation across stages
        initial_Q = 0
        if bps_aware and BPS_AVAILABLE and memory and ENABLE_BPS_CHARGE_TRACKING:
            initial_Q = compute_topological_charge(memory)
            logger.debug(f"Initial Q for multi-stage harvest: {initial_Q}")
        
        for stage in range(stages):
            logger.debug(f"Stage {stage+1}/{stages}")
            
            # Progressive epsilon reduction for controlled harvesting
            stage_epsilon = DEFAULT_EPSILON * (1.0 - 0.3 * stage / stages)
            
            # Respect config bounds
            stage_epsilon = max(ADAPTIVE_EPSILON_MIN, min(ADAPTIVE_EPSILON_MAX, stage_epsilon))
            
            # Stage-specific blowup with BPS safety
            stage_harvest = induce_blowup(
                lattice,
                epsilon=stage_epsilon,
                steps=stage_steps,
                memory=memory,
                bps_aware=bps_aware
            )
            
            harvested_stages.append(stage_harvest)
            stage_energy = np.sum(np.abs(stage_harvest)**2)
            total_harvested_energy += stage_energy
            
            logger.debug(f"Stage {stage+1} harvest: E={stage_energy:.3f}, ε={stage_epsilon:.3f}")
            
            # BPS safety check between stages
            if bps_aware and BPS_AVAILABLE and memory and ENABLE_BPS_SAFETY_CHECKS:
                current_Q = compute_topological_charge(memory)
                charge_drift = abs(current_Q - initial_Q)
                
                if charge_drift > CHARGE_CONSERVATION_TOLERANCE * stages:  # Allow slight drift over stages
                    message = f"Charge drift detected at stage {stage+1}: ΔQ = {charge_drift:.2e}"
                    if STRICT_BPS_MODE:
                        raise RuntimeError(message)
                    else:
                        logger.warning(message)
            
            # Brief recovery between stages (allow lattice to stabilize)
            if stage < stages - 1:
                try:
                    for _ in range(2):
                        lattice.step()
                except Exception as e:
                    logger.warning(f"Recovery step failed after stage {stage+1}: {e}")
                    if STRICT_BPS_MODE:
                        raise
        
        # Final verification
        if bps_aware and BPS_AVAILABLE and memory and ENABLE_BPS_CHARGE_TRACKING:
            final_Q = compute_topological_charge(memory)
            total_charge_error = abs(final_Q - initial_Q)
            
            if total_charge_error > CHARGE_CONSERVATION_TOLERANCE * stages:
                message = f"Total charge conservation violation: ΔQ = {total_charge_error:.2e}"
                if STRICT_BPS_MODE:
                    raise RuntimeError(message)
                else:
                    logger.warning(message)
        
        logger.info(f"Multi-stage harvest complete: {stages} stages, total E={total_harvested_energy:.3f}")
        
        # Add performance metrics to the last harvest if profiling enabled
        if PERFORMANCE_PROFILING_ENABLED and harvested_stages:
            # Store metrics as metadata (could be extended to return metrics dict)
            logger.debug(f"Performance: avg E/stage = {total_harvested_energy/stages:.3f}")
        
        return harvested_stages
        
    except Exception as e:
        logger.error(f"Multi-stage harvest failed: {e}")
        if STRICT_BPS_MODE:
            raise
        return [np.array([0.0])]

def reset_lattice_safely(
    lattice,
    memory=None,
    preserve_topology: bool = True
):
    """
    Safely reset lattice state while preserving topological structure.
    
    Args:
        lattice: Lattice system to reset
        memory: Soliton memory for topology preservation
        preserve_topology: Whether to maintain topological charge
    """
    try:
        if preserve_topology and memory and BPS_AVAILABLE and ENABLE_BPS_CHARGE_TRACKING:
            # Preserve topological structure during reset
            initial_Q = compute_topological_charge(memory)
            
            # Gentle lattice reset
            if hasattr(lattice, 'psi'):
                # Scale down amplitude but preserve phase structure
                lattice.psi *= 0.1
                logger.debug(f"Lattice reset with topology preservation (Q={initial_Q})")
            
            # Verify topology preservation
            if ENABLE_BPS_SAFETY_CHECKS:
                final_Q = compute_topological_charge(memory)
                if abs(final_Q - initial_Q) > CHARGE_CONSERVATION_TOLERANCE:
                    if STRICT_BPS_MODE:
                        raise RuntimeError(f"Topology not preserved during reset: {initial_Q} → {final_Q}")
                    else:
                        logger.warning(f"Topology change during reset: {initial_Q} → {final_Q}")
        else:
            # Standard reset without topology preservation
            if hasattr(lattice, 'psi'):
                lattice.psi.fill(0.1)  # Small non-zero value for numerical stability
                logger.debug("Standard lattice reset (no topology preservation)")
                
    except Exception as e:
        logger.error(f"Lattice reset failed: {e}")
        if STRICT_BPS_MODE:
            raise
        # Emergency fallback
        if hasattr(lattice, 'psi'):
            lattice.psi.fill(0.01)

def profile_blowup_performance(
    lattice,
    memory=None,
    test_params: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Profile blowup performance with various parameter combinations.
    
    Args:
        lattice: Test lattice system
        memory: Test memory system
        test_params: Custom test parameters
        
    Returns:
        Dictionary with performance metrics
    """
    if not PERFORMANCE_PROFILING_ENABLED:
        logger.debug("Performance profiling disabled")
        return {'profiling_disabled': True}
        
    try:
        import time
        
        # Default test parameters
        default_params = {
            'epsilon_values': [0.1, 0.3, 0.5, 0.7],
            'step_counts': [5, 10, 15, 20],
            'trials_per_config': 3
        }
        
        params = {**default_params, **(test_params or {})}
        results = {}
        
        logger.info("Starting blowup performance profiling...")
        
        for epsilon in params['epsilon_values']:
            for steps in params['step_counts']:
                config_key = f"ε{epsilon}_s{steps}"
                times = []
                
                for trial in range(params['trials_per_config']):
                    # Reset for clean trial
                    if hasattr(lattice, 'psi'):
                        lattice.psi.fill(0.1 + 0.01 * trial)  # Slight variation
                    
                    start_time = time.time()
                    try:
                        harvested = induce_blowup(
                            lattice, 
                            epsilon=epsilon, 
                            steps=steps, 
                            memory=memory,
                            bps_aware=True
                        )
                        end_time = time.time()
                        times.append(end_time - start_time)
                        
                    except Exception as e:
                        logger.warning(f"Trial failed for {config_key}: {e}")
                        continue
                
                if times:
                    results[config_key] = {
                        'avg_time': np.mean(times),
                        'min_time': np.min(times),
                        'max_time': np.max(times),
                        'std_time': np.std(times),
                        'trials': len(times)
                    }
                    
        # Summary statistics
        if results:
            all_times = [r['avg_time'] for r in results.values()]
            results['summary'] = {
                'fastest_config': min(results.keys(), key=lambda k: results[k]['avg_time']),
                'slowest_config': max(results.keys(), key=lambda k: results[k]['avg_time']),
                'overall_avg': np.mean(all_times),
                'config_count': len(results) - 1  # Exclude summary itself
            }
            
        logger.info(f"Performance profiling complete: {len(results)-1} configurations tested")
        return results
        
    except Exception as e:
        logger.error(f"Performance profiling failed: {e}")
        return {'error': str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
# Legacy Compatibility Layer
# ═══════════════════════════════════════════════════════════════════════════════

def induce_blowup_legacy(lattice, epsilon: float = 0.5, steps: int = 10):
    """
    Legacy blowup function for backward compatibility.
    
    This maintains the original interface but logs a deprecation warning.
    New code should use induce_blowup() with BPS awareness.
    """
    logger.warning("Using legacy blowup function - consider upgrading to BPS-aware version")
    return induce_blowup(lattice, epsilon, steps, memory=None, bps_aware=False)

def extract_energy_legacy(lattice):
    """
    Legacy energy extraction without BPS awareness.
    """
    logger.warning("Using legacy energy extraction - no BPS protection")
    return extract_energy_from_lattice(lattice, memory=None, bps_aware=False)

# Alias for very old code
blowup_induction = induce_blowup_legacy
energy_extraction = extract_energy_legacy

# ═══════════════════════════════════════════════════════════════════════════════
# Module Validation and Health Checks
# ═══════════════════════════════════════════════════════════════════════════════

def validate_blowup_harness() -> Dict[str, Any]:
    """
    Comprehensive validation of blowup harness functionality.
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        'status': 'unknown',
        'bps_integration': BPS_AVAILABLE,
        'config_available': BPS_CONFIG_AVAILABLE,
        'features': {
            'adaptive_scaling': ENABLE_BPS_ADAPTIVE_SCALING if BPS_CONFIG_AVAILABLE else False,
            'energy_harvest': ENABLE_BPS_ENERGY_HARVEST if BPS_CONFIG_AVAILABLE else True,
            'safety_checks': ENABLE_BPS_SAFETY_CHECKS if BPS_CONFIG_AVAILABLE else True,
            'performance_profiling': PERFORMANCE_PROFILING_ENABLED if BPS_CONFIG_AVAILABLE else False
        },
        'configuration': {},
        'issues': []
    }
    
    try:
        # Validate configuration parameters
        if BPS_CONFIG_AVAILABLE:
            validation['configuration'] = {
                'default_epsilon': DEFAULT_EPSILON,
                'default_steps': DEFAULT_STEPS,
                'max_amplification': MAX_AMPLIFICATION_FACTOR,
                'energy_extraction_efficiency': ENERGY_EXTRACTION_EFFICIENCY,
                'strict_mode': STRICT_BPS_MODE
            }
            
            # Parameter range checks
            if not (0.01 <= DEFAULT_EPSILON <= 2.0):
                validation['issues'].append(f"Default epsilon out of range: {DEFAULT_EPSILON}")
                
            if not (1 <= DEFAULT_STEPS <= 100):
                validation['issues'].append(f"Default steps out of range: {DEFAULT_STEPS}")
                
            if not (2.0 <= MAX_AMPLIFICATION_FACTOR <= 100.0):
                validation['issues'].append(f"Max amplification factor suspicious: {MAX_AMPLIFICATION_FACTOR}")
        
        # Test basic functionality
        try:
            # Create minimal test lattice
            test_psi = np.array([0.1 + 0.1j, 0.2 + 0.0j])
            
            class TestLattice:
                def __init__(self):
                    self.psi = test_psi.copy()
                def step(self):
                    pass  # Minimal step function
            
            test_lattice = TestLattice()
            
            # Test basic blowup
            result = induce_blowup(
                test_lattice, 
                epsilon=0.1, 
                steps=2, 
                memory=None, 
                bps_aware=False
            )
            
            if isinstance(result, np.ndarray) and len(result) > 0:
                validation['basic_functionality'] = True
            else:
                validation['issues'].append("Basic blowup test failed")
                validation['basic_functionality'] = False
                
        except Exception as e:
            validation['issues'].append(f"Functionality test error: {e}")
            validation['basic_functionality'] = False
        
        # Overall status
        if not validation['issues']:
            validation['status'] = 'healthy'
        elif len(validation['issues']) <= 2:
            validation['status'] = 'warnings'
        else:
            validation['status'] = 'issues'
            
        logger.debug(f"Blowup harness validation: {validation['status']}")
        return validation
        
    except Exception as e:
        validation['status'] = 'error'
        validation['issues'].append(f"Validation failed: {e}")
        logger.error(f"Blowup harness validation error: {e}")
        return validation

# Export key functions for easy importing
__all__ = [
    # Core functions
    'induce_blowup',
    'extract_energy_from_lattice', 
    'adaptive_blowup_induction',
    'multi_stage_harvest',
    'reset_lattice_safely',
    
    # Profiling and validation
    'profile_blowup_performance',
    'validate_blowup_harness',
    
    # Legacy compatibility
    'induce_blowup_legacy',
    'extract_energy_legacy',
    'blowup_induction',  # Alias
    'energy_extraction',  # Alias
    
    # Configuration flags
    'BPS_AVAILABLE',
    'BPS_CONFIG_AVAILABLE'
]

if __name__ == "__main__":
    # Module verification and health check
    logger.info("BPS-Aware Blowup Harness loaded successfully")
    logger.info("Key functions:")
    logger.info("  • induce_blowup() - BPS-preserving energy amplification")
    logger.info("  • extract_energy_from_lattice() - Direct energy extraction")
    logger.info("  • adaptive_blowup_induction() - Smart parameter adjustment")
    logger.info("  • multi_stage_harvest() - Gradual energy accumulation")
    logger.info("  • reset_lattice_safely() - Topology-preserving reset")
    logger.info("  • profile_blowup_performance() - Performance benchmarking")
    logger.info(f"BPS integration: {'ENABLED' if BPS_AVAILABLE else 'DISABLED'}")
    logger.info(f"Config available: {'YES' if BPS_CONFIG_AVAILABLE else 'NO'}")
    
    # Run validation if requested
    import sys
    if '--validate' in sys.argv or '--health-check' in sys.argv:
        logger.info("Running comprehensive validation...")
        validation = validate_blowup_harness()
        logger.info(f"Validation status: {validation['status'].upper()}")
        
        if validation['issues']:
            logger.warning("Issues detected:")
            for issue in validation['issues']:
                logger.warning(f"  - {issue}")
        
        if validation['status'] == 'healthy':
            logger.info("All systems operational - ready for BPS energy harvesting!")
    
    logger.info("Blowup harness initialization complete")
