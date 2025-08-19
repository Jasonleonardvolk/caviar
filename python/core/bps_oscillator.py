# bps_oscillator_with_sponge.py
# -------------------------------------------------
# BPS-aware oscillator class with Phase Sponge implementation
# Includes boundary absorbing layers to prevent edge reflections
# PRODUCTION VERSION - Fully integrated with centralized BPS config
# -------------------------------------------------

import numpy as np
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Import centralized BPS configuration
try:
    from .bps_config import (
        # Feature flags
        ENABLE_BPS_PHASE_LOCKING, ENABLE_BPS_CHARGE_TRACKING, ENABLE_BPS_SAFETY_CHECKS,
        STRICT_BPS_MODE, ENABLE_DETAILED_LOGGING,
        
        # Phase locking parameters
        BPS_PHASE_LOCK_GAIN, KURAMOTO_COUPLING_STRENGTH, MAX_PHASE_CORRECTION,
        PHASE_COHERENCE_THRESHOLD,
        
        # Phase sponge parameters
        ENABLE_PHASE_SPONGE, PHASE_SPONGE_DAMPING_FACTOR,
        PHASE_SPONGE_BOUNDARY_WIDTH, PHASE_SPONGE_PROFILE,
        
        # Charge parameters
        ALLOWED_Q_VALUES, MAX_ALLOWED_CHARGE_MAGNITUDE, CHARGE_QUANTIZATION_THRESHOLD,
        
        # Tolerances
        CHARGE_CONSERVATION_TOLERANCE, BPS_BOUND_VIOLATION_TOLERANCE,
        
        # Symbolic tags
        SOLITON_TAGS, STATE_TAGS,
        
        # Performance
        SLOW_OPERATION_THRESHOLD, PERFORMANCE_PROFILING_ENABLED
    )
    BPS_CONFIG_AVAILABLE = True
    logger = logging.getLogger("BPSOscillator")
    logger.info("BPS oscillator using centralized configuration with Phase Sponge")
    
except ImportError:
    logger = logging.getLogger("BPSOscillator")
    logger.warning("BPS config unavailable - using fallback constants")
    
    # Feature flags (conservative defaults)
    ENABLE_BPS_PHASE_LOCKING = True
    ENABLE_BPS_CHARGE_TRACKING = True
    ENABLE_BPS_SAFETY_CHECKS = True
    STRICT_BPS_MODE = False
    ENABLE_DETAILED_LOGGING = True
    
    # Phase locking parameters
    BPS_PHASE_LOCK_GAIN = 0.5
    KURAMOTO_COUPLING_STRENGTH = 0.3
    MAX_PHASE_CORRECTION = 0.5
    PHASE_COHERENCE_THRESHOLD = 0.1
    
    # Phase sponge defaults
    ENABLE_PHASE_SPONGE = True
    PHASE_SPONGE_DAMPING_FACTOR = 0.95
    PHASE_SPONGE_BOUNDARY_WIDTH = 5
    PHASE_SPONGE_PROFILE = "tanh"
    
    # Charge parameters
    ALLOWED_Q_VALUES = {-2, -1, 0, 1, 2}
    MAX_ALLOWED_CHARGE_MAGNITUDE = 2
    CHARGE_QUANTIZATION_THRESHOLD = 0.5
    
    # Tolerances
    CHARGE_CONSERVATION_TOLERANCE = 1e-10
    BPS_BOUND_VIOLATION_TOLERANCE = 1e-6
    
    # Tags
    SOLITON_TAGS = {'bright_bps': "Bright BPS", 'dark_bps': "Dark BPS"}
    STATE_TAGS = {'phase_locked': "Phase Locked"}
    
    # Performance
    SLOW_OPERATION_THRESHOLD = 1.0
    PERFORMANCE_PROFILING_ENABLED = False
    
    BPS_CONFIG_AVAILABLE = False

class BPSOscillator:
    """
    Extension of base oscillator with BPS tagging, phase tracking,
    topological charge integration, and Phase Sponge damping.
    
    Production-grade oscillator with centralized BPS configuration,
    feature flag controls, comprehensive error handling, and
    boundary absorbing layers for edge reflection prevention.
    """
    
    def __init__(self, index: int, theta: float = 0.0, omega: float = 0.0,
                 enable_bps_features: Optional[bool] = None,
                 position: Optional[Tuple[float, float]] = None,
                 lattice_bounds: Optional[Tuple[int, int]] = None):
        """
        Initialize BPS-aware oscillator with config-driven behavior and phase sponge.
        
        Args:
            index: Oscillator index/ID
            theta: Initial phase
            omega: Natural frequency  
            enable_bps_features: Override BPS feature activation
            position: (x, y) position in lattice for boundary detection
            lattice_bounds: (width, height) of lattice for phase sponge calculation
        """
        self.index = index
        self.theta = theta        # Phase
        self.omega = omega        # Natural frequency
        self.charge = 0           # Default: not a soliton
        self.phase_locked = False
        self.symbolic_tag = None
        
        # Position and boundary information for phase sponge
        self.position = position if position else (0, 0)
        self.lattice_bounds = lattice_bounds if lattice_bounds else (100, 100)
        self.is_boundary = False
        self.sponge_damping = 1.0  # No damping by default
        
        # BPS feature control
        if enable_bps_features is None:
            self.bps_features_enabled = BPS_CONFIG_AVAILABLE
        else:
            self.bps_features_enabled = enable_bps_features
            
        # Configuration-driven initialization
        self.phase_lock_gain = BPS_PHASE_LOCK_GAIN
        self.coupling_strength = KURAMOTO_COUPLING_STRENGTH
        self.max_phase_correction = MAX_PHASE_CORRECTION
        
        # Phase sponge parameters
        self.phase_sponge_enabled = ENABLE_PHASE_SPONGE
        self.phase_sponge_damping_factor = PHASE_SPONGE_DAMPING_FACTOR
        self.phase_sponge_boundary_width = PHASE_SPONGE_BOUNDARY_WIDTH
        self.phase_sponge_profile = PHASE_SPONGE_PROFILE
        
        # Calculate initial sponge damping based on position
        if self.phase_sponge_enabled:
            self._update_sponge_damping()
        
        # Performance tracking
        self.operation_count = 0
        self.phase_lock_count = 0
        self.last_operation_time = 0.0
        self.damping_applications = 0
        
        if ENABLE_DETAILED_LOGGING:
            logger.debug(f"BPS oscillator {index} initialized: θ={theta:.3f}, ω={omega:.3f}, "
                        f"BPS={'ON' if self.bps_features_enabled else 'OFF'}, "
                        f"PhSponge={'ON' if self.phase_sponge_enabled else 'OFF'}")

    def _update_sponge_damping(self):
        """
        Calculate phase sponge damping based on distance from boundary.
        Uses different profiles (linear, quadratic, tanh) for smooth damping.
        """
        if not self.phase_sponge_enabled:
            self.sponge_damping = 1.0
            return
        
        x, y = self.position
        width, height = self.lattice_bounds
        
        # Calculate minimum distance to any boundary
        dist_to_left = x
        dist_to_right = width - 1 - x
        dist_to_bottom = y
        dist_to_top = height - 1 - y
        
        min_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
        
        # Check if within boundary layer
        if min_dist < self.phase_sponge_boundary_width:
            self.is_boundary = True
            
            # Normalized distance within boundary layer (0 at edge, 1 at inner edge)
            norm_dist = min_dist / self.phase_sponge_boundary_width
            
            # Apply damping profile
            if self.phase_sponge_profile == "linear":
                # Linear damping from full at edge to none at boundary width
                damping_strength = 1.0 - norm_dist
            elif self.phase_sponge_profile == "quadratic":
                # Quadratic damping for smoother transition
                damping_strength = (1.0 - norm_dist) ** 2
            elif self.phase_sponge_profile == "tanh":
                # Hyperbolic tangent for very smooth transition
                damping_strength = 0.5 * (1.0 - np.tanh(3.0 * (norm_dist - 0.5)))
            else:
                # Default to linear
                damping_strength = 1.0 - norm_dist
            
            # Apply damping factor (1.0 = no damping, smaller = more damping)
            self.sponge_damping = 1.0 - damping_strength * (1.0 - self.phase_sponge_damping_factor)
            
            if ENABLE_DETAILED_LOGGING and self.damping_applications % 100 == 0:
                logger.debug(f"[PhSponge] Osc {self.index} at ({x},{y}): "
                           f"dist={min_dist}, damping={self.sponge_damping:.3f}")
        else:
            self.is_boundary = False
            self.sponge_damping = 1.0

    def set_position(self, position: Tuple[float, float], 
                    lattice_bounds: Optional[Tuple[int, int]] = None):
        """
        Update oscillator position and recalculate phase sponge damping.
        
        Args:
            position: New (x, y) position
            lattice_bounds: Optional new lattice bounds
        """
        self.position = position
        if lattice_bounds:
            self.lattice_bounds = lattice_bounds
        
        if self.phase_sponge_enabled:
            self._update_sponge_damping()

    def assign_bps_charge(self, q: int):
        """
        Assigns topological charge to oscillator with config validation.
        
        Args:
            q: Raw charge value (will be quantized to ±1)
        """
        # Feature flag check
        if not ENABLE_BPS_CHARGE_TRACKING:
            if ENABLE_DETAILED_LOGGING:
                logger.debug(f"Charge tracking disabled - ignoring charge assignment for osc {self.index}")
            return
            
        # Quantize charge using config threshold
        if abs(q) > CHARGE_QUANTIZATION_THRESHOLD:
            quantized_charge = int(np.sign(q))
        else:
            quantized_charge = 0
            
        # Validate against allowed values
        if ENABLE_BPS_SAFETY_CHECKS and quantized_charge not in ALLOWED_Q_VALUES:
            if STRICT_BPS_MODE:
                raise ValueError(f"Invalid charge {quantized_charge} not in {ALLOWED_Q_VALUES}")
            else:
                logger.warning(f"Charge {quantized_charge} not in allowed values {ALLOWED_Q_VALUES}")
                # Clamp to allowed range
                quantized_charge = max(-MAX_ALLOWED_CHARGE_MAGNITUDE, 
                                     min(MAX_ALLOWED_CHARGE_MAGNITUDE, quantized_charge))
        
        old_charge = self.charge
        self.charge = quantized_charge
        
        # Update symbolic tag based on charge
        if quantized_charge > 0:
            self.symbolic_tag = SOLITON_TAGS.get('bright_bps', 'Positive Soliton')
        elif quantized_charge < 0:
            self.symbolic_tag = SOLITON_TAGS.get('dark_bps', 'Negative Soliton')
        else:
            self.symbolic_tag = None
            
        if ENABLE_DETAILED_LOGGING:
            logger.debug(f"[BPS-Osc] Oscillator {self.index} charge: {old_charge} → {self.charge} "
                        f"(tag: {self.symbolic_tag})")

    def set_symbolic_id(self, label: str):
        """
        Attach human-readable label to oscillator.
        
        Args:
            label: Symbolic identifier for debugging/UI
        """
        self.symbolic_tag = label
        
        if ENABLE_DETAILED_LOGGING:
            logger.debug(f"[BPS-Osc] Oscillator {self.index} tagged as '{label}'")

    def lock_to_phase(self, target_phase: float, gain: Optional[float] = None):
        """
        Applies soft phase-locking to a BPS soliton-defined target phase.
        This nudges the oscillator toward stable solitonic alignment.
        
        Args:
            target_phase: Target phase for locking
            gain: Phase lock strength (default from config)
        """
        # Feature flag check
        if not ENABLE_BPS_PHASE_LOCKING:
            if ENABLE_DETAILED_LOGGING:
                logger.debug(f"Phase locking disabled - ignoring lock request for osc {self.index}")
            return
            
        # Use config default if not specified
        if gain is None:
            gain = self.phase_lock_gain
            
        # Apply phase sponge damping to gain if in boundary layer
        if self.phase_sponge_enabled and self.is_boundary:
            gain *= self.sponge_damping
            
        # Compute phase difference
        delta = target_phase - self.theta
        
        # Normalize delta to [-π, π]
        while delta > np.pi:
            delta -= 2 * np.pi
        while delta < -np.pi:
            delta += 2 * np.pi
            
        # Apply correction with config-driven limits
        correction = gain * np.sin(delta)
        
        # Limit correction magnitude using config
        if abs(correction) > self.max_phase_correction:
            correction = np.sign(correction) * self.max_phase_correction
            if ENABLE_DETAILED_LOGGING:
                logger.debug(f"Phase correction limited to {self.max_phase_correction}")
        
        old_theta = self.theta
        self.theta += correction
        
        # Check if phase lock achieved
        if abs(delta) < PHASE_COHERENCE_THRESHOLD:
            if not self.phase_locked:
                self.phase_locked = True
                if ENABLE_DETAILED_LOGGING:
                    logger.debug(f"[BPS-Osc] Oscillator {self.index} achieved phase lock")
        else:
            self.phase_locked = False
            
        self.phase_lock_count += 1
        
        if ENABLE_DETAILED_LOGGING:
            logger.debug(f"[BPS-Osc] Phase lock {self.index}: θ {old_theta:.3f}→{self.theta:.3f} "
                        f"(Δ={delta:.3f}, correction={correction:.3f}, sponge={self.sponge_damping:.3f})")

    def step(self, dt: float = 1.0, external_coupling: float = 0.0):
        """
        Advance oscillator by one time step with config-driven behavior and phase sponge.
        
        Args:
            dt: Time step size
            external_coupling: External coupling term (e.g., from other oscillators)
        """
        start_time = time.time() if PERFORMANCE_PROFILING_ENABLED else 0.0
        
        # Natural evolution (unless strictly phase-locked)
        if not self.phase_locked:
            # Apply phase sponge damping to frequency evolution
            effective_omega = self.omega
            if self.phase_sponge_enabled and self.is_boundary:
                effective_omega *= self.sponge_damping
                self.damping_applications += 1
            
            self.theta += effective_omega * dt
        
        # Add external coupling with config-driven strength
        if external_coupling != 0.0:
            # Apply phase sponge to coupling as well
            effective_coupling_strength = self.coupling_strength
            if self.phase_sponge_enabled and self.is_boundary:
                effective_coupling_strength *= self.sponge_damping
            
            coupling_effect = effective_coupling_strength * external_coupling * dt
            self.theta += coupling_effect
            
            if ENABLE_DETAILED_LOGGING and abs(coupling_effect) > PHASE_COHERENCE_THRESHOLD:
                logger.debug(f"[BPS-Osc] Oscillator {self.index} coupling effect: {coupling_effect:.3f} "
                           f"(sponge={self.sponge_damping:.3f})")
        
        # Normalize phase to [0, 2π]
        self.theta = self.theta % (2 * np.pi)
        
        # Update performance tracking
        self.operation_count += 1
        
        if PERFORMANCE_PROFILING_ENABLED:
            operation_time = time.time() - start_time
            self.last_operation_time = operation_time
            
            # Warn about slow operations
            if operation_time > SLOW_OPERATION_THRESHOLD / 1000:  # Convert to milliseconds
                logger.warning(f"[BPS-Osc] Slow operation on oscillator {self.index}: {operation_time*1000:.2f}ms")
            
            # Periodic performance summary
            if self.operation_count % 10000 == 0:
                logger.debug(f"[BPS-Osc] Oscillator {self.index} completed {self.operation_count} operations, "
                           f"damping applied {self.damping_applications} times")

    def kuramoto_coupling(self, other_oscillators: List['BPSOscillator'], 
                         coupling_strength: Optional[float] = None) -> float:
        """
        Compute Kuramoto coupling term from neighboring oscillators with phase sponge.
        
        Args:
            other_oscillators: List of coupled oscillators
            coupling_strength: Override coupling strength
            
        Returns:
            Coupling term for use in step()
        """
        if not other_oscillators:
            return 0.0
            
        if coupling_strength is None:
            coupling_strength = self.coupling_strength
            
        # Apply phase sponge damping to coupling calculation
        if self.phase_sponge_enabled and self.is_boundary:
            coupling_strength *= self.sponge_damping
            
        # Standard Kuramoto coupling: K * Σ sin(θⱼ - θᵢ)
        coupling_sum = 0.0
        for other in other_oscillators:
            if other.index != self.index:  # Don't couple to self
                phase_diff = other.theta - self.theta
                
                # Apply distance-based weighting if both have positions
                weight = 1.0
                if hasattr(other, 'position') and self.position and other.position:
                    # Simple inverse distance weighting
                    dx = other.position[0] - self.position[0]
                    dy = other.position[1] - self.position[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    if distance > 0:
                        weight = 1.0 / (1.0 + distance)
                
                coupling_sum += weight * np.sin(phase_diff)
                
        # Normalize by number of neighbors
        if len(other_oscillators) > 1:
            coupling_sum /= (len(other_oscillators) - 1)
            
        return coupling_strength * coupling_sum

    def get_phase_coherence(self, other_oscillators: List['BPSOscillator']) -> float:
        """
        Compute phase coherence with other oscillators.
        
        Args:
            other_oscillators: List of oscillators to check coherence with
            
        Returns:
            Phase coherence measure [0, 1]
        """
        if not other_oscillators:
            return 1.0
            
        # Compute complex order parameter
        exp_sum = sum(np.exp(1j * osc.theta) for osc in other_oscillators + [self])
        coherence = abs(exp_sum) / (len(other_oscillators) + 1)
        
        return coherence

    def snapshot(self) -> Dict[str, Any]:
        """Return comprehensive state snapshot for introspection or export."""
        snapshot = {
            "index": self.index,
            "theta": self.theta,
            "omega": self.omega,
            "charge": self.charge,
            "symbol": self.symbolic_tag,
            "phase_locked": self.phase_locked,
            "bps_features_enabled": self.bps_features_enabled,
            "phase_sponge_enabled": self.phase_sponge_enabled,
            "position": self.position,
            "is_boundary": self.is_boundary,
            "sponge_damping": self.sponge_damping,
            "operation_count": self.operation_count,
            "phase_lock_count": self.phase_lock_count,
            "damping_applications": self.damping_applications
        }
        
        # Add performance data if available
        if PERFORMANCE_PROFILING_ENABLED:
            snapshot.update({
                "last_operation_time": self.last_operation_time,
                "avg_operations_per_lock": (self.phase_lock_count / max(self.operation_count, 1))
            })
            
        return snapshot

    def reset(self, theta: Optional[float] = None, omega: Optional[float] = None,
             preserve_charge: bool = True):
        """
        Reset oscillator state with optional parameter updates.
        
        Args:
            theta: New phase (None to keep current)
            omega: New frequency (None to keep current)
            preserve_charge: Whether to keep current topological charge
        """
        if theta is not None:
            self.theta = theta
        if omega is not None:
            self.omega = omega
            
        # Reset state
        self.phase_locked = False
        
        if not preserve_charge:
            self.charge = 0
            self.symbolic_tag = None
            
        # Reset performance counters
        self.operation_count = 0
        self.phase_lock_count = 0
        self.last_operation_time = 0.0
        self.damping_applications = 0
        
        # Recalculate sponge damping
        if self.phase_sponge_enabled:
            self._update_sponge_damping()
        
        if ENABLE_DETAILED_LOGGING:
            logger.debug(f"[BPS-Osc] Oscillator {self.index} reset: θ={self.theta:.3f}, ω={self.omega:.3f}")

    def __repr__(self):
        status = "🔒" if self.phase_locked else "~"
        charge_symbol = {-1: "⊖", 0: "○", 1: "⊕"}.get(self.charge, str(self.charge))
        boundary_symbol = "🧽" if self.is_boundary else ""
        return f"<BPSOscillator #{self.index} θ={self.theta:.2f} ω={self.omega:.2f} Q={charge_symbol} {status}{boundary_symbol}>"

# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def create_bps_oscillator_network(count: int, omega_range: Tuple[float, float] = (0.5, 1.5),
                                  charge_density: float = 0.1,
                                  lattice_size: Optional[Tuple[int, int]] = None) -> List[BPSOscillator]:
    """
    Create a network of BPS oscillators with random parameters and phase sponge.
    
    Args:
        count: Number of oscillators
        omega_range: Range for random frequencies
        charge_density: Fraction of oscillators with non-zero charge
        lattice_size: (width, height) for phase sponge calculation
        
    Returns:
        List of configured BPS oscillators
    """
    oscillators = []
    
    # Determine lattice dimensions
    if lattice_size:
        width, height = lattice_size
    else:
        # Assume square lattice
        side = int(np.sqrt(count))
        width = height = side
    
    for i in range(count):
        # Calculate position on lattice
        x = i % width
        y = i // width
        
        # Random initial conditions
        theta = np.random.uniform(0, 2 * np.pi)
        omega = np.random.uniform(*omega_range)
        
        osc = BPSOscillator(i, theta, omega, 
                           position=(x, y), 
                           lattice_bounds=(width, height))
        
        # Assign charges to some oscillators
        if np.random.random() < charge_density:
            charge = np.random.choice([-1, 1])
            osc.assign_bps_charge(charge)
            
        oscillators.append(osc)
    
    # Count boundary oscillators
    boundary_count = sum(1 for o in oscillators if o.is_boundary)
    
    logger.info(f"Created BPS oscillator network: {count} oscillators, "
               f"{sum(1 for o in oscillators if o.charge != 0)} charged, "
               f"{boundary_count} in boundary layer")
    
    return oscillators

def apply_phase_sponge_to_field(field: np.ndarray, 
                               damping_factor: float = PHASE_SPONGE_DAMPING_FACTOR,
                               boundary_width: int = PHASE_SPONGE_BOUNDARY_WIDTH,
                               profile: str = PHASE_SPONGE_PROFILE) -> np.ndarray:
    """
    Apply phase sponge damping directly to a 2D field array.
    
    Args:
        field: 2D numpy array representing the field
        damping_factor: Damping coefficient (0-1)
        boundary_width: Width of boundary layer
        profile: Damping profile type
        
    Returns:
        Field with phase sponge applied
    """
    if not ENABLE_PHASE_SPONGE:
        return field
    
    height, width = field.shape
    damped_field = field.copy()
    
    # Create damping mask
    damping_mask = np.ones_like(field)
    
    for i in range(height):
        for j in range(width):
            # Distance to boundaries
            dist_to_left = j
            dist_to_right = width - 1 - j
            dist_to_top = i
            dist_to_bottom = height - 1 - i
            
            min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
            
            if min_dist < boundary_width:
                norm_dist = min_dist / boundary_width
                
                if profile == "linear":
                    damping_strength = 1.0 - norm_dist
                elif profile == "quadratic":
                    damping_strength = (1.0 - norm_dist) ** 2
                elif profile == "tanh":
                    damping_strength = 0.5 * (1.0 - np.tanh(3.0 * (norm_dist - 0.5)))
                else:
                    damping_strength = 1.0 - norm_dist
                
                damping_mask[i, j] = 1.0 - damping_strength * (1.0 - damping_factor)
    
    # Apply damping
    damped_field *= damping_mask
    
    return damped_field

def validate_phase_sponge() -> Dict[str, Any]:
    """
    Validate phase sponge functionality.
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        'status': 'unknown',
        'phase_sponge_enabled': ENABLE_PHASE_SPONGE,
        'damping_factor': PHASE_SPONGE_DAMPING_FACTOR,
        'boundary_width': PHASE_SPONGE_BOUNDARY_WIDTH,
        'profile': PHASE_SPONGE_PROFILE,
        'issues': []
    }
    
    try:
        # Test oscillator with phase sponge at boundary
        boundary_osc = BPSOscillator(0, 0.0, 1.0, position=(0, 0), lattice_bounds=(10, 10))
        
        if ENABLE_PHASE_SPONGE:
            if not boundary_osc.is_boundary:
                validation['issues'].append("Boundary detection not working")
            
            if boundary_osc.sponge_damping >= 1.0:
                validation['issues'].append("Sponge damping not applied at boundary")
        
        # Test oscillator in center (no damping)
        center_osc = BPSOscillator(1, 0.0, 1.0, position=(5, 5), lattice_bounds=(10, 10))
        
        if ENABLE_PHASE_SPONGE:
            if center_osc.is_boundary:
                validation['issues'].append("False boundary detection in center")
            
            if center_osc.sponge_damping < 1.0:
                validation['issues'].append("Spurious damping in center")
        
        # Test field damping
        test_field = np.ones((10, 10))
        damped_field = apply_phase_sponge_to_field(test_field)
        
        if ENABLE_PHASE_SPONGE:
            # Check that boundaries are damped
            if damped_field[0, 0] >= 1.0:
                validation['issues'].append("Field damping not applied at corners")
            
            # Check that center is not damped
            if damped_field[5, 5] < 1.0:
                validation['issues'].append("Field damping applied in center")
        
        # Validate configuration
        if not (0.0 <= PHASE_SPONGE_DAMPING_FACTOR <= 1.0):
            validation['issues'].append(f"Damping factor out of range: {PHASE_SPONGE_DAMPING_FACTOR}")
        
        if PHASE_SPONGE_BOUNDARY_WIDTH < 1:
            validation['issues'].append(f"Invalid boundary width: {PHASE_SPONGE_BOUNDARY_WIDTH}")
        
        if PHASE_SPONGE_PROFILE not in ["linear", "quadratic", "tanh"]:
            validation['issues'].append(f"Unknown profile: {PHASE_SPONGE_PROFILE}")
        
        # Overall status
        if not validation['issues']:
            validation['status'] = 'healthy'
        elif len(validation['issues']) <= 2:
            validation['status'] = 'warnings'
        else:
            validation['status'] = 'issues'
            
        return validation
        
    except Exception as e:
        validation['status'] = 'error'
        validation['issues'].append(f"Validation failed: {e}")
        return validation

def validate_bps_oscillator() -> Dict[str, Any]:
    """
    Validate BPS oscillator functionality and configuration.
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        'status': 'unknown',
        'config_available': BPS_CONFIG_AVAILABLE,
        'features': {
            'phase_locking': ENABLE_BPS_PHASE_LOCKING,
            'charge_tracking': ENABLE_BPS_CHARGE_TRACKING,
            'safety_checks': ENABLE_BPS_SAFETY_CHECKS,
            'phase_sponge': ENABLE_PHASE_SPONGE
        },
        'issues': []
    }
    
    try:
        # Test basic oscillator creation
        test_osc = BPSOscillator(0, 0.0, 1.0)
        
        # Test charge assignment
        test_osc.assign_bps_charge(1)
        if ENABLE_BPS_CHARGE_TRACKING and test_osc.charge != 1:
            validation['issues'].append("Charge assignment failed")
            
        # Test phase locking
        initial_theta = test_osc.theta
        test_osc.lock_to_phase(np.pi)
        if ENABLE_BPS_PHASE_LOCKING and test_osc.theta == initial_theta:
            validation['issues'].append("Phase locking not working")
            
        # Test time evolution
        test_osc.step(0.1)
        
        # Validate phase sponge
        sponge_validation = validate_phase_sponge()
        if sponge_validation['status'] != 'healthy':
            validation['issues'].extend(sponge_validation['issues'])
        
        # Configuration validation
        if BPS_CONFIG_AVAILABLE:
            if not (0.0 <= BPS_PHASE_LOCK_GAIN <= 2.0):
                validation['issues'].append(f"Phase lock gain out of range: {BPS_PHASE_LOCK_GAIN}")
                
            if not (0.0 <= KURAMOTO_COUPLING_STRENGTH <= 1.0):
                validation['issues'].append(f"Coupling strength out of range: {KURAMOTO_COUPLING_STRENGTH}")
                
        # Overall status
        if not validation['issues']:
            validation['status'] = 'healthy'
        elif len(validation['issues']) <= 2:
            validation['status'] = 'warnings'
        else:
            validation['status'] = 'issues'
            
        return validation
        
    except Exception as e:
        validation['status'] = 'error'
        validation['issues'].append(f"Validation failed: {e}")
        return validation

# Export key components
__all__ = [
    'BPSOscillator',
    'create_bps_oscillator_network',
    'apply_phase_sponge_to_field',
    'validate_bps_oscillator',
    'validate_phase_sponge',
    'BPS_CONFIG_AVAILABLE'
]

if __name__ == "__main__":
    # Module verification and testing
    logger.info("BPS Oscillator System with Phase Sponge loaded successfully")
    logger.info(f"Config available: {'YES' if BPS_CONFIG_AVAILABLE else 'NO'}")
    logger.info(f"Phase locking: {'ENABLED' if ENABLE_BPS_PHASE_LOCKING else 'DISABLED'}")
    logger.info(f"Charge tracking: {'ENABLED' if ENABLE_BPS_CHARGE_TRACKING else 'DISABLED'}")
    logger.info(f"Phase Sponge: {'ENABLED' if ENABLE_PHASE_SPONGE else 'DISABLED'}")
    
    if ENABLE_PHASE_SPONGE:
        logger.info(f"  - Damping factor: {PHASE_SPONGE_DAMPING_FACTOR}")
        logger.info(f"  - Boundary width: {PHASE_SPONGE_BOUNDARY_WIDTH}")
        logger.info(f"  - Profile: {PHASE_SPONGE_PROFILE}")
    
    # Run validation if requested
    import sys
    if '--validate' in sys.argv or '--test' in sys.argv:
        logger.info("Running oscillator validation...")
        validation = validate_bps_oscillator()
        logger.info(f"Validation status: {validation['status'].upper()}")
        
        if validation['issues']:
            logger.warning("Issues detected:")
            for issue in validation['issues']:
                logger.warning(f"  - {issue}")
        
        if validation['status'] == 'healthy':
            logger.info("All oscillator systems operational!")
    
    # Demo network creation with phase sponge
    if '--demo' in sys.argv:
        logger.info("Creating demo oscillator network with phase sponge...")
        demo_network = create_bps_oscillator_network(25, charge_density=0.4, lattice_size=(5, 5))
        
        logger.info("Demo network:")
        for osc in demo_network:
            if osc.is_boundary:
                logger.info(f"  {osc} [BOUNDARY]")
            else:
                logger.info(f"  {osc}")
        
        # Test Kuramoto coupling with phase sponge
        for i, osc in enumerate(demo_network):
            # Get neighboring oscillators (simple nearest neighbors)
            neighbors = []
            x, y = osc.position
            for other in demo_network:
                ox, oy = other.position
                if abs(ox - x) + abs(oy - y) == 1:  # Manhattan distance = 1
                    neighbors.append(other)
            
            coupling = osc.kuramoto_coupling(neighbors)
            osc.step(0.1, coupling)
        
        logger.info("After one coupling step with phase sponge:")
        boundary_dampings = []
        for osc in demo_network:
            if osc.is_boundary:
                logger.info(f"  {osc} damping={osc.sponge_damping:.3f}")
                boundary_dampings.append(osc.sponge_damping)
        
        if boundary_dampings:
            avg_damping = sum(boundary_dampings) / len(boundary_dampings)
            logger.info(f"Average boundary damping: {avg_damping:.3f}")
    
    logger.info("BPS oscillator with Phase Sponge initialization complete")
