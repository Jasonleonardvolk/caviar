#!/usr/bin/env python3
"""
BPS Configuration - Centralized Supersymmetric Compute Parameters
═══════════════════════════════════════════════════════════════════

Complete configuration management for the BPS supersymmetric compute flow.
All BPS modules import from this centralized config for consistent behavior.

Features:
• Feature flags for selective BPS subsystem activation
• Charge quantization and energy normalization parameters
• Tolerance thresholds for all constraint checking
• Behavioral controls for phase locking and transitions
• Symbolic tagging system for debugging and UI
• Experimental feature switches for advanced modes
• Runtime identity management for diagnostics
"""

import os
from typing import Dict, List, Set, Union
import logging

logger = logging.getLogger("BPSConfig")

# ═══════════════════════════════════════════════════════════════════════════════
# Runtime Identity and Versioning
# ═══════════════════════════════════════════════════════════════════════════════

RUNTIME_IDENTITY = "TORI-BPS-SUPERSYMMETRIC-v2.1"
BPS_CONFIG_VERSION = "2.1.0"
COMPATIBILITY_LEVEL = "PRODUCTION"

# ═══════════════════════════════════════════════════════════════════════════════
# Feature Flags - Master Switches for BPS Subsystems
# ═══════════════════════════════════════════════════════════════════════════════

# Core BPS features
ENABLE_BPS_HOT_SWAP = bool(int(os.getenv("TORI_BPS_HOT_SWAP", "1")))
ENABLE_BPS_ENERGY_HARVEST = bool(int(os.getenv("TORI_BPS_ENERGY_HARVEST", "1")))
ENABLE_BPS_DIAGNOSTICS = bool(int(os.getenv("TORI_BPS_DIAGNOSTICS", "1")))
ENABLE_BPS_CHARGE_TRACKING = bool(int(os.getenv("TORI_BPS_CHARGE_TRACKING", "1")))

# Advanced features
ENABLE_BPS_INTERPOLATION = bool(int(os.getenv("TORI_BPS_INTERPOLATION", "1")))
ENABLE_BPS_PHASE_LOCKING = bool(int(os.getenv("TORI_BPS_PHASE_LOCKING", "1")))
ENABLE_BPS_SPATIAL_ANALYSIS = bool(int(os.getenv("TORI_BPS_SPATIAL", "0")))
ENABLE_BPS_ADAPTIVE_SCALING = bool(int(os.getenv("TORI_BPS_ADAPTIVE", "1")))

# Phase Sponge - Boundary absorbing layers for edge reflection prevention
ENABLE_PHASE_SPONGE = bool(int(os.getenv("TORI_PHASE_SPONGE", "1")))  # Enable phase sponge damping
PHASE_SPONGE_DAMPING_FACTOR = float(os.getenv("TORI_PHASE_SPONGE_DAMPING", "0.95"))  # Damping coefficient (0-1)
PHASE_SPONGE_BOUNDARY_WIDTH = int(os.getenv("TORI_PHASE_SPONGE_WIDTH", "5"))  # Width of boundary layer in nodes
PHASE_SPONGE_PROFILE = os.getenv("TORI_PHASE_SPONGE_PROFILE", "tanh")  # Profile: 'linear', 'quadratic', 'tanh'

# Safety and fallback controls
STRICT_BPS_MODE = bool(int(os.getenv("TORI_BPS_STRICT", "0")))
ALLOW_FALLBACK_ON_SWAP_FAILURE = bool(int(os.getenv("TORI_BPS_FALLBACK", "1")))
ENABLE_BPS_SAFETY_CHECKS = bool(int(os.getenv("TORI_BPS_SAFETY", "1")))

# ═══════════════════════════════════════════════════════════════════════════════
# Charge & Energy Logic - Fundamental BPS Parameters
# ═══════════════════════════════════════════════════════════════════════════════

# Allowed topological charge values (discrete quantum numbers)
ALLOWED_Q_VALUES: Set[int] = {-2, -1, 0, 1, 2}
MAX_ALLOWED_CHARGE_MAGNITUDE = max(abs(q) for q in ALLOWED_Q_VALUES)

# Energy normalization (BPS saturation condition: E = |Q|)
ENERGY_PER_Q = 1.0  # Energy units per unit topological charge
MIN_BPS_ENERGY = 0.1  # Minimum energy for BPS states
MAX_BPS_ENERGY_MULTIPLIER = 10.0  # Safety bound: E <= 10|Q|

# Soliton energy scaling
DEFAULT_SOLITON_AMPLITUDE = 1.0
SOLITON_ENERGY_SCALING_FACTOR = 1.0
BPS_ENERGY_QUANTUM = ENERGY_PER_Q  # Fundamental energy quantum

# ═══════════════════════════════════════════════════════════════════════════════
# Tolerances & Constraints - Numerical Precision Controls
# ═══════════════════════════════════════════════════════════════════════════════

# BPS constraint tolerances
LAGRANGIAN_TOLERANCE = 1e-6  # E >= |Q| - tolerance
BPS_BOUND_VIOLATION_TOLERANCE = 1e-6
BPS_SATURATION_TOLERANCE = 1e-8  # For E = |Q| detection

# Charge conservation
CHARGE_CONSERVATION_TOLERANCE = 1e-10
MAX_ALLOWED_CHARGE_DRIFT = 1e-8
CHARGE_QUANTIZATION_THRESHOLD = 0.5  # For discrete charge assignment

# Energy conservation
ENERGY_CONSERVATION_TOLERANCE = 1e-8
ENERGY_CONSISTENCY_TOLERANCE = 1e-8
ENERGY_EXTRACTION_EFFICIENCY = 0.95

# Numerical stability
NUMERICAL_STABILITY_THRESHOLD = 1e-12
EIGENVALUE_TOLERANCE = 1e-10
SPECTRAL_GAP_TOLERANCE = 1e-8

# ═══════════════════════════════════════════════════════════════════════════════
# Behavioral Controls - Dynamic System Parameters
# ═══════════════════════════════════════════════════════════════════════════════

# Phase locking and synchronization (Kuramoto-style)
BPS_PHASE_LOCK_GAIN = 0.5  # Strength of phase entrainment
PHASE_COHERENCE_THRESHOLD = 0.1  # Radians
MAX_PHASE_CORRECTION = 0.5  # Maximum phase adjustment per step
KURAMOTO_COUPLING_STRENGTH = 0.3

# Topology transition controls
DEFAULT_SWAP_RAMP_DURATION = 1.0  # Seconds for interpolated transitions
SWAP_INTERPOLATION_STEPS = 10
TOPOLOGY_TRANSITION_DAMPING = 0.1
BERRY_PHASE_SCALING = 1.0

# Energy harvesting parameters
DEFAULT_HARVEST_EPSILON = 0.5
DEFAULT_HARVEST_STEPS = 10
MAX_AMPLIFICATION_FACTOR = 10.0
HARVEST_SAFETY_MARGIN = 0.1

# Adaptive behavior
ADAPTIVE_EPSILON_MIN = 0.1
ADAPTIVE_EPSILON_MAX = 1.0
ADAPTIVE_SCALING_FACTOR = 0.5
CONVERGENCE_PATIENCE = 5  # Steps to wait for convergence

# ═══════════════════════════════════════════════════════════════════════════════
# Symbolic Tags - Human-Readable Identifiers
# ═══════════════════════════════════════════════════════════════════════════════

# Soliton type tags
SOLITON_TAGS = {
    'bright_bps': "Bright BPS",
    'dark_bps': "Dark BPS", 
    'composite': "Composite",
    'anti_soliton': "Anti-Soliton",
    'topological_defect': "Topological Defect",
    'vacuum_bubble': "Vacuum Bubble",
    'kink': "Kink Soliton",
    'breather': "Breather Mode"
}

# Topology tags
TOPOLOGY_TAGS = {
    'kagome': "Kagome Lattice",
    'honeycomb': "Honeycomb Lattice",
    'triangular': "Triangular Lattice", 
    'small_world': "Small-World Network",
    'penrose': "Penrose Quasicrystal",
    'exotic': "Exotic Topology"
}

# Operation tags
OPERATION_TAGS = {
    'hot_swap': "Hot Topology Swap",
    'bps_harvest': "BPS Energy Harvest",
    'charge_extraction': "Charge Extraction",
    'phase_lock': "Phase Locking",
    'conservation_check': "Conservation Check",
    'stability_verify': "Stability Verification"
}

# State tags
STATE_TAGS = {
    'bps_saturated': "BPS Saturated",
    'bps_violating': "BPS Violating",
    'charge_conserved': "Charge Conserved",
    'charge_violated': "Charge Violated",
    'phase_locked': "Phase Locked",
    'unstable': "Unstable"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Experimental Switches - Advanced Feature Gates
# ═══════════════════════════════════════════════════════════════════════════════

# Quantum field extensions
BPS_QUANTUM_FIELD_MODE = bool(int(os.getenv("TORI_BPS_QUANTUM", "0")))
ENABLE_BERRY_CURVATURE = bool(int(os.getenv("TORI_BERRY_CURVATURE", "0")))
ENABLE_CHERN_SIMONS_TERMS = bool(int(os.getenv("TORI_CHERN_SIMONS", "0")))

# Lattice geometry
BPS_LATTICE_CURVATURE = bool(int(os.getenv("TORI_LATTICE_CURVATURE", "0")))
ENABLE_HYPERBOLIC_GEOMETRY = bool(int(os.getenv("TORI_HYPERBOLIC", "0")))
ENABLE_FRACTAL_BOUNDARIES = bool(int(os.getenv("TORI_FRACTAL", "0")))

# Advanced analytics
ENABLE_MACHINE_LEARNING_OPTIMIZATION = bool(int(os.getenv("TORI_ML_OPT", "0")))
ENABLE_REAL_TIME_VISUALIZATION = bool(int(os.getenv("TORI_REALTIME_VIZ", "0")))
ENABLE_DISTRIBUTED_COMPUTATION = bool(int(os.getenv("TORI_DISTRIBUTED", "0")))

# Research modes
RESEARCH_MODE = bool(int(os.getenv("TORI_RESEARCH_MODE", "0")))
BENCHMARK_MODE = bool(int(os.getenv("TORI_BENCHMARK_MODE", "0")))
DEBUG_VERBOSE_MODE = bool(int(os.getenv("TORI_VERBOSE_DEBUG", "0")))

# ═══════════════════════════════════════════════════════════════════════════════
# Performance and Resource Limits
# ═══════════════════════════════════════════════════════════════════════════════

# Memory management
MAX_DIAGNOSTIC_HISTORY = int(os.getenv("TORI_MAX_DIAG_HISTORY", "1000"))
MAX_SOLITON_COUNT = int(os.getenv("TORI_MAX_SOLITONS", "1000"))
MAX_SWAP_HISTORY = int(os.getenv("TORI_MAX_SWAP_HISTORY", "10"))

# Performance thresholds
SLOW_OPERATION_THRESHOLD = float(os.getenv("TORI_SLOW_OP_THRESHOLD", "1.0"))
MEMORY_WARNING_THRESHOLD = int(os.getenv("TORI_MEMORY_WARNING", "1000"))
PERFORMANCE_PROFILING_ENABLED = bool(int(os.getenv("TORI_PERF_PROFILE", "0")))

# Computational limits
MAX_EIGENVALUE_COMPUTATION = int(os.getenv("TORI_MAX_EIGENVALS", "6"))
MAX_INTERPOLATION_STEPS = int(os.getenv("TORI_MAX_INTERP_STEPS", "100"))
MAX_HARVEST_STEPS = int(os.getenv("TORI_MAX_HARVEST_STEPS", "50"))

# ═══════════════════════════════════════════════════════════════════════════════
# Logging and Diagnostics Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Logging levels and formats
BPS_LOG_LEVEL = os.getenv("TORI_BPS_LOG_LEVEL", "INFO")
ENABLE_DETAILED_LOGGING = bool(int(os.getenv("TORI_DETAILED_LOG", "1")))
LOG_BPS_OPERATIONS = bool(int(os.getenv("TORI_LOG_BPS_OPS", "1")))
LOG_CONSERVATION_CHECKS = bool(int(os.getenv("TORI_LOG_CONSERVATION", "1")))

# Diagnostic output
DIAGNOSTIC_OUTPUT_FORMAT = os.getenv("TORI_DIAG_FORMAT", "structured")
ENABLE_DIAGNOSTIC_EXPORT = bool(int(os.getenv("TORI_DIAG_EXPORT", "0")))
DIAGNOSTIC_EXPORT_INTERVAL = int(os.getenv("TORI_DIAG_INTERVAL", "100"))

# ═══════════════════════════════════════════════════════════════════════════════
# Validation and Consistency Checks
# ═══════════════════════════════════════════════════════════════════════════════

def validate_bps_config() -> Dict[str, Union[bool, str]]:
    """
    Validate BPS configuration consistency and report issues.
    
    Returns:
        Dictionary with validation results
    """
    validation = {'valid': True, 'issues': []}
    
    try:
        # Check charge constraints
        if MAX_ALLOWED_CHARGE_MAGNITUDE <= 0:
            validation['issues'].append("Invalid max charge magnitude")
            validation['valid'] = False
        
        # Check energy parameters
        if ENERGY_PER_Q <= 0:
            validation['issues'].append("Invalid energy per charge")
            validation['valid'] = False
        
        # Check tolerances
        if LAGRANGIAN_TOLERANCE <= 0:
            validation['issues'].append("Invalid Lagrangian tolerance")
            validation['valid'] = False
        
        # Check phase parameters
        if not (0 <= BPS_PHASE_LOCK_GAIN <= 1):
            validation['issues'].append("Phase lock gain out of range")
            validation['valid'] = False
        
        # Check resource limits
        if MAX_DIAGNOSTIC_HISTORY <= 0:
            validation['issues'].append("Invalid diagnostic history limit")
            validation['valid'] = False
        
        # Log validation results
        if validation['valid']:
            logger.info("BPS configuration validation passed")
        else:
            logger.error(f"BPS configuration issues: {validation['issues']}")
            
    except Exception as e:
        validation['valid'] = False
        validation['issues'].append(f"Validation error: {e}")
        logger.error(f"Config validation failed: {e}")
    
    return validation

def get_bps_config_summary() -> Dict[str, any]:
    """Get summary of current BPS configuration"""
    return {
        'runtime_identity': RUNTIME_IDENTITY,
        'version': BPS_CONFIG_VERSION,
        'compatibility': COMPATIBILITY_LEVEL,
        'core_features': {
            'hot_swap': ENABLE_BPS_HOT_SWAP,
            'energy_harvest': ENABLE_BPS_ENERGY_HARVEST,
            'diagnostics': ENABLE_BPS_DIAGNOSTICS,
            'charge_tracking': ENABLE_BPS_CHARGE_TRACKING
        },
        'advanced_features': {
            'interpolation': ENABLE_BPS_INTERPOLATION,
            'phase_locking': ENABLE_BPS_PHASE_LOCKING,
            'spatial_analysis': ENABLE_BPS_SPATIAL_ANALYSIS,
            'adaptive_scaling': ENABLE_BPS_ADAPTIVE_SCALING,
            'phase_sponge': ENABLE_PHASE_SPONGE
        },
        'phase_sponge': {
            'enabled': ENABLE_PHASE_SPONGE,
            'damping_factor': PHASE_SPONGE_DAMPING_FACTOR,
            'boundary_width': PHASE_SPONGE_BOUNDARY_WIDTH,
            'profile': PHASE_SPONGE_PROFILE
        },
        'experimental': {
            'quantum_field': BPS_QUANTUM_FIELD_MODE,
            'lattice_curvature': BPS_LATTICE_CURVATURE,
            'research_mode': RESEARCH_MODE
        },
        'tolerances': {
            'lagrangian': LAGRANGIAN_TOLERANCE,
            'charge_conservation': CHARGE_CONSERVATION_TOLERANCE,
            'energy_conservation': ENERGY_CONSERVATION_TOLERANCE
        },
        'limits': {
            'max_charge': MAX_ALLOWED_CHARGE_MAGNITUDE,
            'max_solitons': MAX_SOLITON_COUNT,
            'max_history': MAX_DIAGNOSTIC_HISTORY
        }
    }

def override_config_from_env():
    """Apply any additional environment variable overrides"""
    try:
        # Allow runtime override of key parameters
        global ENERGY_PER_Q, BPS_PHASE_LOCK_GAIN, HARVEST_SAFETY_MARGIN
        
        if 'TORI_ENERGY_PER_Q' in os.environ:
            ENERGY_PER_Q = float(os.environ['TORI_ENERGY_PER_Q'])
            logger.info(f"Override: ENERGY_PER_Q = {ENERGY_PER_Q}")
        
        if 'TORI_PHASE_LOCK_GAIN' in os.environ:
            BPS_PHASE_LOCK_GAIN = float(os.environ['TORI_PHASE_LOCK_GAIN'])
            logger.info(f"Override: BPS_PHASE_LOCK_GAIN = {BPS_PHASE_LOCK_GAIN}")
        
        if 'TORI_HARVEST_SAFETY' in os.environ:
            HARVEST_SAFETY_MARGIN = float(os.environ['TORI_HARVEST_SAFETY'])
            logger.info(f"Override: HARVEST_SAFETY_MARGIN = {HARVEST_SAFETY_MARGIN}")
            
    except Exception as e:
        logger.warning(f"Failed to apply environment overrides: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Initialize Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Apply environment overrides on import
override_config_from_env()

# Validate configuration
_validation_result = validate_bps_config()
if not _validation_result['valid']:
    logger.warning("BPS configuration has validation issues - some features may not work correctly")

# Log configuration summary if verbose mode enabled
if DEBUG_VERBOSE_MODE:
    config_summary = get_bps_config_summary()
    logger.info(f"BPS Configuration loaded: {RUNTIME_IDENTITY}")
    logger.debug(f"Config summary: {config_summary}")

if __name__ == "__main__":
    # Configuration verification and reporting
    print(f"BPS Configuration {BPS_CONFIG_VERSION}")
    print(f"Runtime Identity: {RUNTIME_IDENTITY}")
    print("=" * 50)
    
    summary = get_bps_config_summary()
    for category, settings in summary.items():
        if isinstance(settings, dict):
            print(f"\n{category.upper()}:")
            for key, value in settings.items():
                print(f"  {key}: {value}")
        else:
            print(f"{category}: {settings}")
    
    print("\nValidation:", "PASSED" if _validation_result['valid'] else "FAILED")
    if not _validation_result['valid']:
        print("Issues:", _validation_result['issues'])
