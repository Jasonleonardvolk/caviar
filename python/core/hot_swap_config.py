# hot_swap_config.py - Centralized Configuration
"""
Hot-Swappable Laplacian Configuration
═══════════════════════════════════════

Centralized configuration constants for the hot-swappable Laplacian system.
All tunables in one place for easy modification and BPS integration.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Energy Management Constants
# ═══════════════════════════════════════════════════════════════════════════════
CRITICAL_ENERGY_THRESHOLD = 1_000.0
WARNING_ENERGY_THRESHOLD = 500.0
MAX_ENERGY_MULTIPLIER = 2.0

# ═══════════════════════════════════════════════════════════════════════════════
# System Parameters
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_LATTICE_SIZE = (20, 20)
MAX_SWAP_HISTORY = 10
MIN_EIGENVALUES_TO_COMPUTE = 6

# ═══════════════════════════════════════════════════════════════════════════════
# Soliton Configuration
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_SOLITON_WIDTH = 2.0
DEFAULT_SOLITON_VELOCITY = 1.0
SEARCH_SOLITON_WIDTH = 5.0
SEARCH_SOLITON_VELOCITY = 2.0
OPTIMIZATION_SOLITON_WIDTH = 1.0
OPTIMIZATION_SOLITON_VELOCITY = 0.5
MAX_SOLITONS_PER_INJECTION = 10
SOLITON_ENERGY_THRESHOLD = 100.0

# ═══════════════════════════════════════════════════════════════════════════════
# Numerical Stability
# ═══════════════════════════════════════════════════════════════════════════════
MIN_SPECTRAL_GAP_RATIO = 0.5
STABILIZATION_SLEEP_TIME = 0.1
INTERFERENCE_STRENGTH = 0.1
FLUX_PHASE_FACTOR = 2.0
DEFAULT_EPSILON = 0.3
DEFAULT_BLOWUP_STEPS = 5

# ═══════════════════════════════════════════════════════════════════════════════
# Environment Variables
# ═══════════════════════════════════════════════════════════════════════════════
ENV_EXOTIC_ENABLE = "TORI_ENABLE_EXOTIC"
ENV_DEBUG_LEVEL = "TORI_DEBUG_LEVEL"
ENV_LOG_TO_FILE = "TORI_LOG_TO_FILE"

# ═══════════════════════════════════════════════════════════════════════════════
# BPS Integration Constants (Ready for Future Use)
# ═══════════════════════════════════════════════════════════════════════════════
BPS_CHARGE_CONSERVATION_TOLERANCE = 1e-12
BPS_ENERGY_HARVEST_EFFICIENCY = 0.95
BPS_TOPOLOGY_TRANSITION_DAMPING = 0.1
BPS_SOLITON_PHASE_LOCK_STRENGTH = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# Logging Configuration
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_LOG_LEVEL = 'INFO'
