"""
BPS Soliton Configuration Module
Central configuration for Bogomol'nyi-Prasad-Sommerfield soliton support
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class SolitonPolarity(Enum):
    """Soliton polarity types with BPS support"""
    BRIGHT = "bright"  # Standard peak soliton
    DARK = "dark"      # Inverted notch soliton  
    BPS = "BPS"        # Topologically protected state

@dataclass
class BPSConfig:
    """Configuration parameters for BPS soliton behavior"""
    
    # Core BPS parameters
    enable_bps: bool = True
    enable_bps_harvest: bool = False  # Don't harvest BPS energy by default
    enable_bps_hot_swap: bool = True  # Preserve BPS during hot-swaps
    strict_bps_mode: bool = False     # Enforce strict invariant checking
    
    # Energy and charge parameters
    bps_energy_tolerance: float = 0.01  # Tolerance for E = |Q| saturation
    max_bps_oscillators: Optional[int] = None  # No limit by default
    bps_charge_quantum: float = 1.0  # Unit topological charge
    
    # Coupling parameters
    bps_coupling_inertia: float = 1000.0  # Effective infinite inertia
    bps_phase_lock_strength: float = 10.0  # Strong phase locking
    bps_neighbor_influence: float = 0.0  # BPS doesn't get influenced
    
    # Diagnostics and monitoring
    enable_bps_diagnostics: bool = True
    log_charge_conservation: bool = True
    charge_conservation_tolerance: float = 1e-6
    
    # Hot-swap specific
    bps_hot_swap_pause: bool = True  # Pause BPS dynamics during swap
    bps_position_mapping: str = "concept_id"  # How to map positions
    
    # API and schema
    require_polarity_field: bool = True  # Enforce in API
    default_polarity: SolitonPolarity = SolitonPolarity.BRIGHT
    
    def validate(self) -> bool:
        """Validate configuration consistency"""
        if self.enable_bps_harvest and self.strict_bps_mode:
            raise ValueError("Cannot harvest BPS energy in strict mode")
        
        if self.bps_energy_tolerance < 0:
            raise ValueError("Energy tolerance must be positive")
        
        if self.bps_coupling_inertia < 1.0:
            raise ValueError("BPS coupling inertia must be >= 1.0")
        
        return True

# Global configuration instance
BPS_CONFIG = BPSConfig()
