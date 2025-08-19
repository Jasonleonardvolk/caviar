"""
Banksy - Phase oscillator network implementation for the ALAN cognitive system.

This package provides the core oscillator dynamics and phase synchronization
capabilities that power ALAN's concept memory and attractor-based reasoning.
The system combines Kuramoto oscillator networks with Koopman eigenfunction analysis
to create a ψ-phase dynamics framework that ensures stability during inference.
"""

# Core ψ-Sync monitor components
from .psi_sync_monitor import (
    PsiSyncMonitor, 
    PsiPhaseState, 
    PsiSyncMetrics, 
    SyncAction,
    SyncState,
    get_psi_sync_monitor
)

# Koopman integration components
from .psi_koopman_integration import (
    PsiKoopmanIntegrator,
    generate_synthetic_time_series
)

# ALAN integration bridge
from .alan_psi_sync_bridge import (
    AlanPsiSyncBridge,
    AlanPsiState,
    get_alan_psi_bridge
)

# 🔧 CRITICAL FIX: Import oscillator functions from banksy_spin
from .banksy_spin import step, oscillator_update

__all__ = [
    # Core monitor
    'PsiSyncMonitor',
    'PsiPhaseState',
    'PsiSyncMetrics', 
    'SyncAction',
    'SyncState',
    'get_psi_sync_monitor',
    
    # Koopman integration
    'PsiKoopmanIntegrator',
    'generate_synthetic_time_series',
    
    # ALAN bridge
    'AlanPsiSyncBridge',
    'AlanPsiState',
    'get_alan_psi_bridge',
    
    # 🔧 CRITICAL FIX: Export oscillator functions
    'step',
    'oscillator_update'
]
