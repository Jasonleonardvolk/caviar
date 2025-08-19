"""
BdG Integration Patches for TORI System
Apply these patches to integrate BdG spectral stability analysis
"""

# ============================================================================
# PATCH 1: eigensentry_guard.py - Add BdG spectral stability polling
# ============================================================================

EIGENSENTRY_PATCH = """
# Add to imports at the top:
from alan_backend.lyap_exporter import LyapunovExporter
import asyncio

# Add to __init__ method after self.ws_clients = set():
        # BdG spectral stability integration
        self.lyap_exporter = LyapunovExporter()
        self.poll_counter = 0
        self.POLL_INTERVAL = 256  # Update every N steps
        self.nonlinearity = 1.0  # Default g parameter
        self.dx = 0.1  # Default spatial step

# Add new method after register_websocket:
    def poll_spectral_stability(self, soliton_state: np.ndarray):
        '''Poll spectral stability every N steps'''
        self.poll_counter += 1
        
        if self.poll_counter % self.POLL_INTERVAL == 0:
            # Get current lattice parameters
            params = {'g': self.nonlinearity, 'dx': self.dx}
            
            # Update spectrum
            metrics = self.lyap_exporter.update_spectrum(soliton_state, params)
            
            # Update internal state
            self.metrics['lambda_max'] = metrics['lambda_max']
            self.metrics['unstable_modes'] = metrics['unstable_count']
            
            # Broadcast update
            asyncio.create_task(self._broadcast_metrics())

# Modify check_eigenvalues method to include BdG polling:
# Add after curvature_metrics = self.compute_local_curvature(state):
        
        # Poll BdG spectral stability
        self.poll_spectral_stability(state)
"""

# ============================================================================
# PATCH 2: chaos_control_layer.py - Add adaptive timestep
# ============================================================================

CCL_PATCH = """
# Add to imports:
from python.core.adaptive_timestep import AdaptiveTimestep

# In __init__ method, add after self.dt = dt:
        # Adaptive timestep based on spectral stability
        self.adaptive_dt = AdaptiveTimestep(dt_base=self.dt)
        self.eigen_sentry = None  # Will be set during integration

# In fdtd_step_dark_soliton method, replace:
# evolution = np.exp(-1j * (kinetic + potential + nonlinear) * self.dt)
# with:
        # Adaptive timestep based on Lyapunov exponents
        if hasattr(self, 'eigen_sentry') and self.eigen_sentry is not None:
            lambda_max = self.eigen_sentry.metrics.get('lambda_max', 0.0)
            dt = self.adaptive_dt.compute_timestep(lambda_max)
        else:
            dt = self.dt
            
        evolution = np.exp(-1j * (kinetic + potential + nonlinear) * dt)

# Add method to set EigenSentry reference:
    def set_eigen_sentry(self, eigen_sentry):
        '''Set reference to EigenSentry for adaptive timestep'''
        self.eigen_sentry = eigen_sentry
"""

# ============================================================================
# PATCH 3: tori_master.py - Wire BdG components together
# ============================================================================

TORI_MASTER_PATCH = """
# In start() method, after initializing eigen_guard:
            
            # Wire BdG spectral stability
            if 'chaos_controller' in self.components and 'eigen_guard' in self.components:
                # Connect EigenSentry to CCL for adaptive timestep
                ccl = self.components['chaos_controller']
                if hasattr(ccl, 'chaos_processor') and hasattr(ccl.chaos_processor, 'set_eigen_sentry'):
                    ccl.chaos_processor.set_eigen_sentry(self.components['eigen_guard'])
                    logger.info("✅ Connected BdG spectral stability to chaos control")

# In _handle_integrations() method, add BdG monitoring:
                # Monitor Lyapunov exponents for system health
                if 'eigen_guard' in self.components:
                    guard = self.components['eigen_guard']
                    lambda_max = guard.metrics.get('lambda_max', 0.0)
                    
                    # Log warnings for high Lyapunov exponents
                    if lambda_max > 0.1:
                        logger.warning(f"⚠️ Positive Lyapunov exponent detected: {lambda_max:.3f}")
"""

# ============================================================================
# PATCH 4: Add BdG monitoring to WebSocket broadcast
# ============================================================================

WEBSOCKET_PATCH = """
# In services/metrics_ws.py, modify the metrics broadcast to include BdG data:
# In broadcast_metrics method, add to metrics dict:
            'bdg_stability': {
                'lambda_max': self.guard.metrics.get('lambda_max', 0.0),
                'unstable_modes': self.guard.metrics.get('unstable_modes', 0),
                'adaptive_dt': self.guard.metrics.get('adaptive_dt', self.dt)
            }
"""

# ============================================================================
# Function to apply patches
# ============================================================================

def apply_bdg_patches():
    """
    Instructions for applying BdG patches to TORI system
    """
    print("🔧 BdG Integration Patches")
    print("=" * 60)
    print("\n1. EIGENSENTRY GUARD PATCH (alan_backend/eigensentry_guard.py):")
    print(EIGENSENTRY_PATCH)
    print("\n2. CHAOS CONTROL LAYER PATCH (python/core/chaos_control_layer.py):")
    print(CCL_PATCH)
    print("\n3. TORI MASTER PATCH (tori_master.py):")
    print(TORI_MASTER_PATCH)
    print("\n4. WEBSOCKET PATCH (services/metrics_ws.py):")
    print(WEBSOCKET_PATCH)
    print("\n" + "=" * 60)
    print("✅ Apply these patches to integrate BdG spectral stability!")

if __name__ == "__main__":
    apply_bdg_patches()
