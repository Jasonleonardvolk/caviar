#!/usr/bin/env python3
"""
Optional BdG Status Check for enhanced_launcher.py
Add this to the launcher if you want explicit BdG visibility
"""

def check_bdg_components():
    """Check if BdG components are available"""
    bdg_status = {
        "bdg_solver": False,
        "lyap_exporter": False,
        "adaptive_timestep": False
    }
    
    try:
        from python.core.bdg_solver import assemble_bdg
        bdg_status["bdg_solver"] = True
    except ImportError:
        pass
    
    try:
        from alan_backend.lyap_exporter import LyapunovExporter
        bdg_status["lyap_exporter"] = True
    except ImportError:
        pass
    
    try:
        from python.core.adaptive_timestep import AdaptiveTimestep
        bdg_status["adaptive_timestep"] = True
    except ImportError:
        pass
    
    return bdg_status

# Add this to the status display section:
"""
# BdG Spectral Stability status
bdg_status = check_bdg_components()
if all(bdg_status.values()):
    self.logger.info("   ✅ BdG Spectral Stability: Active (Lyapunov monitoring enabled)")
    self.logger.info("   🌊 Adaptive Timestep: Based on spectral stability")
else:
    self.logger.info("   ⚠️ BdG Components: Partially available")
    for component, available in bdg_status.items():
        status = "✅" if available else "❌"
        self.logger.info(f"      {status} {component}")
"""
