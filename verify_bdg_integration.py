#!/usr/bin/env python3
"""
BdG Integration Verification Script
Tests that all components are properly wired together
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add paths
sys.path.extend([
    str(Path(__file__).parent),
    str(Path(__file__).parent / "python" / "core"),
    str(Path(__file__).parent / "alan_backend")
])

def verify_bdg_integration():
    """Verify BdG components are properly integrated"""
    print("🔍 Verifying BdG Integration...")
    print("=" * 60)
    
    errors = []
    
    # 1. Check BdG solver
    try:
        from python.core.bdg_solver import assemble_bdg, compute_spectrum, analyze_stability
        print("✅ BdG solver imported successfully")
        
        # Test basic functionality
        test_field = np.tanh(np.linspace(-5, 5, 64))
        H = assemble_bdg(test_field.reshape(8, 8))
        print("✅ BdG operator assembled")
    except Exception as e:
        errors.append(f"❌ BdG solver error: {e}")
    
    # 2. Check Lyapunov exporter
    try:
        from alan_backend.lyap_exporter import LyapunovExporter
        exporter = LyapunovExporter("test_lyapunov.json")
        print("✅ Lyapunov exporter initialized")
    except Exception as e:
        errors.append(f"❌ Lyapunov exporter error: {e}")
    
    # 3. Check adaptive timestep
    try:
        from python.core.adaptive_timestep import AdaptiveTimestep
        adaptive_dt = AdaptiveTimestep()
        dt = adaptive_dt.compute_timestep(0.1)
        print(f"✅ Adaptive timestep computed: {dt:.4f}")
    except Exception as e:
        errors.append(f"❌ Adaptive timestep error: {e}")
    
    # 4. Check eigensentry integration
    try:
        from alan_backend.eigensentry_guard import get_guard
        guard = get_guard()
        
        # Check if BdG attributes exist
        assert hasattr(guard, 'lyap_exporter'), "Missing lyap_exporter"
        assert hasattr(guard, 'poll_spectral_stability'), "Missing poll_spectral_stability"
        print("✅ EigenSentry BdG integration verified")
    except Exception as e:
        errors.append(f"❌ EigenSentry integration error: {e}")
    
    # 5. Check chaos control layer
    try:
        from python.core.chaos_control_layer import ChaosControlLayer
        # Just check imports work
        print("✅ Chaos control layer imports verified")
    except Exception as e:
        errors.append(f"❌ Chaos control layer error: {e}")
    
    print("\n" + "=" * 60)
    
    if errors:
        print("❌ INTEGRATION ISSUES FOUND:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("✅ ALL BDG COMPONENTS SUCCESSFULLY INTEGRATED!")
        print("\nNext steps:")
        print("1. Run: python tori_master.py")
        print("2. Monitor: tail -f lyapunov_watchlist.json")
        print("3. Connect to WebSocket at ws://localhost:8765/ws/eigensentry")
        return True

if __name__ == "__main__":
    # Run verification
    success = verify_bdg_integration()
    
    # Clean up test file
    test_file = Path("test_lyapunov.json")
    if test_file.exists():
        test_file.unlink()
    
    sys.exit(0 if success else 1)
