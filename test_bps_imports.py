#!/usr/bin/env python3
"""
Test BPS imports to verify everything is working
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing BPS module imports...")

try:
    print("1. Testing BPS config...")
    from python.core.bps_config_enhanced import BPS_CONFIG, SolitonPolarity
    print("   ✅ BPS config imported successfully")
    
    print("2. Testing BPS oscillator...")
    from python.core.bps_oscillator_enhanced import BPSEnhancedLattice
    print("   ✅ BPS oscillator imported successfully")
    
    print("3. Testing BPS memory...")
    from python.core.bps_soliton_memory_enhanced import BPSEnhancedSolitonMemory
    print("   ✅ BPS memory imported successfully")
    
    print("4. Testing BPS blowup harness...")
    from python.core.bps_blowup_harness import BPSBlowupHarness
    print("   ✅ BPS blowup harness imported successfully")
    
    print("5. Testing BPS hot swap...")
    from python.core.bps_hot_swap_laplacian import BPSHotSwapLaplacian
    print("   ✅ BPS hot swap imported successfully")
    
    print("6. Testing BPS diagnostics...")
    from python.monitoring.bps_diagnostics import BPSDiagnostics
    print("   ✅ BPS diagnostics imported successfully")
    
    print("\n✅ All imports successful! BPS modules are ready.")
    
    print("\n7. Quick functionality test...")
    # Create a small lattice
    lattice = BPSEnhancedLattice(size=5)
    lattice.create_bps_soliton(0, charge=1.0)
    
    # Check diagnostics
    diagnostics = BPSDiagnostics(lattice)
    charge = diagnostics.compute_total_charge()
    print(f"   Total charge: {charge}")
    print("   ✅ Basic functionality working!")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)

print("\n🎉 All tests passed! BPS system is ready to integrate.")
