#!/usr/bin/env python3
"""
Test ALBERT - General Relativity computations in TORI
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import albert
    print("✅ ALBERT module imported successfully")
    
    # Initialize a Kerr metric for a rotating black hole
    print("\n🌌 Initializing Kerr metric for rotating black hole...")
    metric = albert.init_metric("kerr", params={"M": 1, "a": 0.7})
    
    print(f"\nMetric initialized: {metric}")
    print(f"Description: {albert.describe()}")
    
    # Get specific metric components
    print("\n📊 Metric components:")
    coords = ['t', 'r', 'theta', 'phi']
    
    # g_tt component (time-time)
    g_tt = metric.get_component((0, 0))
    print(f"g_tt = {g_tt}")
    
    # g_rr component (radial-radial)
    g_rr = metric.get_component((1, 1))
    print(f"g_rr = {g_rr}")
    
    # g_tφ component (time-phi coupling - frame dragging!)
    g_tphi = metric.get_component((0, 3))
    print(f"g_tφ = {g_tphi} (frame dragging term)")
    
    print("\n🎉 ALBERT is working! General relativity computations ready.")
    
except ImportError as e:
    print(f"❌ Failed to import ALBERT: {e}")
    print("Make sure sympy is installed: pip install sympy")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
