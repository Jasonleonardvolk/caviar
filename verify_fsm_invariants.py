#!/usr/bin/env python
"""
One-shot verification script for FSM invariants and benches
"""
import sys
import os
sys.path.append(r"D:\Dev\kha")
os.chdir(r"D:\Dev\kha")

print("=" * 60)
print("FSM INVARIANTS VERIFICATION")
print("=" * 60)

# 1) Verify Laplacian is present
print("\n[1] Checking Laplacian...")
try:
    from python.core.graph_ops import load_laplacian
    L = load_laplacian(r"D:\Dev\kha\data\concept_mesh\L_norm.npz")
    print(f"  OK: Laplacian loaded, shape {L.shape}")
except Exception as e:
    print(f"  ERROR: {e}")

# 2) Verify FSM hooks work
print("\n[2] Testing FSM hooks...")
try:
    import numpy as np
    from python.core.fractal_soliton_memory import FractalSolitonMemory
    from python.core.graph_ops import load_laplacian
    
    # Create FSM instance
    fsm = FractalSolitonMemory.get()
    
    # Initialize state
    fsm._psi = (np.random.randn(128) + 1j * np.random.randn(128))
    
    # Set Laplacian
    fsm.set_laplacian(path=r"D:\Dev\kha\data\concept_mesh\L_norm.npz")
    
    # Test invariants before step
    C0 = fsm.coherence()
    E0 = fsm.energy()
    print(f"  Before step: C={C0:.4f}, E={E0:.4f}")
    
    # Take a step
    fsm.step(dt=0.2, kappa=1.0, lambda_=0.1)
    
    # Test invariants after step
    C1 = fsm.coherence()
    E1 = fsm.energy()
    Lver = fsm.laplacian_version
    
    print(f"  After step:  C={C1:.4f}, E={E1:.4f}, Lver={Lver}")
    print(f"  Delta:       dC={abs(C1-C0):.4f}, dE={E1-E0:.4f}")
    
    # Verify energy decreased (gradient flow)
    if E1 <= E0 + 1e-10:
        print("  OK: Energy decreased or stayed same (gradient flow working)")
    else:
        print("  WARNING: Energy increased (may need smaller dt)")
    
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

# 3) Test API endpoint
print("\n[3] Testing API endpoint...")
try:
    import requests
    response = requests.get("http://127.0.0.1:8002/api/memory/state/me")
    if response.status_code == 200:
        data = response.json()
        print(f"  OK: API responded")
        print(f"    - Coherence: {data.get('coherence', 'N/A')}")
        print(f"    - Energy: {data.get('energy', 'N/A')}")
        print(f"    - Laplacian version: {data.get('laplacian_version', 'N/A')}")
    else:
        print(f"  API returned status {response.status_code}")
        print("  (Server may not be running)")
except Exception as e:
    print(f"  Could not reach API: {e}")
    print("  (This is normal if server is not running)")

# 4) Check benchmark outputs
print("\n[4] Checking benchmark outputs...")
from pathlib import Path

benchmarks = [
    ("Resonance", Path(r"D:\Dev\kha\reports\resonance_bench\topk_pr.json")),
    ("Coherence", Path(r"D:\Dev\kha\reports\coherence_bench\coherence_vs_edit.json")),
    ("Energy", Path(r"D:\Dev\kha\reports\energy_bench\energy_traces.json"))
]

for name, path in benchmarks:
    if path.exists():
        print(f"  {name}: EXISTS ({path.stat().st_size} bytes)")
    else:
        print(f"  {name}: NOT FOUND")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
