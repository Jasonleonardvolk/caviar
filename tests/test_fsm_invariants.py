import os, sys, math
import numpy as np
import pytest

# Ensure project root is importable
ROOT = r"D:\\Dev\\kha"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from python.core.fractal_soliton_memory import FractalSolitonMemory

@pytest.mark.parametrize("method", ["rk4-builtin", "rk4-ext"])  # rk4-ext will be skipped if hook missing
def test_mass_energy_coherence_invariants(method):
    n = 256
    L = np.eye(n)
    fsm = FractalSolitonMemory.from_random(n=n, laplacian=L, seed=7)
    s0 = fsm.status()
    
    # Try to use external only if the hook exists
    if method == "rk4-ext":
        try:
            from python.core.fsm_lattice_integration import rk4_step  # noqa: F401
        except Exception:
            pytest.skip("rk4-ext not available (no external integrator module)")
    
    fsm.step(steps=20_000, dt=1e-3, method=method, conserve_mass=True)
    s1 = fsm.status()
    
    dM = abs(s1["mass"] - s0["mass"])           # Mass should be conserved (post-renorm)
    dE = abs(s1["energy"] - s0["energy"])       # Energy small drift acceptable for RK4 + renorm
    dC = abs(s1["coherence"] - s0["coherence"]) # Coherence should be near-stable on I
    
    # Tight thresholds tuned to identity L runs; adjust if you change dynamics
    assert dM < 1e-10, f"mass drift too high: {dM:g}"
    assert dE < 1e-9, f"energy drift too high: {dE:g}"
    assert dC < 1e-9, f"coherence drift too high: {dC:g}"
