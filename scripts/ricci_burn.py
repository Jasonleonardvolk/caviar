import sys, os, json, time, numpy as np
sys.path.insert(0, r"D:\Dev\kha")
from python.core.fractal_soliton_memory import FractalSolitonMemory, CurvatureField, load_laplacian_from_npz

ROOT = r"D:\Dev\kha"
ART = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)

# Load mesh + curvature
L = load_laplacian_from_npz(os.path.join(ROOT, "data", "concept_mesh", "L_norm.npz"))
ricci = np.load(os.path.join(ROOT, "data", "curvature", "ricci.npy"))
kretz = np.load(os.path.join(ROOT, "data", "curvature", "kretschmann.npy"))

# FSM init
fsm = FractalSolitonMemory.from_random(n=L.shape[0], laplacian=L, seed=42)
fsm.curvature = CurvatureField(n=fsm.n, ricci_scalar=ricci, kretschmann=kretz)

# Optional: add slow potential bias (uncomment if desired)
# fsm.inject_curvature_as_potential(prefer="kretschmann", scale=0.02)

# Pre metrics
s0 = fsm.status()
t0 = time.time()

# Long evolve: 30 minutes with checkpoints every 60s
fsm.evolve_lattice(seconds=1800, dt=5e-4, method="auto",
                   conserve_mass=True, curvature_every=100,
                   checkpoint_every_s=60.0)

t1 = time.time()
s1 = fsm.status()

# Save outputs
stamp = time.strftime("%Y%m%d_%H%M%S")
npz_path = os.path.join(ART, f"fsm_final_{stamp}.npz")
fsm.to_npz(npz_path)

record = {
    "stamp": stamp,
    "dt": 5e-4,
    "method": "auto",
    "curvature_every": 100,
    "secs": t1 - t0,
    "before": s0,
    "after": s1,
    "drift": {k: s1[k]-s0[k] for k in s0},
    "snapshot": npz_path
}

logfile = os.path.join(ART, "ricci_burn_log.jsonl")
with open(logfile, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\\n")

print(json.dumps(record, indent=2))
