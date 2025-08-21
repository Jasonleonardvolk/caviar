import os, sys, time, argparse, json
import numpy as np

ROOT = r"D:\\Dev\\kha"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from python.core.fractal_soliton_memory import (
    FractalSolitonMemory, load_laplacian_from_npz
)

def main():
    p = argparse.ArgumentParser(description="FSM pre-burn acceptance check")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--method", default="auto", choices=["auto","rk4-builtin","rk4-ext","stormer","jit"])
    p.add_argument("--lap", default=None, help="Path to Laplacian .npz (optional)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--outfile", default=os.path.join(ROOT, "artifacts", "fsm_preburn.json"))
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    if args.lap and os.path.exists(args.lap):
        L = load_laplacian_from_npz(args.lap)
        n = L.shape[0]
    else:
        n = args.n
        L = np.eye(n, dtype=float)

    fsm = FractalSolitonMemory.from_random(n=n, laplacian=L, seed=args.seed)
    s0 = fsm.status()

    t0 = time.time()
    fsm.step(steps=args.steps, dt=args.dt, method=args.method, conserve_mass=True)
    t1 = time.time()

    s1 = fsm.status()

    out = {
        "n": n,
        "steps": args.steps,
        "dt": args.dt,
        "method": args.method,
        "t_sec": t1 - t0,
        "before": s0,
        "after": s1,
        "drift": {
            "mass": s1["mass"] - s0["mass"],
            "energy": s1["energy"] - s0["energy"],
            "coherence": s1["coherence"] - s0["coherence"],
        },
    }

    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
