import sys, os, json, time, argparse
import numpy as np

# Ensure project import
sys.path.insert(0, r"D:\Dev\kha")
from python.core.fractal_soliton_memory import (
    FractalSolitonMemory,
    CurvatureField,
    load_laplacian_from_npz,
)

"""
2-minute curvature burn with explicit cadence and gain control.
Precisely schedules injections at global step boundaries (no accidental batching).
Writes JSON + NPZ artifacts to artifacts/.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=120.0)
    ap.add_argument('--dt', type=float, default=5e-4)
    ap.add_argument('--method', default='auto', choices=['auto','rk4-builtin','rk4-ext','stormer','jit'])
    ap.add_argument('--curvature-every', type=int, default=100)
    ap.add_argument('--phase-gain', type=float, default=0.15)
    ap.add_argument('--kcrit', type=float, default=6.0)
    ap.add_argument('--prefer', default='kretschmann', choices=['kretschmann','ricci','mean'])
    ap.add_argument('--potential-scale', type=float, default=0.0, help='Optional slow bias; 0 disables')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lap', default=r'D:\\Dev\\kha\\data\\concept_mesh\\L_norm.npz')
    ap.add_argument('--outdir', default=r'D:\\Dev\\kha\\artifacts')
    args = ap.parse_args()

    ROOT = r'D:\\Dev\\kha'
    ART = args.outdir
    os.makedirs(ART, exist_ok=True)

    # Load mesh + curvature
    L = load_laplacian_from_npz(args.lap)
    n = L.shape[0]
    ricci = np.load(os.path.join(ROOT, 'data', 'curvature', 'ricci.npy'))
    kretz = np.load(os.path.join(ROOT, 'data', 'curvature', 'kretschmann.npy'))

    # FSM init
    fsm = FractalSolitonMemory.from_random(n=n, laplacian=L, seed=args.seed)
    fsm.curvature = CurvatureField(n=n, ricci_scalar=ricci, kretschmann=kretz)

    # Optional: slow potential bias once at start
    if args.potential_scale and args.potential_scale > 0.0:
        fsm.inject_curvature_as_potential(prefer=args.prefer, scale=float(args.potential_scale))

    # Pre metrics
    s0 = fsm.status()
    t0 = time.time()

    total_steps = int(args.seconds / args.dt)
    done = 0
    inject_every = max(1, int(args.curvature_every))
    next_inject = inject_every

    progress_grain = max(10_000, total_steps // 50)
    next_ckpt_t = t0 + 60.0

    while done < total_steps:
        # Step just enough to hit the next injection boundary (or finish)
        if done < next_inject <= total_steps:
            chunk = min(next_inject - done, total_steps - done)
        else:
            chunk = min(progress_grain, total_steps - done)
        fsm.step(steps=chunk, dt=args.dt, method=args.method, conserve_mass=True, curvature_every=None)
        done += chunk

        if done == next_inject:
            fsm.inject_curvature_field(
                phase_mode='log_tanh',
                phase_gain=float(args.phase_gain),
                kcrit=float(args.kcrit),
                prefer=args.prefer,
                inplace=True,
            )
            next_inject += inject_every

        if time.time() >= next_ckpt_t:
            fsm.save_checkpoint(reason='ricci_burn_2min_progress')
            next_ckpt_t += 60.0

    t1 = time.time()
    s1 = fsm.status()

    stamp = time.strftime('%Y%m%d_%H%M%S')
    base = f'fsm_2min_{stamp}_ce{args.curvature_every}_pg{args.phase_gain:.2f}'
    npz_path = os.path.join(ART, base + '.npz')
    fsm.to_npz(npz_path)

    record = {
        'stamp': stamp,
        'seconds': args.seconds,
        'dt': args.dt,
        'method': args.method,
        'curvature_every': args.curvature_every,
        'phase_gain': args.phase_gain,
        'kcrit': args.kcrit,
        'prefer': args.prefer,
        'potential_scale': args.potential_scale,
        'secs': t1 - t0,
        'before': s0,
        'after': s1,
        'drift': {k: s1[k] - s0[k] for k in s0},
        'snapshot': npz_path,
    }

    out_json = os.path.join(ART, base + '.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2)

    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    main()
