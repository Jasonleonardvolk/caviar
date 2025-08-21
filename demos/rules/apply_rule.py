import sys, os, json, argparse
import numpy as np
sys.path.insert(0, r"D:\Dev\kha")
from python.core.fractal_soliton_memory import FractalSolitonMemory, CurvatureField, load_laplacian_from_npz

"""
Apply a rule from ledger to an FSM snapshot and produce a proof JSON.
Supported rules:
  - bias_potential: {"scale": float, "prefer": "kretschmann"|"ricci"|"mean"}
Usage:
  python demos\rules\apply_rule.py --npz artifacts\fsm_final_*.npz --rule '{"type":"bias_potential","payload":{"scale":0.02,"prefer":"kretschmann"}}'
"""

ROOT = r'D:\\Dev\\kha'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--rule', required=True, help='JSON of a single rule')
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--dt', type=float, default=5e-4)
    args = ap.parse_args()

    rule = json.loads(args.rule)
    L = load_laplacian_from_npz(os.path.join(ROOT, 'data', 'concept_mesh', 'L_norm.npz'))
    fsm = FractalSolitonMemory.from_npz(args.npz, laplacian=L)

    # load curvature if rule needs it
    ricci = np.load(os.path.join(ROOT, 'data', 'curvature', 'ricci.npy'))
    kretz = np.load(os.path.join(ROOT, 'data', 'curvature', 'kretschmann.npy'))
    fsm.curvature = CurvatureField(n=fsm.n, ricci_scalar=ricci, kretschmann=kretz)

    s0 = fsm.status()

    if rule.get('type') == 'bias_potential':
        payload = rule.get('payload', {})
        scale = float(payload.get('scale', 0.02))
        prefer = payload.get('prefer', 'kretschmann')
        fsm.inject_curvature_as_potential(prefer=prefer, scale=scale)
    else:
        raise ValueError('Unsupported rule: ' + str(rule))

    fsm.step(steps=args.steps, dt=args.dt, method='auto', conserve_mass=True)

    s1 = fsm.status()
    out = {
        'npz_in': args.npz,
        'rule': rule,
        'before': s0,
        'after': s1,
        'drift': {k: s1[k]-s0[k] for k in s0}
    }
    proof = os.path.join(ROOT, 'artifacts', 'demo', 'rules', 'apply_rule_proof.json')
    os.makedirs(os.path.dirname(proof), exist_ok=True)
    with open(proof, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
