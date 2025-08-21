import sys, os, json, argparse, numpy as np
sys.path.insert(0, r"D:\Dev\kha")
from python.core.ricci_flow import (
    load_laplacian_from_npz,
    compute_curvature_fields_from_L,
    save_curvature_fields,
)


def main():
    p = argparse.ArgumentParser(description="Compute discrete curvature fields (Ricci, Kretschmann, mean) from a Laplacian npz")
    p.add_argument("--lap", required=True, help="Path to Laplacian .npz (L_norm.npz or compatible)")
    p.add_argument("--mode", default="auto", choices=["auto","combinatorial","normalized"], help="Laplacian type")
    p.add_argument("--outdir", default=r"D:\\Dev\\kha\\data\\curvature")
    p.add_argument("--outfile", default=r"D:\\Dev\\kha\\artifacts\\curvature_report.json")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    L = load_laplacian_from_npz(args.lap)
    fields = compute_curvature_fields_from_L(L, mode=args.mode)
    r,k,m = save_curvature_fields(args.outdir, fields)

    out = {
        "lap": args.lap,
        "mode": args.mode,
        "outdir": args.outdir,
        "saved": {"ricci": r, "kretschmann": k, "mean_curvature": m},
        "stats": {
            "ricci": {"mean": float(np.mean(fields.ricci)), "std": float(np.std(fields.ricci))},
            "kretschmann": {"mean": float(np.mean(fields.kretschmann)), "std": float(np.std(fields.kretschmann))},
            "mean_curvature": {"mean": float(np.mean(fields.mean_curvature)), "std": float(np.std(fields.mean_curvature))},
        }
    }
    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
