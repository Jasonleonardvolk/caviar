# D:\Dev\kha\python\bench\resonance_bench.py
from __future__ import annotations
import argparse, json, platform, subprocess, os, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
try:
    import scipy.sparse as sp
except Exception:
    sp = None

from python.core.graph_ops import load_laplacian
from python.bench.baselines import ppr_scores, heat_kernel_scores, simrank_lite_scores

def load_concepts(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))

def cosine_scores(E: np.ndarray, q_idx: int) -> np.ndarray:
    q = E[q_idx]
    num = E @ q
    denom = (np.linalg.norm(E, axis=1) * (np.linalg.norm(q) + 1e-12)) + 1e-12
    return num / denom

def resonance_scores(L, n: int, q_idx: int, T: int = 20, dt: float = 0.2, kappa: float = 1.0, lam: float = 0.0) -> np.ndarray:
    psi = np.zeros(n, dtype=np.complex128); psi[q_idx] = 1.0 + 0.0j
    def Ldot(x):
        if sp is not None and sp.issparse(L): return L @ x
        return L.dot(x)
    for _ in range(T):
        grad = kappa * Ldot(psi) + lam * (np.abs(psi) ** 2) * psi
        psi = psi - dt * grad
    return np.abs(psi)

def pr_at_k(truth: List[int], pred: List[int], k: int) -> Tuple[float,float]:
    P = sum(1 for x in pred[:k] if x in truth) / max(1, k)
    R = sum(1 for x in pred[:k] if x in truth) / max(1, len(truth))
    return P, R

def auc_binary(scores: np.ndarray, truth_idx: List[int]) -> float:
    y_true = np.zeros_like(scores); y_true[truth_idx] = 1.0
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tp = tp / max(1, tp[-1])
    fp = fp / max(1, fp[-1])
    # trapezoidal
    return float(np.trapz(tp, fp))

def hot_swap(L, m: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    if sp is not None and sp.issparse(L):
        L = L.tolil(copy=True); n = L.shape[0]; picks = 0
        while picks < m:
            i = rng.integers(0, n); j = rng.integers(0, n)
            if i == j or L[i, j] == 0: continue
            factor = float(rng.uniform(0.9, 1.1))
            v = L[i, j] * factor
            L[i, j] = v; L[j, i] = v; picks += 1
        return L.tocsr()
    else:
        L = L.copy(); n = L.shape[0]; picks = 0
        while picks < m:
            i = rng.integers(0, n); j = rng.integers(0, n)
            if i == j or L[i, j] == 0: continue
            factor = float(rng.uniform(0.9, 1.1))
            v = L[i, j] * factor
            L[i, j] = v; L[j, i] = v; picks += 1
        return L

def jaccard(a: List[int], b: List[int], k: int) -> float:
    A = set(a[:k]); B = set(b[:k])
    inter = len(A & B); union = len(A | B)
    return float(inter / max(1, union))

def write_manifest(outdir: Path, lap_path: str):
    def git_head():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=outdir.parent).decode().strip()
        except Exception:
            return None
    def sha256(p: str):
        import hashlib
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1<<20), b""):
                h.update(chunk)
        return h.hexdigest()
    manifest = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "git_head": git_head(),
        "laplacian_path": lap_path,
        "laplacian_sha256": sha256(lap_path),
        "env": {k:v for k,v in os.environ.items() if k.startswith("IRIS_") or k.startswith("VITE_")}
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default=r"D:\Dev\kha\data\bench\concepts.json")
    ap.add_argument("--laplacian", default=r"D:\Dev\kha\data\concept_mesh\L_norm.npz")
    ap.add_argument("--outdir", default=r"D:\Dev\kha\reports\resonance_bench")
    ap.add_argument("--klist", default="1,3,5,10")
    ap.add_argument("--edit_m", type=int, default=0, help="if >0, evaluate Jaccard stability under m random reweights")
    ap.add_argument("--include", default="cosine,resonance,ppr,heat,simrank", help="comma list")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    write_manifest(outdir, args.laplacian)

    cfg = load_concepts(Path(args.concepts))
    ids: List[str] = cfg["ids"]
    id2idx = {s:i for i,s in enumerate(ids)}
    queries = cfg["queries"]
    L = load_laplacian(args.laplacian)
    n = len(ids)

    E = None
    if "embeddings" in cfg:
        embp = Path(cfg["embeddings"])
        if embp.suffix == ".npz":
            npz = np.load(embp); E = npz["vecs"]
        elif embp.suffix == ".npy":
            E = np.load(embp)
        assert E is None or E.shape[0] == n

    klist = [int(x) for x in args.klist.split(",") if x.strip()]
    which = set(s.strip() for s in args.include.split(","))
    buckets = {name: {k: {"P": [], "R": [], "AUC": []} for k in klist} for name in which}

    # main scoring loop
    for q in queries:
        q_idx = id2idx[q["id"]]
        truth = [id2idx[x] for x in q["positives"] if x in id2idx]

        SCORES = {}
        if "cosine" in which and E is not None:
            SCORES["cosine"] = cosine_scores(E, q_idx)
        if "resonance" in which:
            SCORES["resonance"] = resonance_scores(L, n, q_idx)
        if "ppr" in which:
            SCORES["ppr"] = ppr_scores(L, q_idx)
        if "heat" in which:
            SCORES["heat"] = heat_kernel_scores(L, q_idx)
        if "simrank" in which:
            SCORES["simrank"] = simrank_lite_scores(L, q_idx)

        for name, s in SCORES.items():
            rank = list(np.argsort(-s))
            for k in klist:
                P, R = pr_at_k(truth, rank, k)
                buckets[name][k]["P"].append(P)
                buckets[name][k]["R"].append(R)
            buckets[name][klist[-1]]["AUC"].append(auc_binary(s, truth))

    # aggregate & write PR/AUC
    out = {name: {k: {"P": float(np.mean(v["P"])), "R": float(np.mean(v["R"])),
                      "AUC": float(np.mean(v["AUC"])) if v["AUC"] else None}
                  for k, v in kk.items()} for name, kk in buckets.items()}
    (outdir / "topk_pr.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # stability under hot-swap
    if args.edit_m > 0 and "resonance" in which:
        L2 = hot_swap(L, m=args.edit_m, seed=42)
        stab = {}
        for k in klist:
            j_list = []
            for q in queries:
                q_idx = id2idx[q["id"]]
                s1 = resonance_scores(L, n, q_idx)
                s2 = resonance_scores(L2, n, q_idx)
                r1 = list(np.argsort(-s1)); r2 = list(np.argsort(-s2))
                j_list.append(jaccard(r1, r2, k))
            stab[k] = {"Jaccard": float(np.mean(j_list))}
        (outdir / "stability.json").write_text(json.dumps(stab, indent=2), encoding="utf-8")

    print(f"[OK] Wrote {outdir}\\topk_pr.json", ("and stability.json" if args.edit_m>0 else ""))

if __name__ == "__main__":
    main()
