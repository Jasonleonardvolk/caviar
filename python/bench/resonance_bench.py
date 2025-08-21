from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    import scipy.sparse as sp
except Exception:
    sp = None

from python.core.graph_ops import load_laplacian

def load_concepts(path: Path) -> Dict:
    """
    concepts.json schema:
    {
      "ids": ["a","b","c",...],
      "queries": [{"id": "a", "positives": ["b","c"]}, ...],
      "embeddings": "D:\\Dev\\kha\\data\\bench\\embeddings.npz"  # optional
    }
    """
    return json.loads(path.read_text(encoding="utf-8"))

def cosine_scores(E: np.ndarray, q_idx: int) -> np.ndarray:
    q = E[q_idx]
    num = E @ q
    denom = (np.linalg.norm(E, axis=1) * (np.linalg.norm(q) + 1e-12)) + 1e-12
    return num / denom

def resonance_scores(L, n: int, q_idx: int, T: int = 20, dt: float = 0.2, kappa: float = 1.0, lam: float = 0.0) -> np.ndarray:
    """
    Simulate gradient flow dpsi/dt = - (kappa L psi + lambda |psi|^2 psi) from a 1-hot at q_idx.
    Score = |psi| after T steps.
    """
    psi = np.zeros(n, dtype=np.complex128)
    psi[q_idx] = 1.0 + 0.0j

    def Ldot(x):
        if sp is not None and sp.issparse(L):
            return L @ x
        return L.dot(x)

    for _ in range(T):
        grad = kappa * Ldot(psi) + lam * (np.abs(psi) ** 2) * psi
        psi = psi - dt * grad
    return np.abs(psi)

def pr_at_k(truth: List[int], pred: List[int], k: int) -> Tuple[float,float]:
    P = sum(1 for x in pred[:k] if x in truth) / max(1, k)
    R = sum(1 for x in pred[:k] if x in truth) / max(1, len(truth))
    return P, R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default=r"D:\Dev\kha\data\bench\concepts.json")
    ap.add_argument("--laplacian", default=r"D:\Dev\kha\data\concept_mesh\L_norm.npz")
    ap.add_argument("--out", default=r"D:\Dev\kha\reports\resonance_bench\topk_pr.json")
    ap.add_argument("--klist", default="1,3,5,10")
    args = ap.parse_args()

    cfg = load_concepts(Path(args.concepts))
    ids: List[str] = cfg["ids"]
    id2idx = {s:i for i,s in enumerate(ids)}
    queries = cfg["queries"]

    L = load_laplacian(args.laplacian)
    n = len(ids)

    # embeddings optional
    E = None
    if "embeddings" in cfg:
        embp = Path(cfg["embeddings"])
        if embp.suffix == ".npz":
            npz = np.load(embp)
            E = npz["vecs"]
        elif embp.suffix == ".npy":
            E = np.load(embp)
        else:
            raise ValueError("embeddings must be .npz (vecs) or .npy")
        assert E.shape[0] == n, "Embedding rows must match ids."

    klist = [int(x) for x in args.klist.split(",") if x.strip()]
    agg = {"cosine": {k: {"P": [], "R": []} for k in klist},
           "resonance": {k: {"P": [], "R": []} for k in klist}}

    for q in queries:
        q_idx = id2idx[q["id"]]
        truth = [id2idx[x] for x in q["positives"] if x in id2idx]

        # cosine baseline (if available)
        if E is not None:
            cos = cosine_scores(E, q_idx)
            cos_rank = list(np.argsort(-cos))
            for k in klist:
                P, R = pr_at_k(truth, cos_rank, k)
                agg["cosine"][k]["P"].append(P)
                agg["cosine"][k]["R"].append(R)

        # resonance
        res = resonance_scores(L, n, q_idx)
        res_rank = list(np.argsort(-res))
        for k in klist:
            P, R = pr_at_k(truth, res_rank, k)
            agg["resonance"][k]["P"].append(P)
            agg["resonance"][k]["R"].append(R)

    # aggregate mean
    out = {m: {k: {"P": float(np.mean(v["P"])) if v["P"] else None,
                   "R": float(np.mean(v["R"])) if v["R"] else None}
               for k, v in res.items()}
           for m, res in agg.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {out_path}")

if __name__ == "__main__":
    main()
