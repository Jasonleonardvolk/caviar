# D:\Dev\kha\python\tools\build_typed_laplacian.py
"""
Standalone typed Laplacian builder for multi-relation graphs.
Avoids package-level imports to prevent torch DLL issues.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np

try:
    import scipy.sparse as sp
except Exception:
    sp = None
    raise RuntimeError("scipy is required for sparse matrix operations. Install with: pip install scipy")


def adjacency_from_edgelist(n: int, edges: Iterable[Tuple[int, int, float]]):
    """Build symmetric weighted adjacency from (i, j, w) edges for i,j in [0,n)."""
    rows, cols, vals = [], [], []
    for i, j, w in edges:
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    return A


def normalize_laplacian_from_adjacency(A) -> "sp.csr_matrix | np.ndarray":
    """
    Symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    Works with scipy.sparse or dense np.ndarray.
    """
    if sp is not None and sp.issparse(A):
        d = np.asarray(A.sum(axis=1)).ravel()
        d_inv_sqrt = np.zeros_like(d)
        nz = d > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        I = sp.eye(A.shape[0], format="csr")
        L = I - (D_inv_sqrt @ (A @ D_inv_sqrt))
        return L.tocsr()
    else:
        d = A.sum(axis=1)
        d_inv_sqrt = np.zeros_like(d)
        nz = d > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        n = A.shape[0]
        I = np.eye(n)
        return I - D_inv_sqrt @ A @ D_inv_sqrt


def save_laplacian(L, path: str | Path) -> None:
    """Save Laplacian matrix to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if sp is not None and sp.issparse(L):
        sp.save_npz(path, L.tocsr())
    else:
        np.save(path.with_suffix(".npy"), L)


def read_edges_json(p: Path):
    """Read edges from JSON file"""
    obj = json.loads(p.read_text(encoding="utf-8"))
    return int(obj["n"]), [(int(i), int(j), float(w)) for i,j,w in obj["edges"]]


def main(schema_path: str = r"D:\Dev\kha\data\concept_mesh\schema.json",
         out_path: str = r"D:\Dev\kha\data\concept_mesh\L_norm_typed.npz"):
    """
    Build a typed Laplacian from multiple relation types.
    
    Schema format:
    {
      "relations": [
        { "name": "theory",  "edges": "path/to/theory_edges.json",  "weight": 0.5 },
        { "name": "hardware","edges": "path/to/hardware_edges.json","weight": 0.3 },
        { "name": "people",  "edges": "path/to/people_edges.json",  "weight": 0.2 }
      ]
    }
    
    Edge file format:
    {
      "n": 100,
      "edges": [[0, 1, 1.0], [1, 2, 0.5], ...]
    }
    """
    schema_path = Path(schema_path)
    out_path = Path(out_path)
    
    if not schema_path.exists():
        print(f"Error: Schema file not found at {schema_path}")
        print("\nCreating example schema.json...")
        example_schema = {
            "relations": [
                {
                    "name": "example",
                    "edges": str(schema_path.parent / "edges.json"),
                    "weight": 1.0,
                    "description": "Example relation type"
                }
            ]
        }
        schema_path.write_text(json.dumps(example_schema, indent=2))
        print(f"Created example schema at {schema_path}")
        print("Please update it with your actual edge files and run again.")
        return
    
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Lsum = None
    n_ref = None
    
    print(f"Building typed Laplacian from {len(schema['relations'])} relation types...")
    
    for rel in schema["relations"]:
        edge_path = Path(rel["edges"])
        if not edge_path.exists():
            print(f"Warning: Edge file not found for '{rel['name']}' at {edge_path}")
            print("Skipping this relation type...")
            continue
            
        n, edges = read_edges_json(edge_path)
        if n_ref is None:
            n_ref = n
            print(f"Graph size: {n} nodes")
        
        if n != n_ref:
            raise ValueError(f"All relations must have the same node count. Expected {n_ref}, got {n} for '{rel['name']}'")
        
        print(f"  - Processing '{rel['name']}' with {len(edges)} edges, weight={rel.get('weight', 1.0)}")
        
        A = adjacency_from_edgelist(n, edges)
        Lr = normalize_laplacian_from_adjacency(A)
        w = float(rel.get("weight", 1.0))
        
        if Lsum is None:
            Lsum = Lr * w
        else:
            Lsum = Lsum + Lr * w
    
    if Lsum is None:
        print("Error: No valid edge files found. Cannot build Laplacian.")
        return
    
    save_laplacian(Lsum, out_path)
    print(f"\n[OK] Wrote typed Laplacian -> {out_path}")
    
    # Print some statistics
    if sp.issparse(Lsum):
        print(f"     Matrix size: {Lsum.shape}")
        print(f"     Non-zero entries: {Lsum.nnz}")
        print(f"     Sparsity: {1 - Lsum.nnz / (Lsum.shape[0] * Lsum.shape[1]):.2%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build typed Laplacian from multi-relation graph")
    parser.add_argument("--schema", default=r"D:\Dev\kha\data\concept_mesh\schema.json",
                        help="Path to schema JSON file")
    parser.add_argument("--output", default=r"D:\Dev\kha\data\concept_mesh\L_norm_typed.npz",
                        help="Output path for typed Laplacian")
    args = parser.parse_args()
    
    main(schema_path=args.schema, out_path=args.output)
