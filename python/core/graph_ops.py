# SPDX-License-Identifier: Apache-2.0
# Location: python/core/graph_ops.py

from __future__ import annotations
import numpy as np
import scipy.sparse as sp

def load_laplacian(path_npz: str) -> sp.csr_matrix:
    """
    Load a normalized Laplacian stored via scipy.sparse.save_npz(...).
    """
    L = sp.load_npz(path_npz)
    return L.tocsr()
