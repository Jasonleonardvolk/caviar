from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
### File: {PROJECT_ROOT}\albert\core\curvature.py

import sympy as sp
from albert.core.tensors import TensorField
from typing import List


def compute_riemann_tensor(metric: TensorField) -> TensorField:
    coords = metric.coords
    dim = len(coords)
    g = metric.symbols.reshape((dim, dim))
    g_inv = sp.Matrix(g).inv()

    Gamma = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]

    for mu in range(dim):
        for nu in range(dim):
            for lam in range(dim):
                term = 0
                for sigma in range(dim):
                    term += g_inv[mu, sigma] * (
                        sp.diff(g[sigma, nu], coords[lam]) +
                        sp.diff(g[sigma, lam], coords[nu]) -
                        sp.diff(g[nu, lam], coords[sigma])
                    )
                Gamma[mu][nu][lam] = sp.simplify(0.5 * term)

    Riemann = TensorField("Riemann", (1, 3), coords)

    for rho in range(dim):
        for sigma in range(dim):
            for mu in range(dim):
                for nu in range(dim):
                    term1 = sp.diff(Gamma[rho][nu][sigma], coords[mu])
                    term2 = sp.diff(Gamma[rho][mu][sigma], coords[nu])
                    term3 = sum(Gamma[rho][mu][lam] * Gamma[lam][nu][sigma] for lam in range(dim))
                    term4 = sum(Gamma[rho][nu][lam] * Gamma[lam][mu][sigma] for lam in range(dim))
                    component = sp.simplify(term1 - term2 + term3 - term4)
                    Riemann.set_component((rho, sigma, mu, nu), component)

    return Riemann


def compute_ricci_tensor(riemann: TensorField) -> TensorField:
    coords = riemann.coords
    dim = len(coords)
    Ricci = TensorField("Ricci", (0, 2), coords)

    for mu in range(dim):
        for nu in range(dim):
            term = 0
            for rho in range(dim):
                term += riemann.get_component((rho, mu, rho, nu))
            Ricci.set_component((mu, nu), sp.simplify(term))

    return Ricci


def compute_kretschmann_scalar(riemann: TensorField) -> sp.Expr:
    coords = riemann.coords
    dim = len(coords)
    total = 0
    for rho in range(dim):
        for sigma in range(dim):
            for mu in range(dim):
                for nu in range(dim):
                    R = riemann.get_component((rho, sigma, mu, nu))
                    total += R * R
    return sp.simplify(total)