from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
### File: {PROJECT_ROOT}\albert\core\tetrads.py

import sympy as sp
from typing import List, Dict
from albert.core.tensors import TensorField

def canonical_kerr_tetrad(coords: List[str], r_val: float, theta_val: float, M=1, a=0.5) -> Dict[str, List[sp.Expr]]:
    """
    Returns a symbolic null tetrad for Kerr spacetime at given (r, θ)
    in Boyer-Lindquist coordinates.
    """
    t, r, th, phi = sp.symbols(coords)
    Σ = r**2 + a**2 * sp.cos(th)**2
    Δ = r**2 - 2 * M * r + a**2

    # Coordinate basis vectors (not normalized):
    l_vec = [(r**2 + a**2)/Δ, 1, 0, a/Δ]  # outgoing null
    n_vec = [0.5 * (r**2 + a**2)/Σ, -0.5 * Δ/Σ, 0, 0.5 * a/Σ]  # ingoing null
    m_vec = [0, 0, 1/sp.sqrt(2*Σ), sp.I / (sp.sqrt(2*Σ) * sp.sin(th))]

    # Substitute point
    subs = {r: r_val, th: theta_val}
    l_sub = [sp.simplify(expr.subs(subs)) for expr in l_vec]
    n_sub = [sp.simplify(expr.subs(subs)) for expr in n_vec]
    m_sub = [sp.simplify(expr.subs(subs)) for expr in m_vec]
    m_bar_sub = [sp.conjugate(e) for e in m_sub]

    return {
        "l": l_sub,
        "n": n_sub,
        "m": m_sub,
        "m_bar": m_bar_sub
    }