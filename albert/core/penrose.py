import sympy as sp
from albert.core.tensors import TensorField
from albert.core.curvature import compute_ricci_tensor


def energy_condition_satisfied(ricci: TensorField, null_vector: list[sp.Expr]) -> sp.Expr:
    coords = ricci.coords
    dim = len(coords)
    Rkk = 0
    for mu in range(dim):
        for nu in range(dim):
            R = ricci.get_component((mu, nu))
            Rkk += R * null_vector[mu] * null_vector[nu]
    return sp.simplify(Rkk)


def null_vector_template(coords):
    return [sp.Function(f'k_{i}')(sp.Symbol('λ')) for i in range(len(coords))]


def convergence_scalars(metric: TensorField, surface_point: dict, tetrad: dict[str, list[sp.Expr]]) -> tuple[sp.Expr, sp.Expr]:
    # Computes expansion (real part) of null congruences in directions l and n
    # tetrad: {'l': [...], 'n': [...], 'm': [...], 'm_bar': [...]}
    coords = metric.coords
    dim = len(coords)
    m = tetrad['m']
    m_bar = tetrad['m_bar']
    l = tetrad['l']
    n = tetrad['n']

    # Construct Levi-Civita connection (Christoffel symbols)
    g = metric.symbols.reshape((dim, dim))
    Gamma = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
    g_inv = sp.Matrix(g).inv()
    for mu in range(dim):
        for nu in range(dim):
            for lam in range(dim):
                Gamma[mu][nu][lam] = 0.5 * sum(
                    g_inv[mu, sigma] * (
                        sp.diff(g[sigma, nu], coords[lam]) +
                        sp.diff(g[sigma, lam], coords[nu]) -
                        sp.diff(g[nu, lam], coords[sigma])
                    )
                    for sigma in range(dim)
                )

    # Compute ρ = mᵃ m̄ᵇ ∇_b l_a
    ρ = sum(
        m[a] * m_bar[b] * sum(
            l[c] * Gamma[c][b][a] for c in range(dim)
        ) for a in range(dim) for b in range(dim)
    )

    # Compute ρ′ = m̄ᵃ mᵇ ∇_b n_a
    ρp = sum(
        m_bar[a] * m[b] * sum(
            n[c] * Gamma[c][b][a] for c in range(dim)
        ) for a in range(dim) for b in range(dim)
    )

    return (sp.simplify(ρ), sp.simplify(ρp))


def is_trapped_surface(ρ: sp.Expr, ρp: sp.Expr) -> bool:
    ρ_real = sp.re(ρ)
    ρp_real = sp.re(ρp)
    return (ρ_real > 0 and ρp_real > 0)


def is_geodesic_incomplete(λ_end: float, λ_max: float = 100) -> bool:
    return λ_end < λ_max


def inject_trapped_surface_flag_into_ψmesh(surface_id: str, ρ: sp.Expr, ρp: sp.Expr):
    print(f"🧠 ψMesh ⟶ Injecting Trapped Surface: {surface_id}")
    print(f"   ρ  = {ρ}")
    print(f"   ρ′ = {ρp}")
    print(f"   ⇒ Setting memory compression zone at {surface_id}")


def inject_incompleteness_flag_into_ψmesh(path_id: str, λ_end: float):
    print(f"🧠 ψMesh ⟶ Incomplete Geodesic Detected: {path_id}")
    print(f"   λ_end = {λ_end:.3f} ⇒ signaling memory collapse")
