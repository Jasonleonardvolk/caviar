"""
ALBERT Geodesics Module
Numerical integration of geodesic equations in curved spacetime
"""

import numpy as np
import sympy as sp
from typing import List, Tuple, Optional, Dict, Any
from scipy.integrate import solve_ivp
from albert.core.tensors import TensorField


def christoffel_symbols(metric: TensorField) -> Dict[Tuple[int, int, int], Any]:
    """
    Compute Christoffel symbols from metric tensor
    
    Γ^μ_νρ = (1/2) g^μσ (∂_ν g_σρ + ∂_ρ g_σν - ∂_σ g_νρ)
    
    Args:
        metric: The metric tensor field
        
    Returns:
        Dictionary of Christoffel symbols indexed by (mu, nu, rho)
    """
    dim = len(metric.coords)
    coord_symbols = sp.symbols(metric.coords)
    
    # First, we need the inverse metric
    # Convert metric to matrix form
    g_matrix = sp.Matrix.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            g_matrix[i, j] = metric.get_component((i, j))
    
    # Compute inverse metric
    g_inv = g_matrix.inv()
    
    # Compute Christoffel symbols
    christoffel = {}
    
    for mu in range(dim):
        for nu in range(dim):
            for rho in range(dim):
                # Sum over sigma
                gamma_sum = 0
                for sigma in range(dim):
                    # Get metric components
                    g_sigma_rho = metric.get_component((sigma, rho))
                    g_sigma_nu = metric.get_component((sigma, nu))
                    g_nu_rho = metric.get_component((nu, rho))
                    
                    # Compute derivatives
                    dg_sigma_rho_dnu = sp.diff(g_sigma_rho, coord_symbols[nu])
                    dg_sigma_nu_drho = sp.diff(g_sigma_nu, coord_symbols[rho])
                    dg_nu_rho_dsigma = sp.diff(g_nu_rho, coord_symbols[sigma])
                    
                    # Apply Christoffel formula
                    term = g_inv[mu, sigma] * (
                        dg_sigma_rho_dnu + dg_sigma_nu_drho - dg_nu_rho_dsigma
                    )
                    gamma_sum += term
                
                christoffel[(mu, nu, rho)] = sp.simplify(gamma_sum / 2)
    
    return christoffel


def geodesic_equations(t, y, metric_func, christoffel_func, coords):
    """
    Geodesic equations in first-order form
    
    d²x^μ/dλ² + Γ^μ_νρ (dx^ν/dλ)(dx^ρ/dλ) = 0
    
    Args:
        t: Affine parameter λ
        y: State vector [x^0, x^1, x^2, x^3, dx^0/dλ, dx^1/dλ, dx^2/dλ, dx^3/dλ]
        metric_func: Function to evaluate metric at given coordinates
        christoffel_func: Function to evaluate Christoffel symbols
        coords: Coordinate symbols
        
    Returns:
        Derivatives [velocities, accelerations]
    """
    dim = len(coords)
    pos = y[:dim]  # positions
    vel = y[dim:]  # velocities
    
    # Evaluate coordinate values
    coord_vals = {coords[i]: pos[i] for i in range(dim)}
    
    # Compute accelerations from geodesic equation
    acc = np.zeros(dim)
    
    for mu in range(dim):
        # Sum over nu and rho
        gamma_sum = 0
        for nu in range(dim):
            for rho in range(dim):
                # Get Christoffel symbol value
                gamma = christoffel_func[(mu, nu, rho)]
                if gamma != 0:
                    # Evaluate at current position
                    gamma_val = float(gamma.subs(coord_vals))
                    gamma_sum += gamma_val * vel[nu] * vel[rho]
        
        acc[mu] = -gamma_sum
    
    # Return derivatives: [velocities, accelerations]
    return np.concatenate([vel, acc])


def numeric_geodesic_solver(
    metric: TensorField,
    initial_position: List[float],
    initial_velocity: List[float],
    lam_span: Tuple[float, float] = (0, 10),
    steps: int = 500,
    method: str = 'RK45',
    rtol: float = 1e-8,
    atol: float = 1e-10
):
    """
    Numerically integrate geodesic equations
    
    Args:
        metric: The metric tensor field
        initial_position: Initial coordinates [t, r, theta, phi]
        initial_velocity: Initial 4-velocity [dt/dλ, dr/dλ, dθ/dλ, dφ/dλ]
        lam_span: Integration range for affine parameter λ
        steps: Number of evaluation points
        method: Integration method ('RK45', 'RK23', 'DOP853', etc.)
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Solution object with trajectory data
    """
    # Get coordinate symbols
    coord_symbols = sp.symbols(metric.coords)
    dim = len(coord_symbols)
    
    # Validate inputs
    if len(initial_position) != dim:
        raise ValueError(f"Initial position must have {dim} components")
    if len(initial_velocity) != dim:
        raise ValueError(f"Initial velocity must have {dim} components")
    
    print("🔧 Computing Christoffel symbols...")
    # Compute Christoffel symbols symbolically
    christoffel = christoffel_symbols(metric)
    
    # Cache for numerical evaluation
    christoffel_lambdified = {}
    for key, expr in christoffel.items():
        if expr != 0:
            christoffel_lambdified[key] = sp.lambdify(coord_symbols, expr, 'numpy')
        else:
            christoffel_lambdified[key] = lambda *args: 0
    
    print("🚀 Integrating geodesic equations...")
    
    # Define the ODE system
    def ode_func(t, y):
        pos = y[:dim]
        vel = y[dim:]
        
        # Compute accelerations
        acc = np.zeros(dim)
        
        for mu in range(dim):
            gamma_sum = 0
            for nu in range(dim):
                for rho in range(dim):
                    if (mu, nu, rho) in christoffel_lambdified:
                        try:
                            gamma_val = christoffel_lambdified[(mu, nu, rho)](*pos)
                            gamma_sum += gamma_val * vel[nu] * vel[rho]
                        except:
                            # Handle singularities
                            pass
            
            acc[mu] = -gamma_sum
        
        return np.concatenate([vel, acc])
    
    # Initial conditions
    y0 = np.concatenate([initial_position, initial_velocity])
    
    # Solve the ODE
    sol = solve_ivp(
        ode_func,
        lam_span,
        y0,
        method=method,
        t_eval=np.linspace(lam_span[0], lam_span[1], steps),
        rtol=rtol,
        atol=atol,
        dense_output=True
    )
    
    # Add coordinate labels to solution
    sol.coords = metric.coords
    sol.positions = sol.y[:dim, :]
    sol.velocities = sol.y[dim:, :]
    
    # Compute 4-velocity norm (should be constant)
    def four_velocity_norm(pos, vel):
        """Compute g_μν v^μ v^ν"""
        norm = 0
        coord_dict = {coord_symbols[i]: pos[i] for i in range(dim)}
        
        for mu in range(dim):
            for nu in range(dim):
                g_mu_nu = metric.get_component((mu, nu))
                if g_mu_nu != 0:
                    g_val = float(g_mu_nu.subs(coord_dict))
                    norm += g_val * vel[mu] * vel[nu]
        return norm
    
    # Check conservation
    sol.velocity_norms = []
    for i in range(len(sol.t)):
        norm = four_velocity_norm(sol.positions[:, i], sol.velocities[:, i])
        sol.velocity_norms.append(norm)
    
    sol.velocity_norms = np.array(sol.velocity_norms)
    
    print(f"✅ Geodesic integrated successfully!")
    print(f"   Steps: {len(sol.t)}")
    print(f"   λ range: [{lam_span[0]}, {lam_span[1]}]")
    print(f"   Velocity norm variation: {np.std(sol.velocity_norms):.2e}")
    
    return sol


def is_null_geodesic(solution, tolerance: float = 1e-6) -> bool:
    """
    Check if a geodesic is null (lightlike)
    
    Args:
        solution: Solution from numeric_geodesic_solver
        tolerance: Tolerance for null condition
        
    Returns:
        True if geodesic is null (photon trajectory)
    """
    mean_norm = np.mean(solution.velocity_norms)
    return abs(mean_norm) < tolerance


def is_timelike_geodesic(solution, tolerance: float = 1e-6) -> bool:
    """
    Check if a geodesic is timelike (massive particle)
    
    Args:
        solution: Solution from numeric_geodesic_solver
        tolerance: Tolerance for checking
        
    Returns:
        True if geodesic is timelike
    """
    mean_norm = np.mean(solution.velocity_norms)
    return mean_norm < -tolerance


def extract_trajectory(solution) -> Dict[str, np.ndarray]:
    """
    Extract trajectory components from solution
    
    Args:
        solution: Solution from numeric_geodesic_solver
        
    Returns:
        Dictionary with coordinate trajectories
    """
    result = {'lambda': solution.t}
    
    for i, coord in enumerate(solution.coords):
        result[coord] = solution.positions[i, :]
        result[f'd{coord}_dlambda'] = solution.velocities[i, :]
    
    result['velocity_norm'] = solution.velocity_norms
    
    return result
