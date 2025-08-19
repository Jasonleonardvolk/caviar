"""
🌌 GEODESIC INTEGRATION ENGINE
True path-resolved phase twist interpolator with ψ-trajectories

Computes total phase accumulation along geodesics through curved spacetime
"""

import numpy as np
import sympy as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad, simpson, odeint
from scipy.optimize import minimize_scalar
from typing import Tuple, List, Dict, Optional, Callable, Union
import logging
from dataclasses import dataclass
import json

# JIT support
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    njit = jit

logger = logging.getLogger("physics.geodesics")


@dataclass
class GeodesicPath:
    """Represents a geodesic path through spacetime"""
    parameter: np.ndarray  # Affine parameter values
    coordinates: np.ndarray  # Shape: (n_points, n_dims)
    metric: Optional[np.ndarray] = None  # Metric tensor along path
    connection: Optional[np.ndarray] = None  # Christoffel symbols
    curvature: Optional[np.ndarray] = None  # Curvature along path
    phase_twist: float = 0.0  # Total phase accumulated
    proper_length: float = 0.0  # Proper length of path


class GeodesicIntegrator:
    """
    🌀 Integrates geodesic equations and computes phase twists
    
    Features:
    - Symbolic geodesic equations from metric
    - Numerical integration with adaptive stepping
    - Phase accumulation along paths
    - Support for null, timelike, and spacelike geodesics
    """
    
    def __init__(self, metric_tensor: Optional[sp.Matrix] = None):
        self.metric = metric_tensor
        self.christoffel_symbols = None
        self.coordinates = []
        self.phase_interpolators = {}
        
        if metric_tensor is not None:
            self._compute_christoffel_symbols()
    
    def set_schwarzschild_metric(self, mass: float = 1.0):
        """Set up Schwarzschild metric"""
        # Coordinates: (t, r, theta, phi)
        t, r, theta, phi = sp.symbols('t r theta phi')
        self.coordinates = [t, r, theta, phi]
        
        # Schwarzschild metric components
        g = sp.zeros(4, 4)
        g[0, 0] = -(1 - 2*mass/r)  # g_tt
        g[1, 1] = 1/(1 - 2*mass/r)  # g_rr
        g[2, 2] = r**2  # g_θθ
        g[3, 3] = r**2 * sp.sin(theta)**2  # g_φφ
        
        self.metric = g
        self.mass = mass
        self._compute_christoffel_symbols()
        
        logger.info(f"🌌 Set Schwarzschild metric with M={mass}")
    
    def set_kerr_metric(self, mass: float = 1.0, spin: float = 0.5):
        """Set up Kerr metric (rotating black hole)"""
        t, r, theta, phi = sp.symbols('t r theta phi')
        self.coordinates = [t, r, theta, phi]
        
        # Kerr metric in Boyer-Lindquist coordinates
        a = spin
        Sigma = r**2 + a**2 * sp.cos(theta)**2
        Delta = r**2 - 2*mass*r + a**2
        
        g = sp.zeros(4, 4)
        g[0, 0] = -(1 - 2*mass*r/Sigma)
        g[0, 3] = -2*mass*r*a*sp.sin(theta)**2/Sigma
        g[3, 0] = g[0, 3]
        g[1, 1] = Sigma/Delta
        g[2, 2] = Sigma
        g[3, 3] = (r**2 + a**2 + 2*mass*r*a**2*sp.sin(theta)**2/Sigma) * sp.sin(theta)**2
        
        self.metric = g
        self.mass = mass
        self.spin = spin
        self._compute_christoffel_symbols()
        
        logger.info(f"🌌 Set Kerr metric with M={mass}, a={spin}")
    
    def _compute_christoffel_symbols(self):
        """Compute Christoffel symbols from metric"""
        if self.metric is None:
            raise ValueError("Metric not set")
        
        n = len(self.coordinates)
        g = self.metric
        
        # Inverse metric
        g_inv = g.inv()
        
        # Christoffel symbols: Γ^i_jk = 1/2 * g^il * (∂_j g_lk + ∂_k g_jl - ∂_l g_jk)
        self.christoffel_symbols = {}
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    gamma = 0
                    for l in range(n):
                        term1 = sp.diff(g[l, k], self.coordinates[j])
                        term2 = sp.diff(g[j, l], self.coordinates[k])
                        term3 = sp.diff(g[j, k], self.coordinates[l])
                        gamma += g_inv[i, l] * (term1 + term2 - term3) / 2
                    
                    # Simplify and store
                    gamma = sp.simplify(gamma)
                    if gamma != 0:
                        self.christoffel_symbols[(i, j, k)] = gamma
        
        logger.info(f"💫 Computed {len(self.christoffel_symbols)} non-zero Christoffel symbols")
    
    @staticmethod
    @njit
    def _geodesic_rhs_schwarzschild(state: np.ndarray, tau: float, M: float) -> np.ndarray:
        """
        JIT-compiled RHS for Schwarzschild geodesic equations
        state = [t, r, theta, phi, dt/dτ, dr/dτ, dθ/dτ, dφ/dτ]
        """
        t, r, theta, phi, dt_dtau, dr_dtau, dtheta_dtau, dphi_dtau = state
        
        # Prevent singularities
        if r <= 2*M:
            r = 2*M + 1e-6
        
        # Second derivatives (geodesic equation)
        d2t_dtau2 = 2*M/(r*(r-2*M)) * dt_dtau * dr_dtau
        
        d2r_dtau2 = (M*(r-2*M)/r**3) * dt_dtau**2 - (M/(r*(r-2*M))) * dr_dtau**2 + \
                    (r-2*M) * (dtheta_dtau**2 + np.sin(theta)**2 * dphi_dtau**2)
        
        d2theta_dtau2 = -2/r * dr_dtau * dtheta_dtau + np.sin(theta)*np.cos(theta) * dphi_dtau**2
        
        d2phi_dtau2 = -2/r * dr_dtau * dphi_dtau - 2*np.cos(theta)/np.sin(theta) * dtheta_dtau * dphi_dtau
        
        return np.array([dt_dtau, dr_dtau, dtheta_dtau, dphi_dtau,
                        d2t_dtau2, d2r_dtau2, d2theta_dtau2, d2phi_dtau2])
    
    def integrate_geodesic(self, 
                          initial_pos: np.ndarray,
                          initial_vel: np.ndarray,
                          tau_span: Tuple[float, float],
                          n_points: int = 100,
                          geodesic_type: str = "timelike") -> GeodesicPath:
        """
        🚀 Integrate geodesic equations
        
        Args:
            initial_pos: Initial coordinates [t, r, θ, φ]
            initial_vel: Initial 4-velocity
            tau_span: (tau_start, tau_end) affine parameter range
            n_points: Number of integration points
            geodesic_type: "timelike", "null", or "spacelike"
            
        Returns:
            GeodesicPath object with integrated path
        """
        logger.info(f"🚀 Integrating {geodesic_type} geodesic from r={initial_pos[1]:.2f}")
        
        # Normalize initial velocity based on geodesic type
        if self.metric is not None and hasattr(self, 'mass'):
            # For Schwarzschild
            if geodesic_type == "null":
                # Null geodesics: g_μν v^μ v^ν = 0
                # Adjust time component to satisfy null condition
                r = initial_pos[1]
                theta = initial_pos[2]
                
                # Calculate required dt/dτ for null geodesic
                spatial_norm = ((1/(1-2*self.mass/r)) * initial_vel[1]**2 + 
                               r**2 * initial_vel[2]**2 + 
                               r**2 * np.sin(theta)**2 * initial_vel[3]**2)
                
                initial_vel[0] = np.sqrt(spatial_norm / (1 - 2*self.mass/r))
        
        # Initial state vector
        initial_state = np.concatenate([initial_pos, initial_vel])
        
        # Integration parameter points
        tau_points = np.linspace(tau_span[0], tau_span[1], n_points)
        
        # Integrate using ODE solver
        if NUMBA_AVAILABLE and hasattr(self, 'mass'):
            # Use JIT-compiled RHS for Schwarzschild
            solution = odeint(
                self._geodesic_rhs_schwarzschild,
                initial_state,
                tau_points,
                args=(self.mass,)
            )
        else:
            # Generic symbolic integration (slower)
            solution = self._integrate_generic(initial_state, tau_points)
        
        # Extract coordinates
        coordinates = solution[:, :4]  # [t, r, θ, φ]
        velocities = solution[:, 4:]   # 4-velocities
        
        # Calculate curvature along path
        curvature = self._compute_curvature_along_path(coordinates)
        
        # Calculate proper length
        proper_length = self._compute_proper_length(coordinates, velocities, tau_points)
        
        # Create geodesic path
        path = GeodesicPath(
            parameter=tau_points,
            coordinates=coordinates,
            curvature=curvature,
            proper_length=proper_length
        )
        
        logger.info(f"✅ Integrated path with {n_points} points, length={proper_length:.3f}")
        
        return path
    
    def _integrate_generic(self, initial_state: np.ndarray, tau_points: np.ndarray) -> np.ndarray:
        """Generic symbolic integration (fallback)"""
        # This would use the symbolic Christoffel symbols
        # For now, return a simple approximation
        logger.warning("⚠️ Using simplified geodesic integration")
        
        n_points = len(tau_points)
        n_coords = len(initial_state) // 2
        
        # Simple straight-line approximation
        solution = np.zeros((n_points, len(initial_state)))
        for i, tau in enumerate(tau_points):
            # Linear evolution
            solution[i, :n_coords] = initial_state[:n_coords] + tau * initial_state[n_coords:]
            solution[i, n_coords:] = initial_state[n_coords:]  # Constant velocity
        
        return solution
    
    def _compute_curvature_along_path(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute curvature scalar along geodesic path"""
        curvature = np.zeros(len(coordinates))
        
        if hasattr(self, 'mass'):
            # Schwarzschild Kretschmann scalar
            for i, coord in enumerate(coordinates):
                r = coord[1]
                if r > 2 * self.mass:
                    curvature[i] = 48 * self.mass**2 / r**6
                else:
                    curvature[i] = np.inf  # Singularity
        
        return curvature
    
    def _compute_proper_length(self, 
                              coordinates: np.ndarray, 
                              velocities: np.ndarray,
                              tau_points: np.ndarray) -> float:
        """Compute proper length/time along path"""
        if len(coordinates) < 2:
            return 0.0
        
        # Integrate ds² along path
        total_length = 0.0
        
        for i in range(len(coordinates) - 1):
            # Metric at this point
            if hasattr(self, 'mass'):
                # Schwarzschild
                r = coordinates[i, 1]
                theta = coordinates[i, 2]
                
                # Line element
                dt = velocities[i, 0]
                dr = velocities[i, 1]
                dtheta = velocities[i, 2]
                dphi = velocities[i, 3]
                
                ds2 = (-(1 - 2*self.mass/r) * dt**2 + 
                       1/(1 - 2*self.mass/r) * dr**2 + 
                       r**2 * dtheta**2 + 
                       r**2 * np.sin(theta)**2 * dphi**2)
                
                # Proper length element
                dtau = tau_points[i+1] - tau_points[i]
                if ds2 > 0:
                    total_length += np.sqrt(ds2) * dtau
        
        return total_length
    
    def compute_phase_twist_along_geodesic(self,
                                          geodesic: GeodesicPath,
                                          phase_field: np.ndarray,
                                          coordinates_dict: Dict[str, np.ndarray]) -> float:
        """
        🌀 Compute total phase twist along geodesic path
        
        This is the key function that integrates ψ-phase along the path
        """
        logger.info("🌀 Computing phase twist along geodesic")
        
        # Create phase interpolator
        coord_arrays = [coordinates_dict[str(coord)] for coord in self.coordinates]
        phase_interpolator = RegularGridInterpolator(
            coord_arrays, 
            phase_field,
            bounds_error=False,
            fill_value=0.0
        )
        
        # Integrate phase along path
        total_phase = 0.0
        
        for i in range(len(geodesic.coordinates) - 1):
            # Points along path
            point1 = geodesic.coordinates[i]
            point2 = geodesic.coordinates[i + 1]
            
            # Interpolate phase
            try:
                phase1 = float(phase_interpolator(point1))
                phase2 = float(phase_interpolator(point2))
                
                # Accumulate phase difference
                dphase = np.angle(np.exp(1j * (phase2 - phase1)))
                
                # Weight by proper distance
                if geodesic.curvature is not None:
                    # Weight by curvature (stronger twist in high curvature)
                    weight = 1.0 + 0.1 * geodesic.curvature[i]
                    dphase *= weight
                
                total_phase += dphase
                
            except Exception as e:
                logger.warning(f"Interpolation failed at point {i}: {e}")
                continue
        
        # Wrap to [-π, π]
        total_phase = np.angle(np.exp(1j * total_phase))
        
        # Store in geodesic
        geodesic.phase_twist = total_phase
        
        logger.info(f"✅ Total phase twist: {total_phase:.3f} radians")
        return total_phase
    
    def find_shortest_geodesic(self,
                              start_pos: np.ndarray,
                              end_pos: np.ndarray,
                              geodesic_type: str = "spacelike",
                              n_attempts: int = 10) -> Optional[GeodesicPath]:
        """
        🎯 Find shortest geodesic between two points
        
        Uses variational principle to minimize path length
        """
        logger.info(f"🎯 Finding shortest {geodesic_type} geodesic")
        
        best_path = None
        min_length = np.inf
        
        for attempt in range(n_attempts):
            # Random initial velocity direction
            if attempt == 0:
                # First attempt: straight line approximation
                direction = end_pos - start_pos
                direction = direction / np.linalg.norm(direction)
                initial_vel = direction
            else:
                # Random perturbation
                initial_vel = np.random.randn(4)
                initial_vel = initial_vel / np.linalg.norm(initial_vel)
            
            # Scale velocity
            initial_vel *= 1.0  # Unit velocity
            
            # Estimate parameter range
            coord_distance = np.linalg.norm(end_pos - start_pos)
            tau_max = 2.0 * coord_distance  # Overestimate
            
            # Integrate geodesic
            try:
                path = self.integrate_geodesic(
                    initial_pos=start_pos,
                    initial_vel=initial_vel,
                    tau_span=(0, tau_max),
                    n_points=100,
                    geodesic_type=geodesic_type
                )
                
                # Find closest approach to endpoint
                distances = np.linalg.norm(path.coordinates - end_pos, axis=1)
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                
                # If close enough to endpoint
                if min_dist < 0.1:  # Tolerance
                    # Truncate path at closest point
                    path.parameter = path.parameter[:min_dist_idx+1]
                    path.coordinates = path.coordinates[:min_dist_idx+1]
                    
                    if path.proper_length < min_length:
                        min_length = path.proper_length
                        best_path = path
                        logger.info(f"   Found path with length {min_length:.3f}")
                
            except Exception as e:
                logger.warning(f"   Attempt {attempt+1} failed: {e}")
                continue
        
        if best_path is None:
            logger.warning("❌ Failed to find geodesic")
        else:
            logger.info(f"✅ Best geodesic has length {min_length:.3f}")
        
        return best_path
    
    def parallel_transport_phase(self,
                                geodesic: GeodesicPath,
                                initial_phase: float) -> np.ndarray:
        """
        🔄 Parallel transport phase along geodesic
        
        This computes how phase evolves when parallel transported
        """
        n_points = len(geodesic.coordinates)
        transported_phase = np.zeros(n_points)
        transported_phase[0] = initial_phase
        
        # Parallel transport equation: ∇_v φ = 0
        # In coordinates: dφ/dτ + Γ^i_jk v^j ∂_i φ = 0
        
        for i in range(1, n_points):
            # Simple approximation: phase stays constant in flat regions
            # but gets twisted by curvature
            if geodesic.curvature is not None:
                # Phase twist proportional to integrated curvature
                curvature_integral = simpson(
                    geodesic.curvature[:i+1], 
                    geodesic.parameter[:i+1]
                )
                transported_phase[i] = initial_phase + 0.1 * curvature_integral
            else:
                transported_phase[i] = initial_phase
        
        return transported_phase
    
    def compute_holonomy(self, loop_path: GeodesicPath) -> complex:
        """
        🔮 Compute holonomy (phase factor) around closed loop
        
        For closed paths, this gives the geometric phase
        """
        if not np.allclose(loop_path.coordinates[0], loop_path.coordinates[-1], atol=0.1):
            logger.warning("⚠️ Path is not closed for holonomy calculation")
        
        # Compute phase around loop
        phase_shift = loop_path.phase_twist
        
        # Holonomy is the phase factor
        holonomy = np.exp(1j * phase_shift)
        
        logger.info(f"🔮 Holonomy: |h| = {np.abs(holonomy):.6f}, arg(h) = {np.angle(holonomy):.3f}")
        
        return holonomy
    
    def export_geodesic(self, geodesic: GeodesicPath, filename: str):
        """📤 Export geodesic data for visualization"""
        export_data = {
            'parameter': geodesic.parameter.tolist(),
            'coordinates': geodesic.coordinates.tolist(),
            'curvature': geodesic.curvature.tolist() if geodesic.curvature is not None else None,
            'phase_twist': geodesic.phase_twist,
            'proper_length': geodesic.proper_length,
            'coordinate_names': [str(coord) for coord in self.coordinates] if self.coordinates else None
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📤 Exported geodesic to {filename}")


# Example usage
if __name__ == "__main__":
    # Create geodesic integrator
    integrator = GeodesicIntegrator()
    
    # Set up Schwarzschild metric
    integrator.set_schwarzschild_metric(mass=1.0)
    
    # Initial conditions: start at r=10M
    initial_pos = np.array([0.0, 10.0, np.pi/2, 0.0])  # [t, r, θ, φ]
    initial_vel = np.array([1.0, -0.1, 0.0, 0.01])     # Inward radial motion with angular momentum
    
    # Integrate geodesic
    geodesic = integrator.integrate_geodesic(
        initial_pos=initial_pos,
        initial_vel=initial_vel,
        tau_span=(0, 50),
        n_points=200,
        geodesic_type="timelike"
    )
    
    print(f"Integrated geodesic with {len(geodesic.coordinates)} points")
    print(f"Final position: r={geodesic.coordinates[-1, 1]:.2f}")
    print(f"Proper length: {geodesic.proper_length:.3f}")
    
    # Example phase field (spiral)
    r_values = np.linspace(2, 20, 50)
    theta_values = np.linspace(0, np.pi, 30)
    phi_values = np.linspace(0, 2*np.pi, 40)
    t_values = np.array([0.0])  # Static field
    
    # Create phase field
    phase_field = np.zeros((1, 50, 30, 40))
    for i, r in enumerate(r_values):
        for j, theta in enumerate(theta_values):
            for k, phi in enumerate(phi_values):
                # Spiral phase pattern
                phase_field[0, i, j, k] = phi + 0.1 * r
    
    # Compute phase twist
    coordinates_dict = {
        't': t_values,
        'r': r_values,
        'theta': theta_values,
        'phi': phi_values
    }
    
    total_twist = integrator.compute_phase_twist_along_geodesic(
        geodesic, phase_field[0], coordinates_dict
    )
    
    print(f"Total phase twist: {total_twist:.3f} radians")
    
    # Find shortest path between two points
    start = np.array([0.0, 10.0, np.pi/2, 0.0])
    end = np.array([0.0, 5.0, np.pi/2, np.pi/4])
    
    shortest = integrator.find_shortest_geodesic(start, end)
    if shortest:
        print(f"Shortest geodesic length: {shortest.proper_length:.3f}")
    
    # Export for visualization
    integrator.export_geodesic(geodesic, "example_geodesic.json")
