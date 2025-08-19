#!/usr/bin/env python3
"""
Bad Ass FDTD Stepper with Adaptive Timestep Control
GPU-accelerated, stability-aware finite difference time domain evolution
Real-time Lyapunov monitoring with κ-law timestep adaptation
"""

import numpy as np
try:
    import cupy as xp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as xp
    GPU_AVAILABLE = False

import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field

# Import our bad ass physics modules
try:
    from . import bdg_solver
    from . import adaptive_timestep
    from . import strang_integrator
    from . import side_mode
except ImportError:
    import bdg_solver
    import adaptive_timestep
    import strang_integrator
    import side_mode

logger = logging.getLogger(__name__)


@dataclass
class FDTDMetrics:
    """Real-time performance and physics metrics"""
    step_count: int = 0
    total_time: float = 0.0
    current_dt: float = 0.0
    lambda_max: float = 0.0
    energy: float = 0.0
    norm: float = 0.0
    energy_drift: float = 0.0
    norm_drift: float = 0.0
    adaptation_ratio: float = 1.0
    gpu_memory_used: float = 0.0
    steps_per_second: float = 0.0
    stability_status: str = "unknown"
    conserved_quantities: Dict[str, float] = field(default_factory=dict)


class BadAssFDTDStepper:
    """
    Ultimate FDTD stepper with all the bad ass features:
    - Real-time Lyapunov monitoring via BdG analysis
    - κ-law adaptive timestep: dt = dt₀/(1 + κλ_max)
    - GPU acceleration with automatic fallback
    - Chi-reduced interferometric lattice support
    - Energy/norm conservation tracking
    - Performance monitoring and diagnostics
    - Symplectic Strang splitting integration
    - Automatic stability management
    """
    
    def __init__(
        self,
        spatial_grid: np.ndarray,
        dt_base: float = 0.01,
        kappa: float = 0.75,
        g: float = 1.0,
        mu: Optional[float] = None,
        chi: float = 0.0,
        pole_list: Optional[List[Tuple[float, float]]] = None,
        boundary: str = 'periodic',
        stability_check_interval: int = 10,
        energy_check_interval: int = 50,
        gpu_enabled: bool = True,
        max_dt_reduction: float = 1000.0,
        emergency_dt_floor: float = 1e-8
    ):
        """
        Initialize the bad ass FDTD stepper
        
        Args:
            spatial_grid: Spatial discretization
            dt_base: Base timestep (dt₀ in κ-law)
            kappa: Adaptation strength (κ in κ-law) 
            g: Nonlinearity strength
            mu: Chemical potential (auto-estimated if None)
            chi: Chi reduction factor for interferometric lattices
            pole_list: Pole structure for chi-reduced self-energy
            boundary: Boundary conditions ('periodic', 'dirichlet', 'neumann')
            stability_check_interval: Steps between BdG stability checks
            energy_check_interval: Steps between energy conservation checks
            gpu_enabled: Enable GPU acceleration if available
            max_dt_reduction: Maximum allowed dt reduction factor
            emergency_dt_floor: Absolute minimum timestep (safety)
        """
        self.grid = spatial_grid
        self.dx = spatial_grid[1] - spatial_grid[0] if spatial_grid.ndim == 1 else 1.0
        self.g = g
        self.mu = mu
        self.chi = chi
        self.pole_list = pole_list
        self.boundary = boundary
        self.stability_check_interval = stability_check_interval
        self.energy_check_interval = energy_check_interval
        self.max_dt_reduction = max_dt_reduction
        self.emergency_dt_floor = emergency_dt_floor
        
        # GPU configuration
        self.gpu_enabled = gpu_enabled and GPU_AVAILABLE
        if self.gpu_enabled:
            logger.info("Bad Ass FDTD: GPU acceleration ENABLED")
        else:
            logger.info("Bad Ass FDTD: Running on CPU")
        
        # Initialize adaptive timestep controller
        self.timestep_controller = adaptive_timestep.AdaptiveTimestep(
            dt_base=dt_base,
            kappa=kappa,
            dt_min=max(dt_base / max_dt_reduction, emergency_dt_floor),
            dt_max=dt_base
        )
        
        # Initialize Strang integrator
        self.integrator = strang_integrator.StrangIntegrator(
            spatial_grid=spatial_grid,
            laplacian=self._build_laplacian(),
            dt=dt_base
        )
        
        # Initialize BdG solver for stability monitoring
        self.bdg_solver = bdg_solver.BdGSolver(dx=self.dx, boundary=boundary)
        
        # Performance tracking
        self.metrics = FDTDMetrics()
        self.step_timer = time.perf_counter()
        self.initial_energy = None
        self.initial_norm = None
        
        # Emergency stability tracking
        self.consecutive_unstable_steps = 0
        self.max_consecutive_unstable = 100
        
        logger.info(f"Bad Ass FDTD initialized: dt_base={dt_base}, κ={kappa}, χ={chi}")
    
    def _build_laplacian(self) -> np.ndarray:
        """Build discrete Laplacian operator"""
        if self.grid.ndim == 1:
            return self.bdg_solver.build_laplacian_1d(len(self.grid)).toarray()
        elif self.grid.ndim == 2:
            return self.bdg_solver.build_laplacian_2d(self.grid.shape).toarray()
        else:
            raise ValueError("Only 1D and 2D grids supported")
    
    def _transfer_to_gpu(self, psi: np.ndarray) -> np.ndarray:
        """Transfer data to GPU if enabled"""
        if self.gpu_enabled and GPU_AVAILABLE:
            return xp.asarray(psi)
        return psi
    
    def _transfer_from_gpu(self, psi) -> np.ndarray:
        """Transfer data from GPU to CPU"""
        if self.gpu_enabled and GPU_AVAILABLE and hasattr(psi, 'get'):
            return psi.get()
        return np.asarray(psi)
    
    def _estimate_chemical_potential(self, psi: np.ndarray) -> float:
        """Estimate chemical potential from wavefunction density"""
        if self.mu is not None:
            return self.mu
        # For cubic nonlinearity: μ ≈ g⟨|ψ|²⟩
        return self.g * float(np.mean(np.abs(psi)**2))
    
    def _monitor_stability(self, psi: np.ndarray) -> Tuple[float, str]:
        """
        Monitor current stability via BdG analysis
        Returns (lambda_max, status_string)
        """
        try:
            # Estimate chemical potential
            mu_current = self._estimate_chemical_potential(psi)
            
            # Compute maximum Lyapunov exponent with chi reduction
            lambda_max = bdg_solver.bdg_lambda_max(
                psi=psi,
                dx=self.dx,
                g=self.g,
                chi=self.chi,
                pole_list=self.pole_list
            )
            
            # Determine stability status
            if lambda_max > 1e-6:
                status = f"UNSTABLE(λ={lambda_max:.2e})"
                self.consecutive_unstable_steps += 1
            elif lambda_max > -1e-10:
                status = f"MARGINAL(λ={lambda_max:.2e})"
                self.consecutive_unstable_steps = 0
            else:
                status = f"STABLE(λ={lambda_max:.2e})"
                self.consecutive_unstable_steps = 0
            
            # Emergency check
            if self.consecutive_unstable_steps > self.max_consecutive_unstable:
                logger.error(f"EMERGENCY: {self.consecutive_unstable_steps} consecutive unstable steps!")
                status = "EMERGENCY_UNSTABLE"
            
            return float(lambda_max), status
            
        except Exception as e:
            logger.warning(f"Stability monitoring failed: {e}")
            return 0.0, "MONITOR_ERROR"
    
    def _check_conservation(self, psi: np.ndarray) -> Dict[str, float]:
        """Check energy and norm conservation"""
        try:
            current_energy = self.integrator.compute_energy(psi)
            current_norm = self.integrator.compute_norm(psi)
            
            if self.initial_energy is None:
                self.initial_energy = current_energy
                self.initial_norm = current_norm
            
            energy_drift = abs(current_energy - self.initial_energy) / abs(self.initial_energy)
            norm_drift = abs(current_norm - self.initial_norm) / self.initial_norm
            
            return {
                'energy': float(current_energy),
                'norm': float(current_norm),
                'energy_drift': float(energy_drift),
                'norm_drift': float(norm_drift)
            }
        except Exception as e:
            logger.warning(f"Conservation check failed: {e}")
            return {'energy': 0.0, 'norm': 0.0, 'energy_drift': 0.0, 'norm_drift': 0.0}
    
    def _update_metrics(self, psi: np.ndarray, dt: float, lambda_max: float, 
                       status: str, conservation: Dict[str, float]) -> None:
        """Update performance and physics metrics"""
        current_time = time.perf_counter()
        step_duration = current_time - self.step_timer
        
        self.metrics.step_count += 1
        self.metrics.total_time += step_duration
        self.metrics.current_dt = dt
        self.metrics.lambda_max = lambda_max
        self.metrics.stability_status = status
        self.metrics.energy = conservation['energy']
        self.metrics.norm = conservation['norm']
        self.metrics.energy_drift = conservation['energy_drift']
        self.metrics.norm_drift = conservation['norm_drift']
        
        # Timestep adaptation metrics
        timestep_metrics = self.timestep_controller.get_metrics()
        self.metrics.adaptation_ratio = timestep_metrics['adaptation_ratio']
        
        # Performance metrics
        self.metrics.steps_per_second = 1.0 / step_duration if step_duration > 0 else 0.0
        
        # GPU memory if available
        if self.gpu_enabled and GPU_AVAILABLE:
            try:
                mempool = xp.get_default_memory_pool()
                self.metrics.gpu_memory_used = mempool.used_bytes() / 1024**2  # MB
            except:
                self.metrics.gpu_memory_used = 0.0
        
        self.step_timer = current_time
    
    def step(self, psi: np.ndarray, force_stability_check: bool = False) -> Tuple[np.ndarray, FDTDMetrics]:
        """
        Perform one bad ass adaptive FDTD step
        
        Args:
            psi: Current wavefunction
            force_stability_check: Force BdG stability check this step
            
        Returns:
            (evolved_psi, current_metrics)
        """
        step_start_time = time.perf_counter()
        
        # Transfer to GPU if enabled
        psi_gpu = self._transfer_to_gpu(psi)
        
        # Stability monitoring (adaptive frequency)
        lambda_max = 0.0
        status = "SKIPPED"
        
        if (force_stability_check or 
            self.metrics.step_count % self.stability_check_interval == 0 or
            self.consecutive_unstable_steps > 0):
            
            # Transfer back to CPU for BdG analysis (for now)
            psi_cpu = self._transfer_from_gpu(psi_gpu)
            lambda_max, status = self._monitor_stability(psi_cpu)
        
        # Adaptive timestep computation
        dt_adaptive = self.timestep_controller.compute_timestep(lambda_max)
        
        # Emergency timestep floor
        if dt_adaptive < self.emergency_dt_floor:
            logger.warning(f"Emergency timestep floor activated: {dt_adaptive:.2e} -> {self.emergency_dt_floor:.2e}")
            dt_adaptive = self.emergency_dt_floor
        
        # Update integrator timestep
        self.integrator.dt = dt_adaptive
        
        # Perform the integration step
        try:
            psi_evolved = self.integrator.step(psi_gpu if isinstance(psi_gpu, np.ndarray) else self._transfer_from_gpu(psi_gpu))
            
            # Transfer evolved state to appropriate device
            if self.gpu_enabled:
                psi_evolved_gpu = self._transfer_to_gpu(psi_evolved)
            else:
                psi_evolved_gpu = psi_evolved
                
        except Exception as e:
            logger.error(f"Integration step failed: {e}")
            # Emergency: return original state with minimal timestep
            self.timestep_controller.dt_base *= 0.5
            return psi, self.metrics
        
        # Conservation monitoring (less frequent)
        conservation = {'energy': 0.0, 'norm': 0.0, 'energy_drift': 0.0, 'norm_drift': 0.0}
        if self.metrics.step_count % self.energy_check_interval == 0:
            psi_cpu = self._transfer_from_gpu(psi_evolved_gpu)
            conservation = self._check_conservation(psi_cpu)
        
        # Update metrics
        self._update_metrics(psi_evolved, dt_adaptive, lambda_max, status, conservation)
        
        # Return CPU version for compatibility
        psi_final = self._transfer_from_gpu(psi_evolved_gpu)
        
        return psi_final, self.metrics
    
    def evolve(
        self,
        psi0: np.ndarray,
        t_final: float,
        progress_callback: Optional[Callable[[int, FDTDMetrics], None]] = None,
        save_interval: Optional[int] = None
    ) -> Tuple[np.ndarray, List[np.ndarray], List[FDTDMetrics]]:
        """
        Evolve system with bad ass adaptive stepping
        
        Args:
            psi0: Initial wavefunction
            t_final: Final evolution time
            progress_callback: Optional callback(step, metrics)
            save_interval: Save trajectory every N steps
            
        Returns:
            (final_psi, trajectory, metrics_history)
        """
        logger.info(f"Bad Ass FDTD Evolution: t_final={t_final}, χ={self.chi}")
        
        psi = psi0.copy()
        trajectory = [psi0.copy()] if save_interval else []
        metrics_history = []
        
        # Reset metrics
        self.metrics = FDTDMetrics()
        self.step_timer = time.perf_counter()
        self.initial_energy = None
        self.initial_norm = None
        
        evolution_start = time.perf_counter()
        
        while self.metrics.total_time < t_final:
            # Adaptive step
            psi, current_metrics = self.step(psi)
            
            # Save trajectory
            if save_interval and self.metrics.step_count % save_interval == 0:
                trajectory.append(psi.copy())
            
            # Store metrics
            metrics_history.append(current_metrics)
            
            # Progress callback
            if progress_callback:
                progress_callback(self.metrics.step_count, current_metrics)
            
            # Periodic logging
            if self.metrics.step_count % 1000 == 0:
                logger.info(
                    f"Step {self.metrics.step_count}: "
                    f"t={self.metrics.total_time:.3f}, "
                    f"dt={self.metrics.current_dt:.2e}, "
                    f"λ_max={self.metrics.lambda_max:.2e}, "
                    f"status={self.metrics.stability_status}, "
                    f"E_drift={self.metrics.energy_drift:.2e}"
                )
            
            # Emergency break on severe instability
            if self.consecutive_unstable_steps > self.max_consecutive_unstable:
                logger.error("EMERGENCY STOP: Severe instability detected!")
                break
        
        evolution_time = time.perf_counter() - evolution_start
        logger.info(
            f"Bad Ass FDTD Complete: "
            f"{self.metrics.step_count} steps in {evolution_time:.2f}s, "
            f"avg {self.metrics.step_count/evolution_time:.1f} steps/s"
        )
        
        return psi, trajectory, metrics_history
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics"""
        timestep_metrics = self.timestep_controller.get_metrics()
        
        return {
            'physics': {
                'lambda_max': self.metrics.lambda_max,
                'stability_status': self.metrics.stability_status,
                'energy': self.metrics.energy,
                'norm': self.metrics.norm,
                'energy_drift': self.metrics.energy_drift,
                'norm_drift': self.metrics.norm_drift,
                'chi_factor': self.chi,
                'nonlinearity': self.g
            },
            'performance': {
                'steps_completed': self.metrics.step_count,
                'total_time': self.metrics.total_time,
                'steps_per_second': self.metrics.steps_per_second,
                'gpu_enabled': self.gpu_enabled,
                'gpu_memory_mb': self.metrics.gpu_memory_used
            },
            'adaptation': {
                'current_dt': self.metrics.current_dt,
                'adaptation_ratio': self.metrics.adaptation_ratio,
                'kappa': self.timestep_controller.kappa,
                'dt_base': self.timestep_controller.dt_base,
                'dt_min': self.timestep_controller.dt_min,
                'dt_max': self.timestep_controller.dt_max
            },
            'stability': {
                'consecutive_unstable': self.consecutive_unstable_steps,
                'max_allowed_unstable': self.max_consecutive_unstable,
                'emergency_dt_floor': self.emergency_dt_floor
            }
        }


# Convenience functions for common use cases
def create_1d_dark_soliton_stepper(
    x: np.ndarray,
    v: float = 0.0,
    dt_base: float = 0.01,
    kappa: float = 0.75,
    **kwargs
) -> Tuple[BadAssFDTDStepper, np.ndarray]:
    """Create stepper pre-configured for 1D dark soliton"""
    stepper = BadAssFDTDStepper(
        spatial_grid=x,
        dt_base=dt_base,
        kappa=kappa,
        g=1.0,
        boundary='periodic',
        **kwargs
    )
    
    # Create dark soliton initial condition
    psi0, mu = stepper.bdg_solver.create_dark_soliton_1d(x, v=v)
    stepper.mu = mu
    
    return stepper, psi0


def create_chi_reduced_stepper(
    x: np.ndarray,
    chi: float,
    pole_list: List[Tuple[float, float]],
    dt_base: float = 0.01,
    kappa: float = 1.0,  # Higher kappa for chi-reduced systems
    **kwargs
) -> BadAssFDTDStepper:
    """Create stepper for chi-reduced interferometric lattice"""
    return BadAssFDTDStepper(
        spatial_grid=x,
        dt_base=dt_base,
        kappa=kappa,
        chi=chi,
        pole_list=pole_list,
        stability_check_interval=5,  # More frequent stability checks
        **kwargs
    )


# Test and demo functions
def demo_adaptive_stability():
    """Demonstrate adaptive timestep responding to instability"""
    logger.info("Bad Ass FDTD Demo: Adaptive Stability Response")
    
    # 1D grid
    L = 40.0
    N = 256
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    
    # Create dark soliton stepper
    stepper, psi0 = create_1d_dark_soliton_stepper(x, v=0.1, kappa=1.5)
    
    # Add some noise to trigger instability
    psi_noisy = psi0 + 0.01 * (np.random.random(N) + 1j * np.random.random(N))
    
    # Evolution with progress tracking
    def progress_tracker(step: int, metrics: FDTDMetrics):
        if step % 100 == 0:
            print(f"Step {step}: dt={metrics.current_dt:.2e}, λ_max={metrics.lambda_max:.2e}, {metrics.stability_status}")
    
    # Evolve
    psi_final, trajectory, metrics = stepper.evolve(
        psi_noisy,
        t_final=5.0,
        progress_callback=progress_tracker,
        save_interval=50
    )
    
    # Final diagnostics
    diagnostics = stepper.get_diagnostics()
    print("\nBad Ass FDTD Demo Complete!")
    print(f"Final energy drift: {diagnostics['physics']['energy_drift']:.2e}")
    print(f"Average adaptation ratio: {diagnostics['adaptation']['adaptation_ratio']:.3f}")
    print(f"Performance: {diagnostics['performance']['steps_per_second']:.1f} steps/s")
    
    return psi_final, diagnostics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    print("🚀 Bad Ass FDTD Stepper - Real-time Adaptive Stability Control")
    print("=" * 60)
    
    # Run demo
    psi_final, diagnostics = demo_adaptive_stability()
    
    print("\n✅ Bad Ass FDTD: MISSION ACCOMPLISHED")
    print(f"GPU: {'ENABLED' if diagnostics['performance']['gpu_enabled'] else 'CPU ONLY'}")
    print(f"Stability: {diagnostics['physics']['stability_status']}")
    print(f"Performance: {diagnostics['performance']['steps_per_second']:.1f} steps/s")
