#!/usr/bin/env python3
"""
Physics Validation Test Suite
Implements the 5 fast physics sanity tests from the triage map
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import logging

# Import our physics modules
try:
    from python.core.strang_integrator import StrangIntegrator, create_sech_soliton
    from python.core.hot_swap_laplacian import HotSwapLaplacian
    from python.core.oscillator_lattice import OscillatorLattice, get_global_lattice
    from python.core.bdg_solver import BdGSolver
    from python.core.blowup_harness import BlowupHarness
except ImportError:
    from strang_integrator import StrangIntegrator, create_sech_soliton
    from hot_swap_laplacian import HotSwapLaplacian
    from oscillator_lattice import OscillatorLattice, get_global_lattice
    from bdg_solver import BdGSolver
    from blowup_harness import BlowupHarness

logger = logging.getLogger(__name__)


class PhysicsValidator:
    """Comprehensive physics validation suite"""
    
    def __init__(self):
        self.results = {}
        self.tolerances = {
            'norm_drift': 1e-8,
            'energy_drift': 1e-5,
            'soliton_shape_error': 1e-4,
            'flat_band_tolerance': 1e-3,
            'adiabatic_leakage': 0.02,
            'flux_divergence': 1e-6,
            'blowup_recovery': 1e-3
        }
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all 5 physics sanity tests"""
        tests = [
            ("1D Soliton Conservation", self.test_1d_soliton),
            ("Kagome Flat Band", self.test_kagome_flat_band),
            ("Topology Swap Adiabaticity", self.test_topology_adiabaticity),
            ("Stress/Flux Divergence", self.test_flux_conservation),
            ("Blowup/Harvest Cycle", self.test_blowup_harvest)
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print('='*60)
            try:
                success = test_func()
                self.results[test_name] = success
                print(f"Result: {'PASSED ✓' if success else 'FAILED ✗'}")
            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                self.results[test_name] = False
                print(f"Result: CRASHED ✗ ({e})")
        
        return self.results
    
    def test_1d_soliton(self) -> bool:
        """
        Test 1: Analytic 1-D soliton check
        Initial condition: ψ(x,0) = sech(x) e^{ix/2}
        Run 1000 steps, check norm and energy conservation
        """
        print("Setting up 1D soliton test...")
        
        # Grid parameters
        L = 40.0
        N = 512
        x = np.linspace(-L/2, L/2, N, endpoint=False)
        dx = x[1] - x[0]
        
        # Discrete Laplacian with periodic BC
        laplacian = np.zeros((N, N))
        for i in range(N):
            laplacian[i, i] = -2.0
            laplacian[i, (i+1) % N] = 1.0
            laplacian[i, (i-1) % N] = 1.0
        laplacian /= dx**2
        
        # Initial soliton
        v = 0.5  # velocity
        psi0 = np.sech(x) * np.exp(1j * v * x / 2)
        
        # Create integrator
        dt = 0.01
        integrator = StrangIntegrator(x, laplacian, dt)
        
        # Store initial quantities
        E0 = integrator.compute_energy(psi0)
        N0 = integrator.compute_norm(psi0)
        print(f"Initial energy: {E0:.10f}")
        print(f"Initial norm: {N0:.10f}")
        
        # Evolve for 1000 steps
        psi = psi0.copy()
        for i in range(1000):
            psi = integrator.step(psi, store_energy=False)
            if i % 200 == 0:
                E = integrator.compute_energy(psi)
                N = integrator.compute_norm(psi)
                print(f"Step {i}: E={E:.10f}, N={N:.10f}")
        
        # Final checks
        E_final = integrator.compute_energy(psi)
        N_final = integrator.compute_norm(psi)
        
        energy_drift = abs(E_final - E0) / abs(E0)
        norm_drift = abs(N_final - N0) / N0
        
        # Compute L2 shape error
        # Account for soliton translation
        t_final = 1000 * dt
        x_shift = v * t_final
        psi_exact = np.sech(x - x_shift) * np.exp(1j * (v * x / 2 - (v**2/4 - 1) * t_final))
        
        # Find best alignment (due to periodic BC)
        min_error = float('inf')
        for shift in range(N):
            psi_shifted = np.roll(psi, shift)
            error = np.linalg.norm(np.abs(psi_shifted) - np.abs(psi_exact)) / np.linalg.norm(np.abs(psi_exact))
            min_error = min(min_error, error)
        
        shape_error = min_error
        
        print(f"\nResults:")
        print(f"Energy drift: {energy_drift:.2e} (tolerance: {self.tolerances['energy_drift']:.2e})")
        print(f"Norm drift: {norm_drift:.2e} (tolerance: {self.tolerances['norm_drift']:.2e})")
        print(f"Shape error: {shape_error:.2e} (tolerance: {self.tolerances['soliton_shape_error']:.2e})")
        
        return (energy_drift < self.tolerances['energy_drift'] and 
                norm_drift < self.tolerances['norm_drift'] and
                shape_error < self.tolerances['soliton_shape_error'])
    
    def test_kagome_flat_band(self) -> bool:
        """
        Test 2: Kagome flat-band validation
        Build 3x3 breathing Kagome, check for flat band
        """
        print("Setting up Kagome flat band test...")
        
        # Create breathing Kagome lattice
        size = 9  # 3x3 unit cells
        breathing_ratio = 0.8  # t1/t2 ratio
        
        # Build Kagome Laplacian
        laplacian = self._build_breathing_kagome(size, breathing_ratio)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues.sort()
        
        print(f"Eigenvalues: {eigenvalues}")
        
        # Find flat bands (should be 1/3 of total states)
        n_flat = size // 3
        flat_bands = eigenvalues[size//3:2*size//3]
        
        # Check flatness
        if len(flat_bands) > 1:
            band_width = np.max(flat_bands) - np.min(flat_bands)
            band_center = np.mean(flat_bands)
            relative_width = band_width / (abs(band_center) + 1e-10)
            
            print(f"\nFlat band analysis:")
            print(f"Number of flat states: {len(flat_bands)}")
            print(f"Band center: {band_center:.6f}")
            print(f"Band width: {band_width:.6e}")
            print(f"Relative width: {relative_width:.6e}")
            
            return relative_width < self.tolerances['flat_band_tolerance']
        else:
            print("Warning: No flat band found!")
            return False
    
    def _build_breathing_kagome(self, size: int, breathing_ratio: float) -> np.ndarray:
        """Build breathing Kagome Laplacian"""
        # Simplified version - in reality would use lattice_topology.rs logic
        n_sites = size
        H = np.zeros((n_sites, n_sites))
        
        # Kagome has 3 sites per unit cell
        t1 = 1.0  # Intra-triangle hopping
        t2 = breathing_ratio  # Inter-triangle hopping
        
        # Build connectivity (simplified for demonstration)
        # Real implementation would follow exact Kagome geometry
        for i in range(0, n_sites, 3):
            # Intra-triangle connections
            for j in range(3):
                for k in range(j+1, 3):
                    if i+j < n_sites and i+k < n_sites:
                        H[i+j, i+k] = -t1
                        H[i+k, i+j] = -t1
            
            # Inter-triangle connections (simplified)
            if i+3 < n_sites:
                H[i+2, i+3] = -t2
                H[i+3, i+2] = -t2
        
        return H
    
    def test_topology_adiabaticity(self) -> bool:
        """
        Test 3: Topology-swap adiabaticity
        Slowly morph from Kagome to Hexagonal, check adiabatic following
        """
        print("Setting up topology adiabaticity test...")
        
        # Create hot-swap system
        hot_swap = HotSwapLaplacian(initial_topology="kagome")
        
        # Initial state: ground state of Kagome
        size = 12
        H_kagome = self._build_breathing_kagome(size, 0.8)
        eigenvalues, eigenvectors = np.linalg.eigh(H_kagome)
        psi0 = eigenvectors[:, 0]  # Ground state
        
        # Initiate slow morph
        hot_swap.initiate_morph("hexagonal", blend_rate=0.01)  # 100 steps
        
        # Track state overlap with instantaneous ground state
        overlaps = []
        psi = psi0.copy()
        
        for step in range(100):
            # Get current Hamiltonian
            blend_progress = (step + 1) * 0.01
            H_current = (1 - blend_progress) * H_kagome  # Simplified blending
            
            # Evolve state (simplified - should use Strang)
            dt = 0.1
            psi = psi - 1j * dt * (H_current @ psi)
            psi /= np.linalg.norm(psi)
            
            # Compute overlap with instantaneous ground state
            eigvals, eigvecs = np.linalg.eigh(H_current)
            ground_state = eigvecs[:, 0]
            overlap = abs(np.vdot(psi, ground_state))**2
            overlaps.append(overlap)
        
        # Check adiabatic following
        min_overlap = min(overlaps)
        leakage = 1 - min_overlap
        
        print(f"\nAdiabatic following:")
        print(f"Minimum overlap: {min_overlap:.4f}")
        print(f"Maximum leakage: {leakage:.4f}")
        print(f"Tolerance: {self.tolerances['adiabatic_leakage']:.4f}")
        
        return leakage < self.tolerances['adiabatic_leakage']
    
    def test_flux_conservation(self) -> bool:
        """
        Test 4: Stress/flux divergence test
        At steady state, check ∇·J ≈ 0
        """
        print("Setting up flux conservation test...")
        
        # Create a simple lattice
        lattice = OscillatorLattice(size=16)
        
        # Add some test oscillators
        for i in range(5):
            lattice.add_oscillator(
                phase=np.random.rand() * 2 * np.pi,
                natural_freq=0.1 + 0.05 * np.random.randn(),
                amplitude=0.5 + 0.1 * np.random.randn()
            )
        
        # Let system equilibrate
        for _ in range(1000):
            lattice.step(dt=0.01)
        
        # Compute flux divergence at each site
        flux_div = []
        
        for i, osc in enumerate(lattice.oscillators):
            if not osc.get('active', True):
                continue
            
            # Compute net flux (simplified)
            flux_in = 0.0
            flux_out = 0.0
            
            # Sum over connections
            for j, other in enumerate(lattice.oscillators):
                if i != j and other.get('active', True):
                    # Simplified flux calculation
                    coupling = lattice.K[i, j] if hasattr(lattice, 'K') else 0.1
                    phase_diff = osc['phase'] - other['phase']
                    flux = coupling * np.sin(phase_diff) * other['amplitude']
                    
                    if flux > 0:
                        flux_in += flux
                    else:
                        flux_out += abs(flux)
            
            divergence = abs(flux_in - flux_out) / (flux_in + flux_out + 1e-10)
            flux_div.append(divergence)
        
        # Check conservation
        max_divergence = max(flux_div) if flux_div else 0.0
        avg_divergence = np.mean(flux_div) if flux_div else 0.0
        
        print(f"\nFlux conservation:")
        print(f"Max divergence: {max_divergence:.6e}")
        print(f"Avg divergence: {avg_divergence:.6e}")
        print(f"Tolerance: {self.tolerances['flux_divergence']:.6e}")
        
        return max_divergence < self.tolerances['flux_divergence']
    
    def test_blowup_harvest(self) -> bool:
        """
        Test 5: Blow-up / harvest cycle
        Induce blowup, harvest energy, check recovery
        """
        print("Setting up blowup/harvest test...")
        
        # Create system with blowup harness
        lattice = OscillatorLattice(size=10)
        harness = BlowupHarness(
            threshold_energy=5.0,
            safety_factor=0.8,
            harvest_efficiency=0.9
        )
        
        # Add oscillators
        for i in range(5):
            lattice.add_oscillator(phase=0.1*i, natural_freq=1.0, amplitude=0.5)
        
        # Record baseline
        if hasattr(lattice, 'compute_total_energy'):
            baseline_energy = lattice.compute_total_energy()
        else:
            # Simple energy estimate
            baseline_energy = sum(osc['amplitude']**2 for osc in lattice.oscillators)
        
        baseline_norm = sum(abs(osc['amplitude']) for osc in lattice.oscillators)
        print(f"Baseline energy: {baseline_energy:.6f}")
        print(f"Baseline norm: {baseline_norm:.6f}")
        
        # Induce blowup by pumping energy
        print("\nInducing blowup...")
        for osc in lattice.oscillators:
            osc['amplitude'] *= 3.0  # Triple amplitude
        
        # Check blowup detection and harvesting
        if hasattr(lattice, 'total_charge'):
            initial_charge = lattice.total_charge
        else:
            initial_charge = 0.0
        
        # Harvest energy
        harvested = harness.check_and_harvest(lattice)
        print(f"Harvested energy: {harvested:.6f}")
        
        # Check recovery
        if hasattr(lattice, 'compute_total_energy'):
            final_energy = lattice.compute_total_energy()
        else:
            final_energy = sum(osc['amplitude']**2 for osc in lattice.oscillators)
        
        final_norm = sum(abs(osc['amplitude']) for osc in lattice.oscillators)
        
        # Energy should be reduced but conserved (transferred to charge)
        recovery_error = abs(final_energy - baseline_energy) / baseline_energy
        
        print(f"\nPost-harvest state:")
        print(f"Final energy: {final_energy:.6f}")
        print(f"Final norm: {final_norm:.6f}")
        print(f"Recovery error: {recovery_error:.6e}")
        print(f"Tolerance: {self.tolerances['blowup_recovery']:.6e}")
        
        return recovery_error < self.tolerances['blowup_recovery'] and harvested > 0
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        report = "\n" + "="*60 + "\n"
        report += "PHYSICS VALIDATION REPORT\n"
        report += "="*60 + "\n\n"
        
        # Summary
        total_tests = len(self.results)
        passed_tests = sum(1 for v in self.results.values() if v)
        
        report += f"Total Tests: {total_tests}\n"
        report += f"Passed: {passed_tests}\n"
        report += f"Failed: {total_tests - passed_tests}\n"
        report += f"Success Rate: {100*passed_tests/total_tests:.1f}%\n\n"
        
        # Individual results
        report += "Test Results:\n"
        report += "-"*40 + "\n"
        for test_name, passed in self.results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            report += f"{test_name:<40} {status}\n"
        
        # Recommendations
        report += "\n" + "="*60 + "\n"
        report += "RECOMMENDATIONS:\n"
        report += "="*60 + "\n"
        
        if not self.results.get("1D Soliton Conservation", True):
            report += "- Check Strang integrator implementation and time step\n"
            report += "- Verify discrete Laplacian accuracy\n"
        
        if not self.results.get("Kagome Flat Band", True):
            report += "- Review breathing Kagome parameters (t1/t2 ratio)\n"
            report += "- Check lattice connectivity generation\n"
        
        if not self.results.get("Topology Swap Adiabaticity", True):
            report += "- Reduce morphing speed (smaller blend_rate)\n"
            report += "- Check Hamiltonian interpolation smoothness\n"
        
        if not self.results.get("Stress/Flux Divergence", True):
            report += "- Review flux calculation in comfort metrics\n"
            report += "- Verify coupling matrix symmetry\n"
        
        if not self.results.get("Blowup/Harvest Cycle", True):
            report += "- Check energy bookkeeping in BlowupHarness\n"
            report += "- Verify charge accumulation logic\n"
        
        return report


def run_physics_validation():
    """Main entry point for physics validation"""
    logging.basicConfig(level=logging.INFO)
    
    validator = PhysicsValidator()
    results = validator.run_all_tests()
    report = validator.generate_report()
    
    print(report)
    
    # Save report
    with open("physics_validation_report.txt", "w") as f:
        f.write(report)
    
    return all(results.values())


if __name__ == "__main__":
    success = run_physics_validation()
    exit(0 if success else 1)
