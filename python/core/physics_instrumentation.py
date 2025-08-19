#!/usr/bin/env python3
"""
Physics Instrumentation Toolkit
Decorators and utilities for monitoring conservation laws and physics quantities
"""

import numpy as np
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PhysicalQuantity:
    """
    Physical quantity with units and uncertainty
    Guards against unit errors and tracks propagation
    """
    value: float
    unit: str
    uncertainty: float = 0.0
    
    def __add__(self, other):
        if isinstance(other, PhysicalQuantity):
            if self.unit != other.unit:
                raise ValueError(f"Unit mismatch: {self.unit} vs {other.unit}")
            return PhysicalQuantity(
                self.value + other.value,
                self.unit,
                np.sqrt(self.uncertainty**2 + other.uncertainty**2)
            )
        return PhysicalQuantity(self.value + other, self.unit, self.uncertainty)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return PhysicalQuantity(
                self.value * other,
                self.unit,
                abs(other) * self.uncertainty
            )
        elif isinstance(other, PhysicalQuantity):
            # Simple unit multiplication (would need full unit system for general case)
            new_unit = f"({self.unit})·({other.unit})"
            rel_unc = np.sqrt((self.uncertainty/abs(self.value))**2 + 
                             (other.uncertainty/abs(other.value))**2)
            new_value = self.value * other.value
            return PhysicalQuantity(new_value, new_unit, abs(new_value) * rel_unc)
        return NotImplemented
    
    def __repr__(self):
        if self.uncertainty > 0:
            return f"{self.value:.6g} ± {self.uncertainty:.2g} {self.unit}"
        return f"{self.value:.6g} {self.unit}"


class EnergyTracker:
    """
    Decorator and context manager for tracking energy conservation
    """
    
    def __init__(self, name: str = "System", log_every: int = 100):
        self.name = name
        self.log_every = log_every
        self.history = []
        self.call_count = 0
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator mode"""
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            # Get energy before
            if hasattr(obj, 'compute_total_energy'):
                E_before = obj.compute_total_energy()
            elif hasattr(obj, 'compute_energy'):
                E_before = obj.compute_energy()
            else:
                E_before = self._estimate_energy(obj)
            
            # Execute function
            result = func(obj, *args, **kwargs)
            
            # Get energy after
            if hasattr(obj, 'compute_total_energy'):
                E_after = obj.compute_total_energy()
            elif hasattr(obj, 'compute_energy'):
                E_after = obj.compute_energy()
            else:
                E_after = self._estimate_energy(obj)
            
            # Track
            delta_E = E_after - E_before
            self.history.append({
                'call': self.call_count,
                'function': func.__name__,
                'E_before': E_before,
                'E_after': E_after,
                'delta_E': delta_E,
                'time': time.time()
            })
            
            self.call_count += 1
            
            # Log if needed
            if self.call_count % self.log_every == 0:
                rel_change = abs(delta_E) / (abs(E_before) + 1e-10)
                if rel_change > 1e-6:
                    logger.warning(
                        f"{self.name} energy change in {func.__name__}: "
                        f"ΔE = {delta_E:.6e} ({rel_change:.2e} relative)"
                    )
            
            return result
        
        return wrapper
    
    def _estimate_energy(self, obj) -> float:
        """Fallback energy estimation"""
        energy = 0.0
        
        # Look for oscillators
        if hasattr(obj, 'oscillators'):
            for osc in obj.oscillators:
                if isinstance(osc, dict) and osc.get('active', True):
                    energy += osc.get('amplitude', 0)**2
        
        # Look for memories
        if hasattr(obj, 'memories'):
            energy += len(obj.memories) * 0.1  # Rough estimate
        
        return energy
    
    def report(self) -> Dict[str, Any]:
        """Generate energy tracking report"""
        if not self.history:
            return {'total_calls': 0, 'total_drift': 0}
        
        E_initial = self.history[0]['E_before']
        E_final = self.history[-1]['E_after']
        total_drift = E_final - E_initial
        
        # Find largest changes
        largest_changes = sorted(
            self.history,
            key=lambda x: abs(x['delta_E']),
            reverse=True
        )[:5]
        
        return {
            'total_calls': self.call_count,
            'total_drift': total_drift,
            'relative_drift': abs(total_drift) / (abs(E_initial) + 1e-10),
            'E_initial': E_initial,
            'E_final': E_final,
            'largest_changes': [
                {
                    'function': x['function'],
                    'delta_E': x['delta_E'],
                    'call': x['call']
                }
                for x in largest_changes
            ]
        }


class EigenSpectrumProbe:
    """
    Utility to monitor eigenvalue spectrum during topology changes
    """
    
    def __init__(self, save_path: Optional[str] = None):
        self.save_path = save_path
        self.spectra = []
        
    def probe(self, matrix: np.ndarray, label: str = "", metadata: Dict = None):
        """Compute and store eigenvalue spectrum"""
        eigenvalues = np.linalg.eigvalsh(matrix)
        eigenvalues.sort()
        
        spectrum_data = {
            'label': label,
            'timestamp': time.time(),
            'eigenvalues': eigenvalues.tolist(),
            'metadata': metadata or {},
            'statistics': {
                'min': float(np.min(eigenvalues)),
                'max': float(np.max(eigenvalues)),
                'mean': float(np.mean(eigenvalues)),
                'std': float(np.std(eigenvalues)),
                'gap': float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0
            }
        }
        
        self.spectra.append(spectrum_data)
        
        # Check for flat bands
        self._check_flat_bands(eigenvalues, label)
        
        # Save if requested
        if self.save_path:
            self._save_spectrum(spectrum_data)
        
        return eigenvalues
    
    def _check_flat_bands(self, eigenvalues: np.ndarray, label: str):
        """Detect and report flat bands"""
        # Group nearby eigenvalues
        bands = []
        current_band = [eigenvalues[0]]
        
        for i in range(1, len(eigenvalues)):
            if abs(eigenvalues[i] - eigenvalues[i-1]) < 1e-6:
                current_band.append(eigenvalues[i])
            else:
                if len(current_band) > 1:
                    bands.append(current_band)
                current_band = [eigenvalues[i]]
        
        if len(current_band) > 1:
            bands.append(current_band)
        
        # Report flat bands
        for band in bands:
            if len(band) >= 3:  # At least 3 degenerate states
                center = np.mean(band)
                width = np.max(band) - np.min(band)
                logger.info(
                    f"Flat band detected in {label}: "
                    f"{len(band)} states at E={center:.6f} (width={width:.2e})"
                )
    
    def _save_spectrum(self, spectrum_data: Dict):
        """Save spectrum to file"""
        path = Path(self.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to JSONL file
        with open(path, 'a') as f:
            f.write(json.dumps(spectrum_data) + '\n')
    
    def plot_evolution(self, save_fig: Optional[str] = None):
        """Plot eigenvalue evolution"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return
        
        if not self.spectra:
            logger.warning("No spectra recorded")
            return
        
        # Extract data
        times = [s['timestamp'] for s in self.spectra]
        times = np.array(times) - times[0]  # Relative time
        
        # Plot each eigenvalue's evolution
        n_eigen = len(self.spectra[0]['eigenvalues'])
        eigenvalue_traces = np.zeros((len(self.spectra), n_eigen))
        
        for i, spectrum in enumerate(self.spectra):
            eigenvalue_traces[i] = spectrum['eigenvalues']
        
        plt.figure(figsize=(10, 6))
        for j in range(n_eigen):
            plt.plot(times, eigenvalue_traces[:, j], 'b-', alpha=0.3, linewidth=0.5)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalue Spectrum Evolution')
        plt.grid(True, alpha=0.3)
        
        if save_fig:
            plt.savefig(save_fig, dpi=150, bbox_inches='tight')
        else:
            plt.show()


class PhysicsMonitor:
    """
    Comprehensive physics monitoring system
    Tracks multiple conservation laws and physics invariants
    """
    
    def __init__(self, system_name: str = "System"):
        self.system_name = system_name
        self.monitors = {
            'energy': EnergyTracker(f"{system_name}_energy"),
            'spectrum': EigenSpectrumProbe(f"spectra/{system_name}_spectrum.jsonl")
        }
        self.invariants = {}
        self.history = []
        
    def register_invariant(self, name: str, compute_func: Callable, tolerance: float = 1e-6):
        """Register a physics invariant to monitor"""
        self.invariants[name] = {
            'compute': compute_func,
            'tolerance': tolerance,
            'initial': None,
            'violations': 0
        }
    
    def checkpoint(self, system: Any, label: str = ""):
        """Take a physics checkpoint of the system"""
        checkpoint_data = {
            'label': label,
            'timestamp': time.time(),
            'invariants': {}
        }
        
        # Check all registered invariants
        for name, inv_data in self.invariants.items():
            try:
                current_value = inv_data['compute'](system)
                
                if inv_data['initial'] is None:
                    inv_data['initial'] = current_value
                
                drift = abs(current_value - inv_data['initial'])
                relative_drift = drift / (abs(inv_data['initial']) + 1e-10)
                
                checkpoint_data['invariants'][name] = {
                    'value': current_value,
                    'drift': drift,
                    'relative_drift': relative_drift
                }
                
                # Check violation
                if relative_drift > inv_data['tolerance']:
                    inv_data['violations'] += 1
                    logger.warning(
                        f"Invariant violation in {name}: "
                        f"drift = {relative_drift:.2e} > {inv_data['tolerance']:.2e}"
                    )
                
            except Exception as e:
                logger.error(f"Failed to compute invariant {name}: {e}")
                checkpoint_data['invariants'][name] = {'error': str(e)}
        
        self.history.append(checkpoint_data)
        return checkpoint_data
    
    def report(self) -> Dict[str, Any]:
        """Generate comprehensive physics report"""
        report = {
            'system': self.system_name,
            'checkpoints': len(self.history),
            'invariants': {}
        }
        
        # Summarize each invariant
        for name, inv_data in self.invariants.items():
            if self.history:
                values = [
                    h['invariants'].get(name, {}).get('value', np.nan)
                    for h in self.history
                    if name in h['invariants'] and 'value' in h['invariants'][name]
                ]
                
                if values:
                    report['invariants'][name] = {
                        'initial': inv_data['initial'],
                        'final': values[-1],
                        'max_drift': max(
                            h['invariants'].get(name, {}).get('relative_drift', 0)
                            for h in self.history
                        ),
                        'violations': inv_data['violations'],
                        'tolerance': inv_data['tolerance']
                    }
        
        # Add energy tracking report
        if 'energy' in self.monitors:
            report['energy'] = self.monitors['energy'].report()
        
        return report


def create_physics_notebook(output_path: str = "physics_validation_notebook.ipynb"):
    """
    Create a Jupyter notebook for physics validation
    """
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Physics Validation Notebook\n",
                    "Interactive tests for soliton memory physics"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "from python.core.strang_integrator import StrangIntegrator, create_sech_soliton\n",
                    "from tests.test_physics_validation import PhysicsValidator\n",
                    "\n",
                    "%matplotlib inline"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Analytic Soliton Test"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Create 1D grid\n",
                    "L = 40.0\n",
                    "N = 512\n",
                    "x = np.linspace(-L/2, L/2, N, endpoint=False)\n",
                    "\n",
                    "# Initial soliton\n",
                    "psi0 = create_sech_soliton(x, v=0.5)\n",
                    "\n",
                    "# Plot\n",
                    "plt.figure(figsize=(10, 4))\n",
                    "plt.subplot(121)\n",
                    "plt.plot(x, np.abs(psi0)**2, 'b-', label='|ψ|²')\n",
                    "plt.xlabel('x')\n",
                    "plt.ylabel('Density')\n",
                    "plt.title('Soliton Density')\n",
                    "plt.grid(True)\n",
                    "\n",
                    "plt.subplot(122)\n",
                    "plt.plot(x, np.angle(psi0), 'r-', label='arg(ψ)')\n",
                    "plt.xlabel('x')\n",
                    "plt.ylabel('Phase')\n",
                    "plt.title('Soliton Phase')\n",
                    "plt.grid(True)\n",
                    "plt.tight_layout()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Energy Conservation Test"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Run physics validation\n",
                    "validator = PhysicsValidator()\n",
                    "result = validator.test_1d_soliton()\n",
                    "print(f\"Test result: {'PASSED' if result else 'FAILED'}\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save notebook
    import json
    with open(output_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    logger.info(f"Created physics notebook: {output_path}")


# Example usage decorators
@EnergyTracker(name="Lattice")
def evolve_lattice(lattice, dt: float):
    """Example of decorated evolution function"""
    # Evolution code here
    pass


if __name__ == "__main__":
    # Demo the physics instrumentation
    print("Physics Instrumentation Toolkit Demo")
    print("="*50)
    
    # Test PhysicalQuantity
    energy = PhysicalQuantity(1.234, "eV", 0.001)
    momentum = PhysicalQuantity(0.567, "eV/c", 0.0005)
    print(f"Energy: {energy}")
    print(f"Momentum: {momentum}")
    
    # Test unit safety
    try:
        bad_sum = energy + momentum
    except ValueError as e:
        print(f"Unit safety works: {e}")
    
    # Create physics notebook
    create_physics_notebook()
    print("\nCreated physics_validation_notebook.ipynb")
