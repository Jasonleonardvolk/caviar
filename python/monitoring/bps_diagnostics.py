"""
BPS Soliton Diagnostics Module
Monitoring and validation for BPS soliton behavior
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json

from python.core.bps_oscillator_enhanced import BPSEnhancedLattice
from python.core.bps_soliton_memory_enhanced import BPSEnhancedSolitonMemory
from python.core.bps_config_enhanced import SolitonPolarity, BPS_CONFIG

logger = logging.getLogger(__name__)


class BPSDiagnostics:
    """Diagnostic tools for BPS soliton monitoring"""
    
    def __init__(self, lattice: Optional[BPSEnhancedLattice] = None,
                 memory: Optional[BPSEnhancedSolitonMemory] = None):
        self.lattice = lattice
        self.memory = memory
        self.diagnostics_history = []
        
        logger.info("🔬 BPS Diagnostics initialized")
    
    def compute_total_charge(self, lattice: Optional[BPSEnhancedLattice] = None) -> float:
        """
        Compute total topological charge Q
        
        Args:
            lattice: Lattice to analyze (uses self.lattice if None)
        
        Returns:
            Total topological charge
        """
        
        if lattice is None:
            lattice = self.lattice
        
        if lattice is None:
            logger.error("No lattice available for charge computation")
            return 0.0
        
        total_charge = 0.0
        
        for osc in lattice.oscillator_objects:
            if osc.polarity == SolitonPolarity.BPS:
                total_charge += osc.charge
        
        return total_charge
    
    def bps_energy_report(self, lattice: Optional[BPSEnhancedLattice] = None) -> Dict[str, Any]:
        """
        Generate energy vs charge report for BPS solitons
        
        Verifies E = |Q| saturation for each BPS soliton
        
        Args:
            lattice: Lattice to analyze
        
        Returns:
            Detailed energy report
        """
        
        if lattice is None:
            lattice = self.lattice
        
        if lattice is None:
            return {"error": "No lattice available"}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "num_bps_solitons": 0,
            "total_charge": 0.0,
            "total_energy": 0.0,
            "solitons": [],
            "compliance_summary": {
                "compliant": 0,
                "non_compliant": 0,
                "max_deviation": 0.0
            }
        }
        
        for i, osc in enumerate(lattice.oscillator_objects):
            if osc.polarity == SolitonPolarity.BPS:
                energy = osc.amplitude ** 2
                target_energy = osc.charge ** 2
                deviation = abs(energy - target_energy)
                compliant = deviation < BPS_CONFIG.bps_energy_tolerance
                
                soliton_info = {
                    "index": i,
                    "charge": osc.charge,
                    "energy": energy,
                    "target_energy": target_energy,
                    "deviation": deviation,
                    "compliant": compliant,
                    "phase": osc.phase,
                    "amplitude": osc.amplitude
                }
                
                report["solitons"].append(soliton_info)
                report["num_bps_solitons"] += 1
                report["total_charge"] += osc.charge
                report["total_energy"] += energy
                
                if compliant:
                    report["compliance_summary"]["compliant"] += 1
                else:
                    report["compliance_summary"]["non_compliant"] += 1
                    report["compliance_summary"]["max_deviation"] = max(
                        report["compliance_summary"]["max_deviation"],
                        deviation
                    )
        
        # Log warnings for non-compliance
        if report["compliance_summary"]["non_compliant"] > 0:
            logger.warning(f"BPS energy compliance violation: "
                          f"{report['compliance_summary']['non_compliant']} solitons "
                          f"with max deviation {report['compliance_summary']['max_deviation']:.6f}")
        
        return report
    
    def verify_charge_conservation(self, before_lattice: BPSEnhancedLattice,
                                  after_lattice: BPSEnhancedLattice) -> bool:
        """
        Verify charge conservation between two lattice states
        
        Args:
            before_lattice: Lattice state before operation
            after_lattice: Lattice state after operation
        
        Returns:
            True if charge is conserved within tolerance
        """
        
        charge_before = self.compute_total_charge(before_lattice)
        charge_after = self.compute_total_charge(after_lattice)
        
        deviation = abs(charge_after - charge_before)
        conserved = deviation < BPS_CONFIG.charge_conservation_tolerance
        
        if not conserved:
            logger.error(f"❌ Charge conservation violated: "
                        f"Q_before={charge_before:.6f}, Q_after={charge_after:.6f}, "
                        f"deviation={deviation:.6f}")
        else:
            logger.info(f"✅ Charge conserved: Q={charge_after:.6f}")
        
        # Record in history
        self.diagnostics_history.append({
            "timestamp": datetime.now().isoformat(),
            "operation": "charge_conservation_check",
            "charge_before": charge_before,
            "charge_after": charge_after,
            "deviation": deviation,
            "conserved": conserved
        })
        
        return conserved
    
    def monitor_bps_stability(self, duration: float = 1.0, 
                             sample_interval: float = 0.1) -> Dict[str, Any]:
        """
        Monitor BPS soliton stability over time
        
        Args:
            duration: Monitoring duration in seconds
            sample_interval: Sampling interval in seconds
        
        Returns:
            Stability report
        """
        
        if self.lattice is None:
            return {"error": "No lattice available"}
        
        import time
        
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Take sample
            sample = {
                "timestamp": time.time() - start_time,
                "total_charge": self.compute_total_charge(),
                "energy_report": self.bps_energy_report()
            }
            samples.append(sample)
            
            time.sleep(sample_interval)
        
        # Analyze stability
        charges = [s["total_charge"] for s in samples]
        charge_std = np.std(charges) if charges else 0.0
        
        stability_report = {
            "duration": duration,
            "num_samples": len(samples),
            "charge_mean": np.mean(charges) if charges else 0.0,
            "charge_std": charge_std,
            "charge_stable": charge_std < BPS_CONFIG.charge_conservation_tolerance,
            "samples": samples
        }
        
        return stability_report
    
    def stress_test_bps(self, num_solitons: int = 10,
                       num_operations: int = 100) -> Dict[str, Any]:
        """
        Stress test BPS soliton operations
        
        Args:
            num_solitons: Number of BPS solitons to create
            num_operations: Number of operations to perform
        
        Returns:
            Stress test results
        """
        
        if self.lattice is None:
            return {"error": "No lattice available"}
        
        results = {
            "num_solitons": num_solitons,
            "num_operations": num_operations,
            "creation_success": 0,
            "annihilation_success": 0,
            "charge_violations": 0,
            "energy_violations": 0
        }
        
        # Test creation
        created_indices = []
        for i in range(num_solitons):
            charge = (-1) ** i  # Alternate charges
            success = self.lattice.create_bps_soliton(i, charge)
            if success:
                results["creation_success"] += 1
                created_indices.append(i)
        
        initial_charge = self.compute_total_charge()
        
        # Test operations
        for op in range(num_operations):
            # Random operation
            if np.random.random() < 0.5 and created_indices:
                # Remove random soliton
                idx = np.random.choice(created_indices)
                self.lattice.remove_bps_soliton(idx)
                created_indices.remove(idx)
            elif len(created_indices) < self.lattice.size:
                # Add new soliton
                idx = np.random.randint(0, self.lattice.size)
                if idx not in created_indices:
                    charge = np.random.choice([-1, 1])
                    if self.lattice.create_bps_soliton(idx, charge):
                        created_indices.append(idx)
            
            # Check invariants
            report = self.bps_energy_report()
            if report["compliance_summary"]["non_compliant"] > 0:
                results["energy_violations"] += 1
        
        # Final charge check
        final_charge = self.compute_total_charge()
        
        results["final_charge"] = final_charge
        results["charge_preserved"] = abs(final_charge) < 0.1  # Should sum to ~0 with alternating charges
        
        return results
    
    def export_diagnostics(self, filename: Optional[str] = None) -> str:
        """
        Export diagnostics history to JSON
        
        Args:
            filename: Output filename (auto-generated if None)
        
        Returns:
            Filename written
        """
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bps_diagnostics_{timestamp}.json"
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "enable_bps": BPS_CONFIG.enable_bps,
                "strict_mode": BPS_CONFIG.strict_bps_mode,
                "energy_tolerance": BPS_CONFIG.bps_energy_tolerance,
                "charge_tolerance": BPS_CONFIG.charge_conservation_tolerance
            },
            "current_state": {
                "total_charge": self.compute_total_charge() if self.lattice else None,
                "energy_report": self.bps_energy_report() if self.lattice else None
            },
            "history": self.diagnostics_history
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📊 Diagnostics exported to {filename}")
        return filename
    
    def real_time_monitor(self):
        """
        Real-time monitoring dashboard output
        
        This can be called periodically to update a UI or log
        """
        
        if self.lattice is None:
            return {"error": "No lattice available"}
        
        # Current snapshot
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "bps_solitons": len(self.lattice.bps_indices),
            "total_charge": self.compute_total_charge(),
            "energy_report": self.bps_energy_report()
        }
        
        # Format for logging
        logger.info(f"[BPS Monitor] Solitons: {snapshot['bps_solitons']}, "
                   f"Q: {snapshot['total_charge']:.3f}, "
                   f"Compliant: {snapshot['energy_report']['compliance_summary']['compliant']}/{snapshot['bps_solitons']}")
        
        return snapshot


# Convenience functions for integration
def log_bps_diagnostics(lattice: BPSEnhancedLattice):
    """Log BPS diagnostics for a lattice"""
    diagnostics = BPSDiagnostics(lattice)
    report = diagnostics.bps_energy_report()
    logger.info(f"BPS Diagnostics: {report['num_bps_solitons']} solitons, "
               f"Q={report['total_charge']:.3f}, "
               f"Energy compliance: {report['compliance_summary']['compliant']}/{report['num_bps_solitons']}")
    return report


def verify_hot_swap_conservation(before: BPSEnhancedLattice, 
                                after: BPSEnhancedLattice) -> bool:
    """Verify charge conservation in hot-swap"""
    diagnostics = BPSDiagnostics()
    return diagnostics.verify_charge_conservation(before, after)
