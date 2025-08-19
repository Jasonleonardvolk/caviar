#!/usr/bin/env python3
"""
Test energy conservation after dark-bright soliton suppression
Ensures mass/energy conservation within 0.1% tolerance
"""

import numpy as np
import pytest
from typing import List, Tuple

# Import with fallbacks
try:
    from python.core.oscillator_lattice import OscillatorLattice, get_global_lattice
    from python.core.soliton_memory import SolitonMemory
except ImportError:
    from oscillator_lattice import OscillatorLattice, get_global_lattice
    from soliton_memory import SolitonMemory

class TestDarkBrightEnergyConservation:
    """Test suite for energy conservation in dark-bright soliton interactions"""
    
    def setup_method(self):
        """Set up test environment"""
        self.lattice = OscillatorLattice(size=100)
        self.tolerance = 0.001  # 0.1% tolerance
        
    def create_bright_soliton(self, amplitude: float = 1.0, position: int = 25) -> int:
        """Create a bright soliton and return its oscillator index"""
        phase = 0.0
        frequency = 0.1
        
        idx = self.lattice.add_oscillator(
            phase=phase,
            natural_freq=frequency,
            amplitude=amplitude,
            stability=1.0
        )
        
        # Set soliton profile (sech shape for bright soliton)
        if hasattr(self.lattice, 'oscillators'):
            self.lattice.oscillators[idx]['type'] = 'bright'
            self.lattice.oscillators[idx]['position'] = position
            
        return idx
    
    def create_dark_soliton(self, amplitude: float = -0.8, position: int = 75) -> int:
        """Create a dark soliton and return its oscillator index"""
        phase = np.pi  # π phase shift for dark soliton
        frequency = 0.1
        
        idx = self.lattice.add_oscillator(
            phase=phase,
            natural_freq=frequency,
            amplitude=amplitude,
            stability=0.9
        )
        
        # Dark soliton properties
        if hasattr(self.lattice, 'oscillators'):
            self.lattice.oscillators[idx]['type'] = 'dark'
            self.lattice.oscillators[idx]['position'] = position
            self.lattice.oscillators[idx]['baseline'] = 1.0  # Background amplitude
            
        return idx
    
    def calculate_total_energy(self) -> float:
        """Calculate total energy in the system"""
        total_energy = 0.0
        
        # Kinetic energy from oscillator motion
        for osc in self.lattice.oscillators:
            if osc.get('active', True):
                # E_kinetic = 0.5 * |amplitude|^2 * frequency^2
                amplitude = osc.get('amplitude', 0.0)
                frequency = osc.get('natural_freq', 0.0)
                kinetic = 0.5 * amplitude**2 * frequency**2
                total_energy += kinetic
        
        # Potential energy from couplings
        if hasattr(self.lattice, 'K') and self.lattice.K is not None:
            K = self.lattice.K
            for i in range(len(self.lattice.oscillators)):
                for j in range(i+1, len(self.lattice.oscillators)):
                    if i < K.shape[0] and j < K.shape[1]:
                        coupling = K[i, j]
                        if coupling != 0:
                            # E_potential = coupling * amplitude_i * amplitude_j * cos(phase_diff)
                            amp_i = self.lattice.oscillators[i].get('amplitude', 0.0)
                            amp_j = self.lattice.oscillators[j].get('amplitude', 0.0)
                            phase_i = self.lattice.oscillators[i].get('phase', 0.0)
                            phase_j = self.lattice.oscillators[j].get('phase', 0.0)
                            phase_diff = phase_i - phase_j
                            
                            potential = coupling * amp_i * amp_j * np.cos(phase_diff)
                            total_energy += potential
        
        # Background field energy (for dark solitons)
        for osc in self.lattice.oscillators:
            if osc.get('type') == 'dark' and osc.get('active', True):
                baseline = osc.get('baseline', 0.0)
                total_energy += 0.5 * baseline**2
        
        return total_energy
    
    def simulate_collision(self, bright_idx: int, dark_idx: int, steps: int = 100) -> Tuple[float, float]:
        """
        Simulate collision between bright and dark solitons
        Returns (initial_energy, final_energy)
        """
        # Record initial energy
        initial_energy = self.calculate_total_energy()
        
        # Simulate approach (move solitons toward each other)
        bright_pos = self.lattice.oscillators[bright_idx].get('position', 25)
        dark_pos = self.lattice.oscillators[dark_idx].get('position', 75)
        
        for step in range(steps // 2):
            # Move solitons
            bright_pos += 0.5
            dark_pos -= 0.5
            
            self.lattice.oscillators[bright_idx]['position'] = bright_pos
            self.lattice.oscillators[dark_idx]['position'] = dark_pos
            
            # Check for collision
            if abs(bright_pos - dark_pos) < 5.0:
                # Collision detected - apply suppression logic
                self._apply_dark_bright_suppression(bright_idx, dark_idx)
                break
        
        # Final energy after collision
        final_energy = self.calculate_total_energy()
        
        return initial_energy, final_energy
    
    def _apply_dark_bright_suppression(self, bright_idx: int, dark_idx: int):
        """Apply dark-bright soliton suppression rules"""
        bright = self.lattice.oscillators[bright_idx]
        dark = self.lattice.oscillators[dark_idx]
        
        # Energy before suppression
        bright_energy = 0.5 * bright['amplitude']**2 * bright['natural_freq']**2
        dark_energy = 0.5 * dark['amplitude']**2 * dark['natural_freq']**2
        dark_baseline_energy = 0.5 * dark.get('baseline', 1.0)**2
        
        total_before = bright_energy + abs(dark_energy) + dark_baseline_energy
        
        # Suppression: bright soliton is "vaulted" (phase shifted)
        bright['phase'] += np.pi/2  # 90° phase shift
        bright['amplitude'] *= 0.5  # Reduced amplitude
        bright['vaulted'] = True
        
        # Dark soliton fills the gap
        dark['amplitude'] = -np.sqrt(abs(dark['amplitude']**2 + bright_energy/dark['natural_freq']**2))
        
        # Energy should be conserved (redistributed)
        # Some energy may go into background field oscillations
        background_boost = total_before * 0.1  # 10% to background
        dark['baseline'] = np.sqrt(dark.get('baseline', 1.0)**2 + 2*background_boost)
    
    def test_single_collision_energy_conservation(self):
        """Test energy conservation in a single bright-dark collision"""
        # Create solitons
        bright_idx = self.create_bright_soliton(amplitude=1.5)
        dark_idx = self.create_dark_soliton(amplitude=-1.0)
        
        # Add coupling between them
        self.lattice.set_coupling(bright_idx, dark_idx, 0.1)
        
        # Simulate collision
        initial_energy, final_energy = self.simulate_collision(bright_idx, dark_idx)
        
        # Check energy conservation
        energy_change = abs(final_energy - initial_energy) / initial_energy
        
        assert energy_change < self.tolerance, \
            f"Energy not conserved: {energy_change:.4%} change (initial: {initial_energy:.6f}, final: {final_energy:.6f})"
        
        # Verify bright soliton was vaulted
        assert self.lattice.oscillators[bright_idx].get('vaulted', False), \
            "Bright soliton should be vaulted after collision"
    
    def test_multiple_collision_cascade(self):
        """Test energy conservation in cascade of collisions"""
        # Create multiple solitons
        bright_indices = []
        dark_indices = []
        
        for i in range(3):
            bright_idx = self.create_bright_soliton(amplitude=1.0 + i*0.2, position=20 + i*10)
            dark_idx = self.create_dark_soliton(amplitude=-0.8 - i*0.1, position=80 - i*10)
            
            bright_indices.append(bright_idx)
            dark_indices.append(dark_idx)
            
            # Add couplings
            if i > 0:
                self.lattice.set_coupling(bright_indices[i], bright_indices[i-1], 0.05)
                self.lattice.set_coupling(dark_indices[i], dark_indices[i-1], 0.05)
        
        # Initial total energy
        initial_energy = self.calculate_total_energy()
        
        # Simulate multiple collisions
        for bright_idx, dark_idx in zip(bright_indices, dark_indices):
            _, _ = self.simulate_collision(bright_idx, dark_idx, steps=50)
        
        # Final total energy
        final_energy = self.calculate_total_energy()
        
        # Check conservation
        energy_change = abs(final_energy - initial_energy) / initial_energy
        
        assert energy_change < self.tolerance, \
            f"Energy not conserved in cascade: {energy_change:.4%} change"
    
    def test_energy_redistribution_profile(self):
        """Test that energy is properly redistributed, not lost"""
        # Create high-energy collision scenario
        bright_idx = self.create_bright_soliton(amplitude=2.0)
        dark_idx = self.create_dark_soliton(amplitude=-1.5)
        
        # Strong coupling for dramatic collision
        self.lattice.set_coupling(bright_idx, dark_idx, 0.3)
        
        # Track energy components
        initial_bright = 0.5 * self.lattice.oscillators[bright_idx]['amplitude']**2
        initial_dark = 0.5 * self.lattice.oscillators[dark_idx]['amplitude']**2
        initial_baseline = 0.5 * self.lattice.oscillators[dark_idx].get('baseline', 1.0)**2
        
        # Collision
        initial_total, final_total = self.simulate_collision(bright_idx, dark_idx)
        
        # Check redistribution
        final_bright = 0.5 * self.lattice.oscillators[bright_idx]['amplitude']**2
        final_dark = 0.5 * self.lattice.oscillators[dark_idx]['amplitude']**2
        final_baseline = 0.5 * self.lattice.oscillators[dark_idx].get('baseline', 1.0)**2
        
        # Energy should move from bright to dark/baseline
        assert final_bright < initial_bright, "Bright soliton should lose energy"
        assert abs(final_dark) > abs(initial_dark), "Dark soliton should gain energy"
        assert final_baseline > initial_baseline, "Background should gain energy"
        
        # Total conservation
        energy_change = abs(final_total - initial_total) / initial_total
        assert energy_change < self.tolerance, \
            f"Total energy not conserved: {energy_change:.4%} change"
    
    def test_phase_space_volume_conservation(self):
        """Test Liouville's theorem - phase space volume conservation"""
        # Create ensemble of solitons
        num_solitons = 10
        bright_indices = []
        dark_indices = []
        
        for i in range(num_solitons // 2):
            b_idx = self.create_bright_soliton(
                amplitude=0.5 + np.random.random(),
                position=np.random.randint(10, 40)
            )
            d_idx = self.create_dark_soliton(
                amplitude=-(0.5 + np.random.random()),
                position=np.random.randint(60, 90)
            )
            
            bright_indices.append(b_idx)
            dark_indices.append(d_idx)
        
        # Calculate initial phase space volume (simplified)
        initial_volume = 1.0
        for osc in self.lattice.oscillators:
            # Volume ~ product of amplitude * phase uncertainties
            initial_volume *= (1.0 + abs(osc['amplitude'])) * 2 * np.pi
        
        # Run dynamics
        for _ in range(5):
            idx = np.random.randint(0, len(bright_indices))
            self.simulate_collision(bright_indices[idx], dark_indices[idx], steps=20)
        
        # Calculate final phase space volume
        final_volume = 1.0
        for osc in self.lattice.oscillators:
            if osc.get('active', True):
                final_volume *= (1.0 + abs(osc['amplitude'])) * 2 * np.pi
        
        # Check volume conservation (log scale for numerical stability)
        volume_ratio = np.log(final_volume) / np.log(initial_volume)
        
        assert abs(volume_ratio - 1.0) < 0.1, \
            f"Phase space volume not conserved: ratio = {volume_ratio:.4f}"


if __name__ == "__main__":
    # Run tests
    test = TestDarkBrightEnergyConservation()
    
    print("Running energy conservation tests...")
    
    test.setup_method()
    test.test_single_collision_energy_conservation()
    print("✓ Single collision energy conservation")
    
    test.setup_method()
    test.test_multiple_collision_cascade()
    print("✓ Multiple collision cascade")
    
    test.setup_method()
    test.test_energy_redistribution_profile()
    print("✓ Energy redistribution profile")
    
    test.setup_method()
    test.test_phase_space_volume_conservation()
    print("✓ Phase space volume conservation")
    
    print("\n✅ All energy conservation tests passed!")
