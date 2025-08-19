#!/usr/bin/env python3
"""
Test ALBERT Phase Encoding
Demonstrates curvature-to-phase conversion for ψ-mesh integration
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import albert
    
    print("🌌 ALBERT Phase Encoding Test")
    print("=" * 50)
    
    # Initialize Kerr metric for a rotating black hole
    print("\n1️⃣ Initializing Kerr spacetime...")
    M = 1.0  # Black hole mass
    a = 0.9  # High spin parameter
    
    metric = albert.init_metric("kerr", params={"M": M, "a": a})
    print(f"✅ Kerr metric initialized (M={M}, a={a})")
    
    # Define region to sample - near the horizon
    print("\n2️⃣ Defining spacetime region...")
    horizon_radius = M + np.sqrt(M**2 - a**2)  # Outer horizon
    
    region_sample = {
        "r": np.linspace(1.9, 3.0, 100),  # From near horizon to 3M
        "theta": np.array([np.pi / 2])    # Equatorial slice
    }
    
    print(f"📍 Sampling region:")
    print(f"   r: {region_sample['r'][0]:.2f}M to {region_sample['r'][-1]:.2f}M")
    print(f"   θ: π/2 (equatorial plane)")
    print(f"   Horizon at r = {horizon_radius:.2f}M")
    
    # Generate phase modulation
    print("\n3️⃣ Computing phase modulation from curvature...")
    phase_data = albert.generate_phase_modulation(region_sample)
    
    print(f"\n✅ Phase modulation computed!")
    print(f"   Grid shape: {phase_data['grid_shape']}")
    print(f"   Phase range: [{np.min(phase_data['psi_phase']):.3f}, "
          f"{np.max(phase_data['psi_phase']):.3f}] radians")
    print(f"   Amplitude range: [{np.min(phase_data['psi_amplitude']):.3f}, "
          f"{np.max(phase_data['psi_amplitude']):.3f}]")
    
    # Extract data for plotting
    r_values = region_sample['r']
    psi_phase = phase_data['psi_phase'].flatten()
    psi_amplitude = phase_data['psi_amplitude'].flatten()
    curvature = phase_data['curvature_values'].flatten()
    
    # Create visualization
    print("\n4️⃣ Creating visualization...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    # Plot 1: Curvature
    ax1.semilogy(r_values, np.abs(curvature), 'b-', linewidth=2)
    ax1.axvline(x=horizon_radius, color='k', linestyle='--', alpha=0.5, label='Horizon')
    ax1.set_ylabel('|Kretschmann Scalar|')
    ax1.set_title('Spacetime Curvature near Rotating Black Hole')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Phase modulation
    ax2.plot(r_values, psi_phase, 'r-', linewidth=2)
    ax2.axvline(x=horizon_radius, color='k', linestyle='--', alpha=0.5)
    ax2.set_ylabel('ψ Phase (radians)')
    ax2.set_title('Curvature-Induced Phase Shifts')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-np.pi, np.pi)
    
    # Plot 3: Amplitude modulation
    ax3.plot(r_values, psi_amplitude, 'g-', linewidth=2)
    ax3.axvline(x=horizon_radius, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('r/M')
    ax3.set_ylabel('ψ Amplitude')
    ax3.set_title('Memory Density Modulation')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('phase_encoding_test.png', dpi=150)
    print("📊 Visualization saved as 'phase_encoding_test.png'")
    
    # Demonstrate conceptual impact
    print("\n5️⃣ Conceptual implications for ψ-mesh:")
    
    # Find maximum curvature point
    max_curv_idx = np.argmax(np.abs(curvature))
    r_max_curv = r_values[max_curv_idx]
    
    print(f"\n🌀 Maximum curvature at r = {r_max_curv:.2f}M:")
    print(f"   Phase twist: {psi_phase[max_curv_idx]:.3f} radians")
    print(f"   Amplitude compression: {1 - psi_amplitude[max_curv_idx]:.1%}")
    
    # Simulate concept traversal
    print("\n🎯 Simulated concept trajectory through curved region:")
    
    # Sample points along trajectory
    traj_indices = [10, 30, 50, 70, 90]
    accumulated_phase = 0
    
    for i, idx in enumerate(traj_indices):
        r = r_values[idx]
        phase = psi_phase[idx]
        amp = psi_amplitude[idx]
        accumulated_phase += phase
        
        print(f"   Step {i+1}: r={r:.2f}M")
        print(f"      Local phase: {phase:+.3f} rad")
        print(f"      Local amplitude: {amp:.3f}")
        print(f"      Accumulated phase: {accumulated_phase:+.3f} rad")
    
    print("\n🔮 Physical interpretations:")
    print("   • Strong curvature creates phase twists in concept relationships")
    print("   • Near horizon: extreme phase gradients = attention singularity")
    print("   • Amplitude compression = memory density collapse")
    print("   • Trapped surfaces would show as phase vortices")
    
    print("\n✅ Phase encoding test complete!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure to install required packages:")
    print("  pip install numpy matplotlib sympy scipy")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
