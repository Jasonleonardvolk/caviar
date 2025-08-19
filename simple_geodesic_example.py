#!/usr/bin/env python3
"""
Simple ALBERT Geodesic Example
Trace a light ray near a black hole
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import albert
import numpy as np

# Initialize Kerr metric
print("🌌 Setting up rotating black hole...")
metric = albert.init_metric("kerr", params={"M": 1, "a": 0.5})

# Light ray starting at r=5M, moving inward
print("\n💫 Tracing light ray...")
initial_pos = [0, 5, np.pi/2, 0]  # t, r, θ, φ
initial_vel = [1, -0.3, 0, 0.1]   # photon 4-velocity

# Trace the geodesic
solution = albert.trace_geodesic(
    initial_position=initial_pos,
    initial_velocity=initial_vel,
    lam_span=(0, 20),
    steps=500
)

# Extract results
trajectory = albert.core.extract_trajectory(solution)

print(f"\n📊 Results:")
print(f"   Starting radius: {trajectory['r'][0]:.2f}M")
print(f"   Final radius: {trajectory['r'][-1]:.2f}M")
print(f"   Min radius reached: {np.min(trajectory['r']):.2f}M")

# Check if it's a photon
if albert.core.is_null_geodesic(solution):
    print("   ✅ Confirmed: Light ray (null geodesic)")

# Did it escape or fall in?
if trajectory['r'][-1] > trajectory['r'][0]:
    print("   🌟 Light ray escaped!")
else:
    print("   🕳️ Light ray fell into black hole!")
