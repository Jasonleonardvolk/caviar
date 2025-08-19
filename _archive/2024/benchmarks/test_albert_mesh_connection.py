#!/usr/bin/env python3
"""
Test ALBERT → ψMesh Communication
"""

from albert.api.interface import init_metric, push_phase_to_ψmesh
import numpy as np

print("🧪 Testing ALBERT → ψMesh connection...")
print("=" * 50)

# Initialize Kerr metric
print("\n1️⃣ Initializing Kerr metric...")
init_metric("kerr", {"M": 1.0, "a": 0.5})
print("✅ Metric initialized")

# Define spacetime region
print("\n2️⃣ Defining spacetime region...")
region = {
    "r": np.linspace(1.9, 3.0, 128),
    "theta": np.array([np.pi / 2]),
}
print(f"📍 Region: r ∈ [{region['r'][0]:.2f}, {region['r'][-1]:.2f}], θ = π/2")

# Push phase to ψMesh
print("\n3️⃣ Pushing phase modulation to ψMesh...")
push_phase_to_ψmesh("equatorial_belt", region)

print("\n✅ Test complete! If you see both the 'ψMesh ⟵ Phase injected' and '[ψMesh] registered' messages above, the connection is live!")
