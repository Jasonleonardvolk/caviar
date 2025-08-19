# Copyright 2025 ALAN Team and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patent Peace / Retaliation Notice:
#   As stated in Section 3 of the Apache 2.0 License, any entity that
#   initiates patent litigation (including a cross-claim or counterclaim)
#   alleging that this software or a contribution embodied within it
#   infringes a patent shall have all patent licenses granted herein
#   terminated as of the date such litigation is filed.

"""
Tests for the snapshot serialization module.

These tests verify that state snapshots can be correctly serialized and
deserialized using FlatBuffers, ensuring portability across different
processes and languages.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
import pytest
import numpy as np

from alan_backend.core.oscillator.banksy_oscillator import BanksyOscillator, BanksyConfig, SpinVector
from alan_backend.snapshot import StateSnapshot, to_bytes, from_bytes


def test_snapshot_basic_serialization():
    """Test basic serialization and deserialization of snapshots."""
    # Create sample data
    n_oscillators = 16
    
    # Phase and momentum
    theta = np.linspace(0, 2*np.pi, n_oscillators, endpoint=False)
    p_theta = np.random.normal(0, 0.1, n_oscillators)
    
    # Spin vectors and momenta (normalized)
    sigma = np.random.normal(0, 1, (n_oscillators, 3))
    sigma = sigma / np.linalg.norm(sigma, axis=1, keepdims=True)
    p_sigma = np.random.normal(0, 0.1, (n_oscillators, 3))
    
    # Timesteps
    dt_phase = 0.01
    dt_spin = 0.00125
    
    # Create snapshot
    snapshot = StateSnapshot(
        theta=theta,
        p_theta=p_theta,
        sigma=sigma,
        p_sigma=p_sigma,
        dt_phase=dt_phase,
        dt_spin=dt_spin,
    )
    
    # Serialize to bytes
    buffer = snapshot.to_bytes()
    
    # Verify it's a bytes object with the correct identifier
    assert isinstance(buffer, bytes)
    assert buffer[4:8] == b"ALSN"
    
    # Deserialize
    restored = StateSnapshot.from_bytes(buffer)
    
    # Check that the data is preserved
    np.testing.assert_allclose(restored.theta, theta)
    np.testing.assert_allclose(restored.p_theta, p_theta)
    np.testing.assert_allclose(restored.sigma, sigma)
    np.testing.assert_allclose(restored.p_sigma, p_sigma)
    assert restored.dt_phase == dt_phase
    assert restored.dt_spin == dt_spin
    assert restored.version == 2


def test_snapshot_from_oscillator():
    """Test creating a snapshot from an oscillator state."""
    # Create an oscillator
    n_oscillators = 32
    config = BanksyConfig(gamma=0.1, epsilon=0.01, eta_damp=1e-4, dt=0.01)
    oscillator = BanksyOscillator(n_oscillators, config)
    
    # Run for a few steps
    for _ in range(10):
        oscillator.step()
    
    # Create a snapshot
    theta = oscillator.phases
    p_theta = oscillator.momenta
    sigma = np.array([s.as_array() for s in oscillator.spins])
    
    # We don't have direct access to spin momenta in the oscillator,
    # so we'll use zeros for this test
    p_sigma = np.zeros_like(sigma)
    
    snapshot = StateSnapshot(
        theta=theta,
        p_theta=p_theta,
        sigma=sigma,
        p_sigma=p_sigma,
        dt_phase=config.dt,
        dt_spin=config.dt / 8,  # Typical sub-step ratio
    )
    
    # Serialize and deserialize
    buffer = snapshot.to_bytes()
    restored = StateSnapshot.from_bytes(buffer)
    
    # Check that it matches the original oscillator state
    np.testing.assert_allclose(restored.theta, oscillator.phases)
    np.testing.assert_allclose(restored.p_theta, oscillator.momenta)
    
    for i, spin in enumerate(oscillator.spins):
        np.testing.assert_allclose(restored.sigma[i], spin.as_array())


@pytest.mark.slow
def test_cross_process_snapshot(tmp_path):
    """Test snapshot round-trip through a different Python process.
    
    This test creates a snapshot, saves it to disk, and loads it in a
    separate Python process to ensure cross-process compatibility.
    """
    # Create an oscillator and run it
    n_oscillators = 16
    config = BanksyConfig(gamma=0.1, epsilon=0.01, eta_damp=1e-4, dt=0.01)
    oscillator = BanksyOscillator(n_oscillators, config)
    
    # Run for a while to get interesting state
    for _ in range(100):
        oscillator.step()
    
    # Create a snapshot
    theta = oscillator.phases
    p_theta = oscillator.momenta
    sigma = np.array([s.as_array() for s in oscillator.spins])
    p_sigma = np.zeros_like(sigma)  # Placeholder
    
    snapshot = StateSnapshot(
        theta=theta,
        p_theta=p_theta,
        sigma=sigma,
        p_sigma=p_sigma,
        dt_phase=config.dt,
        dt_spin=config.dt / 8,
    )
    
    # Save to file
    snapshot_file = tmp_path / "snapshot.bin"
    with open(snapshot_file, "wb") as f:
        f.write(snapshot.to_bytes())
    
    # Create a Python script to load the snapshot in a new process
    test_script = f"""
import sys
import numpy as np
from pathlib import Path
from alan_backend.snapshot import StateSnapshot, from_bytes

# Load the snapshot from file
with open("{snapshot_file}", "rb") as f:
    buffer = f.read()

# Deserialize
snapshot = StateSnapshot.from_bytes(buffer)

# Verify dimensions match
assert snapshot.n_oscillators == {n_oscillators}
assert snapshot.theta.shape == ({n_oscillators},)
assert snapshot.p_theta.shape == ({n_oscillators},)
assert snapshot.sigma.shape == ({n_oscillators}, 3)
assert snapshot.p_sigma.shape == ({n_oscillators}, 3)

# Print a success message
print("Cross-process snapshot validation successful")
sys.exit(0)
"""
    
    script_file = tmp_path / "test_script.py"
    with open(script_file, "w") as f:
        f.write(test_script)
    
    # Run the script in a new process
    result = subprocess.run(
        [sys.executable, str(script_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    # Check that it succeeded
    assert result.returncode == 0, f"Cross-process test failed: {result.stderr}"
    assert "Cross-process snapshot validation successful" in result.stdout


@pytest.mark.slow
def test_snapshot_continuity(tmp_path):
    """Test continuity of simulation across snapshot boundary.
    
    This tests that we can save a snapshot, restore it, continue
    simulation, and get results that are consistent with continuous
    simulation.
    """
    # Create an oscillator and run it
    n_oscillators = 16
    config = BanksyConfig(gamma=0.1, epsilon=0.01, eta_damp=1e-4, dt=0.01)
    oscillator1 = BanksyOscillator(n_oscillators, config)
    
    # Set a fixed coupling matrix for reproducibility
    np.random.seed(42)
    coupling = np.random.uniform(0, 0.1, (n_oscillators, n_oscillators))
    coupling = (coupling + coupling.T) / 2  # Symmetrize
    np.fill_diagonal(coupling, 0.0)
    oscillator1.set_coupling(coupling)
    
    # Run for 50 steps
    for _ in range(50):
        oscillator1.step()
    
    # Create a snapshot and save to file
    theta = oscillator1.phases
    p_theta = oscillator1.momenta
    sigma = np.array([s.as_array() for s in oscillator1.spins])
    p_sigma = np.zeros_like(sigma)  # Placeholder
    
    snapshot = StateSnapshot(
        theta=theta,
        p_theta=p_theta,
        sigma=sigma,
        p_sigma=p_sigma,
        dt_phase=config.dt,
        dt_spin=config.dt / 8,
    )
    
    snapshot_file = tmp_path / "continuity_snapshot.bin"
    with open(snapshot_file, "wb") as f:
        f.write(snapshot.to_bytes())
    
    # Continue running the first oscillator for 50 more steps
    for _ in range(50):
        oscillator1.step()
    
    # Now create a new oscillator and initialize from the snapshot
    oscillator2 = BanksyOscillator(n_oscillators, config)
    oscillator2.set_coupling(coupling)
    
    # Load the snapshot
    with open(snapshot_file, "rb") as f:
        buffer = f.read()
    
    restored = StateSnapshot.from_bytes(buffer)
    
    # Initialize the new oscillator from the snapshot
    oscillator2.phases = restored.theta.copy()
    oscillator2.momenta = restored.p_theta.copy()
    
    # Reset the spins
    for i, spin_array in enumerate(restored.sigma):
        oscillator2.spins[i] = SpinVector(
            spin_array[0], spin_array[1], spin_array[2]
        )
    
    # Run the second oscillator for 50 steps
    for _ in range(50):
        oscillator2.step()
    
    # Compare the states - they should be very close
    np.testing.assert_allclose(
        oscillator1.phases, oscillator2.phases, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        oscillator1.momenta, oscillator2.momenta, rtol=1e-6, atol=1e-6
    )
    
    for i in range(n_oscillators):
        np.testing.assert_allclose(
            oscillator1.spins[i].as_array(),
            oscillator2.spins[i].as_array(),
            rtol=1e-6, atol=1e-6,
        )


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
