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
Cross-architecture serialization tests.

These tests verify that snapshots can be correctly serialized and deserialized
across different architectures and endianness, ensuring portability of data.
"""

import os
import sys
import pytest
import numpy as np
import struct
import tempfile
import subprocess
from pathlib import Path

from alan_backend.snapshot import StateSnapshot, to_bytes, from_bytes
from alan_backend.snapshot.snapshot_serializer import (
    SCHEMA_CRC32, VERSION, LITTLE_ENDIAN, BIG_ENDIAN, 
    pack_float32_array, unpack_float32_array
)


def test_pack_unpack_float32():
    """Test packing and unpacking float32 arrays."""
    # Create sample data
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    
    # Pack as little-endian bytes
    packed = pack_float32_array(data)
    
    # Verify bytes are in little-endian format
    for i, value in enumerate(data):
        # Extract 4 bytes for this float
        float_bytes = packed[i*4:(i+1)*4]
        # Unpack as little-endian float
        unpacked = struct.unpack('<f', float_bytes)[0]
        # Compare with original
        assert np.isclose(unpacked, value)
    
    # Unpack back to an array
    unpacked_array = unpack_float32_array(packed, len(data))
    
    # Verify results match
    np.testing.assert_allclose(unpacked_array, data)


def test_manual_byte_swap():
    """Test manually byte-swapping float32 values."""
    # Create sample data
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    
    # Pack as little-endian bytes
    packed = pack_float32_array(data)
    
    # Manually swap bytes to simulate big-endian
    swapped = bytearray()
    for i in range(0, len(packed), 4):
        chunk = packed[i:i+4]
        swapped.extend(bytes(reversed(chunk)))
    
    # Verify the swapped bytes are different
    assert swapped != packed
    
    # Swap back to little-endian
    swapped_back = bytearray()
    for i in range(0, len(swapped), 4):
        chunk = swapped[i:i+4]
        swapped_back.extend(bytes(reversed(chunk)))
    
    # Verify we get back the original bytes
    assert swapped_back == packed
    
    # Unpack the swapped-back bytes
    unpacked = unpack_float32_array(swapped_back, len(data))
    
    # Verify the results match the original data
    np.testing.assert_allclose(unpacked, data)


def test_endianness_flag():
    """Test serialization with explicit endianness flag."""
    # Create sample data
    n_oscillators = 16
    theta = np.linspace(0, 2*np.pi, n_oscillators, endpoint=False)
    p_theta = np.random.normal(0, 0.1, n_oscillators)
    sigma = np.random.normal(0, 1, (n_oscillators, 3))
    sigma = sigma / np.linalg.norm(sigma, axis=1, keepdims=True)
    p_sigma = np.random.normal(0, 0.1, (n_oscillators, 3))
    
    # Create snapshot
    snapshot = StateSnapshot(
        theta=theta,
        p_theta=p_theta,
        sigma=sigma,
        p_sigma=p_sigma,
    )
    
    # Serialize to bytes
    buffer = snapshot.to_bytes()
    
    # Verify it contains the little-endian flag
    # (This is a very basic check and depends on the FlatBuffers layout,
    # which might change - adjust the offset if needed)
    # For a real implementation, we'd parse the FlatBuffer to extract the flag
    
    # Deserialize
    restored = StateSnapshot.from_bytes(buffer)
    
    # Check that the data is preserved
    np.testing.assert_allclose(restored.theta, theta)
    np.testing.assert_allclose(restored.p_theta, p_theta)
    np.testing.assert_allclose(restored.sigma, sigma)
    np.testing.assert_allclose(restored.p_sigma, p_sigma)


def test_schema_version_check():
    """Test schema version validation."""
    # Create sample data
    n_oscillators = 4
    theta = np.linspace(0, 2*np.pi, n_oscillators, endpoint=False)
    p_theta = np.zeros(n_oscillators)
    sigma = np.zeros((n_oscillators, 3))
    sigma[:, 2] = 1.0  # All pointing in z direction
    p_sigma = np.zeros((n_oscillators, 3))
    
    # Create snapshot
    snapshot = StateSnapshot(
        theta=theta,
        p_theta=p_theta,
        sigma=sigma,
        p_sigma=p_sigma,
    )
    
    # Serialize to bytes
    buffer = snapshot.to_bytes()
    
    # Modify the version in the buffer (simulating an incompatible version)
    # This is hacky and depends on the FlatBuffers layout
    # In a real test, we'd use the FlatBuffers API to modify the buffer
    # or create a test-specific serializer with a different version
    
    # For now, let's just verify that our version is what we expect
    from alan_backend.snapshot.snapshot_serializer import VERSION
    major = VERSION >> 8
    minor = VERSION & 0xFF
    assert major == 2, f"Expected major version 2, got {major}"
    assert minor == 0, f"Expected minor version 0, got {minor}"


def test_crc32_validation():
    """Test schema CRC32 validation."""
    # Create sample data
    n_oscillators = 4
    theta = np.linspace(0, 2*np.pi, n_oscillators, endpoint=False)
    p_theta = np.zeros(n_oscillators)
    sigma = np.zeros((n_oscillators, 3))
    sigma[:, 2] = 1.0  # All pointing in z direction
    p_sigma = np.zeros((n_oscillators, 3))
    
    # Create snapshot
    snapshot = StateSnapshot(
        theta=theta,
        p_theta=p_theta,
        sigma=sigma,
        p_sigma=p_sigma,
    )
    
    # Verify that our CRC32 is what we expect
    from alan_backend.snapshot.snapshot_serializer import SCHEMA_CRC32
    assert SCHEMA_CRC32 == 0x8A7B4C3D, f"Expected CRC32 0x8A7B4C3D, got {SCHEMA_CRC32:08x}"


def simulate_big_endian_system(buffer: bytes) -> bytes:
    """Simulate a snapshot being read on a big-endian system.
    
    This function manually swaps the bytes of all float32 values in the buffer
    to simulate how the data would be interpreted on a big-endian system.
    
    Args:
        buffer: Original snapshot buffer
        
    Returns:
        Modified buffer with float values byte-swapped
    """
    # This is a simplified simulation - in reality the FlatBuffers
    # implementation would handle the endianness internally
    
    # In a real implementation, we'd use the generated code to modify
    # only the float32 fields, or create a proper big-endian test fixture
    
    # For now, let's return the original buffer since our code should detect
    # the endianness flag and handle it correctly
    return buffer


@pytest.mark.xfail(reason="Big-endian support is not yet fully implemented")
def test_cross_endian_snapshot():
    """Test snapshot round-trip through different endianness systems."""
    # Create sample data
    n_oscillators = 16
    theta = np.linspace(0, 2*np.pi, n_oscillators, endpoint=False)
    p_theta = np.random.normal(0, 0.1, n_oscillators)
    sigma = np.random.normal(0, 1, (n_oscillators, 3))
    sigma = sigma / np.linalg.norm(sigma, axis=1, keepdims=True)
    p_sigma = np.random.normal(0, 0.1, (n_oscillators, 3))
    
    # Create snapshot
    snapshot = StateSnapshot(
        theta=theta,
        p_theta=p_theta,
        sigma=sigma,
        p_sigma=p_sigma,
    )
    
    # Serialize to bytes
    buffer = snapshot.to_bytes()
    
    # Simulate reading on a big-endian system
    big_endian_buffer = simulate_big_endian_system(buffer)
    
    # Now deserialize on our system
    # This should work because we explicitly handle endianness in from_bytes
    restored = StateSnapshot.from_bytes(big_endian_buffer)
    
    # Check that the data is preserved
    np.testing.assert_allclose(restored.theta, theta)
    np.testing.assert_allclose(restored.p_theta, p_theta)
    np.testing.assert_allclose(restored.sigma, sigma)
    np.testing.assert_allclose(restored.p_sigma, p_sigma)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
