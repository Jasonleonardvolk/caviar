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
Pytest configuration for ALAN core tests.

This module contains fixtures and setup code for the ALAN core test suite,
including configuration for deterministic testing across multiple workers.
"""

import os
import sys
import pytest
import random
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base seed for deterministic tests
BASE_SEED = 1337


def pytest_configure(config):
    """Configure pytest.
    
    This function is called before tests are collected and executed.
    """
    # Make tests deterministic by setting seed
    seed = BASE_SEED
    
    # Check if we're running in a pytest-xdist worker
    # This prevents all workers from using the same seed, which would result in
    # identical test sequences across workers, reducing coverage
    worker_input = getattr(config, 'workerinput', None)
    if worker_input is not None:
        # We're in a pytest-xdist worker
        worker_id = worker_input['workerid']
        # Generate a unique seed based on the worker ID
        # XOR with base seed to maintain some predictability
        worker_seed = seed ^ hash(worker_id) % (2**32)
        logger.info(f"Worker {worker_id} using seed {worker_seed}")
        seed = worker_seed
    else:
        logger.info(f"Using base seed {seed}")
    
    # Set deterministic seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Register a custom marker for deterministic tests
    config.addinivalue_line(
        "markers", "deterministic: mark test as deterministic with fixed seed"
    )
    
    # Register a custom marker for tests that require hardware
    config.addinivalue_line(
        "markers", "hardware: mark test as requiring hardware connection"
    )
    
    # Register a custom marker for slow tests
    config.addinivalue_line(
        "markers", "slow: mark test as slow (use -m 'not slow' to skip)"
    )


@pytest.fixture
def fixed_seed():
    """Fixture to provide a fixed seed for deterministic tests."""
    return BASE_SEED


@pytest.fixture
def test_data_dir():
    """Fixture to provide path to test data directory."""
    # Get the directory of this file
    conftest_dir = Path(__file__).parent
    
    # Test data is in a 'data' subdirectory
    data_dir = conftest_dir / 'data'
    
    # Create the directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    return data_dir


@pytest.fixture
def temp_snapshot_file(tmpdir):
    """Fixture to provide a temporary file path for snapshot tests."""
    return Path(tmpdir) / "test_snapshot.bin"


@pytest.fixture
def deterministic_config(pytestconfig, request):
    """Fixture to enable deterministic configuration based on test marker.
    
    If the test is marked with @pytest.mark.deterministic, this will configure
    NumPy, PyTorch (if available), and TensorFlow (if available) for deterministic
    operation.
    """
    # Check if the test is marked as deterministic
    is_deterministic = request.node.get_closest_marker("deterministic") is not None
    
    if is_deterministic:
        # Save the current state
        np_random_state = np.random.get_state()
        
        # Set seeds
        np.random.seed(BASE_SEED)
        
        # Configure PyTorch if available
        try:
            import torch
            torch.manual_seed(BASE_SEED)
            torch.cuda.manual_seed_all(BASE_SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("PyTorch configured for deterministic operation")
        except ImportError:
            pass
        
        # Configure TensorFlow if available
        try:
            import tensorflow as tf
            tf.random.set_seed(BASE_SEED)
            logger.info("TensorFlow configured for deterministic operation")
        except ImportError:
            pass
        
        yield {
            "deterministic": True,
            "seed": BASE_SEED,
        }
        
        # Restore the state
        np.random.set_state(np_random_state)
    else:
        yield {
            "deterministic": False,
        }


@pytest.fixture
def mock_hardware_backend():
    """Fixture to provide a mock hardware backend for testing.
    
    This allows tests to run without actual hardware present, while still
    exercising the code paths that would interact with hardware.
    """
    class MockHardwareBackend:
        def __init__(self):
            self.connected = True
            self.values = {}
            logger.info("Using mock hardware backend")
        
        def read_phase(self, channel=0):
            return np.random.uniform(0, 2*np.pi)
        
        def write_bias(self, value, channel=0):
            self.values[channel] = value
            return True
        
        def close(self):
            self.connected = False
    
    return MockHardwareBackend()
