#!/usr/bin/env python
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
AVX-512 Compatibility Check

This script checks if the system supports AVX-512 instructions and sets NumPy
environment variables accordingly. This prevents segmentation faults in environments
like SGX enclaves or VMs that don't support all AVX-512 instructions.

Usage:
    Import and call `check_avx512()` before importing NumPy in your application.
    
Example:
    from alan_backend.scripts.check_avx512 import check_avx512
    check_avx512()
    import numpy as np
"""

import os
import sys
import logging
import platform

logger = logging.getLogger(__name__)


def detect_avx512():
    """Detect if the CPU supports AVX-512 instructions.
    
    Returns:
        bool: True if AVX-512 is supported, False otherwise
    """
    # For Linux, we can check /proc/cpuinfo
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                return "avx512" in cpuinfo.lower()
        except Exception as e:
            logger.warning(f"Could not read /proc/cpuinfo: {e}")
    
    # For Windows, we need to use more complex methods
    # For simplicity and safety, default to assuming it's not available
    # This could be improved with actual CPU feature detection
    
    # If NumPy is already imported, we can use its CPU detection
    try:
        import numpy as np
        if hasattr(np, "__config__") and hasattr(np.__config__, "cpu_features"):
            return "avx512" in np.__config__.cpu_features
    except (ImportError, AttributeError):
        pass
    
    # If we couldn't detect it, assume it's not available for safety
    return False


def check_avx512():
    """Check for AVX-512 support and set NumPy environment variables accordingly.
    
    This should be called before importing NumPy to prevent any segfaults in
    environments without full AVX-512 support.
    """
    try:
        # Only proceed if NumPy isn't already imported
        if "numpy" not in sys.modules:
            has_avx512 = detect_avx512()
            
            if not has_avx512:
                logger.info("AVX-512 not detected, setting NUMPY_DISABLE_AVX512=1")
                os.environ["NUMPY_DISABLE_AVX512"] = "1"
            else:
                logger.info("AVX-512 instructions supported")
        else:
            # NumPy is already imported, warn that it's too late to set the flag
            logger.warning(
                "NumPy is already imported, cannot configure AVX-512 settings. "
                "Ensure check_avx512() is called before importing NumPy."
            )
    except Exception as e:
        # Don't let any errors here crash the application
        logger.error(f"Error during AVX-512 check: {e}")


if __name__ == "__main__":
    # Configure logging when run as a script
    logging.basicConfig(level=logging.INFO)
    
    # Run check and print results
    check_avx512()
    
    # Print the current environment variable status
    avx512_env = os.environ.get("NUMPY_DISABLE_AVX512", "(not set)")
    print(f"NUMPY_DISABLE_AVX512 = {avx512_env}")
