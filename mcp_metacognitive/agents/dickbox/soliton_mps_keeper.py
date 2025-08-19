"""
Soliton MPS Keeper - GPU Warmup Service
=======================================

Keeps GPUs initialized with MPS by running minimal CUDA kernels.
"""

import argparse
import time
import logging
import sys
import os

# Try importing CUDA libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


class SolitonMPSKeeper:
    """Keeps GPU warm with minimal CUDA operations"""
    
    def __init__(self, gpu_index: int, sleep_interval: int = 10):
        self.gpu_index = gpu_index
        self.sleep_interval = sleep_interval
        self.running = True
        
        # Initialize CUDA
        if CUPY_AVAILABLE:
            self._init_cupy()
        elif PYCUDA_AVAILABLE:
            self._init_pycuda()
        else:
            raise RuntimeError("No CUDA library available (install cupy or pycuda)")
    
    def _init_cupy(self):
        """Initialize CuPy for the specified GPU"""
        try:
            cp.cuda.Device(self.gpu_index).use()
            self.device = cp.cuda.Device(self.gpu_index)
            logger.info(f"Initialized CuPy on GPU {self.gpu_index}")
        except Exception as e:
            logger.error(f"Failed to initialize CuPy on GPU {self.gpu_index}: {e}")
            raise
    
    def _init_pycuda(self):
        """Initialize PyCUDA for the specified GPU"""
        try:
            cuda.init()
            self.device = cuda.Device(self.gpu_index)
            self.context = self.device.make_context()
            logger.info(f"Initialized PyCUDA on GPU {self.gpu_index}")
        except Exception as e:
            logger.error(f"Failed to initialize PyCUDA on GPU {self.gpu_index}: {e}")
            raise
    
    def run_nop_kernel(self):
        """Run a minimal CUDA kernel to keep GPU active"""
        if CUPY_AVAILABLE:
            self._run_cupy_kernel()
        else:
            self._run_pycuda_kernel()
    
    def _run_cupy_kernel(self):
        """Run minimal CuPy operation"""
        try:
            # Allocate small array
            a = cp.ones((128, 128), dtype=cp.float32)
            # Perform minimal operation
            b = a * 2.0
            # Force synchronization
            cp.cuda.Stream.null.synchronize()
            # Clean up
            del a, b
        except Exception as e:
            logger.error(f"CuPy kernel error: {e}")
    
    def _run_pycuda_kernel(self):
        """Run minimal PyCUDA kernel"""
        try:
            # Simple kernel that does almost nothing
            mod = SourceModule("""
            __global__ void nop_kernel(float *data)
            {
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                if (idx < 128) {
                    data[idx] = data[idx] * 2.0f;
                }
            }
            """)
            
            nop_kernel = mod.get_function("nop_kernel")
            
            # Allocate small array
            import numpy as np
            a = np.ones(128, dtype=np.float32)
            a_gpu = cuda.mem_alloc(a.nbytes)
            cuda.memcpy_htod(a_gpu, a)
            
            # Run kernel
            nop_kernel(
                a_gpu,
                block=(128, 1, 1),
                grid=(1, 1)
            )
            
            # Synchronize
            self.context.synchronize()
            
            # Clean up
            a_gpu.free()
            
        except Exception as e:
            logger.error(f"PyCUDA kernel error: {e}")
    
    def run(self):
        """Main loop - run NOP kernel periodically"""
        logger.info(f"Starting Soliton MPS keeper for GPU {self.gpu_index}")
        
        while self.running:
            try:
                # Run minimal kernel
                self.run_nop_kernel()
                
                # Log heartbeat every 10 iterations
                if hasattr(self, '_iteration'):
                    self._iteration += 1
                else:
                    self._iteration = 1
                
                if self._iteration % 10 == 0:
                    logger.info(f"GPU {self.gpu_index} heartbeat - iteration {self._iteration}")
                
                # Sleep
                time.sleep(self.sleep_interval)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                self.running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(self.sleep_interval)
    
    def cleanup(self):
        """Clean up CUDA context"""
        if PYCUDA_AVAILABLE and hasattr(self, 'context'):
            self.context.pop()
            logger.info(f"Cleaned up CUDA context for GPU {self.gpu_index}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Soliton MPS GPU Keeper")
    parser.add_argument(
        "--gpu-index",
        type=int,
        required=True,
        help="GPU index to keep warm"
    )
    parser.add_argument(
        "--sleep-interval",
        type=int,
        default=10,
        help="Sleep interval between kernels (seconds)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Create and run keeper
    keeper = SolitonMPSKeeper(
        gpu_index=args.gpu_index,
        sleep_interval=args.sleep_interval
    )
    
    try:
        keeper.run()
    finally:
        keeper.cleanup()


if __name__ == "__main__":
    main()
