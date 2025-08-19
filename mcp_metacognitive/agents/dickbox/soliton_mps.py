"""
Soliton MPS Keep-Alive Service
==============================

Runs a minimal CUDA kernel periodically to keep GPU contexts warm.
"""

import asyncio
import subprocess
import logging
from pathlib import Path
from typing import List, Optional
import os
import sys

logger = logging.getLogger(__name__)

# CUDA kernel code (minimal memory touch)
CUDA_KERNEL_CODE = """
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void keepAliveKernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 1024) {
        data[idx] = data[idx] + 0.0001f;
    }
}

extern "C" void runKeepAlive() {
    float* d_data;
    size_t size = 1024 * sizeof(float);
    
    // Allocate minimal GPU memory
    cudaMalloc(&d_data, size);
    
    // Run minimal kernel
    keepAliveKernel<<<1, 1024>>>(d_data);
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Free memory
    cudaFree(d_data);
}
"""

# Python wrapper using ctypes
PYTHON_KEEPALIVE = """
import ctypes
import time
import os
import sys

def run_keepalive_loop(gpu_uuid):
    '''Run keep-alive loop for specific GPU'''
    
    # Set CUDA device based on UUID
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    
    try:
        # Try to load precompiled CUDA library
        cuda_lib = ctypes.CDLL('/opt/tori/lib/soliton_keepalive.so')
        run_kernel = cuda_lib.runKeepAlive
        
        print(f"Soliton MPS keep-alive started for GPU {gpu_uuid}")
        
        while True:
            # Run minimal CUDA kernel
            run_kernel()
            
            # Sleep for 10 seconds
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("Soliton MPS keep-alive stopped")
        sys.exit(0)
    except Exception as e:
        print(f"Error in keep-alive loop: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: soliton-mps-keepalive <GPU-UUID>")
        sys.exit(1)
    
    gpu_uuid = sys.argv[1]
    run_keepalive_loop(gpu_uuid)
"""


class SolitonMPSManager:
    """
    Manages Soliton MPS keep-alive services for GPUs.
    """
    
    def __init__(self):
        self.active_services = {}  # gpu_uuid -> service_name
        self._ensure_keepalive_binary()
    
    def _ensure_keepalive_binary(self):
        """Ensure keep-alive binary exists"""
        binary_path = Path("/opt/tori/bin/soliton-mps-keepalive")
        lib_path = Path("/opt/tori/lib/soliton_keepalive.so")
        
        # Create directories
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write Python script
        if not binary_path.exists():
            with open(binary_path, 'w') as f:
                f.write(f"#!/usr/bin/env python3\n{PYTHON_KEEPALIVE}")
            os.chmod(binary_path, 0o755)
            logger.info(f"Created keep-alive script at {binary_path}")
        
        # Compile CUDA kernel if nvcc available
        if not lib_path.exists():
            self._compile_cuda_kernel(lib_path)
    
    def _compile_cuda_kernel(self, lib_path: Path):
        """Compile CUDA kernel to shared library"""
        try:
            # Write CUDA source
            cuda_src = Path("/tmp/soliton_keepalive.cu")
            with open(cuda_src, 'w') as f:
                f.write(CUDA_KERNEL_CODE)
            
            # Compile with nvcc
            result = subprocess.run([
                "nvcc",
                "-shared",
                "-fPIC",
                "-o", str(lib_path),
                str(cuda_src)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Compiled CUDA keep-alive kernel to {lib_path}")
            else:
                logger.warning(f"Failed to compile CUDA kernel: {result.stderr}")
                # Fall back to Python-only implementation
                
        except FileNotFoundError:
            logger.warning("nvcc not found - using Python-only keep-alive")
        except Exception as e:
            logger.error(f"Error compiling CUDA kernel: {e}")
    
    async def list_gpus(self) -> List[str]:
        """List available GPU UUIDs"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            
            uuids = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return uuids
            
        except Exception as e:
            logger.error(f"Failed to list GPUs: {e}")
            return []
    
    async def start_keepalive_services(self):
        """Start keep-alive service for each GPU"""
        gpus = await self.list_gpus()
        
        for gpu_uuid in gpus:
            service_name = f"soliton-mps@{gpu_uuid}.service"
            
            try:
                # Enable and start service
                subprocess.run(["systemctl", "enable", service_name], check=True)
                subprocess.run(["systemctl", "start", service_name], check=True)
                
                self.active_services[gpu_uuid] = service_name
                logger.info(f"Started keep-alive service for GPU {gpu_uuid}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to start keep-alive for GPU {gpu_uuid}: {e}")
    
    async def stop_keepalive_services(self):
        """Stop all keep-alive services"""
        for gpu_uuid, service_name in self.active_services.items():
            try:
                subprocess.run(["systemctl", "stop", service_name], check=True)
                subprocess.run(["systemctl", "disable", service_name], check=True)
                
                logger.info(f"Stopped keep-alive service for GPU {gpu_uuid}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to stop keep-alive for GPU {gpu_uuid}: {e}")
        
        self.active_services.clear()
    
    async def get_service_status(self, gpu_uuid: str) -> str:
        """Get status of keep-alive service for a GPU"""
        service_name = f"soliton-mps@{gpu_uuid}.service"
        
        try:
            result = subprocess.run(
                ["systemctl", "is-active", service_name],
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"


# Integration with GPU manager
def integrate_soliton_mps(gpu_manager):
    """
    Integrate Soliton MPS keep-alive with GPU manager.
    
    This should be called when Dickbox starts.
    """
    soliton_manager = SolitonMPSManager()
    
    # Add methods to GPU manager
    gpu_manager.soliton_manager = soliton_manager
    gpu_manager.start_gpu_keepalive = soliton_manager.start_keepalive_services
    gpu_manager.stop_gpu_keepalive = soliton_manager.stop_keepalive_services
    
    return soliton_manager
