"""
GPU/MPS Manager for Dickbox
===========================

Manages NVIDIA GPU sharing via Multi-Process Service (MPS).
"""

import asyncio
import subprocess
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GPUMode(str, Enum):
    """GPU sharing modes"""
    EXCLUSIVE = "exclusive"
    MPS = "mps"
    DEFAULT = "default"


@dataclass
class GPUInfo:
    """Information about a GPU"""
    index: int
    name: str
    uuid: str
    memory_total: int  # MB
    memory_used: int   # MB
    memory_free: int   # MB
    utilization: int   # Percentage
    temperature: int   # Celsius
    power_draw: float  # Watts
    processes: List[Dict[str, Any]]


class MPSManager:
    """
    Manages NVIDIA Multi-Process Service for GPU sharing.
    
    Features:
    - Start/stop MPS control daemon
    - Configure MPS resource partitioning
    - Monitor GPU usage per process
    - Handle GPU assignment for services
    """
    
    def __init__(self, pipe_dir: Path = Path("/tmp/nvidia-mps")):
        self.pipe_dir = pipe_dir
        self.mps_active = False
        self._check_nvidia_tools()
    
    def _check_nvidia_tools(self):
        """Check if NVIDIA tools are available"""
        try:
            subprocess.run(["nvidia-smi", "--version"], 
                         capture_output=True, check=True)
            self.nvidia_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.nvidia_available = False
            logger.warning("NVIDIA tools not available - GPU features disabled")
    
    async def start_mps(self) -> Dict[str, Any]:
        """Start NVIDIA MPS control daemon"""
        if not self.nvidia_available:
            return {"error": "NVIDIA tools not available"}
        
        # Check if already running
        if await self.is_mps_running():
            return {"status": "already_running"}
        
        try:
            # Create pipe directory
            self.pipe_dir.mkdir(parents=True, exist_ok=True)
            
            # Set environment
            env = os.environ.copy()
            env["CUDA_MPS_PIPE_DIRECTORY"] = str(self.pipe_dir)
            env["CUDA_MPS_LOG_DIRECTORY"] = str(self.pipe_dir / "log")
            
            # Start control daemon
            subprocess.run(
                ["nvidia-cuda-mps-control", "-d"],
                env=env,
                check=True
            )
            
            self.mps_active = True
            logger.info(f"Started MPS control daemon with pipe dir: {self.pipe_dir}")
            
            return {
                "status": "started",
                "pipe_dir": str(self.pipe_dir)
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start MPS: {e}")
            return {
                "error": f"Failed to start MPS: {e}",
                "status": "failed"
            }
    
    async def stop_mps(self) -> Dict[str, Any]:
        """Stop NVIDIA MPS control daemon"""
        if not self.nvidia_available:
            return {"error": "NVIDIA tools not available"}
        
        try:
            # Send quit command to MPS control
            env = os.environ.copy()
            env["CUDA_MPS_PIPE_DIRECTORY"] = str(self.pipe_dir)
            
            process = subprocess.Popen(
                ["nvidia-cuda-mps-control"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            stdout, stderr = process.communicate(input="quit\n")
            
            self.mps_active = False
            logger.info("Stopped MPS control daemon")
            
            return {
                "status": "stopped",
                "output": stdout
            }
            
        except Exception as e:
            logger.error(f"Failed to stop MPS: {e}")
            return {
                "error": f"Failed to stop MPS: {e}",
                "status": "failed"
            }
    
    async def is_mps_running(self) -> bool:
        """Check if MPS is running"""
        if not self.nvidia_available:
            return False
        
        try:
            # Check for MPS server process
            result = subprocess.run(
                ["pgrep", "-f", "nvidia-cuda-mps-server"],
                capture_output=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def set_active_thread_percentage(self, client_id: str, percentage: int) -> Dict[str, Any]:
        """
        Set active thread percentage for an MPS client.
        This limits the GPU SM usage for a specific process.
        
        Args:
            client_id: Process ID or identifier
            percentage: Percentage of GPU SMs (1-100)
        """
        if not self.nvidia_available:
            return {"error": "NVIDIA tools not available"}
        
        if not 1 <= percentage <= 100:
            return {"error": "Percentage must be between 1 and 100"}
        
        try:
            env = os.environ.copy()
            env["CUDA_MPS_PIPE_DIRECTORY"] = str(self.pipe_dir)
            
            # Send command to MPS control
            process = subprocess.Popen(
                ["nvidia-cuda-mps-control"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            command = f"set_active_thread_percentage {client_id} {percentage}\n"
            stdout, stderr = process.communicate(input=command)
            
            if process.returncode == 0:
                logger.info(f"Set GPU limit for {client_id} to {percentage}%")
                return {
                    "status": "success",
                    "client_id": client_id,
                    "percentage": percentage
                }
            else:
                return {
                    "error": f"Failed to set limit: {stderr}",
                    "status": "failed"
                }
                
        except Exception as e:
            logger.error(f"Failed to set thread percentage: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    async def get_gpu_info(self) -> List[GPUInfo]:
        """Get information about all GPUs"""
        if not self.nvidia_available:
            return []
        
        gpus = []
        
        try:
            # Query GPU info using nvidia-smi
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse GPU info
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                parts = line.split(', ')
                if len(parts) >= 9:
                    gpu = GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        uuid=parts[2],
                        memory_total=int(parts[3]),
                        memory_used=int(parts[4]),
                        memory_free=int(parts[5]),
                        utilization=int(parts[6]),
                        temperature=int(parts[7]),
                        power_draw=float(parts[8]) if parts[8] != 'N/A' else 0.0,
                        processes=[]
                    )
                    
                    # Get processes for this GPU
                    gpu.processes = await self._get_gpu_processes(gpu.index)
                    gpus.append(gpu)
            
            return gpus
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to query GPU info: {e}")
            return []
    
    async def _get_gpu_processes(self, gpu_index: int) -> List[Dict[str, Any]]:
        """Get processes running on a specific GPU"""
        processes = []
        
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,name,used_memory",
                    f"--id={gpu_index}",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                parts = line.split(', ')
                if len(parts) >= 3:
                    processes.append({
                        "pid": int(parts[0]),
                        "name": parts[1],
                        "memory_mb": int(parts[2])
                    })
            
        except Exception as e:
            logger.debug(f"Failed to get GPU processes: {e}")
        
        return processes
    
    async def assign_gpu_to_service(self, service_name: str, gpu_indices: List[int]) -> Dict[str, Any]:
        """
        Assign specific GPUs to a service via CUDA_VISIBLE_DEVICES.
        
        Args:
            service_name: Name of the service
            gpu_indices: List of GPU indices to assign
        """
        if not self.nvidia_available:
            return {"error": "NVIDIA tools not available"}
        
        # Validate GPU indices
        gpus = await self.get_gpu_info()
        valid_indices = {gpu.index for gpu in gpus}
        
        for idx in gpu_indices:
            if idx not in valid_indices:
                return {"error": f"Invalid GPU index: {idx}"}
        
        # Create CUDA_VISIBLE_DEVICES string
        cuda_devices = ",".join(str(idx) for idx in gpu_indices)
        
        return {
            "status": "success",
            "service": service_name,
            "gpu_indices": gpu_indices,
            "cuda_visible_devices": cuda_devices,
            "environment": {
                "CUDA_VISIBLE_DEVICES": cuda_devices,
                "CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_dir) if self.mps_active else None
            }
        }
    
    async def get_gpu_allocation_recommendation(self, services: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Recommend GPU allocation for services based on their requirements.
        
        Args:
            services: List of services with GPU requirements
                     [{"name": "service1", "gpu_memory_mb": 2000, "priority": "high"}, ...]
        
        Returns:
            Mapping of service names to recommended GPU indices
        """
        gpus = await self.get_gpu_info()
        if not gpus:
            return {}
        
        allocations = {}
        gpu_available_memory = {gpu.index: gpu.memory_free for gpu in gpus}
        
        # Sort services by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_services = sorted(
            services,
            key=lambda s: (priority_order.get(s.get("priority", "medium"), 1), -s.get("gpu_memory_mb", 0))
        )
        
        # Simple first-fit allocation
        for service in sorted_services:
            name = service["name"]
            required_memory = service.get("gpu_memory_mb", 1000)
            
            # Find GPU with enough free memory
            for gpu_idx, free_memory in gpu_available_memory.items():
                if free_memory >= required_memory:
                    allocations[name] = [gpu_idx]
                    gpu_available_memory[gpu_idx] -= required_memory
                    break
            else:
                # No single GPU has enough memory, try to find least loaded
                if gpu_available_memory:
                    best_gpu = max(gpu_available_memory.items(), key=lambda x: x[1])[0]
                    allocations[name] = [best_gpu]
                    gpu_available_memory[best_gpu] = 0
        
        return allocations
    
    async def monitor_gpu_health(self) -> Dict[str, Any]:
        """Monitor GPU health metrics"""
        if not self.nvidia_available:
            return {"error": "NVIDIA tools not available"}
        
        gpus = await self.get_gpu_info()
        
        health_status = {
            "healthy": True,
            "gpus": [],
            "warnings": [],
            "errors": []
        }
        
        for gpu in gpus:
            gpu_health = {
                "index": gpu.index,
                "name": gpu.name,
                "status": "healthy",
                "metrics": {
                    "temperature": gpu.temperature,
                    "power_draw": gpu.power_draw,
                    "memory_utilization": (gpu.memory_used / gpu.memory_total) * 100,
                    "gpu_utilization": gpu.utilization
                }
            }
            
            # Check temperature
            if gpu.temperature > 85:
                gpu_health["status"] = "warning"
                health_status["warnings"].append(f"GPU {gpu.index} temperature high: {gpu.temperature}°C")
            elif gpu.temperature > 90:
                gpu_health["status"] = "critical"
                health_status["errors"].append(f"GPU {gpu.index} temperature critical: {gpu.temperature}°C")
                health_status["healthy"] = False
            
            # Check memory pressure
            mem_util = (gpu.memory_used / gpu.memory_total) * 100
            if mem_util > 90:
                health_status["warnings"].append(f"GPU {gpu.index} memory pressure: {mem_util:.1f}%")
            
            health_status["gpus"].append(gpu_health)
        
        return health_status

    async def start_soliton_mps_services(self) -> Dict[str, Any]:
        """
        Start soliton-mps services for all GPUs to keep them warm.
        
        Returns:
            Status of service starts
        """
        gpus = await self.get_gpu_info()
        results = {}
        
        for gpu in gpus:
            service_name = f"soliton-mps@{gpu.uuid}.service"
            
            try:
                # Start the service
                subprocess.run(
                    ["systemctl", "start", service_name],
                    check=True
                )
                
                results[gpu.uuid] = {
                    "status": "started",
                    "service": service_name,
                    "gpu_name": gpu.name,
                    "gpu_index": gpu.index
                }
                logger.info(f"Started {service_name} for GPU {gpu.index} ({gpu.name})")
                
            except subprocess.CalledProcessError as e:
                results[gpu.uuid] = {
                    "status": "failed",
                    "error": str(e),
                    "gpu_name": gpu.name,
                    "gpu_index": gpu.index
                }
                logger.error(f"Failed to start {service_name}: {e}")
        
        return {
            "services_started": len([r for r in results.values() if r["status"] == "started"]),
            "total_gpus": len(gpus),
            "results": results
        }
    
    async def stop_soliton_mps_services(self) -> Dict[str, Any]:
        """
        Stop all soliton-mps services.
        
        Returns:
            Status of service stops
        """
        gpus = await self.get_gpu_info()
        results = {}
        
        for gpu in gpus:
            service_name = f"soliton-mps@{gpu.uuid}.service"
            
            try:
                # Stop the service
                subprocess.run(
                    ["systemctl", "stop", service_name],
                    check=True
                )
                
                results[gpu.uuid] = {"status": "stopped", "gpu_index": gpu.index}
                logger.info(f"Stopped {service_name}")
                
            except subprocess.CalledProcessError as e:
                results[gpu.uuid] = {"status": "error", "error": str(e), "gpu_index": gpu.index}
                logger.error(f"Failed to stop {service_name}: {e}")
        
        return results


class GPUScheduler:
    """
    Schedules GPU resources for services based on policies.
    """
    
    def __init__(self, mps_manager: MPSManager):
        self.mps_manager = mps_manager
        self.allocations = {}  # service -> gpu_indices
        self.policies = {}
    
    async def schedule_service(self, service_name: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Schedule GPU resources for a service.
        
        Args:
            service_name: Name of the service
            requirements: GPU requirements (memory, compute, exclusive, etc.)
        """
        # Get current GPU state
        gpus = await self.mps_manager.get_gpu_info()
        if not gpus:
            return {"error": "No GPUs available"}
        
        mode = requirements.get("mode", "mps")
        required_memory = requirements.get("memory_mb", 1000)
        required_compute = requirements.get("compute_percentage", 100)
        prefer_gpu = requirements.get("prefer_gpu", None)
        
        # Find suitable GPU
        selected_gpu = None
        
        if prefer_gpu is not None and 0 <= prefer_gpu < len(gpus):
            # Check if preferred GPU is suitable
            gpu = gpus[prefer_gpu]
            if gpu.memory_free >= required_memory:
                selected_gpu = gpu
        
        if selected_gpu is None:
            # Find GPU with most free memory
            suitable_gpus = [g for g in gpus if g.memory_free >= required_memory]
            if suitable_gpus:
                selected_gpu = max(suitable_gpus, key=lambda g: g.memory_free)
            else:
                # No GPU has enough free memory
                selected_gpu = max(gpus, key=lambda g: g.memory_free)
        
        # Assign GPU
        assignment = await self.mps_manager.assign_gpu_to_service(
            service_name,
            [selected_gpu.index]
        )
        
        # Set MPS limits if needed
        if mode == "mps" and required_compute < 100:
            # We'll use PID later when service starts
            self.policies[service_name] = {
                "compute_percentage": required_compute
            }
        
        # Track allocation
        self.allocations[service_name] = [selected_gpu.index]
        
        return {
            "status": "scheduled",
            "service": service_name,
            "gpu": selected_gpu.index,
            "gpu_name": selected_gpu.name,
            "mode": mode,
            "environment": assignment["environment"]
        }
    
    async def release_service_gpus(self, service_name: str) -> Dict[str, Any]:
        """Release GPU allocation for a service"""
        if service_name in self.allocations:
            gpu_indices = self.allocations.pop(service_name)
            
            if service_name in self.policies:
                del self.policies[service_name]
            
            return {
                "status": "released",
                "service": service_name,
                "gpu_indices": gpu_indices
            }
        
        return {
            "status": "not_found",
            "service": service_name
        }
    
    async def apply_mps_policy(self, service_name: str, pid: int) -> Dict[str, Any]:
        """Apply MPS resource limits for a running service"""
        if service_name not in self.policies:
            return {"status": "no_policy"}
        
        policy = self.policies[service_name]
        compute_pct = policy.get("compute_percentage", 100)
        
        result = await self.mps_manager.set_active_thread_percentage(
            str(pid),
            compute_pct
        )
        
        return result


# Export
__all__ = ['MPSManager', 'GPUScheduler', 'GPUInfo', 'GPUMode']
