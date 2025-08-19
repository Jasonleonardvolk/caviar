"""
Systemd Slice Manager for Dickbox
==================================

Manages systemd slices and cgroup resource controls.
"""

import asyncio
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SliceConfig:
    """Configuration for a systemd slice"""
    name: str
    parent: Optional[str] = None
    cpu_weight: Optional[int] = None
    cpu_quota: Optional[int] = None  # Percentage (100 = 1 core)
    memory_max: Optional[str] = None  # e.g., "2G"
    memory_high: Optional[str] = None
    io_weight: Optional[int] = None
    tasks_max: Optional[int] = None


class SystemdSliceManager:
    """
    Manages systemd slices for resource isolation.
    
    Creates and configures slices like:
    - tori.slice (parent)
      - tori-server.slice (critical services)
      - tori-helper.slice (background jobs)
      - tori-build.slice (build tasks)
    """
    
    def __init__(self):
        self.slices: Dict[str, SliceConfig] = {}
        self._init_default_slices()
    
    def _init_default_slices(self):
        """Initialize default slice hierarchy"""
        # Parent slice for all TORI services
        self.slices["tori.slice"] = SliceConfig(
            name="tori.slice",
            cpu_weight=100
        )
        
        # High-priority services
        self.slices["tori-server.slice"] = SliceConfig(
            name="tori-server.slice",
            parent="tori.slice",
            cpu_weight=200,  # Higher priority
            memory_high="4G",
            io_weight=200
        )
        
        # Background/batch jobs
        self.slices["tori-helper.slice"] = SliceConfig(
            name="tori-helper.slice",
            parent="tori.slice",
            cpu_weight=50,   # Lower priority
            cpu_quota=400,   # Max 4 cores
            memory_high="2G",
            memory_max="3G",
            io_weight=50
        )
        
        # Build tasks (lowest priority)
        self.slices["tori-build.slice"] = SliceConfig(
            name="tori-build.slice",
            parent="tori.slice",
            cpu_weight=10,
            cpu_quota=200,   # Max 2 cores
            memory_max="1G",
            tasks_max=50,
            io_weight=10
        )
    
    async def create_slice(self, config: SliceConfig) -> bool:
        """Create or update a systemd slice"""
        # Generate slice unit file
        unit_content = self._generate_slice_unit(config)
        unit_path = Path(f"/etc/systemd/system/{config.name}")
        
        try:
            # Write unit file (requires permissions)
            with open(unit_path, 'w') as f:
                f.write(unit_content)
            
            # Reload systemd
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            
            # Start slice (slices auto-start when needed, but we can ensure it)
            subprocess.run(["systemctl", "start", config.name], check=False)
            
            logger.info(f"Created/updated slice {config.name}")
            self.slices[config.name] = config
            return True
            
        except Exception as e:
            logger.error(f"Failed to create slice {config.name}: {e}")
            return False
    
    def _generate_slice_unit(self, config: SliceConfig) -> str:
        """Generate systemd slice unit file content"""
        lines = [
            "[Unit]",
            f"Description=TORI Slice: {config.name}",
            "Documentation=man:systemd.slice(5)",
            "Before=slices.target"
        ]
        
        if config.parent:
            lines.append(f"Slice={config.parent}")
        
        lines.extend(["", "[Slice]"])
        
        # Resource controls
        if config.cpu_weight is not None:
            lines.append(f"CPUWeight={config.cpu_weight}")
        
        if config.cpu_quota is not None:
            lines.append(f"CPUQuota={config.cpu_quota}%")
        
        if config.memory_max is not None:
            lines.append(f"MemoryMax={config.memory_max}")
        
        if config.memory_high is not None:
            lines.append(f"MemoryHigh={config.memory_high}")
        
        if config.io_weight is not None:
            lines.append(f"IOWeight={config.io_weight}")
        
        if config.tasks_max is not None:
            lines.append(f"TasksMax={config.tasks_max}")
        
        return "\n".join(lines) + "\n"
    
    async def ensure_slices(self) -> Dict[str, bool]:
        """Ensure all configured slices exist"""
        results = {}
        
        # Create in order (parent first)
        for slice_name in ["tori.slice", "tori-server.slice", "tori-helper.slice", "tori-build.slice"]:
            if slice_name in self.slices:
                success = await self.create_slice(self.slices[slice_name])
                results[slice_name] = success
        
        return results
    
    async def get_slice_status(self, slice_name: str) -> Dict[str, Any]:
        """Get status and resource usage of a slice"""
        try:
            # Get basic status
            status_result = subprocess.run(
                ["systemctl", "show", slice_name, "--property=ActiveState,MemoryCurrent,CPUUsageNSec,NTasks"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse properties
            props = {}
            for line in status_result.stdout.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    props[key] = value
            
            # Get cgroup path
            cgroup_result = subprocess.run(
                ["systemctl", "show", slice_name, "--property=ControlGroup"],
                capture_output=True,
                text=True
            )
            
            control_group = ""
            if '=' in cgroup_result.stdout:
                control_group = cgroup_result.stdout.split('=', 1)[1].strip()
            
            return {
                "slice": slice_name,
                "active": props.get("ActiveState") == "active",
                "memory_bytes": int(props.get("MemoryCurrent", 0)),
                "cpu_usage_ns": int(props.get("CPUUsageNSec", 0)),
                "task_count": int(props.get("NTasks", 0)),
                "control_group": control_group
            }
            
        except Exception as e:
            logger.error(f"Failed to get slice status for {slice_name}: {e}")
            return {
                "slice": slice_name,
                "error": str(e)
            }
    
    async def set_service_slice(self, service_name: str, slice_name: str) -> bool:
        """Assign a service to a specific slice"""
        if slice_name not in self.slices:
            logger.error(f"Unknown slice: {slice_name}")
            return False
        
        try:
            # Use systemctl set-property to update slice assignment
            subprocess.run(
                ["systemctl", "set-property", service_name, f"Slice={slice_name}"],
                check=True
            )
            
            logger.info(f"Assigned {service_name} to slice {slice_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign {service_name} to slice {slice_name}: {e}")
            return False
    
    async def update_resource_limits(self, slice_name: str, **limits) -> bool:
        """
        Update resource limits for a slice dynamically.
        
        Args:
            slice_name: Name of the slice
            **limits: Resource limits (cpu_weight=100, memory_max="2G", etc.)
        """
        if slice_name not in self.slices:
            logger.error(f"Unknown slice: {slice_name}")
            return False
        
        properties = []
        
        # Map Python names to systemd properties
        mapping = {
            "cpu_weight": "CPUWeight",
            "cpu_quota": "CPUQuota",
            "memory_max": "MemoryMax",
            "memory_high": "MemoryHigh",
            "io_weight": "IOWeight",
            "tasks_max": "TasksMax"
        }
        
        for key, value in limits.items():
            if key in mapping:
                systemd_prop = mapping[key]
                if key == "cpu_quota" and isinstance(value, int):
                    value = f"{value}%"
                properties.append(f"{systemd_prop}={value}")
        
        if not properties:
            return True
        
        try:
            # Apply properties
            cmd = ["systemctl", "set-property", slice_name] + properties
            subprocess.run(cmd, check=True)
            
            # Update config
            config = self.slices[slice_name]
            for key, value in limits.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            logger.info(f"Updated resource limits for {slice_name}: {limits}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update limits for {slice_name}: {e}")
            return False
    
    async def get_slice_tree(self) -> Dict[str, Any]:
        """Get the slice hierarchy tree"""
        try:
            # Use systemd-cgls to show cgroup tree
            result = subprocess.run(
                ["systemd-cgls", "--no-pager", "/tori.slice"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Try without specifying slice
                result = subprocess.run(
                    ["systemd-cgls", "--no-pager"],
                    capture_output=True,
                    text=True
                )
            
            return {
                "tree": result.stdout,
                "success": result.returncode == 0
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def get_slice_for_workload(self, workload_type: str) -> str:
        """
        Determine appropriate slice for a workload type.
        
        Args:
            workload_type: Type of workload (api, batch, build, etc.)
            
        Returns:
            Slice name
        """
        mapping = {
            "api": "tori-server.slice",
            "web": "tori-server.slice",
            "grpc": "tori-server.slice",
            "batch": "tori-helper.slice",
            "background": "tori-helper.slice",
            "ingestion": "tori-helper.slice",
            "build": "tori-build.slice",
            "compile": "tori-build.slice",
            "test": "tori-build.slice"
        }
        
        return mapping.get(workload_type.lower(), "tori-helper.slice")


class CgroupMonitor:
    """Monitor cgroup resource usage"""
    
    @staticmethod
    async def get_cgroup_stats(cgroup_path: str) -> Dict[str, Any]:
        """Get detailed cgroup statistics"""
        stats = {}
        base_path = Path("/sys/fs/cgroup") / cgroup_path.lstrip('/')
        
        if not base_path.exists():
            return {"error": f"Cgroup path not found: {cgroup_path}"}
        
        # Read various cgroup files
        files_to_read = {
            "memory.current": "memory_current",
            "memory.max": "memory_max",
            "memory.stat": "memory_stat",
            "cpu.stat": "cpu_stat",
            "io.stat": "io_stat",
            "pids.current": "pids_current",
            "pids.max": "pids_max"
        }
        
        for filename, key in files_to_read.items():
            file_path = base_path / filename
            if file_path.exists():
                try:
                    content = file_path.read_text().strip()
                    if filename.endswith('.stat'):
                        # Parse stat files
                        stats[key] = {}
                        for line in content.split('\n'):
                            if ' ' in line:
                                k, v = line.split(' ', 1)
                                stats[key][k] = v
                    else:
                        stats[key] = content
                except Exception as e:
                    stats[key] = f"Error: {e}"
        
        return stats
    
    @staticmethod
    async def monitor_slice_resources(slice_name: str, duration: int = 60) -> List[Dict[str, Any]]:
        """
        Monitor slice resource usage over time.
        
        Args:
            slice_name: Slice to monitor
            duration: Monitoring duration in seconds
            
        Returns:
            List of samples
        """
        samples = []
        interval = 5  # Sample every 5 seconds
        
        for _ in range(duration // interval):
            manager = SystemdSliceManager()
            status = await manager.get_slice_status(slice_name)
            
            status["timestamp"] = asyncio.get_event_loop().time()
            samples.append(status)
            
            await asyncio.sleep(interval)
        
        return samples


# Export
__all__ = ['SystemdSliceManager', 'SliceConfig', 'CgroupMonitor']
