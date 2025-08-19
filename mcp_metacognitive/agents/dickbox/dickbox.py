"""
Dickbox - Containerless Deployment Agent
========================================

Core implementation of the capsule-based deployment system.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import aiohttp
import yaml

# Try importing signature verification
try:
    import minisign
    MINISIGN_AVAILABLE = True
except ImportError:
    MINISIGN_AVAILABLE = False
    
try:
    from sigstore import dsse, verify
    SIGSTORE_AVAILABLE = True
except ImportError:
    SIGSTORE_AVAILABLE = False

# Try to import config with fallback
try:
    from .dickbox_config import DickboxConfig, CapsuleManifest, ServiceConfig, load_capsule_manifest
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Import core components
try:
    from ..core.agent_registry import Agent
    from ..core.psi_archive import psi_archive
    CORE_AVAILABLE = True
except ImportError:
    # Fallback for standalone testing
    from abc import ABC, abstractmethod
    class Agent(ABC):
        def __init__(self, name: str):
            self.name = name
        @abstractmethod
        async def execute(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
            pass
    psi_archive = None
    CORE_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeploymentState(str, Enum):
    """Deployment states for tracking"""
    PENDING = "pending"
    EXTRACTING = "extracting"
    STARTING = "starting"
    HEALTH_CHECKING = "health_checking"
    SWITCHING = "switching"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class CapsuleInfo:
    """Information about a deployed capsule"""
    
    def __init__(self, capsule_id: str, path: Path, manifest: Optional[Dict] = None):
        self.id = capsule_id
        self.path = path
        self.manifest = manifest or {}
        self.deployed_at = None
        self.systemd_unit = None
        self.status = "unknown"
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "path": str(self.path),
            "manifest": self.manifest,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "systemd_unit": self.systemd_unit,
            "status": self.status
        }


class DickboxAgent(Agent):
    """
    Dickbox - Manages containerless deployments using capsules and systemd.
    
    This agent handles:
    - Capsule lifecycle (fetch, extract, deploy, rollback)
    - Systemd service management
    - Resource isolation via cgroups
    - Blue-green deployments
    - GPU sharing configuration
    - Inter-service communication setup
    """
    
    # Metadata for agent discovery
    _metadata = {
        "name": "dickbox",
        "description": "Containerless capsule deployment and orchestration",
        "enabled": True,
        "auto_start": True,
        "endpoints": [
            {"path": "/capsules", "method": "GET", "description": "List deployed capsules"},
            {"path": "/capsules", "method": "POST", "description": "Deploy new capsule"},
            {"path": "/capsules/{id}", "method": "DELETE", "description": "Remove capsule"},
            {"path": "/services", "method": "GET", "description": "List running services"},
            {"path": "/deploy", "method": "POST", "description": "Deploy service"}
        ],
        "version": "1.0.0"
    }
    
    def __init__(self, name: str = "dickbox", config: Optional[Dict[str, Any]] = None):
        super().__init__(name)
        
        # Load configuration
        if CONFIG_AVAILABLE and config is None:
            self.config = DickboxConfig.from_env()
        elif CONFIG_AVAILABLE and isinstance(config, dict):
            self.config = DickboxConfig(**config)
        else:
            # Minimal fallback config
            self.config = type('Config', (), {
                'releases_dir': Path('/opt/tori/releases'),
                'sockets_dir': Path('/var/run/tori'),
                'keep_releases': 5,
                'parallel_deployments': True,
                'health_check_retries': 10,
                'health_check_interval': 3.0,
                'deployment_timeout': 300,
                'rollback_on_failure': True,
                'enable_mps': True,
                'enable_metrics': True,
                'systemd_unit_template': 'tori@.service'
            })()
        
        # State tracking
        self.deployed_capsules: Dict[str, CapsuleInfo] = {}
        self.active_services: Dict[str, str] = {}  # service_name -> capsule_id
        self.deployment_lock = asyncio.Lock()
        
        # HTTP session for artifact downloads
        self._http_session = None
        
        # Initialize directories
        self._ensure_directories()
        
        # Log initialization
        if psi_archive:
            psi_archive.log_event("dickbox_initialized", {
                "releases_dir": str(self.config.releases_dir),
                "config": self._get_config_summary()
            })
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging"""
        if hasattr(self.config, 'dict'):
            return {
                "releases_dir": str(self.config.releases_dir),
                "keep_releases": self.config.keep_releases,
                "enable_mps": self.config.enable_mps,
                "parallel_deployments": self.config.parallel_deployments
            }
        return {"releases_dir": str(self.config.releases_dir)}
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        dirs = [self.config.releases_dir, self.config.sockets_dir]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    async def execute(self, command: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute Dickbox commands.
        
        Commands:
        - list_capsules: List all deployed capsules
        - deploy_capsule: Deploy a new capsule
        - remove_capsule: Remove a capsule
        - list_services: List running services
        - deploy_service: Full service deployment (fetch, extract, start)
        - rollback_service: Rollback to previous version
        - get_status: Get system status
        """
        
        params = params or {}
        
        try:
            if command == "list_capsules":
                return await self._list_capsules()
                
            elif command == "deploy_capsule":
                return await self._deploy_capsule(params)
                
            elif command == "remove_capsule":
                return await self._remove_capsule(params.get("capsule_id"))
                
            elif command == "list_services":
                return await self._list_services()
                
            elif command == "deploy_service":
                return await self._deploy_service(params)
                
            elif command == "rollback_service":
                return await self._rollback_service(params.get("service_name"))
                
            elif command == "get_status":
                return await self._get_status()
                
            elif command == "cleanup_old_releases":
                return await self._cleanup_old_releases()
                
            else:
                return {
                    "error": f"Unknown command: {command}",
                    "available_commands": [
                        "list_capsules", "deploy_capsule", "remove_capsule",
                        "list_services", "deploy_service", "rollback_service",
                        "get_status", "cleanup_old_releases"
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error executing command {command}: {e}")
            return {"error": str(e), "command": command}
    
    async def _list_capsules(self) -> Dict[str, Any]:
        """List all deployed capsules"""
        capsules = []
        
        # Scan releases directory
        if self.config.releases_dir.exists():
            for capsule_dir in self.config.releases_dir.iterdir():
                if capsule_dir.is_dir():
                    capsule_id = capsule_dir.name
                    
                    # Load manifest if exists
                    manifest = None
                    manifest_path = capsule_dir / "capsule.yml"
                    if manifest_path.exists():
                        try:
                            with open(manifest_path) as f:
                                manifest = yaml.safe_load(f)
                        except Exception as e:
                            logger.error(f"Failed to load manifest for {capsule_id}: {e}")
                    
                    # Get systemd status
                    unit_name = self._get_systemd_unit(capsule_id)
                    status = await self._get_systemd_status(unit_name)
                    
                    capsules.append({
                        "id": capsule_id,
                        "path": str(capsule_dir),
                        "manifest": manifest,
                        "systemd_unit": unit_name,
                        "status": status,
                        "size_mb": self._get_directory_size(capsule_dir) / (1024 * 1024)
                    })
        
        return {
            "capsules": capsules,
            "total": len(capsules),
            "releases_dir": str(self.config.releases_dir)
        }
    
    async def _deploy_capsule(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy a capsule from tarball or URL"""
        source = params.get("source")  # Path or URL to capsule
        service_name = params.get("service_name")
        
        if not source:
            return {"error": "Missing required parameter: source"}
        
        async with self.deployment_lock:
            deployment_id = f"deploy_{datetime.utcnow().timestamp()}"
            state = DeploymentState.PENDING
            
            try:
                # Log deployment start
                if psi_archive:
                    psi_archive.log_event("deployment_started", {
                        "deployment_id": deployment_id,
                        "source": source,
                        "service_name": service_name
                    })
                
                # Fetch capsule if URL
                if source.startswith(("http://", "https://")):
                    state = DeploymentState.EXTRACTING
                    local_path = await self._fetch_capsule(source)
                else:
                    local_path = Path(source)
                
                if not local_path.exists():
                    raise FileNotFoundError(f"Capsule not found: {local_path}")
                
                # Extract capsule
                capsule_id = await self._extract_capsule(local_path)
                capsule_path = self.config.releases_dir / capsule_id
                
                # Load manifest
                manifest = load_capsule_manifest(capsule_path) if CONFIG_AVAILABLE else {}
                
                # Create systemd service
                state = DeploymentState.STARTING
                unit_name = await self._create_systemd_service(capsule_id, manifest)
                
                # Start service
                await self._start_service(unit_name)
                
                # Health check
                state = DeploymentState.HEALTH_CHECKING
                healthy = await self._health_check_service(capsule_id, manifest)
                
                if not healthy:
                    raise Exception("Service failed health check")
                
                # Update tracking
                capsule_info = CapsuleInfo(capsule_id, capsule_path, manifest)
                capsule_info.deployed_at = datetime.utcnow()
                capsule_info.systemd_unit = unit_name
                capsule_info.status = "running"
                
                self.deployed_capsules[capsule_id] = capsule_info
                if service_name:
                    self.active_services[service_name] = capsule_id
                
                state = DeploymentState.COMPLETED
                
                # Log success
                if psi_archive:
                    psi_archive.log_event("deployment_completed", {
                        "deployment_id": deployment_id,
                        "capsule_id": capsule_id,
                        "service_name": service_name,
                        "duration": (datetime.utcnow().timestamp() - float(deployment_id.split('_')[1]))
                    })
                
                return {
                    "success": True,
                    "capsule_id": capsule_id,
                    "systemd_unit": unit_name,
                    "deployment_id": deployment_id,
                    "state": state.value
                }
                
            except Exception as e:
                state = DeploymentState.FAILED
                logger.error(f"Deployment failed: {e}")
                
                # Rollback if configured
                if self.config.rollback_on_failure and service_name in self.active_services:
                    await self._rollback_service(service_name)
                    state = DeploymentState.ROLLED_BACK
                
                return {
                    "success": False,
                    "error": str(e),
                    "deployment_id": deployment_id,
                    "state": state.value
                }
    
    async def _fetch_capsule(self, url: str) -> Path:
        """Download capsule from URL"""
        if not self._http_session:
            self._http_session = aiohttp.ClientSession()
        
        # Download to temp file
        temp_path = Path(tempfile.mktemp(suffix=".tar.gz"))
        
        async with self._http_session.get(url) as response:
            response.raise_for_status()
            
            with open(temp_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
        
        return temp_path
    
    async def _extract_capsule(self, tarball_path: Path) -> str:
        """Extract capsule and return its ID"""
        # Calculate content hash
        hasher = hashlib.sha256()
        with open(tarball_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        capsule_id = hasher.hexdigest()[:12]
        target_dir = self.config.releases_dir / capsule_id
        
        # Skip if already extracted
        if target_dir.exists():
            logger.info(f"Capsule {capsule_id} already extracted")
            return capsule_id
        
        # Extract tarball
        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(target_dir)
        
        logger.info(f"Extracted capsule {capsule_id} to {target_dir}")
        return capsule_id
    
    def _get_systemd_unit(self, capsule_id: str) -> str:
        """Get systemd unit name for capsule"""
        template = self.config.systemd_unit_template
        return template.replace('@.', f'@{capsule_id}.')
    
    async def _create_systemd_service(self, capsule_id: str, manifest: Any) -> str:
        """Create systemd service for capsule"""
        unit_name = self._get_systemd_unit(capsule_id)
        
        # Check if template exists
        template_path = Path(f"/etc/systemd/system/{self.config.systemd_unit_template}")
        if not template_path.exists():
            # Create basic template
            await self._create_systemd_template()
        
        # Reload systemd to pick up changes
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        
        return unit_name
    
    async def _create_systemd_template(self):
        """Create systemd service template"""
        template = """[Unit]
Description=TORI Service %i
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/tori/releases/%i
ExecStart=/opt/tori/releases/%i/bin/start.sh
Restart=on-failure
RestartSec=5
Slice=tori.slice

# Resource limits
CPUWeight=100
MemoryHigh=2G
TasksMax=512

# Environment
Environment="CAPSULE_ID=%i"
Environment="PYTHONPATH=/opt/tori/releases/%i/venv/lib/python3.10/site-packages"
EnvironmentFile=-/opt/tori/releases/%i/config/env

# Security
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/log/tori /var/run/tori

[Install]
WantedBy=multi-user.target
"""
        
        template_path = Path(f"/etc/systemd/system/{self.config.systemd_unit_template}")
        
        # Write template (requires sudo in production)
        try:
            with open(template_path, 'w') as f:
                f.write(template)
            logger.info(f"Created systemd template at {template_path}")
        except PermissionError:
            logger.error("Permission denied creating systemd template - need sudo")
            raise
    
    async def _start_service(self, unit_name: str):
        """Start systemd service"""
        try:
            subprocess.run(["systemctl", "start", unit_name], check=True)
            logger.info(f"Started service {unit_name}")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to start service: {e}")
    
    async def _stop_service(self, unit_name: str):
        """Stop systemd service"""
        try:
            subprocess.run(["systemctl", "stop", unit_name], check=True)
            logger.info(f"Stopped service {unit_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop service: {e}")
    
    async def _get_systemd_status(self, unit_name: str) -> str:
        """Get systemd service status"""
        try:
            result = subprocess.run(
                ["systemctl", "is-active", unit_name],
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"
    
    async def _health_check_service(self, capsule_id: str, manifest: Any) -> bool:
        """Perform health check on service"""
        # Simple implementation - check if process is running
        unit_name = self._get_systemd_unit(capsule_id)
        
        for i in range(self.config.health_check_retries):
            status = await self._get_systemd_status(unit_name)
            if status == "active":
                return True
            
            await asyncio.sleep(self.config.health_check_interval)
        
        return False
    
    async def _deploy_service(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Full service deployment with blue-green switch"""
        service_name = params.get("service_name")
        source = params.get("source")
        
        if not service_name or not source:
            return {"error": "Missing required parameters: service_name, source"}
        
        # Check if service already running
        old_capsule_id = self.active_services.get(service_name)
        
        # Deploy new version
        deploy_result = await self._deploy_capsule(params)
        
        if not deploy_result.get("success"):
            return deploy_result
        
        new_capsule_id = deploy_result["capsule_id"]
        
        # Blue-green switch if old version exists
        if old_capsule_id and self.config.parallel_deployments:
            # Both versions running - switch traffic
            # In production, this would update HAProxy or similar
            logger.info(f"Switching {service_name} from {old_capsule_id} to {new_capsule_id}")
            
            # Update active service
            self.active_services[service_name] = new_capsule_id
            
            # Stop old version after grace period
            await asyncio.sleep(10)  # Grace period for draining
            old_unit = self._get_systemd_unit(old_capsule_id)
            await self._stop_service(old_unit)
        
        return {
            "success": True,
            "service_name": service_name,
            "capsule_id": new_capsule_id,
            "previous_capsule_id": old_capsule_id
        }
    
    async def _rollback_service(self, service_name: str) -> Dict[str, Any]:
        """Rollback service to previous version"""
        current_capsule_id = self.active_services.get(service_name)
        if not current_capsule_id:
            return {"error": f"Service {service_name} not found"}
        
        # Find previous version
        # In production, we'd track deployment history
        capsules = await self._list_capsules()
        service_capsules = [
            c for c in capsules["capsules"]
            if c.get("manifest", {}).get("name") == service_name
        ]
        
        if len(service_capsules) < 2:
            return {"error": "No previous version available for rollback"}
        
        # Sort by deployment time and get previous
        service_capsules.sort(key=lambda x: x.get("deployed_at", ""), reverse=True)
        previous = service_capsules[1]
        
        # Start previous version
        prev_unit = previous["systemd_unit"]
        await self._start_service(prev_unit)
        
        # Wait for health
        healthy = await self._health_check_service(previous["id"], previous.get("manifest"))
        if not healthy:
            return {"error": "Previous version failed health check"}
        
        # Switch active
        self.active_services[service_name] = previous["id"]
        
        # Stop current
        current_unit = self._get_systemd_unit(current_capsule_id)
        await self._stop_service(current_unit)
        
        if psi_archive:
            psi_archive.log_event("service_rolled_back", {
                "service_name": service_name,
                "from_capsule": current_capsule_id,
                "to_capsule": previous["id"]
            })
        
        return {
            "success": True,
            "service_name": service_name,
            "rolled_back_to": previous["id"],
            "rolled_back_from": current_capsule_id
        }
    
    async def _list_services(self) -> Dict[str, Any]:
        """List all running services"""
        services = []
        
        for service_name, capsule_id in self.active_services.items():
            capsule_info = self.deployed_capsules.get(capsule_id)
            unit_name = self._get_systemd_unit(capsule_id)
            status = await self._get_systemd_status(unit_name)
            
            services.append({
                "name": service_name,
                "capsule_id": capsule_id,
                "systemd_unit": unit_name,
                "status": status,
                "deployed_at": capsule_info.deployed_at.isoformat() if capsule_info and capsule_info.deployed_at else None
            })
        
        return {
            "services": services,
            "total": len(services)
        }
    
    async def _remove_capsule(self, capsule_id: str) -> Dict[str, Any]:
        """Remove a capsule"""
        if not capsule_id:
            return {"error": "Missing capsule_id"}
        
        capsule_path = self.config.releases_dir / capsule_id
        if not capsule_path.exists():
            return {"error": f"Capsule {capsule_id} not found"}
        
        # Check if running
        unit_name = self._get_systemd_unit(capsule_id)
        status = await self._get_systemd_status(unit_name)
        
        if status == "active":
            return {"error": "Cannot remove running capsule - stop service first"}
        
        # Remove directory
        shutil.rmtree(capsule_path)
        
        # Remove from tracking
        if capsule_id in self.deployed_capsules:
            del self.deployed_capsules[capsule_id]
        
        return {
            "success": True,
            "removed": capsule_id
        }
    
    async def _cleanup_old_releases(self) -> Dict[str, Any]:
        """Clean up old releases beyond keep_releases limit"""
        capsules = await self._list_capsules()
        all_capsules = capsules["capsules"]
        
        # Sort by modification time
        all_capsules.sort(key=lambda x: os.path.getmtime(x["path"]), reverse=True)
        
        # Find removable capsules
        removable = []
        active_capsules = set(self.active_services.values())
        
        for i, capsule in enumerate(all_capsules):
            if i >= self.config.keep_releases and capsule["id"] not in active_capsules:
                if capsule["status"] != "active":
                    removable.append(capsule["id"])
        
        # Remove old capsules
        removed = []
        for capsule_id in removable:
            result = await self._remove_capsule(capsule_id)
            if result.get("success"):
                removed.append(capsule_id)
        
        return {
            "removed": removed,
            "count": len(removed),
            "space_freed_mb": sum(
                self._get_directory_size(self.config.releases_dir / cid) / (1024 * 1024)
                for cid in removed
                if (self.config.releases_dir / cid).exists()
            )
        }
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes"""
        total = 0
        try:
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception:
            pass
        return total
    
    async def _get_status(self) -> Dict[str, Any]:
        """Get Dickbox system status"""
        capsules = await self._list_capsules()
        services = await self._list_services()
        
        # Check GPU/MPS status
        mps_status = "unknown"
        if self.config.enable_mps:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    mps_status = "available"
            except Exception:
                mps_status = "not available"
        
        return {
            "status": "healthy",
            "config": self._get_config_summary(),
            "capsules": {
                "total": capsules["total"],
                "active": len([c for c in capsules["capsules"] if c["status"] == "active"])
            },
            "services": {
                "total": services["total"],
                "running": len([s for s in services["services"] if s["status"] == "active"])
            },
            "gpu": {
                "mps_enabled": self.config.enable_mps,
                "mps_status": mps_status
            },
            "disk_usage_mb": sum(
                c["size_mb"] for c in capsules["capsules"]
            )
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Dickbox shutting down...")
        
        # Close HTTP session
        if self._http_session:
            await self._http_session.close()
        
        # Log final state
        if psi_archive:
            psi_archive.log_event("dickbox_shutdown", {
                "active_services": len(self.active_services),
                "deployed_capsules": len(self.deployed_capsules)
            })


# Export
__all__ = ['DickboxAgent', 'DeploymentState', 'CapsuleInfo']
