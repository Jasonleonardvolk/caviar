"""
Dickbox Configuration Schema
============================

Type-safe configuration for the containerless deployment system.
"""

from pydantic import BaseModel, Field, validator, DirectoryPath
from typing import Dict, Optional, List, Any
from pathlib import Path
import os
from enum import Enum


class SliceType(str, Enum):
    """Types of systemd slices for workload classification"""
    SERVER = "tori-server.slice"
    HELPER = "tori-helper.slice"
    BUILD = "tori-build.slice"


class ResourceLimits(BaseModel):
    """Resource limits for a service or slice"""
    cpu_quota: Optional[int] = Field(
        default=None,
        description="CPU quota as percentage (e.g., 50 = 50% of one core)",
        ge=0,
        le=10000  # Can be > 100 for multi-core
    )
    cpu_weight: Optional[int] = Field(
        default=None,
        description="CPU weight for proportional sharing (default 100)",
        ge=1,
        le=10000
    )
    memory_max: Optional[str] = Field(
        default=None,
        description="Hard memory limit (e.g., '2G', '512M')",
        pattern=r'^\d+[KMG]?$'
    )
    memory_high: Optional[str] = Field(
        default=None,
        description="Soft memory limit for throttling",
        pattern=r'^\d+[KMG]?$'
    )
    tasks_max: Optional[int] = Field(
        default=None,
        description="Maximum number of tasks/threads",
        ge=1
    )
    io_weight: Optional[int] = Field(
        default=None,
        description="I/O weight (10-1000)",
        ge=10,
        le=1000
    )


class GPUConfig(BaseModel):
    """GPU configuration for services"""
    enabled: bool = Field(
        default=False,
        description="Whether service uses GPU"
    )
    visible_devices: Optional[str] = Field(
        default=None,
        description="CUDA_VISIBLE_DEVICES setting (e.g., '0', '0,1')"
    )
    mps_percentage: Optional[int] = Field(
        default=None,
        description="MPS resource percentage (if using MPS partitioning)",
        ge=1,
        le=100
    )


class ServiceConfig(BaseModel):
    """Configuration for a deployed service"""
    name: str = Field(
        description="Service name"
    )
    slice: SliceType = Field(
        default=SliceType.HELPER,
        description="Systemd slice assignment"
    )
    resource_limits: ResourceLimits = Field(
        default_factory=ResourceLimits,
        description="Resource constraints"
    )
    gpu_config: GPUConfig = Field(
        default_factory=GPUConfig,
        description="GPU settings"
    )
    health_check_url: Optional[str] = Field(
        default=None,
        description="Health check endpoint"
    )
    startup_timeout: int = Field(
        default=30,
        description="Seconds to wait for service startup",
        ge=1,
        le=300
    )
    environment: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional environment variables"
    )


class CapsuleManifest(BaseModel):
    """Manifest file schema (capsule.yml)"""
    name: str
    version: str
    entrypoint: str = Field(
        description="Entry command relative to capsule root"
    )
    dependencies: Dict[str, str] = Field(
        default_factory=dict,
        description="Dependency versions (python, rust, etc.)"
    )
    services: List[ServiceConfig] = Field(
        default_factory=list,
        description="Service configurations"
    )
    build_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Build metadata (git sha, timestamp, etc.)"
    )


class DickboxConfig(BaseModel):
    """
    Main configuration for Dickbox deployment system.
    
    Supports environment variable overrides with DICKBOX_ prefix.
    """
    
    # Paths
    releases_dir: Path = Field(
        default=Path("/opt/tori/releases"),
        description="Base directory for capsule releases",
        env="DICKBOX_RELEASES_DIR"
    )
    
    sockets_dir: Path = Field(
        default=Path("/var/run/tori"),
        description="Directory for Unix domain sockets",
        env="DICKBOX_SOCKETS_DIR"
    )
    
    artifacts_url: Optional[str] = Field(
        default=None,
        description="URL for artifact repository",
        env="DICKBOX_ARTIFACTS_URL"
    )
    

    
    # Systemd settings
    systemd_unit_template: str = Field(
        default="tori@.service",
        description="Systemd template unit name",
        env="DICKBOX_SYSTEMD_TEMPLATE"
    )
    
    enable_overlayfs: bool = Field(
        default=False,
        description="Use OverlayFS for immutable capsules",
        env="DICKBOX_ENABLE_OVERLAYFS"
    )
    
    # Deployment settings
    keep_releases: int = Field(
        default=5,
        description="Number of old releases to keep",
        env="DICKBOX_KEEP_RELEASES",
        ge=1,
        le=20
    )
    
    parallel_deployments: bool = Field(
        default=True,
        description="Allow multiple versions running simultaneously",
        env="DICKBOX_PARALLEL_DEPLOYMENTS"
    )
    
    health_check_retries: int = Field(
        default=10,
        description="Health check retry attempts",
        ge=1,
        le=30
    )
    
    health_check_interval: float = Field(
        default=3.0,
        description="Seconds between health checks",
        ge=0.5,
        le=30.0
    )
    
    # Resource defaults
    default_resource_limits: ResourceLimits = Field(
        default_factory=lambda: ResourceLimits(
            cpu_weight=100,
            memory_high="1G",
            tasks_max=512
        ),
        description="Default limits for services without explicit config"
    )
    
    # Slice configurations
    slice_configs: Dict[str, ResourceLimits] = Field(
        default_factory=lambda: {
            "tori.slice": ResourceLimits(),  # No limits on parent
            "tori-server.slice": ResourceLimits(
                cpu_weight=200,  # Higher priority
                memory_high="4G"
            ),
            "tori-helper.slice": ResourceLimits(
                cpu_weight=50,   # Lower priority
                cpu_quota=400,   # Max 4 cores
                memory_high="2G"
            ),
            "tori-build.slice": ResourceLimits(
                cpu_weight=10,   # Lowest priority
                cpu_quota=200,   # Max 2 cores
                memory_max="1G"
            )
        },
        description="Resource limits per systemd slice"
    )
    
    # GPU settings
    enable_mps: bool = Field(
        default=True,
        description="Use NVIDIA MPS for GPU sharing",
        env="DICKBOX_ENABLE_MPS"
    )
    
    mps_pipe_dir: Path = Field(
        default=Path("/tmp/nvidia-mps"),
        description="NVIDIA MPS pipe directory",
        env="DICKBOX_MPS_PIPE_DIR"
    )
    
    # Communication settings
    enable_zeromq: bool = Field(
        default=True,
        description="Use ZeroMQ for pub/sub",
        env="DICKBOX_ENABLE_ZEROMQ"
    )
    
    zeromq_pub_port: int = Field(
        default=5555,
        description="ZeroMQ publisher port",
        ge=1024,
        le=65535
    )
    
    enable_grpc_unix: bool = Field(
        default=True,
        description="Use Unix sockets for gRPC",
        env="DICKBOX_ENABLE_GRPC_UNIX"
    )
    
    # Observability
    enable_metrics: bool = Field(
        default=True,
        description="Export Prometheus metrics",
        env="DICKBOX_ENABLE_METRICS"
    )
    
    metrics_port: int = Field(
        default=9091,
        description="Prometheus metrics port",
        env="DICKBOX_METRICS_PORT",
        ge=1024,
        le=65535
    )
    
    enable_tracing: bool = Field(
        default=False,
        description="Enable OpenTelemetry tracing",
        env="DICKBOX_ENABLE_TRACING"
    )
    
    # Security
    enable_seccomp: bool = Field(
        default=False,
        description="Apply seccomp profiles to services",
        env="DICKBOX_ENABLE_SECCOMP"
    )
    
    enable_apparmor: bool = Field(
        default=False,
        description="Apply AppArmor profiles",
        env="DICKBOX_ENABLE_APPARMOR"
    )
    
    run_as_user: Optional[str] = Field(
        default=None,
        description="Default user for services",
        env="DICKBOX_RUN_AS_USER"
    )
    
    # Energy budget settings
    energy_budget_path: Path = Field(
        default=Path("/var/tmp/tori_energy.json"),
        description="Path to energy budget state file",
        env="DICKBOX_ENERGY_BUDGET_PATH"
    )
    
    energy_budget_sync_interval: int = Field(
        default=60,
        description="Seconds between energy budget syncs to disk",
        env="DICKBOX_ENERGY_SYNC_INTERVAL",
        ge=10,
        le=3600
    )
    
    # Circuit breaker settings (for deployments)
    deployment_timeout: int = Field(
        default=300,
        description="Total deployment timeout in seconds",
        ge=60,
        le=3600
    )
    
    rollback_on_failure: bool = Field(
        default=True,
        description="Automatically rollback failed deployments",
        env="DICKBOX_AUTO_ROLLBACK"
    )
    
    class Config:
        """Pydantic config"""
        env_prefix = "DICKBOX_"
        case_sensitive = False
        
    @validator('releases_dir', 'sockets_dir', 'mps_pipe_dir')
    def ensure_absolute_path(cls, v):
        """Ensure paths are absolute"""
        if not v.is_absolute():
            v = Path.cwd() / v
        return v
    
    @validator('releases_dir', 'sockets_dir')
    def ensure_directories_exist(cls, v):
        """Create directories if they don't exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @classmethod
    def from_env(cls) -> 'DickboxConfig':
        """Create config from environment variables"""
        return cls()
    
    def get_capsule_path(self, capsule_id: str) -> Path:
        """Get full path for a capsule"""
        return self.releases_dir / capsule_id
    
    def get_socket_path(self, service_name: str, version: Optional[str] = None) -> Path:
        """Get Unix socket path for a service"""
        if version:
            return self.sockets_dir / f"{service_name}_{version}.sock"
        return self.sockets_dir / f"{service_name}.sock"
    
    def get_systemd_instance(self, capsule_id: str) -> str:
        """Get systemd instance name for a capsule"""
        base_name = self.systemd_unit_template.replace('@.service', '')
        return f"{base_name}@{capsule_id}.service"


def load_capsule_manifest(capsule_path: Path) -> CapsuleManifest:
    """Load and validate a capsule manifest"""
    import yaml
    
    manifest_path = capsule_path / "capsule.yml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")
    
    with open(manifest_path) as f:
        data = yaml.safe_load(f)
    
    return CapsuleManifest(**data)


# Example usage
if __name__ == "__main__":
    # Load config from environment
    config = DickboxConfig.from_env()
    print("Dickbox configuration:")
    print(config.json(indent=2))
    
    # Example capsule manifest
    manifest = CapsuleManifest(
        name="tori-ingest",
        version="1.4.3",
        entrypoint="bin/ingest_bus",
        dependencies={
            "python": "3.10",
            "rust": "1.65"
        },
        services=[
            ServiceConfig(
                name="tori-ingest",
                slice=SliceType.SERVER,
                resource_limits=ResourceLimits(
                    cpu_quota=200,  # 2 cores max
                    memory_max="4G"
                ),
                gpu_config=GPUConfig(
                    enabled=True,
                    visible_devices="0"
                )
            )
        ]
    )
    print("\nExample manifest:")
    print(manifest.json(indent=2))
