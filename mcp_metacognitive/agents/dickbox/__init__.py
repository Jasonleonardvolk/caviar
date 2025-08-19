"""
Dickbox - Containerless Deployment System
========================================

A lightweight, secure capsule-based deployment system for TORI.
"""

# Core agent
from .dickbox import DickboxAgent, DeploymentState, CapsuleInfo

# Configuration
try:
    from .dickbox_config import (
        DickboxConfig, 
        CapsuleManifest, 
        ServiceConfig,
        SliceType,
        ResourceLimits,
        GPUConfig,
        load_capsule_manifest
    )
except ImportError:
    # Fallback if pydantic not available
    DickboxConfig = None
    CapsuleManifest = None
    ServiceConfig = None
    SliceType = None
    ResourceLimits = None
    GPUConfig = None
    load_capsule_manifest = None

# Systemd management
from .systemd_manager import SystemdSliceManager, SliceConfig, CgroupMonitor

# GPU management
from .gpu_manager import MPSManager, GPUScheduler, GPUInfo, GPUMode

# Soliton MPS
try:
    from .soliton_mps import SolitonMPSManager, integrate_soliton_mps
except ImportError:
    SolitonMPSManager = None
    integrate_soliton_mps = None

# Signature verification
try:
    from .signature_verification import verify_capsule_signature, extract_capsule_with_verification
except ImportError:
    verify_capsule_signature = None
    extract_capsule_with_verification = None

# ZMQ key rotation
try:
    from .zmq_key_rotation import ZMQKeyManager, ZMQKeyRotationService
except ImportError:
    ZMQKeyManager = None
    ZMQKeyRotationService = None

# Communication
from .communication import (
    CommunicationFabric,
    ZeroMQBus,
    UnixSocketServer,
    UnixSocketClient,
    MessageBus
)

# Metrics
try:
    from .dickbox_metrics import (
        metrics_exporter,
        create_metrics_app,
        metrics_router,
        DickboxMetricsExporter
    )
except ImportError:
    # Fallback if prometheus_client not available
    metrics_exporter = None
    create_metrics_app = None
    metrics_router = None
    DickboxMetricsExporter = None

# Version
__version__ = "1.0.0"

# Export all
__all__ = [
    # Core
    'DickboxAgent',
    'DeploymentState',
    'CapsuleInfo',
    
    # Config
    'DickboxConfig',
    'CapsuleManifest',
    'ServiceConfig',
    'SliceType',
    'ResourceLimits',
    'GPUConfig',
    'load_capsule_manifest',
    
    # Systemd
    'SystemdSliceManager',
    'SliceConfig',
    'CgroupMonitor',
    
    # GPU
    'MPSManager',
    'GPUScheduler',
    'GPUInfo',
    'GPUMode',
    
    # Soliton MPS
    'SolitonMPSManager',
    'integrate_soliton_mps',
    
    # Security
    'verify_capsule_signature',
    'extract_capsule_with_verification',
    'ZMQKeyManager',
    'ZMQKeyRotationService',
    
    # Communication
    'CommunicationFabric',
    'ZeroMQBus',
    'UnixSocketServer',
    'UnixSocketClient',
    'MessageBus',
    
    # Metrics
    'metrics_exporter',
    'create_metrics_app',
    'metrics_router',
    'DickboxMetricsExporter'
]


def create_dickbox_agent(config=None):
    """
    Factory function to create a fully configured Dickbox agent.
    
    Args:
        config: Dictionary config or DickboxConfig instance
        
    Returns:
        Configured DickboxAgent instance
    """
    from pathlib import Path
    
    # Create agent
    agent = DickboxAgent(config=config)
    
    # Initialize managers
    agent.slice_manager = SystemdSliceManager()
    agent.gpu_manager = MPSManager()
    agent.gpu_scheduler = GPUScheduler(agent.gpu_manager)
    
    # Integrate Soliton MPS keep-alive if available
    if integrate_soliton_mps:
        integrate_soliton_mps(agent.gpu_manager)
    
    # Initialize ZMQ key manager if available
    if ZMQKeyManager:
        zmq_keys_dir = getattr(agent.config, 'zmq_keys_dir', Path('/etc/tori/zmq_keys'))
        agent.zmq_key_manager = ZMQKeyManager(zmq_keys_dir)
    
    # Initialize communication if config available
    if hasattr(agent.config, 'dict'):
        comm_config = agent.config.dict()
    else:
        comm_config = {
            'enable_zeromq': True,
            'zeromq_pub_port': 5555,
            'enable_grpc_unix': True
        }
    
    agent.communication = CommunicationFabric(comm_config)
    
    # Set up ZMQ key rotation service if available
    if ZMQKeyRotationService and hasattr(agent, 'zmq_key_manager'):
        agent.zmq_rotation_service = ZMQKeyRotationService(
            agent.zmq_key_manager,
            agent.communication
        )
    
    # Set up metrics if available
    if metrics_exporter:
        metrics_exporter.set_dickbox_agent(agent)
    
    return agent


# Auto-register with agent registry if available
try:
    from ..core.agent_registry import agent_registry
    
    # Create and register default instance
    default_agent = DickboxAgent()
    agent_registry.register(default_agent)
    
except ImportError:
    # Running standalone
    pass
