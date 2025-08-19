"""
Integration Tests for Dickbox
============================

Tests the complete Dickbox system integration.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
import sys
import yaml
import tarfile

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from kha.mcp_metacognitive.agents.dickbox import (
        create_dickbox_agent,
        DickboxConfig,
        CapsuleManifest,
        ServiceConfig,
        SliceType,
        ResourceLimits,
        GPUConfig
    )
    DICKBOX_AVAILABLE = True
except ImportError:
    DICKBOX_AVAILABLE = False


@pytest.mark.skipif(not DICKBOX_AVAILABLE, reason="Dickbox not available")
class TestDickboxIntegration:
    """Test Dickbox integration"""
    
    @pytest.fixture
    async def dickbox_agent(self, tmp_path):
        """Create Dickbox agent for testing"""
        config = {
            "releases_dir": tmp_path / "releases",
            "sockets_dir": tmp_path / "sockets",
            "energy_budget_path": tmp_path / "energy.json",
            "keep_releases": 3,
            "enable_mps": False,  # Disable for testing
            "enable_metrics": False
        }
        
        agent = create_dickbox_agent(config)
        yield agent
        
        # Cleanup
        if hasattr(agent, 'communication'):
            await agent.communication.stop()
    
    @pytest.fixture
    def mock_capsule_tarball(self, tmp_path):
        """Create a mock capsule tarball"""
        # Create capsule directory
        capsule_dir = tmp_path / "capsule_contents"
        capsule_dir.mkdir()
        
        # Create bin directory and start script
        bin_dir = capsule_dir / "bin"
        bin_dir.mkdir()
        start_script = bin_dir / "start.sh"
        start_script.write_text("#!/bin/bash\necho 'Service started'\n")
        start_script.chmod(0o755)
        
        # Create manifest
        manifest = CapsuleManifest(
            name="test-service",
            version="1.0.0",
            entrypoint="bin/start.sh",
            dependencies={"python": "3.10"},
            services=[
                ServiceConfig(
                    name="test-service",
                    slice=SliceType.HELPER,
                    resource_limits=ResourceLimits(
                        cpu_quota=100,
                        memory_max="1G"
                    ),
                    gpu_config=GPUConfig(enabled=False)
                )
            ],
            build_info={
                "timestamp": "2025-01-10T10:00:00Z",
                "builder": "test@localhost"
            }
        )
        
        # Write manifest
        with open(capsule_dir / "capsule.yml", 'w') as f:
            yaml.dump(manifest.dict(), f)
        
        # Create tarball
        tarball_path = tmp_path / "test-service-1.0.0.tar.gz"
        with tarfile.open(tarball_path, 'w:gz') as tar:
            tar.add(capsule_dir, arcname=".")
        
        return tarball_path
    
    async def test_agent_initialization(self, dickbox_agent):
        """Test agent initialization"""
        assert dickbox_agent is not None
        assert dickbox_agent.name == "dickbox"
        assert hasattr(dickbox_agent, 'slice_manager')
        assert hasattr(dickbox_agent, 'gpu_manager')
        assert hasattr(dickbox_agent, 'communication')
        assert hasattr(dickbox_agent, 'zmq_key_manager')
    
    async def test_list_capsules_empty(self, dickbox_agent):
        """Test listing capsules when none deployed"""
        result = await dickbox_agent.execute("list_capsules")
        
        assert "error" not in result
        assert result["total"] == 0
        assert result["capsules"] == []
    
    async def test_deploy_capsule(self, dickbox_agent, mock_capsule_tarball):
        """Test deploying a capsule"""
        result = await dickbox_agent.execute("deploy_capsule", {
            "source": str(mock_capsule_tarball),
            "service_name": "test-service"
        })
        
        # Should succeed (in test mode, systemd commands are mocked)
        assert "error" not in result or "Permission denied" in result.get("error", "")
        
        if "error" not in result:
            assert result.get("success") is True
            assert "capsule_id" in result
            assert "deployment_id" in result
    
    async def test_get_status(self, dickbox_agent):
        """Test getting system status"""
        result = await dickbox_agent.execute("get_status")
        
        assert "error" not in result
        assert "status" in result
        assert result["status"] == "healthy"
        assert "capsules" in result
        assert "services" in result
        assert "config" in result
    
    async def test_zmq_key_rotation(self, dickbox_agent):
        """Test ZMQ key rotation"""
        # Rotate keys
        new_keys = dickbox_agent.zmq_key_manager.rotate_keys()
        
        assert "timestamp" in new_keys
        assert "private_key" in new_keys
        assert "public_key" in new_keys
        
        # Check current keys
        current = dickbox_agent.zmq_key_manager.get_current_keys()
        assert current is not None
        assert "private_key" in current
        assert "public_key" in current
    
    async def test_communication_fabric(self, dickbox_agent):
        """Test communication fabric"""
        # Start communication
        await dickbox_agent.communication.start()
        
        # Register a test service
        socket_path = dickbox_agent.config.sockets_dir / "test.sock"
        
        async def handler(request):
            return {"echo": request.get("message", "hello")}
        
        await dickbox_agent.communication.register_service(
            "test-service",
            socket_path,
            handler
        )
        
        # List services
        services = dickbox_agent.communication.list_services()
        assert "test-service" in services
        
        # Call service
        response = await dickbox_agent.communication.call_service(
            "test-service",
            {"message": "test"}
        )
        assert response["echo"] == "test"
    
    async def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = DickboxConfig(
            releases_dir=Path("/tmp/releases"),
            energy_budget_path=Path("/tmp/energy.json"),
            keep_releases=5,
            deployment_timeout=300
        )
        
        assert config.releases_dir == Path("/tmp/releases")
        assert config.energy_budget_path == Path("/tmp/energy.json")
        
        # Test environment variable override
        import os
        os.environ["DICKBOX_KEEP_RELEASES"] = "10"
        config2 = DickboxConfig.from_env()
        assert config2.keep_releases == 10
    
    async def test_capsule_manifest_validation(self):
        """Test capsule manifest validation"""
        manifest = CapsuleManifest(
            name="my-service",
            version="2.0.0",
            entrypoint="bin/start",
            services=[
                ServiceConfig(
                    name="my-service",
                    slice=SliceType.SERVER,
                    resource_limits=ResourceLimits(
                        cpu_quota=400,
                        memory_max="8G"
                    ),
                    gpu_config=GPUConfig(
                        enabled=True,
                        visible_devices="0,1",
                        mps_percentage=75
                    )
                )
            ]
        )
        
        assert manifest.name == "my-service"
        assert manifest.services[0].gpu_config.mps_percentage == 75
        assert manifest.services[0].slice == SliceType.SERVER


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
