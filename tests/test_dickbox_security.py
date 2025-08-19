"""
Security Tests for Dickbox
==========================

Tests signature verification and security features.
"""

import pytest
import tempfile
import tarfile
import yaml
import hashlib
from pathlib import Path
import shutil
import sys
import os

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from kha.mcp_metacognitive.agents.dickbox.signature_verification import (
        verify_capsule_signature,
        extract_capsule_with_verification
    )
    from kha.mcp_metacognitive.agents.dickbox.dickbox import DickboxAgent
    DICKBOX_AVAILABLE = True
except ImportError:
    DICKBOX_AVAILABLE = False

try:
    import minisign
    MINISIGN_AVAILABLE = True
except ImportError:
    MINISIGN_AVAILABLE = False


class TestCapsuleSecurity:
    """Test capsule signature verification"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_capsule(self, temp_dir):
        """Create a mock capsule for testing"""
        capsule_dir = temp_dir / "capsule_contents"
        capsule_dir.mkdir()
        
        # Create mock files
        (capsule_dir / "bin").mkdir()
        (capsule_dir / "bin" / "start.sh").write_text("#!/bin/bash\necho 'Starting service'")
        
        # Create manifest
        manifest = {
            "name": "test-service",
            "version": "1.0.0",
            "entrypoint": "bin/start.sh"
        }
        
        with open(capsule_dir / "capsule.yml", 'w') as f:
            yaml.dump(manifest, f)
        
        # Create tarball
        tarball_path = temp_dir / "test-capsule.tar.gz"
        with tarfile.open(tarball_path, 'w:gz') as tar:
            tar.add(capsule_dir, arcname=".")
        
        return {
            "tarball": tarball_path,
            "manifest": manifest,
            "contents_dir": capsule_dir
        }
    
    @pytest.mark.skipif(not DICKBOX_AVAILABLE, reason="Dickbox not available")
    def test_extract_without_signature(self, mock_capsule, temp_dir):
        """Test extracting capsule without signature (should succeed)"""
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()
        
        # Should succeed without signature
        result = pytest.helpers.run_async(
            extract_capsule_with_verification(
                mock_capsule["tarball"],
                extract_dir
            )
        )
        
        assert result is True
        assert (extract_dir / "capsule.yml").exists()
        assert (extract_dir / "bin" / "start.sh").exists()
    
    @pytest.mark.skipif(not DICKBOX_AVAILABLE or not MINISIGN_AVAILABLE, 
                       reason="Dickbox or minisign not available")
    def test_valid_signature(self, mock_capsule, temp_dir):
        """Test capsule with valid signature"""
        # Generate test keypair
        private_key = minisign.PrivateKey.generate()
        public_key = private_key.public_key
        
        # Sign the capsule
        with open(mock_capsule["tarball"], 'rb') as f:
            data = f.read()
        
        signature = private_key.sign(data)
        sig_path = temp_dir / "capsule.sig"
        with open(sig_path, 'wb') as f:
            f.write(signature)
        
        # Update manifest with signature info
        manifest_path = mock_capsule["contents_dir"] / "capsule.yml"
        manifest = mock_capsule["manifest"]
        manifest["signature"] = "capsule.sig"
        manifest["public_key"] = public_key.encode_base64()
        
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest, f)
        
        # Recreate tarball with updated manifest
        with tarfile.open(mock_capsule["tarball"], 'w:gz') as tar:
            tar.add(mock_capsule["contents_dir"], arcname=".")
        
        # Extract and verify
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()
        
        # Copy signature to extract dir (simulating extraction)
        shutil.copy(sig_path, extract_dir / "capsule.sig")
        
        result = pytest.helpers.run_async(
            verify_capsule_signature(
                mock_capsule["tarball"],
                extract_dir / "capsule.sig",
                public_key.encode_base64()
            )
        )
        
        assert result is True
    
    @pytest.mark.skipif(not DICKBOX_AVAILABLE, reason="Dickbox not available")
    def test_tampered_capsule(self, mock_capsule, temp_dir):
        """Test detection of tampered capsule"""
        if not MINISIGN_AVAILABLE:
            pytest.skip("Minisign not available")
        
        # Generate test keypair
        private_key = minisign.PrivateKey.generate()
        public_key = private_key.public_key
        
        # Sign the original capsule
        with open(mock_capsule["tarball"], 'rb') as f:
            original_data = f.read()
        
        signature = private_key.sign(original_data)
        sig_path = temp_dir / "capsule.sig"
        with open(sig_path, 'wb') as f:
            f.write(signature)
        
        # Tamper with the capsule
        tampered_path = temp_dir / "tampered.tar.gz"
        tampered_data = original_data + b"TAMPERED"
        with open(tampered_path, 'wb') as f:
            f.write(tampered_data)
        
        # Verification should fail
        result = pytest.helpers.run_async(
            verify_capsule_signature(
                tampered_path,
                sig_path,
                public_key.encode_base64()
            )
        )
        
        assert result is False
    
    @pytest.mark.skipif(not DICKBOX_AVAILABLE, reason="Dickbox not available")
    def test_wrong_public_key(self, mock_capsule, temp_dir):
        """Test verification with wrong public key"""
        if not MINISIGN_AVAILABLE:
            pytest.skip("Minisign not available")
        
        # Generate two different keypairs
        private_key1 = minisign.PrivateKey.generate()
        public_key1 = private_key1.public_key
        
        private_key2 = minisign.PrivateKey.generate()
        public_key2 = private_key2.public_key
        
        # Sign with key1
        with open(mock_capsule["tarball"], 'rb') as f:
            data = f.read()
        
        signature = private_key1.sign(data)
        sig_path = temp_dir / "capsule.sig"
        with open(sig_path, 'wb') as f:
            f.write(signature)
        
        # Try to verify with key2 (should fail)
        result = pytest.helpers.run_async(
            verify_capsule_signature(
                mock_capsule["tarball"],
                sig_path,
                public_key2.encode_base64()  # Wrong key!
            )
        )
        
        assert result is False
    
    @pytest.mark.skipif(not DICKBOX_AVAILABLE, reason="Dickbox not available")
    def test_missing_signature_file(self, mock_capsule, temp_dir):
        """Test handling of missing signature file"""
        # Create manifest with signature reference
        manifest_path = mock_capsule["contents_dir"] / "capsule.yml"
        manifest = mock_capsule["manifest"]
        manifest["signature"] = "capsule.sig"
        manifest["public_key"] = "dummy_key"
        
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest, f)
        
        # Recreate tarball
        with tarfile.open(mock_capsule["tarball"], 'w:gz') as tar:
            tar.add(mock_capsule["contents_dir"], arcname=".")
        
        # Extract without signature file
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()
        
        # Should fail due to missing signature
        with pytest.raises(Exception, match="Capsule verification failed"):
            pytest.helpers.run_async(
                extract_capsule_with_verification(
                    mock_capsule["tarball"],
                    extract_dir
                )
            )


class TestDickboxSecurity:
    """Test Dickbox security features"""
    
    @pytest.mark.skipif(not DICKBOX_AVAILABLE, reason="Dickbox not available")
    def test_path_traversal_protection(self, temp_dir):
        """Test protection against path traversal attacks"""
        # Create malicious tarball with path traversal
        malicious_dir = temp_dir / "malicious"
        malicious_dir.mkdir()
        
        # Try to create file outside capsule directory
        evil_file = malicious_dir / ".." / ".." / "evil.txt"
        
        # Create tarball
        tarball_path = temp_dir / "malicious.tar.gz"
        with tarfile.open(tarball_path, 'w:gz') as tar:
            # This should be caught and prevented
            info = tarfile.TarInfo(name="../../evil.txt")
            info.size = 4
            tar.addfile(info, fileobj=None)
        
        # Extract should sanitize paths
        extract_dir = temp_dir / "safe_extract"
        extract_dir.mkdir()
        
        with tarfile.open(tarball_path, 'r:gz') as tar:
            # Python's tarfile module should prevent extraction outside target
            tar.extractall(extract_dir)
        
        # Evil file should not exist outside extract_dir
        assert not (temp_dir / "evil.txt").exists()
        assert not (temp_dir.parent / "evil.txt").exists()
    
    @pytest.mark.skipif(not DICKBOX_AVAILABLE, reason="Dickbox not available")
    def test_systemd_template_validation(self):
        """Test systemd service template security"""
        agent = DickboxAgent()
        
        # Get systemd unit name - should sanitize capsule ID
        unit_name = agent._get_systemd_unit("test-capsule-123")
        assert "@test-capsule-123." in unit_name
        
        # Test with potentially malicious capsule ID
        malicious_id = "../../etc/passwd"
        unit_name = agent._get_systemd_unit(malicious_id)
        # Should not contain path traversal
        assert ".." not in unit_name
        assert "/" not in unit_name
    
    @pytest.mark.skipif(not DICKBOX_AVAILABLE, reason="Dickbox not available")
    def test_releases_dir_isolation(self, temp_dir):
        """Test that capsules are isolated to releases directory"""
        config = {
            "releases_dir": temp_dir / "releases"
        }
        
        agent = DickboxAgent(config=config)
        
        # Ensure capsule paths are within releases_dir
        capsule_id = "test-capsule-456"
        if hasattr(agent.config, 'get_capsule_path'):
            capsule_path = agent.config.get_capsule_path(capsule_id)
        else:
            capsule_path = agent.config.releases_dir / capsule_id
        
        # Path should be within releases directory
        assert str(capsule_path).startswith(str(agent.config.releases_dir))
        
        # Test with malicious capsule ID
        malicious_id = "../../../etc"
        if hasattr(agent.config, 'get_capsule_path'):
            capsule_path = agent.config.get_capsule_path(malicious_id)
        else:
            capsule_path = agent.config.releases_dir / malicious_id
        
        # Should still be within releases directory
        resolved_path = capsule_path.resolve()
        releases_path = agent.config.releases_dir.resolve()
        assert str(resolved_path).startswith(str(releases_path))


# Test helpers
class AsyncHelpers:
    @staticmethod
    def run_async(coro):
        """Helper to run async functions in tests"""
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


# Register helpers
pytest.helpers = AsyncHelpers()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
