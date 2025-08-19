"""
Unit Tests for Dickbox Security Features
=========================================

Tests signature verification and key rotation functionality.
"""

import unittest
import asyncio
import tempfile
import shutil
from pathlib import Path
import json
import subprocess
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from deployer import SignatureVerifier, SecureDeployer
from zmq_key_rotation import ZMQKeyManager, ZMQKeyRotationService
from communication import ZeroMQBus


class TestSignatureVerification(unittest.TestCase):
    """Test capsule signature verification"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_minisign_verifier_creation(self):
        """Test creating minisign verifier"""
        verifier = SignatureVerifier(verification_method="minisign")
        self.assertEqual(verifier.method, "minisign")
    
    def test_sigstore_verifier_creation(self):
        """Test creating sigstore verifier"""
        verifier = SignatureVerifier(verification_method="sigstore")
        self.assertEqual(verifier.method, "sigstore")
    
    async def test_verify_signature_missing_capsule(self):
        """Test verification with missing capsule file"""
        verifier = SignatureVerifier()
        
        capsule_path = self.test_path / "missing.tar.gz"
        
        with self.assertRaises(FileNotFoundError):
            await verifier.verify_signature(capsule_path)
    
    async def test_verify_signature_missing_sig(self):
        """Test verification with missing signature file"""
        verifier = SignatureVerifier()
        
        # Create capsule file
        capsule_path = self.test_path / "test.tar.gz"
        capsule_path.write_bytes(b"test capsule data")
        
        with self.assertRaises(FileNotFoundError):
            await verifier.verify_signature(capsule_path)
    
    @patch('subprocess.run')
    async def test_verify_valid_signature(self, mock_run):
        """Test successful signature verification"""
        # Mock successful minisign verification
        mock_run.return_value = Mock(returncode=0, stdout=b"", stderr=b"")
        
        verifier = SignatureVerifier()
        
        # Create test files
        capsule_path = self.test_path / "test.tar.gz"
        sig_path = self.test_path / "test.tar.gz.sig"
        public_key = self.test_path / "test.pub"
        
        capsule_path.write_bytes(b"test capsule data")
        sig_path.write_bytes(b"test signature")
        public_key.write_text("test public key")
        
        # Verify
        result = await verifier._verify_minisign(
            capsule_path, sig_path, str(public_key)
        )
        
        self.assertTrue(result)
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    async def test_verify_invalid_signature(self, mock_run):
        """Test failed signature verification"""
        # Mock failed minisign verification
        mock_run.return_value = Mock(
            returncode=1, 
            stdout=b"", 
            stderr=b"Signature verification failed"
        )
        
        verifier = SignatureVerifier()
        
        # Create test files
        capsule_path = self.test_path / "test.tar.gz"
        sig_path = self.test_path / "test.tar.gz.sig"
        public_key = self.test_path / "test.pub"
        
        capsule_path.write_bytes(b"tampered capsule data")
        sig_path.write_bytes(b"test signature")
        public_key.write_text("test public key")
        
        # Verify should raise
        with self.assertRaises(ValueError) as context:
            await verifier._verify_minisign(
                capsule_path, sig_path, str(public_key)
            )
        
        self.assertIn("Signature verification failed", str(context.exception))
    
    async def test_secure_deployer_with_signature(self):
        """Test secure deployer with signature verification"""
        verifier = Mock(spec=SignatureVerifier)
        verifier.verify_signature = AsyncMock(return_value=True)
        
        deployer = SecureDeployer(verifier)
        
        # Create test capsule
        capsule_path = self.test_path / "test.tar.gz"
        target_dir = self.test_path / "target"
        target_dir.mkdir()
        
        # Create simple tarball
        subprocess.run([
            "tar", "-czf", str(capsule_path),
            "-C", str(self.test_path),
            "."
        ], capture_output=True)
        
        # Create manifest with signature info
        manifest = {
            "signature": {
                "public_key": "/path/to/key.pub"
            }
        }
        
        # Extract with verification
        capsule_id = await deployer.extract_capsule_with_verification(
            capsule_path, target_dir, manifest
        )
        
        # Verify signature was checked
        verifier.verify_signature.assert_called_once()
        self.assertIsNotNone(capsule_id)
    
    async def test_secure_deployer_without_signature(self):
        """Test secure deployer without signature info warns"""
        verifier = Mock(spec=SignatureVerifier)
        deployer = SecureDeployer(verifier)
        
        # Create test capsule
        capsule_path = self.test_path / "test.tar.gz"
        target_dir = self.test_path / "target"
        target_dir.mkdir()
        
        # Create simple tarball
        subprocess.run([
            "tar", "-czf", str(capsule_path),
            "-C", str(self.test_path),
            "."
        ], capture_output=True)
        
        # Manifest without signature
        manifest = {}
        
        # Extract without verification
        with self.assertLogs(level='WARNING') as logs:
            capsule_id = await deployer.extract_capsule_with_verification(
                capsule_path, target_dir, manifest
            )
        
        # Should warn about missing signature
        self.assertTrue(any("No signature info" in log for log in logs.output))
        verifier.verify_signature.assert_not_called()


class TestZMQKeyRotation(unittest.TestCase):
    """Test ZeroMQ key rotation"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.keys_dir = Path(self.test_dir) / "keys"
        self.keys_dir.mkdir()
        
    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_key_manager_creation(self):
        """Test creating key manager"""
        manager = ZMQKeyManager(self.keys_dir)
        self.assertEqual(manager.keys_dir, self.keys_dir)
        self.assertTrue(self.keys_dir.exists())
    
    def test_generate_keypair(self):
        """Test ED25519 keypair generation"""
        manager = ZMQKeyManager(self.keys_dir)
        
        private_key, public_key = manager.generate_ed25519_keypair()
        
        self.assertIsInstance(private_key, bytes)
        self.assertIsInstance(public_key, bytes)
        self.assertGreater(len(private_key), 0)
        self.assertGreater(len(public_key), 0)
    
    def test_rotate_keys(self):
        """Test key rotation"""
        manager = ZMQKeyManager(self.keys_dir)
        
        # Rotate keys
        metadata = manager.rotate_keys()
        
        # Check metadata
        self.assertIn("timestamp", metadata)
        self.assertIn("private_key", metadata)
        self.assertIn("public_key", metadata)
        self.assertEqual(metadata["algorithm"], "ed25519")
        
        # Check files created
        private_path = Path(metadata["private_key"])
        public_path = Path(metadata["public_key"])
        
        self.assertTrue(private_path.exists())
        self.assertTrue(public_path.exists())
        
        # Check symlinks
        current_private = self.keys_dir / "current_private.key"
        current_public = self.keys_dir / "current_public.key"
        
        self.assertTrue(current_private.exists())
        self.assertTrue(current_public.exists())
        self.assertTrue(current_private.is_symlink())
        self.assertTrue(current_public.is_symlink())
    
    def test_get_current_keys(self):
        """Test getting current keys"""
        manager = ZMQKeyManager(self.keys_dir)
        
        # No keys initially
        self.assertIsNone(manager.get_current_keys())
        
        # Rotate keys
        manager.rotate_keys()
        
        # Now should have current keys
        current = manager.get_current_keys()
        self.assertIsNotNone(current)
        self.assertIn("private_key", current)
        self.assertIn("public_key", current)
    
    def test_cleanup_old_keys(self):
        """Test cleaning up old keys"""
        manager = ZMQKeyManager(self.keys_dir)
        
        # Create old key files
        old_key = self.keys_dir / "20200101_120000_private.key"
        old_key.write_bytes(b"old key")
        
        # Create recent key
        recent_key = self.keys_dir / "20991231_120000_private.key"
        recent_key.write_bytes(b"recent key")
        
        # Cleanup with 30 day retention
        manager.cleanup_old_keys(keep_days=30)
        
        # Old key should be removed
        self.assertFalse(old_key.exists())
        # Recent key should remain
        self.assertTrue(recent_key.exists())
    
    async def test_key_rotation_service(self):
        """Test key rotation service"""
        manager = ZMQKeyManager(self.keys_dir)
        
        # Mock communication fabric
        mock_fabric = Mock()
        mock_fabric.publish_event = AsyncMock()
        
        service = ZMQKeyRotationService(manager, mock_fabric)
        
        # Rotate and broadcast
        new_keys = await service.rotate_and_broadcast()
        
        # Check rotation happened
        self.assertIn("timestamp", new_keys)
        
        # Check broadcast was called
        mock_fabric.publish_event.assert_called_once_with(
            "zmq.key_update",
            {
                "action": "rotated",
                "timestamp": new_keys["timestamp"],
                "public_key_path": new_keys["public_key"],
                "created_at": new_keys["created_at"]
            }
        )


class TestZeroMQEncryption(unittest.TestCase):
    """Test ZeroMQ encrypted communication"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.keys_dir = Path(self.test_dir) / "keys"
        self.keys_dir.mkdir()
        
    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir)
    
    async def test_encrypted_bus_creation(self):
        """Test creating encrypted ZeroMQ bus"""
        bus = ZeroMQBus(
            pub_endpoint="tcp://127.0.0.1:15555",
            sub_endpoint="tcp://127.0.0.1:15555",
            enable_encryption=True,
            keys_dir=self.keys_dir
        )
        
        self.assertTrue(bus.enable_encryption)
        self.assertEqual(bus.keys_dir, self.keys_dir)
    
    @patch('zmq.auth.AsyncioAuthenticator')
    async def test_encryption_setup(self, mock_auth_class):
        """Test encryption setup"""
        # Create keys first
        manager = ZMQKeyManager(self.keys_dir)
        manager.rotate_keys()
        
        # Mock authenticator
        mock_auth = Mock()
        mock_auth_class.return_value = mock_auth
        
        bus = ZeroMQBus(
            enable_encryption=True,
            keys_dir=self.keys_dir
        )
        
        await bus._setup_encryption()
        
        # Check authenticator was started
        mock_auth.start.assert_called_once()
        mock_auth.configure_curve.assert_called_once()
    
    def test_load_server_keys(self):
        """Test loading server encryption keys"""
        # Create keys
        manager = ZMQKeyManager(self.keys_dir)
        manager.rotate_keys()
        
        bus = ZeroMQBus(
            enable_encryption=True,
            keys_dir=self.keys_dir
        )
        
        # Mock socket
        mock_socket = Mock()
        
        # Load keys
        bus._load_server_keys(mock_socket)
        
        # Check keys were set
        self.assertIsNotNone(mock_socket.curve_secretkey)
        self.assertIsNotNone(mock_socket.curve_publickey)
    
    async def test_key_reload_detection(self):
        """Test key reload detection"""
        # Create initial keys
        manager = ZMQKeyManager(self.keys_dir)
        manager.rotate_keys()
        
        bus = ZeroMQBus(
            enable_encryption=True,
            keys_dir=self.keys_dir
        )
        
        # Mock reload method
        bus._reload_keys = AsyncMock()
        
        # Set initial mtime
        current_private = self.keys_dir / "current_private.key"
        bus._last_key_mtime = current_private.stat().st_mtime - 1
        
        # Run one iteration of key reload loop
        bus._running = True
        reload_task = asyncio.create_task(bus._key_reload_loop())
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop the loop
        bus._running = False
        reload_task.cancel()
        try:
            await reload_task
        except asyncio.CancelledError:
            pass
        
        # Should have detected change
        bus._reload_keys.assert_called()


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
    def tearDown(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir)
    
    async def test_secure_deployment_workflow(self):
        """Test complete secure deployment workflow"""
        # Create test capsule directory
        capsule_dir = self.test_path / "capsule_src"
        capsule_dir.mkdir()
        
        # Create test files
        (capsule_dir / "app.py").write_text("print('Hello TORI')")
        (capsule_dir / "requirements.txt").write_text("flask==2.0.0")
        
        # Create manifest
        manifest = {
            "name": "test-service",
            "version": "1.0.0",
            "entrypoint": "python app.py",
            "signature": {
                "public_key": "/path/to/key.pub"
            }
        }
        
        with open(capsule_dir / "capsule.yml", 'w') as f:
            json.dump(manifest, f)
        
        # Create tarball
        capsule_path = self.test_path / "test.tar.gz"
        subprocess.run([
            "tar", "-czf", str(capsule_path),
            "-C", str(capsule_dir),
            "."
        ], check=True)
        
        # Mock verifier
        verifier = Mock(spec=SignatureVerifier)
        verifier.verify_signature = AsyncMock(return_value=True)
        
        # Deploy
        deployer = SecureDeployer(verifier)
        target_dir = self.test_path / "releases"
        target_dir.mkdir()
        
        capsule_id = await deployer.extract_capsule_with_verification(
            capsule_path, target_dir, manifest
        )
        
        # Check deployment
        self.assertIsNotNone(capsule_id)
        deployed_dir = target_dir / capsule_id
        self.assertTrue(deployed_dir.exists())
        self.assertTrue((deployed_dir / "app.py").exists())
    
    async def test_encrypted_pubsub(self):
        """Test encrypted pub/sub messaging"""
        # Setup keys
        keys_dir = self.test_path / "keys"
        manager = ZMQKeyManager(keys_dir)
        manager.rotate_keys()
        
        # Create buses
        pub_bus = ZeroMQBus(
            pub_endpoint="tcp://127.0.0.1:25555",
            sub_endpoint="tcp://127.0.0.1:25556",
            enable_encryption=False,  # Simplified for test
            keys_dir=keys_dir
        )
        
        sub_bus = ZeroMQBus(
            pub_endpoint="tcp://127.0.0.1:25556",
            sub_endpoint="tcp://127.0.0.1:25555",
            enable_encryption=False,  # Simplified for test
            keys_dir=keys_dir
        )
        
        # Start buses
        await pub_bus.start()
        await sub_bus.start()
        
        # Setup subscription
        received = []
        async def handler(topic, message):
            received.append((topic, message))
        
        await sub_bus.subscribe("test.topic", handler)
        
        # Give time for subscription
        await asyncio.sleep(0.1)
        
        # Publish message
        test_message = {"data": "encrypted test", "seq": 1}
        await pub_bus.publish("test.topic", test_message)
        
        # Give time for delivery
        await asyncio.sleep(0.1)
        
        # Check received
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][0], "test.topic")
        self.assertEqual(received[0][1], test_message)
        
        # Cleanup
        await pub_bus.stop()
        await sub_bus.stop()


def run_async_test(coro):
    """Helper to run async tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Async test runners
class AsyncTestCase(unittest.TestCase):
    """Base class for async tests"""
    
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if name.startswith('test_') and asyncio.iscoroutinefunction(attr):
            return lambda: run_async_test(attr())
        return attr


class TestSignatureVerificationAsync(AsyncTestCase, TestSignatureVerification):
    """Async version of signature tests"""
    pass


class TestZMQKeyRotationAsync(AsyncTestCase, TestZMQKeyRotation):
    """Async version of key rotation tests"""
    pass


class TestZeroMQEncryptionAsync(AsyncTestCase, TestZeroMQEncryption):
    """Async version of ZeroMQ encryption tests"""
    pass


class TestIntegrationAsync(AsyncTestCase, TestIntegration):
    """Async version of integration tests"""
    pass


if __name__ == '__main__':
    unittest.main()
