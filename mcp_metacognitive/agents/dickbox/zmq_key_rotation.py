"""
ZeroMQ Key Rotation for Dickbox
================================

Handles ED25519 key generation and rotation for secure ZeroMQ communications.
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import asyncio

try:
    import zmq
    import zmq.auth
    from zmq.auth.certs import create_certificates
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


class ZMQKeyManager:
    """
    Manages ZeroMQ encryption keys with rotation.
    """
    
    def __init__(self, keys_dir: Path = Path("/etc/tori/zmq_keys")):
        self.keys_dir = keys_dir
        self.current_key_path = None
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_ed25519_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate ED25519 keypair for ZeroMQ.
        
        Returns:
            (private_key, public_key) as bytes
        """
        if CRYPTO_AVAILABLE:
            # Use cryptography library
            private_key = ed25519.Ed25519PrivateKey.generate()
            public_key = private_key.public_key()
            
            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            
            return private_bytes, public_bytes
            
        elif ZMQ_AVAILABLE and hasattr(zmq.auth, 'create_certificates'):
            # Use ZeroMQ's built-in certificate generation
            key_dir = self.keys_dir / f"temp_{datetime.now().timestamp()}"
            key_dir.mkdir(exist_ok=True)
            
            create_certificates(str(key_dir), "key")
            
            # Read generated keys
            with open(key_dir / "key.key", 'rb') as f:
                private_key = f.read()
            with open(key_dir / "key.key_public", 'rb') as f:
                public_key = f.read()
            
            # Cleanup temp dir
            import shutil
            shutil.rmtree(key_dir)
            
            return private_key, public_key
            
        else:
            # Fallback: use ssh-keygen
            temp_dir = Path(f"/tmp/zmq_keys_{datetime.now().timestamp()}")
            temp_dir.mkdir(exist_ok=True)
            
            key_file = temp_dir / "key"
            
            # Generate ED25519 key with ssh-keygen
            subprocess.run([
                "ssh-keygen", "-t", "ed25519",
                "-f", str(key_file),
                "-N", "",  # No passphrase
                "-C", f"zmq@{datetime.now().isoformat()}"
            ], check=True)
            
            # Read keys
            with open(key_file, 'rb') as f:
                private_key = f.read()
            with open(f"{key_file}.pub", 'rb') as f:
                public_key = f.read()
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
            
            return private_key, public_key
    
    def rotate_keys(self) -> Dict[str, Any]:
        """
        Generate new keypair and save with timestamp.
        
        Returns:
            Dict with new key information
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate new keypair
        private_key, public_key = self.generate_ed25519_keypair()
        
        # Save keys
        private_path = self.keys_dir / f"{timestamp}_private.key"
        public_path = self.keys_dir / f"{timestamp}_public.key"
        
        # Save with restricted permissions
        with open(private_path, 'wb') as f:
            f.write(private_key)
        os.chmod(private_path, 0o600)  # Read/write for owner only
        
        with open(public_path, 'wb') as f:
            f.write(public_key)
        os.chmod(public_path, 0o644)  # Read for all, write for owner
        
        # Create symlinks to current keys
        current_private = self.keys_dir / "current_private.key"
        current_public = self.keys_dir / "current_public.key"
        
        # Remove old symlinks
        if current_private.exists():
            current_private.unlink()
        if current_public.exists():
            current_public.unlink()
        
        # Create new symlinks
        current_private.symlink_to(private_path.name)
        current_public.symlink_to(public_path.name)
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "created_at": datetime.now().isoformat(),
            "private_key": str(private_path),
            "public_key": str(public_path),
            "algorithm": "ed25519"
        }
        
        metadata_path = self.keys_dir / f"{timestamp}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Rotated ZeroMQ keys: {timestamp}")
        
        return metadata
    
    def get_current_keys(self) -> Optional[Dict[str, Any]]:
        """
        Get current active keys.
        
        Returns:
            Dict with current key paths or None
        """
        current_private = self.keys_dir / "current_private.key"
        current_public = self.keys_dir / "current_public.key"
        
        if not current_private.exists() or not current_public.exists():
            return None
        
        # Find metadata
        private_target = current_private.resolve()
        timestamp = private_target.stem.split('_')[0]
        metadata_path = self.keys_dir / f"{timestamp}_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        
        return {
            "private_key": str(current_private),
            "public_key": str(current_public)
        }
    
    def cleanup_old_keys(self, keep_days: int = 30):
        """
        Remove keys older than specified days.
        
        Args:
            keep_days: Number of days to keep old keys
        """
        cutoff = datetime.now() - timedelta(days=keep_days)
        
        for key_file in self.keys_dir.glob("*_*.key"):
            try:
                # Parse timestamp from filename
                timestamp_str = key_file.stem.split('_')[0]
                file_date = datetime.strptime(timestamp_str[:8], "%Y%m%d")
                
                if file_date < cutoff:
                    key_file.unlink()
                    logger.info(f"Removed old key: {key_file}")
                    
                    # Remove associated metadata
                    metadata_file = key_file.parent / f"{timestamp_str}_metadata.json"
                    if metadata_file.exists():
                        metadata_file.unlink()
                        
            except Exception as e:
                logger.error(f"Error cleaning up {key_file}: {e}")


class ZMQKeyRotationService:
    """
    Service for automatic key rotation.
    """
    
    def __init__(self, key_manager: ZMQKeyManager, communication_fabric=None):
        self.key_manager = key_manager
        self.communication_fabric = communication_fabric
        
    async def rotate_and_broadcast(self):
        """
        Rotate keys and broadcast update event.
        """
        # Rotate keys
        new_keys = self.key_manager.rotate_keys()
        
        # Broadcast key update event if communication fabric available
        if self.communication_fabric and hasattr(self.communication_fabric, 'publish_event'):
            await self.communication_fabric.publish_event("zmq.key_update", {
                "action": "rotated",
                "timestamp": new_keys["timestamp"],
                "public_key_path": new_keys["public_key"],
                "created_at": new_keys["created_at"]
            })
            
            logger.info("Broadcasted key rotation event")
        
        # Cleanup old keys
        self.key_manager.cleanup_old_keys()
        
        return new_keys


def main():
    """Main function for key rotation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ZeroMQ Key Rotation")
    parser.add_argument("--keys-dir", default="/etc/tori/zmq_keys",
                       help="Directory for ZMQ keys")
    parser.add_argument("--cleanup-days", type=int, default=30,
                       help="Days to keep old keys")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create key manager
    key_manager = ZMQKeyManager(Path(args.keys_dir))
    
    # Rotate keys
    new_keys = key_manager.rotate_keys()
    print(f"Rotated keys: {new_keys['timestamp']}")
    print(f"Public key: {new_keys['public_key']}")
    
    # Cleanup old keys
    key_manager.cleanup_old_keys(args.cleanup_days)
    

if __name__ == "__main__":
    main()
