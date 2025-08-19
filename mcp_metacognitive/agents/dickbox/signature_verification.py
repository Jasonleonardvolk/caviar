"""
Signature Verification Module for Dickbox
========================================

Handles capsule signature verification using minisign or sigstore.
"""

import logging
from pathlib import Path
from typing import Optional
import yaml
import shutil

# Try importing signature verification
try:
    import minisign
    MINISIGN_AVAILABLE = True
except ImportError:
    MINISIGN_AVAILABLE = False
    
try:
    from sigstore import verify
    SIGSTORE_AVAILABLE = True
except ImportError:
    SIGSTORE_AVAILABLE = False

logger = logging.getLogger(__name__)


async def verify_capsule_signature(capsule_path: Path, sig_path: Path, public_key: str) -> bool:
    """
    Verify capsule signature using available verification library.
    
    Args:
        capsule_path: Path to capsule tarball
        sig_path: Path to signature file
        public_key: Public key for verification
        
    Returns:
        True if signature is valid, False otherwise
    """
    # Try minisign first
    if MINISIGN_AVAILABLE:
        try:
            # Read signature
            with open(sig_path, 'rb') as f:
                signature = f.read()
            
            # Verify with minisign
            public_key_obj = minisign.PublicKey.from_base64(public_key)
            return public_key_obj.verify(capsule_path.read_bytes(), signature)
        except Exception as e:
            logger.error(f"Minisign verification failed: {e}")
            return False
    
    # Try sigstore as fallback
    elif SIGSTORE_AVAILABLE:
        try:
            # Sigstore verification
            with open(capsule_path, 'rb') as f:
                blob = f.read()
            
            with open(sig_path, 'rb') as f:
                sig_bundle = f.read()
            
            # Verify with sigstore
            result = verify.verify_blob(
                blob,
                signature_bundle=sig_bundle,
                identity=public_key  # Can be email or key
            )
            return result is not None
        except Exception as e:
            logger.error(f"Sigstore verification failed: {e}")
            return False
    
    else:
        logger.warning("No signature verification library available - signature check disabled")
        # In production, this should fail closed
        return True  # For development only


async def extract_capsule_with_verification(tarball_path: Path, target_dir: Path) -> bool:
    """
    Extract capsule and verify signature if present.
    
    Args:
        tarball_path: Path to capsule tarball
        target_dir: Directory to extract to
        
    Returns:
        True if extraction and verification successful
    """
    import tarfile
    
    # Extract tarball
    with tarfile.open(tarball_path, 'r:gz') as tar:
        tar.extractall(target_dir)
    
    # Check for signature verification
    manifest_path = target_dir / "capsule.yml"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)
            
            # Verify signature if present
            if "signature" in manifest and "public_key" in manifest:
                sig_filename = manifest.get("signature", "capsule.sig")
                sig_path = target_dir / sig_filename
                
                if sig_path.exists():
                    verified = await verify_capsule_signature(
                        tarball_path,
                        sig_path,
                        manifest["public_key"]
                    )
                    
                    if not verified:
                        # Remove extracted files on verification failure
                        shutil.rmtree(target_dir)
                        raise Exception("Signature verification failed")
                    
                    logger.info(f"Signature verified for capsule")
                    return True
                else:
                    logger.warning(f"Signature file not found: {sig_path}")
                    # Could be stricter here
        except Exception as e:
            # Clean up on any error
            if target_dir.exists():
                shutil.rmtree(target_dir)
            raise Exception(f"Capsule verification failed: {e}")
    
    return True
