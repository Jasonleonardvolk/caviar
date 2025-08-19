"""
Dickbox Deployer with Signature Verification
===========================================

Handles secure capsule deployment with cryptographic signatures.
"""

import asyncio
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# Try importing signature verification libraries
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


class SignatureVerifier:
    """Handles capsule signature verification"""
    
    def __init__(self, verification_method: str = "minisign"):
        """
        Initialize verifier.
        
        Args:
            verification_method: "minisign" or "sigstore"
        """
        self.method = verification_method
        
        if verification_method == "minisign" and not MINISIGN_AVAILABLE:
            raise ImportError("python-minisign not installed")
        elif verification_method == "sigstore" and not SIGSTORE_AVAILABLE:
            raise ImportError("sigstore not installed")
    
    async def verify_signature(self, 
                             capsule_path: Path, 
                             sig_path: Optional[Path] = None,
                             public_key: Optional[str] = None) -> bool:
        """
        Verify capsule signature.
        
        Args:
            capsule_path: Path to capsule tarball
            sig_path: Path to signature file (defaults to capsule_path.sig)
            public_key: Public key for verification
            
        Returns:
            True if signature is valid
            
        Raises:
            ValueError: If signature verification fails
        """
        if not capsule_path.exists():
            raise FileNotFoundError(f"Capsule not found: {capsule_path}")
        
        # Default signature path
        if sig_path is None:
            sig_path = capsule_path.with_suffix(capsule_path.suffix + '.sig')
        
        if not sig_path.exists():
            raise FileNotFoundError(f"Signature not found: {sig_path}")
        
        if self.method == "minisign":
            return await self._verify_minisign(capsule_path, sig_path, public_key)
        elif self.method == "sigstore":
            return await self._verify_sigstore(capsule_path, sig_path, public_key)
        else:
            raise ValueError(f"Unknown verification method: {self.method}")
    
    async def _verify_minisign(self, capsule_path: Path, sig_path: Path, public_key: str) -> bool:
        """Verify using minisign"""
        if not public_key:
            raise ValueError("Public key required for minisign verification")
        
        try:
            # Use minisign command line tool
            cmd = [
                "minisign", "-V",
                "-p", public_key,
                "-m", str(capsule_path),
                "-x", str(sig_path)
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                logger.info(f"Signature verified for {capsule_path}")
                return True
            else:
                raise ValueError(f"Signature verification failed: {stderr.decode()}")
                
        except FileNotFoundError:
            # Fallback to python-minisign if available
            if MINISIGN_AVAILABLE:
                # Read public key
                with open(public_key, 'r') as f:
                    pk = minisign.PublicKey.from_base64(f.read().strip())
                
                # Read signature
                with open(sig_path, 'rb') as f:
                    signature = minisign.Signature.decode(f.read())
                
                # Verify
                with open(capsule_path, 'rb') as f:
                    pk.verify(signature, f.read())
                
                logger.info(f"Signature verified for {capsule_path}")
                return True
            else:
                raise ValueError("minisign command not found and python-minisign not available")
    
    async def _verify_sigstore(self, capsule_path: Path, sig_path: Path, public_key: Optional[str]) -> bool:
        """Verify using sigstore"""
        try:
            # Read signature bundle
            with open(sig_path, 'r') as f:
                import json
                bundle = json.load(f)
            
            # Verify with sigstore
            result = verify.verify_blob(
                blob_path=str(capsule_path),
                bundle_path=str(sig_path),
                offline=True  # Don't require online verification
            )
            
            if result:
                logger.info(f"Sigstore verification passed for {capsule_path}")
                return True
            else:
                raise ValueError("Sigstore verification failed")
                
        except Exception as e:
            logger.error(f"Sigstore verification error: {e}")
            raise ValueError(f"Sigstore verification failed: {e}")


class SecureDeployer:
    """Enhanced deployer with signature verification"""
    
    def __init__(self, verifier: Optional[SignatureVerifier] = None):
        self.verifier = verifier or SignatureVerifier()
    
    async def extract_capsule_with_verification(self,
                                               capsule_path: Path,
                                               target_dir: Path,
                                               manifest: Dict[str, Any]) -> str:
        """
        Extract capsule after verifying signature.
        
        Args:
            capsule_path: Path to capsule tarball
            target_dir: Directory to extract to
            manifest: Capsule manifest with signature info
            
        Returns:
            Capsule ID (hash)
        """
        # Check if signature verification is required
        signature_info = manifest.get("signature", {})
        if signature_info:
            sig_path = None
            public_key = signature_info.get("public_key")
            
            # Check for signature file path
            if "signature_file" in signature_info:
                sig_path = capsule_path.parent / signature_info["signature_file"]
            
            # Verify signature
            try:
                await self.verifier.verify_signature(
                    capsule_path,
                    sig_path=sig_path,
                    public_key=public_key
                )
            except Exception as e:
                logger.error(f"Signature verification failed: {e}")
                raise ValueError(f"Capsule signature verification failed: {e}")
        else:
            logger.warning(f"No signature info in manifest for {capsule_path}")
        
        # Calculate content hash
        hasher = hashlib.sha256()
        with open(capsule_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        capsule_id = hasher.hexdigest()[:12]
        extract_dir = target_dir / capsule_id
        
        # Extract only if verified
        if extract_dir.exists():
            logger.info(f"Capsule {capsule_id} already extracted")
        else:
            extract_dir.mkdir(parents=True)
            
            # Extract tarball
            cmd = ["tar", "-xzf", str(capsule_path), "-C", str(extract_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                extract_dir.rmdir()
                raise Exception(f"Failed to extract capsule: {result.stderr}")
            
            logger.info(f"Extracted verified capsule {capsule_id} to {extract_dir}")
        
        return capsule_id


def create_signed_capsule(source_dir: Path, 
                         output_path: Path,
                         private_key: Optional[str] = None,
                         method: str = "minisign") -> Tuple[Path, Path]:
    """
    Create a signed capsule.
    
    Args:
        source_dir: Directory to package
        output_path: Output capsule path
        private_key: Private key for signing
        method: Signing method ("minisign" or "sigstore")
        
    Returns:
        Tuple of (capsule_path, signature_path)
    """
    # Create tarball
    cmd = ["tar", "-czf", str(output_path), "-C", str(source_dir), "."]
    subprocess.run(cmd, check=True)
    
    # Sign the capsule
    sig_path = output_path.with_suffix(output_path.suffix + '.sig')
    
    if method == "minisign" and private_key:
        # Sign with minisign
        sign_cmd = [
            "minisign", "-S",
            "-s", private_key,
            "-m", str(output_path),
            "-x", str(sig_path)
        ]
        subprocess.run(sign_cmd, check=True)
        
    elif method == "sigstore":
        # Sign with sigstore
        import json
        sign_cmd = [
            "cosign", "sign-blob",
            "--bundle", str(sig_path),
            str(output_path)
        ]
        subprocess.run(sign_cmd, check=True)
    
    return output_path, sig_path


# Export
__all__ = ['SignatureVerifier', 'SecureDeployer', 'create_signed_capsule']
