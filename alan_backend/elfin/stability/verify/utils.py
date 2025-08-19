"""
Utility functions for the verification of neural Lyapunov functions.

This module contains reusable functions for verification, including
proof hashing, domain manipulations, and other utilities.
"""

import io
import json
import hashlib
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def compute_proof_hash(
    model: nn.Module, 
    bounds: Tuple[np.ndarray, np.ndarray], 
    solver_tag: str,
    framework_version: Optional[str] = None
) -> str:
    """
    Compute a deterministic hash for the verification problem.
    
    The hash uniquely identifies a verification problem based on:
    1. Network architecture signature (layer sizes, activations, etc.)
    2. Network weights (serialized state_dict)
    3. Verification domain definition (bounds)
    4. Solver and encoding version
    5. (Optional) Framework version for cross-project isolation
    
    This hash can be used to cache verification results and avoid redundant
    verification of the same network.
    
    Args:
        model: PyTorch neural network model
        bounds: Tuple of (lower_bounds, upper_bounds) numpy arrays
        solver_tag: String identifier for solver and encoding version
        framework_version: Optional version string to avoid cross-project collisions
        
    Returns:
        SHA-256 hexadecimal digest string
    """
    # Extract architecture metadata
    arch_info = {
        'shapes': [tuple(p.shape) for p in model.parameters()],
        'activations': [m.__class__.__name__ for m in model.modules() 
                       if not isinstance(m, nn.Sequential) and not isinstance(m, nn.ModuleList)],
        'input_dim': next(iter(model.parameters())).shape[-1] if len(list(model.parameters())) > 0 else None
    }
    
    # If LyapunovNet, get alpha parameter
    if hasattr(model, 'alpha'):
        arch_info['alpha'] = float(model.alpha)
    
    # Extract bounds information
    bounds_info = {
        'lower': bounds[0].tolist(),
        'upper': bounds[1].tolist(),
    }
    
    # Create metadata dictionary
    metadata = {
        'architecture': arch_info,
        'bounds': bounds_info,
        'solver': solver_tag
    }
    
    # Add framework version if provided
    if framework_version is not None:
        metadata['version'] = framework_version
    
    # Serialize model weights to bytes
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer, _use_new_zipfile_serialization=True)
    
    # Create hash
    hasher = hashlib.sha256()
    
    # Add metadata as JSON (sorted for determinism)
    metadata_json = json.dumps(metadata, sort_keys=True).encode()
    hasher.update(metadata_json)
    
    # Add model weights
    hasher.update(buffer.getbuffer())
    
    # Return hexadecimal digest
    return hasher.hexdigest()


def verify_proof_identity(
    original_hash: str,
    model: nn.Module,
    bounds: Tuple[np.ndarray, np.ndarray],
    solver_tag: str,
    framework_version: Optional[str] = None
) -> bool:
    """
    Verify that a model/domain/solver combination matches a previously computed hash.
    
    This is useful to detect if a cached proof can be reused, or if verification
    needs to be redone due to changes in the model or verification problem.
    
    Args:
        original_hash: Original hash string to compare against
        model: PyTorch neural network model
        bounds: Tuple of (lower_bounds, upper_bounds) numpy arrays
        solver_tag: String identifier for solver and encoding version
        framework_version: Optional version string for cross-project isolation
        
    Returns:
        True if the current model/bounds/solver hash matches the original_hash
    """
    current_hash = compute_proof_hash(model, bounds, solver_tag, framework_version)
    return current_hash == original_hash


class ProofCertificate:
    """
    A serializable certificate of successful verification.
    
    This class represents a verified property of a neural Lyapunov function,
    containing all the information needed to reproduce the verification or
    invalidate the proof if conditions change.
    """
    
    def __init__(
        self,
        proof_hash: str,
        solver_name: str,
        solver_version: str,
        encoding_version: str,
        property_type: str,
        verification_time: float,
        extra_info: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a proof certificate.
        
        Args:
            proof_hash: SHA-256 hash identifying the verification problem
            solver_name: Name of the solver used (e.g., 'gurobi', 'z3')
            solver_version: Version of the solver
            encoding_version: Version of the encoding method
            property_type: Type of property verified (e.g., 'positive_definite', 'decrease')
            verification_time: Time taken for verification in seconds
            extra_info: Additional information about the verification
        """
        self.proof_hash = proof_hash
        self.solver_name = solver_name
        self.solver_version = solver_version
        self.encoding_version = encoding_version
        self.property_type = property_type
        self.verification_time = verification_time
        self.extra_info = extra_info or {}
        self.timestamp = None  # Set when serialized
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert certificate to a serializable dictionary."""
        import datetime
        
        # Set timestamp if not already set
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now().isoformat()
            
        return {
            'proof_hash': self.proof_hash,
            'solver_name': self.solver_name,
            'solver_version': self.solver_version,
            'encoding_version': self.encoding_version,
            'property_type': self.property_type,
            'verification_time': self.verification_time,
            'extra_info': self.extra_info,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProofCertificate':
        """Create certificate from dictionary."""
        certificate = cls(
            proof_hash=data['proof_hash'],
            solver_name=data['solver_name'],
            solver_version=data['solver_version'],
            encoding_version=data['encoding_version'],
            property_type=data['property_type'],
            verification_time=data['verification_time'],
            extra_info=data.get('extra_info', {})
        )
        certificate.timestamp = data.get('timestamp')
        return certificate
    
    def to_json(self) -> str:
        """Convert certificate to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ProofCertificate':
        """Create certificate from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class VerificationResult:
    """
    Result of a verification attempt.
    
    This class contains all the information about a verification result,
    including success status, counterexample, certificate, etc.
    """
    
    def __init__(
        self,
        success: bool,
        property_type: str,
        counterexample: Optional[np.ndarray] = None,
        certificate: Optional[ProofCertificate] = None,
        message: str = "",
        stats: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a verification result.
        
        Args:
            success: Whether the verification was successful
            property_type: Type of property verified
            counterexample: Counterexample point if verification failed
            certificate: Proof certificate if verification succeeded
            message: Human-readable message about the verification
            stats: Additional statistics about the verification
        """
        self.success = success
        self.property_type = property_type
        self.counterexample = counterexample
        self.certificate = certificate
        self.message = message
        self.stats = stats or {}
    
    @property
    def has_counterexample(self) -> bool:
        """Check if the result has a counterexample."""
        return self.counterexample is not None
    
    @property
    def has_certificate(self) -> bool:
        """Check if the result has a proof certificate."""
        return self.certificate is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        result = {
            'success': self.success,
            'property_type': self.property_type,
            'message': self.message,
            'stats': self.stats
        }
        
        if self.counterexample is not None:
            result['counterexample'] = self.counterexample.tolist()
            
        if self.certificate is not None:
            result['certificate'] = self.certificate.to_dict()
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationResult':
        """Create result from dictionary."""
        # Convert counterexample to numpy array if present
        counterexample = None
        if 'counterexample' in data:
            counterexample = np.array(data['counterexample'])
            
        # Convert certificate if present
        certificate = None
        if 'certificate' in data:
            certificate = ProofCertificate.from_dict(data['certificate'])
            
        return cls(
            success=data['success'],
            property_type=data['property_type'],
            counterexample=counterexample,
            certificate=certificate,
            message=data.get('message', ""),
            stats=data.get('stats', {})
        )
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'VerificationResult':
        """Create result from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
