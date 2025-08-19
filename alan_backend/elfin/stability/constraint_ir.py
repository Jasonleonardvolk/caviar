"""
Constraint IR (Intermediate Representation) Module.

This module defines a solver-agnostic constraint representation
that can be passed to various verification backends (SOS, SMT, MILP, etc.).
"""

from typing import Dict, List, Optional, Union, Any, Literal
from dataclasses import dataclass, field
from enum import Enum, auto
import hashlib
import json
import time


class ConstraintType(Enum):
    """Types of constraints supported in the IR."""
    
    EQUALITY = auto()     # a == b
    INEQUALITY = auto()   # a <= b or a >= b
    POSITIVE = auto()     # a > 0
    NEGATIVE = auto()     # a < 0
    VANISHING = auto()    # a == 0
    CUSTOM = auto()       # Other constraint types


class VerificationStatus(Enum):
    """Status of a verification result."""
    
    VERIFIED = auto()     # Constraint verified to be true
    REFUTED = auto()      # Constraint verified to be false
    UNKNOWN = auto()      # Unknown status
    IN_PROGRESS = auto()  # Verification in progress
    ERROR = auto()        # Error during verification


@dataclass
class ConstraintIR:
    """
    Constraint Intermediate Representation.
    
    This represents a solver-agnostic constraint that can be passed to
    various verification backends (SOS, SMT, MILP, etc.).
    """
    
    id: str
    variables: List[str]
    expression: str
    constraint_type: Union[ConstraintType, str]
    context: Dict[str, Any] = field(default_factory=dict)
    solver_hint: Optional[str] = None
    proof_needed: bool = True
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "variables": self.variables,
            "expression": self.expression,
            "constraint_type": self.constraint_type.name if isinstance(self.constraint_type, ConstraintType) else self.constraint_type,
            "context": self.context,
        }
        
        if self.solver_hint:
            result["solver_hint"] = self.solver_hint
            
        if not self.proof_needed:
            result["proof_needed"] = False
            
        if self.dependencies:
            result["dependencies"] = self.dependencies
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConstraintIR':
        """Create from dictionary representation."""
        constraint_type = data["constraint_type"]
        if isinstance(constraint_type, str):
            try:
                constraint_type = ConstraintType[constraint_type]
            except KeyError:
                pass  # Keep as string if not a valid enum value
                
        return cls(
            id=data["id"],
            variables=data["variables"],
            expression=data["expression"],
            constraint_type=constraint_type,
            context=data.get("context", {}),
            solver_hint=data.get("solver_hint"),
            proof_needed=data.get("proof_needed", True),
            dependencies=data.get("dependencies", [])
        )
    
    def compute_hash(self) -> str:
        """
        Compute a unique hash for this constraint.
        
        This hash can be used for caching verification results.
        """
        # Convert to JSON-serializable dict (excluding solver-specific hints)
        data = {
            "id": self.id,
            "variables": sorted(self.variables),
            "expression": self.expression,
            "constraint_type": self.constraint_type.name if isinstance(self.constraint_type, ConstraintType) else self.constraint_type,
            "context": {k: v for k, v in sorted(self.context.items()) if k != "solver_specific"}
        }
        
        # Compute hash
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


@dataclass
class VerificationResult:
    """
    Result of a constraint verification.
    
    This includes the status, any counterexample found, and a
    certificate of proof if the verification succeeded.
    """
    
    constraint_id: str
    status: VerificationStatus
    proof_hash: str
    verification_time: float
    solver_time: float = field(default_factory=lambda: time.time())
    counterexample: Optional[Dict[str, Any]] = None
    certificate: Optional[Dict[str, Any]] = None
    solver_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "constraint_id": self.constraint_id,
            "status": self.status.name,
            "proof_hash": self.proof_hash,
            "verification_time": self.verification_time,
            "solver_info": self.solver_info
        }
        
        if self.counterexample:
            result["counterexample"] = self.counterexample
            
        if self.certificate:
            result["certificate"] = self.certificate
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationResult':
        """Create from dictionary representation."""
        status = data["status"]
        if isinstance(status, str):
            try:
                status = VerificationStatus[status]
            except KeyError:
                status = VerificationStatus.UNKNOWN
                
        return cls(
            constraint_id=data["constraint_id"],
            status=status,
            proof_hash=data["proof_hash"],
            verification_time=data["verification_time"],
            counterexample=data.get("counterexample"),
            certificate=data.get("certificate"),
            solver_info=data.get("solver_info", {})
        )


class ProofCache:
    """
    Cache for verification results with dependency tracking.
    
    This enables incremental verification and avoids re-verifying unchanged
    constraints.
    """
    
    def __init__(self):
        """Initialize the proof cache."""
        self.proofs: Dict[str, VerificationResult] = {}
        self.dependencies: Dict[str, List[str]] = {}  # concept_id -> list of proof_hashes
        
    def get(self, constraint_hash: str, grammar_time: Optional[float] = None) -> Optional[VerificationResult]:
        """
        Get a cached verification result.
        
        Args:
            constraint_hash: Hash of the constraint
            grammar_time: Optional timestamp of last grammar modification
            
        Returns:
            Cached result or None if not found or outdated
        """
        result = self.proofs.get(constraint_hash)
        
        # If no result in cache or grammar was modified after proof was generated,
        # we need to re-verify
        if result is None:
            return None
            
        if grammar_time is not None and result.solver_time < grammar_time:
            # Proof is older than grammar, mark as stale but return it
            result.status = VerificationStatus.UNKNOWN
            
        return result
        
    def put(self, result: VerificationResult, dependencies: Optional[List[str]] = None) -> None:
        """
        Add a verification result to the cache.
        
        Args:
            result: Verification result
            dependencies: List of concept IDs that this result depends on
        """
        self.proofs[result.proof_hash] = result
        
        # Record dependencies
        if dependencies:
            for concept_id in dependencies:
                if concept_id not in self.dependencies:
                    self.dependencies[concept_id] = []
                    
                if result.proof_hash not in self.dependencies[concept_id]:
                    self.dependencies[concept_id].append(result.proof_hash)
                    
    def invalidate(self, dependency: str) -> List[str]:
        """
        Invalidate all proofs that depend on a concept.
        
        Args:
            dependency: Concept ID or other dependency
            
        Returns:
            List of invalidated proof hashes
        """
        if dependency not in self.dependencies:
            return []
            
        invalidated = []
        
        for proof_hash in self.dependencies.get(dependency, []):
            if proof_hash in self.proofs:
                # Update status to unknown
                result = self.proofs[proof_hash]
                result.status = VerificationStatus.UNKNOWN
                invalidated.append(proof_hash)
                
        return invalidated


# Examples for documentation
_examples = [
    ConstraintIR(
        id="pd_lyap1",
        variables=["x1", "x2", "x3"],
        expression="(> (V_lyap1 x1 x2 x3) 0)",
        constraint_type=ConstraintType.POSITIVE,
        context={"lyapunov_type": "polynomial", "degree": 2},
        solver_hint="sos",
        proof_needed=True,
        dependencies=["concept_1", "concept_2"]
    ),
    ConstraintIR(
        id="decreasing_lyap1",
        variables=["x1", "x2", "x3"],
        expression="(< (lie_derivative V_lyap1 f_dynamics) 0)",
        constraint_type=ConstraintType.NEGATIVE,
        context={
            "lyapunov_type": "neural", 
            "stability_type": "asymptotic",
            "solver_specific": {
                "milp": {"timeout": 120}
            }
        },
        solver_hint="milp",
        proof_needed=True,
        dependencies=["concept_3", "concept_4"]
    )
]
