#!/usr/bin/env python3
"""
Standalone demo of the StabilityAgent for the ELFIN stability framework.

This script demonstrates how to use the StabilityAgent to verify
stability properties of a system, log interactions, and generate
rich error messages. It uses only the stability components and
does not require the parser components.
"""

import os
import logging
import pathlib
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("elfin.demo")

# ========== INLINE IMPLEMENTATION OF CORE COMPONENTS ==========
# This allows the demo to run without depending on other modules

@dataclass
class Interaction:
    """
    Represents a single interaction with a stability verification component.
    """
    timestamp: str
    action: str
    meta: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    
    @staticmethod
    def now(action: str, **meta) -> "Interaction":
        """Create a new interaction with the current timestamp."""
        return Interaction(
            timestamp=datetime.utcnow().isoformat(),
            action=action,
            meta=meta
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert interaction to a dictionary for serialization."""
        # Convert to dictionary using dataclasses.asdict
        result = asdict(self)
        
        # Make sure all values are JSON serializable
        return self._make_json_serializable(result)
    
    def _make_json_serializable(self, obj):
        """Make an object JSON serializable, handling numpy arrays, etc."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def to_json(self) -> str:
        """Convert interaction to JSON string."""
        return json.dumps(self.to_dict())
    
    def get_reference(self) -> str:
        """Get a unique reference to this interaction."""
        return f"{self.timestamp}#{self.action}"


@dataclass
class InteractionLog:
    """An append-only log of interactions."""
    interactions: List[Interaction] = field(default_factory=list)
    
    def append(self, interaction: Interaction) -> None:
        """Add an interaction to the log."""
        self.interactions.append(interaction)
    
    def append_and_persist(self, interaction: Interaction, path: Union[str, pathlib.Path]) -> None:
        """Add an interaction to the log and append it to a JSONL file."""
        self.append(interaction)
        
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open("a", encoding="utf-8") as f:
            f.write(interaction.to_json() + "\n")
    
    def tail(self, n: int) -> "InteractionLog":
        """Get the last n interactions."""
        return InteractionLog(self.interactions[-n:])


class VerificationError:
    """Standardized error format for verification failures."""
    ERROR_CODES = {
        "LYAP_001": "Lyapunov function not positive definite",
        "LYAP_002": "Lyapunov function not decreasing",
    }
    
    def __init__(
        self,
        code: str,
        detail: str,
        system_id: str,
        interaction_ref: str,
        **extra_fields
    ):
        """Initialize a verification error."""
        self.code = code
        self.title = self.ERROR_CODES.get(code, "Unknown error")
        self.detail = detail
        self.system_id = system_id
        self.interaction_ref = interaction_ref
        self.extra_fields = extra_fields
        self.doc_url = f"https://elfin.dev/errors/{code}"
    
    def __str__(self) -> str:
        """Convert error to a string."""
        return f"E-{self.code}: {self.detail} (see {self.doc_url})"


class LyapunovNet(nn.Module):
    """
    Lyapunov-Net architecture for learning verifiable Lyapunov functions.
    
    Implements the approach from Gaby et al. where V(x) = |phi(x) - phi(0)| + alpha*||x||.
    """
    
    def __init__(
        self, 
        dim: int, 
        hidden_dims: tuple = (64, 64),
        alpha: float = 1e-3,
        activation: nn.Module = nn.Tanh()
    ):
        """Initialize the LyapunovNet."""
        super().__init__()
        
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
            
        self.alpha = alpha
        self.hidden_dims = hidden_dims
        self.activation_type = activation.__class__.__name__
        
        # Build the neural network phi(x)
        layers = []
        in_dim = dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation)
            in_dim = h_dim
            
        # Final output layer (scalar)
        layers.append(nn.Linear(in_dim, 1))
        
        # Create sequential model
        self.phi = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Created LyapunovNet with hidden dims {hidden_dims}, "
                   f"alpha={alpha}, activation={self.activation_type}")
    
    def _initialize_weights(self):
        """Initialize network weights for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization
                nn.init.kaiming_uniform_(m.weight, nonlinearity='tanh')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        """Compute the Lyapunov function value V(x)."""
        # Create a zero vector with the same batch shape as x
        zeros = torch.zeros_like(x)
        
        # Compute |phi(x) - phi(0)|
        phi_diff = torch.abs(self.phi(x) - self.phi(zeros))
        
        # Compute the norm term alpha*||x||
        use_alpha = alpha if alpha is not None else self.alpha
        norm_term = use_alpha * torch.norm(x, dim=-1, keepdim=True)
        
        # Return V(x) = |phi(x) - phi(0)| + alpha*||x||
        return phi_diff + norm_term


class VerificationResult:
    """Result of a verification attempt."""
    def __init__(
        self,
        success: bool,
        property_type: str,
        counterexample: Optional[np.ndarray] = None,
        message: str = "",
        time_taken: float = 0.0
    ):
        self.success = success
        self.property_type = property_type
        self.counterexample = counterexample
        self.message = message
        self.time_taken = time_taken


class SimpleVerifier:
    """A simple Lyapunov function verifier for demo purposes."""
    
    def __init__(
        self,
        torch_net: nn.Module,
        domain: tuple,
        time_limit: float = 10.0,
        verbose: bool = False
    ):
        """Initialize the verifier."""
        self.net = torch_net
        self.low, self.high = domain
        self.time_limit = time_limit
        self.verbose = verbose
        self.verification_time = 0.0
        self.proof_hash = "demo_proof_" + datetime.now().strftime("%Y%m%d%H%M%S")
    
    def find_pd_counterexample(self) -> VerificationResult:
        """Find a counterexample to positive definiteness condition."""
        start_time = time.time()
        
        # For demo purposes, check a grid of points
        if hasattr(self.net, 'id') and self.net.id == "unstable_net":
            # For the unstable demo net, return a counterexample
            counterexample = np.array([0.5, 0.5])
            self.verification_time = time.time() - start_time
            return VerificationResult(
                success=False,
                property_type="positive_definite",
                counterexample=counterexample,
                message="Function not positive definite",
                time_taken=self.verification_time
            )
        else:
            # For the stable demo net, return success
            self.verification_time = time.time() - start_time
            return VerificationResult(
                success=True,
                property_type="positive_definite",
                message="Function is positive definite",
                time_taken=self.verification_time
            )
    
    def find_decrease_counterexample(self, dynamics_fn, gamma: float = 0.0) -> Optional[np.ndarray]:
        """Find a counterexample to the decrease condition."""
        start_time = time.time()
        
        # For demo purposes, we'll just check if the system has negative damping
        if hasattr(dynamics_fn.__self__, 'damping') and dynamics_fn.__self__.damping < 0.1:
            # Return a counterexample for systems with low damping
            counterexample = np.array([1.0, 0.0])
            self.verification_time = time.time() - start_time
            return counterexample
        else:
            # Return success for well-damped systems
            self.verification_time = time.time() - start_time
            return None


class StabilityAgent:
    """Agent for stability verification that tracks all interactions."""
    
    def __init__(
        self,
        name: str,
        cache_db: Union[str, pathlib.Path],
        verifier_cls=SimpleVerifier
    ):
        """Initialize a stability agent."""
        self.name = name
        self.cache_db = pathlib.Path(cache_db)
        self.verifier_cls = verifier_cls
        
        # Initialize interaction log
        self.log_path = self.cache_db / f"{self.name}.log.jsonl"
        self.log = InteractionLog()
        
        logger.info(f"Initialized StabilityAgent '{name}' with cache at {cache_db}")
    
    def verify(
        self,
        system,
        domain,
        **kwargs
    ) -> Dict[str, Any]:
        """Verify a system's stability properties."""
        # Create interaction
        system_id = getattr(system, 'id', str(id(system)))
        interaction = Interaction.now(
            "verify",
            system_id=system_id,
            domain=domain,
            kwargs=kwargs
        )
        
        try:
            # Create and run verifier
            verifier = self.verifier_cls(system, domain, **kwargs)
            result = verifier.find_pd_counterexample()
            
            # Process result
            interaction.result = {
                "status": "VERIFIED" if result.success else "FAILED",
                "solve_time": verifier.verification_time,
                "counterexample": result.counterexample.tolist() if result.counterexample is not None else None,
                "proof_hash": verifier.proof_hash
            }
            
            # Log error for counterexample
            if not result.success and result.counterexample is not None:
                error = VerificationError(
                    code="LYAP_001",
                    detail=f"Function not positive definite at x={result.counterexample}",
                    system_id=system_id,
                    interaction_ref=interaction.get_reference()
                )
                logger.warning(str(error))
            
        except Exception as e:
            # Log exception
            interaction.result = {
                "error": str(e),
                "traceback": "Demo traceback"
            }
            
            # Emit error log
            error = VerificationError(
                code="VERIF_001",
                detail=str(e),
                system_id=system_id,
                interaction_ref=interaction.get_reference()
            )
            logger.error(str(error))
            
            # Re-raise
            raise
        
        finally:
            # Persist interaction
            self._append_and_persist(interaction)
        
        return interaction.result
    
    def verify_decrease(
        self,
        system,
        dynamics_fn,
        domain,
        gamma: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Verify that a system's Lyapunov function is decreasing along trajectories."""
        # Create interaction
        system_id = getattr(system, 'id', str(id(system)))
        interaction = Interaction.now(
            "verify_decrease",
            system_id=system_id,
            domain=domain,
            gamma=gamma,
            kwargs=kwargs
        )
        
        try:
            # Create verifier
            verifier = self.verifier_cls(system, domain, **kwargs)
            
            # Verify decrease condition
            result = verifier.find_decrease_counterexample(dynamics_fn, gamma)
            
            # Process result
            interaction.result = {
                "status": "VERIFIED" if result is None else "FAILED",
                "solve_time": verifier.verification_time,
                "counterexample": result.tolist() if result is not None else None,
                "gamma": gamma
            }
            
            # Log error for counterexample
            if result is not None:
                error = VerificationError(
                    code="LYAP_002",
                    detail=f"Function not decreasing at x={result}",
                    system_id=system_id,
                    interaction_ref=interaction.get_reference(),
                    gamma=gamma
                )
                logger.warning(str(error))
            
        except Exception as e:
            # Log exception
            interaction.result = {
                "error": str(e),
                "traceback": "Demo traceback"
            }
            
            # Log error
            error = VerificationError(
                code="VERIF_001",
                detail=str(e),
                system_id=system_id,
                interaction_ref=interaction.get_reference()
            )
            logger.error(str(error))
            
            # Re-raise
            raise
        
        finally:
            # Persist interaction
            self._append_and_persist(interaction)
        
        return interaction.result
    
    def param_tune(
        self,
        system,
        param_name: str,
        old_value: Any,
        new_value: Any
    ) -> Dict[str, Any]:
        """Record a parameter tuning action."""
        # Create interaction
        system_id = getattr(system, 'id', str(id(system)))
        interaction = Interaction.now(
            "param_tune",
            system_id=system_id,
            param_name=param_name,
            old_value=old_value,
            new_value=new_value
        )
        
        # Set result
        interaction.result = {
            "status": "SUCCESS",
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value
        }
        
        # Update system
        setattr(system, param_name, new_value)
        
        # Persist interaction
        self._append_and_persist(interaction)
        
        return interaction.result
    
    def _append_and_persist(self, interaction: Interaction) -> None:
        """Add an interaction to the log and persist it."""
        self.log.append_and_persist(interaction, self.log_path)
        logger.debug(f"Recorded interaction: {interaction.action} ({interaction.get_reference()})")
    
    def get_summary(self, tail: Optional[int] = None) -> str:
        """Get a human-readable summary of the interaction log."""
        log = self.log
        
        if tail is not None:
            log = log.tail(tail)
        
        if not log.interactions:
            return f"No interactions recorded for agent '{self.name}'"
        
        lines = []
        lines.append(f"Interaction log for agent '{self.name}' ({len(log.interactions)} entries):")
        
        for interaction in log.interactions:
            timestamp = interaction.timestamp.split("T")[0] + " " + interaction.timestamp.split("T")[1][:8]
            action = interaction.action.ljust(15)
            
            if interaction.result is None:
                status = "⏳"
            elif "error" in interaction.result:
                status = "❌"
            elif interaction.result.get("status") == "VERIFIED":
                status = "✅"
            elif interaction.result.get("status") == "FAILED":
                status = "⚠️"
            else:
                status = "ℹ️"
            
            # Additional details based on action
            details = ""
            if interaction.action == "verify" and interaction.result:
                if interaction.result.get("counterexample"):
                    details = f"x={np.round(interaction.result['counterexample'], 3)}"
                else:
                    solve_time = interaction.result.get("solve_time", 0.0)
                    details = f"solve={solve_time:.1f}s"
            elif interaction.action == "param_tune" and interaction.result:
                details = f"{interaction.result.get('param_name')} set {interaction.result.get('old_value')} → {interaction.result.get('new_value')}"
            
            line = f"[{timestamp}] {action} {status}  {details}"
            lines.append(line)
        
        return "\n".join(lines)


# ========== DEMO IMPLEMENTATION ==========

class SimpleOscillator:
    """
    A simple oscillator system for demonstration.
    
    This class implements a simple 2D oscillator system with adjustable
    parameters. It can be stable or unstable depending on the parameters.
    """
    
    def __init__(self, id="simple_oscillator", damping=0.1, stiffness=1.0):
        """
        Initialize the oscillator.
        
        Args:
            id: Identifier for the system
            damping: Damping coefficient (c)
            stiffness: Stiffness coefficient (k)
        """
        self.id = id
        self.damping = damping
        self.stiffness = stiffness
        
        logger.info(f"Created oscillator with damping={damping}, stiffness={stiffness}")
    
    def dynamics(self, x):
        """
        Compute the dynamics of the system.
        
        Args:
            x: State vector [position, velocity]
            
        Returns:
            Derivative of the state vector [velocity, acceleration]
        """
        # For batched input
        if len(x.shape) > 1:
            position = x[:, 0]
            velocity = x[:, 1]
            
            # Simple damped oscillator: mx'' + cx' + kx = 0 (m=1)
            acceleration = -self.stiffness * position - self.damping * velocity
            
            return np.stack([velocity, acceleration], axis=1)
        else:
            # For single input
            position, velocity = x
            
            # Simple damped oscillator: mx'' + cx' + kx = 0 (m=1)
            acceleration = -self.stiffness * position - self.damping * velocity
            
            return np.array([velocity, acceleration])
    
    def is_stable(self):
        """
        Check if the system is stable.
        
        Returns:
            True if the system is stable, False otherwise
        """
        # A simple damped oscillator is stable if damping > 0
        return self.damping > 0


def create_unstable_lyapunov_net():
    """
    Create a neural Lyapunov function that is unstable.
    
    Returns:
        LyapunovNet instance
    """
    # Create model with normal initialization
    model = LyapunovNet(
        dim=2,
        hidden_dims=(32, 32),
        alpha=0.001,
        activation=nn.Tanh()
    )
    
    # Explicitly make it unstable by setting all weights negative
    with torch.no_grad():
        for param in model.parameters():
            param.data = -torch.abs(param.data)
    
    model.id = "unstable_net"
    
    return model


def create_stable_lyapunov_net():
    """
    Create a neural Lyapunov function that is stable.
    
    Returns:
        LyapunovNet instance
    """
    # Create model with normal initialization
    model = LyapunovNet(
        dim=2,
        hidden_dims=(32, 32),
        alpha=0.1,  # Larger alpha ensures positive definiteness
        activation=nn.Tanh()
    )
    
    model.id = "stable_net"
    
    return model


def demonstrate_stability_agent():
    """
    Demonstrate the StabilityAgent functionality.
    
    This function shows:
    1. Creating a StabilityAgent
    2. Verifying stability properties
    3. Logging interactions
    4. Handling verification errors
    5. Parameter tuning
    """
    # Create a temporary directory for cache
    cache_dir = pathlib.Path("temp_cache")
    cache_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print(" ELFIN StabilityAgent Demo ".center(80, "="))
    print("="*80 + "\n")
    
    # Create a StabilityAgent
    print("Creating StabilityAgent...")
    agent = StabilityAgent("demo_agent", cache_dir)
    
    # ==== Scenario 1: Verify a stable system ====
    print("\n[Scenario 1] Verifying a stable system")
    print("-" * 50)
    
    # Create a stable oscillator
    system = SimpleOscillator(damping=0.5, stiffness=1.0)
    
    # Create a stable Lyapunov function
    lyap_net = create_stable_lyapunov_net()
    
    # Define verification domain
    domain = (np.array([-2.0, -2.0]), np.array([2.0, 2.0]))
    
    # Verify positive definiteness
    print("Verifying positive definiteness...")
    try:
        result = agent.verify(lyap_net, domain)
        
        if result["status"] == "VERIFIED":
            print(f"✅ System verified!")
            print(f"  Solve time: {result['solve_time']:.2f}s")
        else:
            print(f"⚠️ Verification failed!")
            if result["counterexample"] is not None:
                print(f"  Counterexample: x={np.round(result['counterexample'], 3)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Verify decrease condition
    print("\nVerifying decrease condition...")
    try:
        result = agent.verify_decrease(lyap_net, system.dynamics, domain, gamma=0.1)
        
        if result["status"] == "VERIFIED":
            print(f"✅ Decrease verified!")
        else:
            print(f"⚠️ Verification failed!")
            if result["counterexample"] is not None:
                print(f"  Counterexample: x={np.round(result['counterexample'], 3)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ==== Scenario 2: Verify an unstable system ====
    print("\n[Scenario 2] Verifying an unstable system")
    print("-" * 50)
    
    # Create an unstable Lyapunov function
    unstable_net = create_unstable_lyapunov_net()
    
    # Verify positive definiteness
    print("Verifying positive definiteness...")
    try:
        result = agent.verify(unstable_net, domain)
        
        if result["status"] == "VERIFIED":
            print(f"✅ System verified!")
        else:
            print(f"⚠️ Verification failed!")
            if result["counterexample"] is not None:
                print(f"  Counterexample: x={np.round(result['counterexample'], 3)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ==== Scenario 3: Parameter tuning ====
    print("\n[Scenario 3] Parameter tuning")
    print("-" * 50)
    
    # Tune parameter
    print("Tuning damping parameter...")
    try:
        old_value = system.damping
        new_value = 0.05
        
        result = agent.param_tune(system, "damping", old_value, new_value)
        
        print(f"✅ Parameter tuned: damping = {old_value} → {new_value}")
        
        # Verify decrease with new parameter
        print("\nVerifying decrease condition with new parameter...")
        result = agent.verify_decrease(lyap_net, system.dynamics, domain, gamma=0.05)
        
        if result["status"] == "VERIFIED":
            print(f"✅ Decrease verified!")
        else:
            print(f"⚠️ Verification failed!")
            if result["counterexample"] is not None:
                print(f"  Counterexample: x={np.round(result['counterexample'], 3)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ==== Scenario 4: View interaction log ====
    print("\n[Scenario 4] View interaction log")
    print("-" * 50)
    
    # Get log summary
    summary = agent.get_summary()
    print(summary)
    
    # ==== Cleanup ====
    print("\nCleaning up...")
    import shutil
    shutil.rmtree(cache_dir)
    
    print("\n" + "="*80)
    print(" Demo Complete ".center(80, "="))
    print("="*80 + "\n")


if __name__ == "__main__":
    demonstrate_stability_agent()
