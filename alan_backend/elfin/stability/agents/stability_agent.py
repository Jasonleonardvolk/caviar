"""
StabilityAgent implementation for the ELFIN stability framework.

This module provides a wrapper around verification components that
adds interaction logging, event emission, and error handling.
"""

import json
import logging
import pathlib
import traceback
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from ..core.interactions import Interaction, InteractionLog
from ..verify import MILPVerifier, DepGraph, ParallelVerifier

# Configure logging
logger = logging.getLogger(__name__)

# Optional import for event handling
try:
    from alan_backend.elfin.events import get_event_bus, publish_event
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False
    logger.warning("Event bus not available, falling back to logging only")

class VerificationError:
    """
    Standardized error format for verification failures.
    
    This class creates a structured error object with a code, message,
    and reference to the interaction that caused the error.
    """
    ERROR_CODES = {
        "LYAP_001": "Lyapunov function not positive definite",
        "LYAP_002": "Lyapunov function not decreasing",
        "VERIF_001": "Verification failed due to solver error",
        "VERIF_002": "Verification timeout",
        "PARAM_001": "Invalid parameter values",
    }
    
    def __init__(
        self,
        code: str,
        detail: str,
        system_id: str,
        interaction_ref: str,
        **extra_fields
    ):
        """
        Initialize a verification error.
        
        Args:
            code: Error code (e.g., "LYAP_001")
            detail: Detailed error message
            system_id: ID of the system being verified
            interaction_ref: Reference to the interaction that caused the error
            **extra_fields: Additional fields to include in the error
        """
        self.code = code
        self.title = self.ERROR_CODES.get(code, "Unknown error")
        self.detail = detail
        self.system_id = system_id
        self.interaction_ref = interaction_ref
        self.extra_fields = extra_fields
        self.doc_url = f"https://elfin.dev/errors/{code}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to a dictionary."""
        error_dict = {
            "code": f"E-{self.code}",
            "title": self.title,
            "detail": self.detail,
            "system_id": self.system_id,
            "doc": self.doc_url,
            "interaction_ref": self.interaction_ref
        }
        error_dict.update(self.extra_fields)
        return error_dict
    
    def to_json(self) -> str:
        """Convert error to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def __str__(self) -> str:
        """Convert error to a string."""
        return f"E-{self.code}: {self.detail} (see {self.doc_url})"


# Create global dependency graph for the verification system
dep_graph = DepGraph()

class StabilityAgent:
    """
    Agent for stability verification that tracks all interactions.
    
    This class wraps verification components and adds interaction logging,
    event emission, and error handling.
    
    Attributes:
        name: Name of the agent
        cache_db: Path to the cache file_storage
        log: Interaction log
        dep_graph: Dependency graph for tracking dependencies between entities and proofs
        parallel_verifier: Parallel verifier for distributing jobs across CPU cores
    """
    
    def __init__(
        self,
        name: str,
        cache_db: Union[str, pathlib.Path],
        verifier_cls: Type = MILPVerifier,
        event_bus: Any = None,
        max_workers: Optional[int] = None
    ):
        """
        Initialize a stability agent.
        
        Args:
            name: Name of the agent
            cache_db: Path to the cache file_storage
            verifier_cls: Verifier class to use
            event_bus: Event bus instance (if None, will use global bus if available)
            max_workers: Maximum number of worker processes for parallel verification
        """
        self.name = name
        self.cache_db = pathlib.Path(cache_db)
        self.verifier_cls = verifier_cls
        
        # Set up event bus
        if event_bus is not None:
            self.event_bus = event_bus
        elif EVENT_BUS_AVAILABLE:
            self.event_bus = get_event_bus()
        else:
            self.event_bus = None
        
        # Initialize interaction log
        self.log_path = self.cache_db / f"{self.name}.log.jsonl"
        self.log = InteractionLog.load(self.log_path) if self.log_path.exists() else InteractionLog()
        
        # Initialize parallel verification components
        self.dep_graph = dep_graph  # Use the global dependency graph
        self.parallel_verifier = ParallelVerifier(self.dep_graph, max_workers)
        
        logger.info(f"Initialized StabilityAgent '{name}' with cache at {cache_db}")
    
    def verify(
        self,
        system,
        domain,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Verify a system's stability properties.
        
        Args:
            system: System to verify
            domain: Verification domain
            **kwargs: Additional arguments for the verifier
            
        Returns:
            Result dictionary
        """
        # Create interaction
        system_id = getattr(system, "id", str(id(system)))
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
                "solve_time": getattr(verifier, "verification_time", 0.0),
                "counterexample": result.counterexample.tolist() if result.counterexample is not None else None,
                "proof_hash": getattr(result, "proof_hash", None)
            }
            
            # Emit events
            if result.success:
                self._emit("proof_added", interaction.result)
            else:
                self._emit("counterexample", {
                    "system_id": system_id,
                    "counterexample": interaction.result["counterexample"]
                })
                
                # Create error object for counterexample
                if interaction.result["counterexample"] is not None:
                    error = VerificationError(
                        code="LYAP_001",
                        detail=f"Function not positive definite at x={interaction.result['counterexample']}",
                        system_id=system_id,
                        interaction_ref=interaction.get_reference()
                    )
                    logger.warning(str(error))
        
        except Exception as e:
            # Log exception
            interaction.result = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
            # Emit error event
            self._emit("verification_error", {
                "system_id": system_id,
                "error": str(e)
            })
            
            # Create error object
            error = VerificationError(
                code="VERIF_001",
                detail=str(e),
                system_id=system_id,
                interaction_ref=interaction.get_reference(),
                traceback=traceback.format_exc()
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
        """
        Verify that a system's Lyapunov function is decreasing along trajectories.
        
        Args:
            system: System to verify
            dynamics_fn: Dynamics function
            domain: Verification domain
            gamma: Margin for decrease condition
            **kwargs: Additional arguments for the verifier
            
        Returns:
            Result dictionary
        """
        # Create interaction
        system_id = getattr(system, "id", str(id(system)))
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
                "solve_time": getattr(verifier, "verification_time", 0.0),
                "counterexample": result.tolist() if result is not None else None,
                "gamma": gamma
            }
            
            # Emit events
            if result is None:
                self._emit("decrease_verified", {
                    "system_id": system_id,
                    "gamma": gamma
                })
            else:
                self._emit("decrease_violation", {
                    "system_id": system_id,
                    "counterexample": interaction.result["counterexample"],
                    "gamma": gamma
                })
                
                # Create error object for counterexample
                error = VerificationError(
                    code="LYAP_002",
                    detail=f"Function not decreasing at x={interaction.result['counterexample']}",
                    system_id=system_id,
                    interaction_ref=interaction.get_reference(),
                    gamma=gamma
                )
                logger.warning(str(error))
        
        except Exception as e:
            # Log exception
            interaction.result = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
            # Emit error event
            self._emit("verification_error", {
                "system_id": system_id,
                "error": str(e)
            })
            
            # Create error object
            error = VerificationError(
                code="VERIF_001",
                detail=str(e),
                system_id=system_id,
                interaction_ref=interaction.get_reference(),
                traceback=traceback.format_exc()
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
        """
        Record a parameter tuning action.
        
        Args:
            system: System being tuned
            param_name: Name of the parameter
            old_value: Old value of the parameter
            new_value: New value of the parameter
            
        Returns:
            Result dictionary
        """
        # Create interaction
        system_id = getattr(system, "id", str(id(system)))
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
        
        # Emit event
        self._emit("param_tuned", {
            "system_id": system_id,
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value
        })
        
        # Persist interaction
        self._append_and_persist(interaction)
        
        return interaction.result
    
    def _emit(self, topic: str, payload: Dict[str, Any]) -> None:
        """
        Emit an event.
        
        Args:
            topic: Event topic
            payload: Event payload
        """
        # Update dependency graph based on events
        if topic == "proof_added" and "proof_hash" in payload:
            self.dep_graph.mark_fresh(payload["proof_hash"])
            logger.debug(f"Marked proof {payload['proof_hash']} as fresh in dependency graph")
            
        # Optional: Add counterexamples to a sampler for training
        # elif topic == "counterexample" and "counterexample" in payload:
        #     sampler.add_counterexamples([np.array(payload["counterexample"])])
        
        # Emit the event
        if self.event_bus is not None:
            try:
                publish_event(self.event_bus, topic, payload)
            except Exception as e:
                logger.warning(f"Failed to publish event: {e}")
        else:
            logger.debug(f"Event [{topic}]: {payload}")
    
    def _append_and_persist(self, interaction: Interaction) -> None:
        """
        Add an interaction to the log and persist it.
        
        Args:
            interaction: Interaction to add
        """
        self.log.append_and_persist(interaction, self.log_path)
        logger.debug(f"Recorded interaction: {interaction.action} ({interaction.get_reference()})")
    
    def get_log(self, tail: Optional[int] = None, **filters) -> InteractionLog:
        """
        Get the interaction log, optionally filtered.
        
        Args:
            tail: If specified, only return the last N interactions
            **filters: Filters to apply to the log
            
        Returns:
            Filtered interaction log
        """
        filtered_log = self.log.filter(**filters)
        
        if tail is not None:
            filtered_log = filtered_log.tail(tail)
        
        return filtered_log
    
    def verify_many(
        self,
        systems: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Verify multiple systems in parallel.
        
        This method uses the parallel verifier to verify multiple systems
        in parallel, skipping proofs that are already verified.
        
        Args:
            systems: List of verification jobs, each with system, domain, hash, and entities
            
        Returns:
            Dictionary mapping proof hash to verification result
        """
        # Create interaction for the batch verification
        interaction = Interaction.now(
            "verify_many",
            system_count=len(systems),
            systems=[s.get("system_id", str(id(s.get("system")))) for s in systems]
        )
        
        # Track results
        results = {}
        
        try:
            # Define callback function for results
            def on_done(result: Dict[str, Any]) -> None:
                """Process a verification result."""
                results[result["hash"]] = result
                
                # Log the result
                self._emit("parallel_verify", result)
                
                # Create an interaction for the individual verification
                system_id = next((s.get("system_id", str(id(s.get("system")))) 
                                for s in systems if s["hash"] == result["hash"]), "unknown")
                
                individual_interaction = Interaction.now(
                    "parallel_verify",
                    system_id=system_id,
                    proof_hash=result["hash"]
                )
                individual_interaction.result = result
                self._append_and_persist(individual_interaction)
            
            # Run verification in parallel
            processed = self.parallel_verifier.verify_many(systems, on_done)
            
            # Update interaction result
            interaction.result = {
                "processed_count": len(processed),
                "total_count": len(systems),
                "skipped_count": len(systems) - len(processed),
                "verification_time": sum(r.get("solve_time", 0.0) for r in results.values())
            }
            
        except Exception as e:
            # Log exception
            interaction.result = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
            # Emit error event
            self._emit("verification_error", {
                "error": str(e)
            })
            
            # Re-raise
            raise
            
        finally:
            # Persist interaction
            self._append_and_persist(interaction)
        
        return results
        
    def mark_entity_dirty(self, entity_id: str) -> None:
        """
        Mark an entity as dirty in the dependency graph.
        
        This will cause all proofs that depend on this entity to be
        re-verified the next time verify_many is called.
        
        Args:
            entity_id: ID of the entity that has changed
        """
        self.dep_graph.mark_dirty(entity_id)
        logger.debug(f"Marked entity {entity_id} as dirty in dependency graph")
    
    def get_summary(self, tail: Optional[int] = None) -> str:
        """
        Get a human-readable summary of the interaction log.
        
        Args:
            tail: If specified, only summarize the last N interactions
            
        Returns:
            Summary string
        """
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
            elif interaction.action == "verify_many" and interaction.result:
                if "error" not in interaction.result:
                    processed = interaction.result.get("processed_count", 0)
                    total = interaction.result.get("total_count", 0)
                    skipped = interaction.result.get("skipped_count", 0)
                    solve_time = interaction.result.get("verification_time", 0.0)
                    details = f"processed={processed}/{total} skipped={skipped} solve={solve_time:.1f}s"
            elif interaction.action == "parallel_verify" and interaction.result:
                if interaction.result.get("counterexample"):
                    details = f"x={np.round(interaction.result['counterexample'], 3)}"
                else:
                    solve_time = interaction.result.get("solve_time", 0.0)
                    details = f"solve={solve_time:.1f}s proof={interaction.result.get('hash', '')[:6]}..."
            elif interaction.action == "param_tune" and interaction.result:
                details = f"{interaction.result.get('param_name')} set {interaction.result.get('old_value')} → {interaction.result.get('new_value')}"
            
            line = f"[{timestamp}] {action} {status}  {details}"
            lines.append(line)
        
        return "\n".join(lines)
