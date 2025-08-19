"""
Integration of error handling with the StabilityAgent.

This module provides utilities for integrating error handling with
the StabilityAgent class, including error formatting and event handling.
"""

import logging
import pathlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from alan_backend.elfin.errors import ErrorHandler, VerificationError
from alan_backend.elfin.stability.core.interactions import InteractionLog

logger = logging.getLogger(__name__)


class ErrorIntegration:
    """
    Integrates error handling with StabilityAgent.
    
    This class provides utilities for handling verification errors
    in the StabilityAgent, including error formatting and tracking.
    """
    
    def __init__(self, agent_name: str, interaction_log: InteractionLog):
        """
        Initialize error integration.
        
        Args:
            agent_name: Name of the agent
            interaction_log: Interaction log for the agent
        """
        self.agent_name = agent_name
        self.interaction_log = interaction_log
        self.error_handler = ErrorHandler()
    
    def handle_verification_error(
        self, 
        error: VerificationError, 
        system_id: str,
        job: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle verification error.
        
        This method formats a verification error into a standardized
        result format and logs it to the interaction log.
        
        Args:
            error: Verification error
            system_id: ID of the system being verified
            job: Verification job that caused the error
            
        Returns:
            Standardized error result
        """
        # Create interaction for this error
        interaction_id = self.interaction_log.add_interaction(
            topic="verification_error",
            source="verification_engine",
            target=system_id,
            payload={
                "error": error.to_dict(),
                "job": job
            }
        )
        
        # Format result
        result = {
            "status": "ERROR",
            "error_code": f"E-{error.code}",
            "error_title": error.title,
            "error_detail": error.detail,
            "system_id": error.system_id,
            "interaction_ref": interaction_id,
            "doc_url": error.doc_url
        }
        
        # Add counterexample if available
        if "counterexample" in error.extra_fields:
            result["counterexample"] = error.extra_fields["counterexample"]
        
        return result
    
    def handle_counterexample(
        self, 
        system_id: str,
        counterexample: np.ndarray,
        trainer_ref: Any = None,
        job: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Handle verification counterexample.
        
        This method logs a counterexample to the interaction log
        and optionally passes it to a trainer for refinement.
        
        Args:
            system_id: ID of the system being verified
            counterexample: Counterexample point as a numpy array
            trainer_ref: Reference to a trainer (optional)
            job: Verification job that produced the counterexample (optional)
            
        Returns:
            Interaction ID
        """
        # Convert counterexample to a regular Python list for JSON serialization
        ce_list = counterexample.tolist()
        
        # Create interaction for this counterexample
        payload = {
            "point": ce_list,
            "system_id": system_id
        }
        
        if job:
            payload["job"] = job
        
        interaction_id = self.interaction_log.add_interaction(
            topic="counterexample",
            source="verification_engine",
            target=system_id,
            payload=payload
        )
        
        # If a trainer is provided, pass the counterexample to it
        if trainer_ref is not None:
            self._add_to_trainer(trainer_ref, counterexample)
            
            # Log this action
            self.interaction_log.add_interaction(
                topic="auto_retrain_triggered",
                source=self.agent_name,
                target=system_id,
                payload={
                    "counterexample": ce_list,
                    "origin_interaction": interaction_id
                }
            )
        
        return interaction_id
    
    def _add_to_trainer(self, trainer_ref: Any, counterexample: np.ndarray) -> None:
        """
        Add a counterexample to a trainer.
        
        Args:
            trainer_ref: Reference to a trainer
            counterexample: Counterexample point as a numpy array
        """
        # This is a simplified version. In a real implementation,
        # this would add the counterexample to the trainer's data set.
        # Note that dependency graph integration would happen here too.
        try:
            # Add the counterexample to the trainer
            trainer_ref.add_counterexamples([counterexample])
            
            # In the final implementation, we would also mark proofs
            # as dirty in the dependency graph
            # dep_graph.mark_dirty(system_id)
            
            logger.info(f"Added counterexample to trainer: {counterexample}")
        except Exception as e:
            logger.error(f"Failed to add counterexample to trainer: {e}")
            
    def format_error(
        self, 
        error_code: str,
        detail: str,
        system_id: str,
        counterexample: Optional[np.ndarray] = None,
        **extra_fields
    ) -> VerificationError:
        """
        Format an error with a standardized structure.
        
        Args:
            error_code: Error code (e.g., "LYAP_001")
            detail: Detailed error message
            system_id: ID of the system being verified
            counterexample: Counterexample point (optional)
            **extra_fields: Additional fields to include in the error
            
        Returns:
            Formatted VerificationError
        """
        # Create additional fields
        fields = dict(extra_fields)
        
        if counterexample is not None:
            fields["counterexample"] = counterexample.tolist()
        
        # Create verification error
        error = VerificationError(
            code=error_code,
            detail=detail,
            system_id=system_id,
            **fields
        )
        
        return error
    
    def get_error_doc(self, error_code: str) -> Optional[str]:
        """
        Get documentation for an error code.
        
        Args:
            error_code: Error code (e.g., "E-LYAP-001" or "LYAP_001")
            
        Returns:
            Error documentation or None if not found
        """
        return self.error_handler.get_error_doc(error_code)


# Integration with StabilityAgent
def integrate_with_stability_agent(agent_class: type) -> type:
    """
    Integrate error handling with StabilityAgent class.
    
    This decorator adds error handling capabilities to the
    StabilityAgent class.
    
    Args:
        agent_class: StabilityAgent class to enhance
        
    Returns:
        Enhanced StabilityAgent class
    """
    original_init = agent_class.__init__
    original_verify = agent_class.verify
    
    def new_init(self, *args, **kwargs):
        """Enhanced initialization with error handling."""
        original_init(self, *args, **kwargs)
        self.error_integration = ErrorIntegration(self.name, self.interaction_log)
    
    def new_verify(self, system, domain, **kwargs):
        """Enhanced verify method with error handling."""
        try:
            return original_verify(self, system, domain, **kwargs)
        except VerificationError as e:
            # Extract job information
            job = {
                "system_id": e.system_id,
                "domain": domain,
                **kwargs
            }
            
            # Handle error
            result = self.error_integration.handle_verification_error(
                e, e.system_id, job
            )
            
            # If there's a counterexample, handle it
            if "counterexample" in result:
                trainer = getattr(self, "trainer", None)
                self.error_integration.handle_counterexample(
                    e.system_id, result["counterexample"], trainer, job
                )
            
            return result
    
    # Replace methods
    agent_class.__init__ = new_init
    agent_class.verify = new_verify
    
    # Add new methods
    agent_class.format_error = lambda self, *args, **kwargs: self.error_integration.format_error(*args, **kwargs)
    agent_class.get_error_doc = lambda self, error_code: self.error_integration.get_error_doc(error_code)
    
    return agent_class


# Usage example:
"""
from alan_backend.elfin.stability.agents import StabilityAgent
from alan_backend.elfin.stability.agents.error_integration import integrate_with_stability_agent

# Enhance StabilityAgent with error handling
@integrate_with_stability_agent
class EnhancedStabilityAgent(StabilityAgent):
    pass

# Create an enhanced agent
agent = EnhancedStabilityAgent("my_agent")

# Alternatively, enhance the class directly
integrate_with_stability_agent(StabilityAgent)
"""
