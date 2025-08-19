"""
Stability Reasoning Factory - Integration with Spectral Monitoring

This module provides factory functions to create and configure StabilityReasoning
instances that are integrated with the Spectral Monitor system. It serves as the
integration point between spectral monitoring and stability reasoning.
"""

import logging
import os
from typing import Dict, Any, Optional, Union

try:
    # Try absolute import first
    from stability_reasoning import StabilityReasoning, default_stability_reasoning
except ImportError:
    # Fallback to relative import
    from .stability_reasoning import StabilityReasoning, default_stability_reasoning
try:
    # Try absolute import first
    from concept_metadata import ConceptMetadata
except ImportError:
    # Fallback to relative import
    from .concept_metadata import ConceptMetadata
try:
    # Try absolute import first
    from time_context import TimeContext, default_time_context
except ImportError:
    # Fallback to relative import
    from .time_context import TimeContext, default_time_context
try:
    # Try absolute import first
    from concept_logger import ConceptLogger, default_concept_logger
except ImportError:
    # Fallback to relative import
    from .concept_logger import ConceptLogger, default_concept_logger

# Import from runtime bridge
from packages.runtime_bridge.python.stability_bridge import (
    default_stability_bridge,
    update_stability_reasoning_context
)

# Configure logger
logger = logging.getLogger("stability_reasoning_factory")

def create_integrated_stability_reasoning(
    concept_store: Optional[Dict[str, Union[ConceptMetadata, Any]]] = None,
    time_context: Optional[TimeContext] = None,
    logger: Optional[ConceptLogger] = None,
    config: Optional[Dict[str, Any]] = None
) -> StabilityReasoning:
    """
    Create a StabilityReasoning instance integrated with spectral monitoring.
    
    This factory function creates a StabilityReasoning instance that is properly
    configured to work with the spectral monitor, with coherence break detection
    and automatic stability updates.
    
    Args:
        concept_store: Dictionary mapping concept IDs to metadata
                      If None, a new empty store will be created
        time_context: TimeContext for temporal awareness
                      If None, the default time context will be used
        logger: ConceptLogger for logging events
                If None, the default concept logger will be used
        config: Optional configuration parameters
                - stability_threshold: Minimum stability for concepts (default: 0.4)
                - coherence_threshold: Minimum coherence for reasoning (default: 0.6)
                - phase_noise: Phase noise level (default: 0.1)
    
    Returns:
        Properly configured StabilityReasoning instance
    """
    # Get configuration values from environment variables or config parameter
    cfg = config or {}
    
    # Import os here to read environment variables
    import os
    
    # Stability thresholds from environment with fallbacks to config values or defaults
    stability_threshold = float(os.environ.get('STABILITY_THRESHOLD', 
                                              str(cfg.get('stability_threshold', 0.4))))
    
    coherence_threshold = float(os.environ.get('STABILITY_COHERENCE_THRESH', 
                                              str(cfg.get('coherence_threshold', 0.65))))
    
    phase_noise = float(os.environ.get('STABILITY_PHASE_NOISE', 
                                      str(cfg.get('phase_noise', 0.1))))
    
    # Log the configuration being used
    logger.info(f"Creating stability reasoning with thresholds: "
                f"stability={stability_threshold:.2f}, "
                f"coherence={coherence_threshold:.2f}, "
                f"phase_noise={phase_noise:.2f}")
    
    # Use provided objects or defaults
    ctx = time_context or default_time_context
    log = logger or default_concept_logger
    store = concept_store or {}
    
    # Create StabilityReasoning instance
    sr = StabilityReasoning(
        concept_store=store,
        time_context=ctx,
        logger=log,
        stability_threshold=stability_threshold,
        coherence_threshold=coherence_threshold
    )
    
    # Register for coherence break events from stability bridge
    default_stability_bridge.on_coherence_break(_create_coherence_break_handler(sr))
    
    # Register for regular updates
    default_stability_bridge.on_update(lambda: _update_stability_context(sr))
    
    # Perform initial update from current spectral state
    try:
        update_stability_reasoning_context(sr)
    except Exception as e:
        logger.warning(f"Could not perform initial spectral update: {e}")
    
    return sr

def _create_coherence_break_handler(sr: StabilityReasoning):
    """Create a handler for coherence break events."""
    def handle_coherence_break(break_info):
        """Handle coherence break events by recording desync."""
        logger.warning(f"Processing coherence break: {break_info['coherence']:.2f}")
        
        # Record desync events for drifting concepts
        for concept_id in break_info["driftingConcepts"]:
            if hasattr(sr, 'record_desync_event'):
                sr.record_desync_event(concept_id, break_info["coherence"])
                logger.info(f"Recorded desync for concept: {concept_id}")
    
    return handle_coherence_break

def _update_stability_context(sr: StabilityReasoning):
    """Update stability context with latest spectral state."""
    try:
        result = update_stability_reasoning_context(sr)
        logger.debug(f"Updated {result['conceptsUpdated']} concepts, {result['coherenceBreaks']} coherence breaks")
    except Exception as e:
        logger.error(f"Error updating stability context: {e}")

def ensure_stability_bridge_running():
    """
    Ensure the stability bridge is running.
    
    This function starts the stability bridge if it's not already running.
    It's useful to call this when initializing the application.
    
    Returns:
        True if bridge is running, False if it failed to start
    """
    from packages.runtime_bridge.python.stability_bridge import start_stability_bridge
    
    # Get configuration from environment variables
    ws_endpoint = os.environ.get("SPECTRAL_WS_ENDPOINT", None)
    sync_interval = float(os.environ.get("SPECTRAL_SYNC_INTERVAL", "5.0"))
    
    # Start the bridge
    try:
        return start_stability_bridge(
            websocket_endpoint=ws_endpoint,
            sync_interval=sync_interval
        )
    except Exception as e:
        logger.error(f"Failed to start stability bridge: {e}")
        return False

# Initialize stability bridge when this module is imported
try:
    ensure_stability_bridge_running()
except Exception as e:
    logger.warning(f"Failed to initialize stability bridge: {e}")
