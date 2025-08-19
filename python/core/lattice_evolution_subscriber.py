"""Subscribe lattice evolution to concept events."""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global oscillator count
oscillator_count = 0
oscillator_phases: Dict[str, float] = {}

def on_concept_added(event):
    """Handle concept_added events from FractalSolitonMemory."""
    global oscillator_count, oscillator_phases
    
    oscillator_count += 1
    oscillator_phases[event.concept_id] = event.phase
    
    logger.info(
        f"[lattice] oscillators={oscillator_count} "
        f"(added {event.concept_id} with phase {event.phase:.3f})"
    )

def setup_lattice_subscription():
    """Wire up the lattice to listen for concept events."""
    try:
        from python.core.fractal_soliton_events import concept_event_bus
        
        # Subscribe to concept_added events
        concept_event_bus.subscribe('concept_added', on_concept_added)
        
        logger.info("✅ Lattice subscribed to concept events")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup lattice subscription: {e}")
        return False

# Auto-setup on import
if __name__ != "__main__":
    setup_lattice_subscription()
