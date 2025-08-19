"""Phase 7: Event system for FractalSolitonMemory."""
import asyncio
import logging
from typing import Callable, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ConceptEvent:
    """Event emitted when a concept is added/modified."""
    concept_id: str
    phase: float
    operation: str  # 'add', 'update', 'delete'
    timestamp: datetime
    metadata: Dict[str, Any] = None

class EventBus:
    """Simple event bus for concept events."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_queue: asyncio.Queue = None
        self._running = False
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.info(f"Subscribed to {event_type}: {callback.__name__}")
    
    async def emit(self, event_type: str, event: ConceptEvent):
        """Emit an event to all subscribers."""
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in event handler {callback.__name__}: {e}")
    
    def emit_sync(self, event_type: str, event: ConceptEvent):
        """Synchronous event emission."""
        asyncio.create_task(self.emit(event_type, event))

# Global event bus instance
concept_event_bus = EventBus()

# Monkey patch for FractalSolitonMemory
def patch_fractal_soliton_memory():
    """Add event emission to FractalSolitonMemory."""
    try:
        from python.core.fractal_soliton_memory import FractalSolitonMemory
        
        # Store original method
        original_add = FractalSolitonMemory.add_concept
        
        def add_concept_with_event(self, concept_id: str, phase: float = None, **kwargs):
            """Wrapped add_concept that emits events."""
            # Call original method
            result = original_add(self, concept_id, phase, **kwargs)
            
            # Emit event
            event = ConceptEvent(
                concept_id=concept_id,
                phase=phase or 0.0,
                operation='add',
                timestamp=datetime.utcnow(),
                metadata=kwargs
            )
            concept_event_bus.emit_sync('concept_added', event)
            logger.info(f"🌊 Emitted concept_added event: {concept_id} with phase {phase}")
            
            return result
        
        # Replace method
        FractalSolitonMemory.add_concept = add_concept_with_event
        logger.info("✅ Patched FractalSolitonMemory with event emission")
        
    except ImportError as e:
        logger.warning(f"Could not patch FractalSolitonMemory: {e}")

# Auto-patch on import
patch_fractal_soliton_memory()
