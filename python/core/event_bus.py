"""
Simple Event Bus for TORI
Provides publish/subscribe functionality for system events
"""

import logging
from typing import Dict, List, Callable, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class EventBus:
    """Simple synchronous event bus"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_count = 0
    
    def subscribe(self, event_name: str, callback: Callable):
        """Subscribe to an event"""
        self.subscribers[event_name].append(callback)
        logger.info(f"📌 Subscribed to '{event_name}' event")
    
    def publish(self, event_name: str, data: Any = None):
        """Publish an event to all subscribers"""
        callbacks = self.subscribers.get(event_name, [])
        self.event_count += 1
        
        if callbacks:
            logger.info(f"📡 Publishing '{event_name}' to {len(callbacks)} subscribers")
            for callback in callbacks:
                try:
                    callback(event_name, data)
                except Exception as e:
                    logger.error(f"❌ Error in event callback: {e}")
        else:
            logger.debug(f"📡 Event '{event_name}' published (no subscribers)")
    
    def clear(self):
        """Clear all subscriptions"""
        self.subscribers.clear()
        self.event_count = 0

# Global event bus instance
_event_bus = EventBus()

# Export convenience functions
def subscribe(event_name: str, callback: Callable):
    """Subscribe to an event on the global bus"""
    _event_bus.subscribe(event_name, callback)

def publish(event_name: str, data: Any = None):
    """Publish an event to the global bus"""
    _event_bus.publish(event_name, data)

def get_event_bus() -> EventBus:
    """Get the global event bus instance"""
    return _event_bus

__all__ = ['EventBus', 'subscribe', 'publish', 'get_event_bus']
