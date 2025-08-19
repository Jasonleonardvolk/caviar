"""
Component Registry - Track initialization state of all TORI components
Provides a centralized way to monitor when the system is truly ready
"""

import threading
import time
from typing import Dict, Set, Optional, Callable, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Tracks the initialization state of all TORI components.
    Components register themselves and report when ready.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def instance(cls) -> 'ComponentRegistry':
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        self._components: Dict[str, Dict] = {}
        self._required_components: Set[str] = set()
        self._ready_callbacks: List[Callable] = []
        self._start_time = time.time()
        
        # Define required components for full system readiness
        self._required_components = {
            'cognitive_engine',
            'memory_vault', 
            'concept_mesh',
            'mcp_metacognitive',
            'prajna_api',
            'soliton_memory',
            'oscillator_lattice'
        }
    
    def register_component(self, 
                         name: str, 
                         required: bool = True,
                         dependencies: Optional[Set[str]] = None) -> None:
        """Register a component that needs to be tracked"""
        with self._lock:
            self._components[name] = {
                'status': 'initializing',
                'required': required,
                'dependencies': dependencies or set(),
                'registered_at': datetime.now(),
                'ready_at': None,
                'error': None,
                'metadata': {}
            }
            
            if required:
                self._required_components.add(name)
                
            logger.info(f"📝 Registered component: {name} (required: {required})")
    
    def mark_ready(self, name: str, metadata: Optional[Dict] = None) -> None:
        """Mark a component as ready"""
        with self._lock:
            if name not in self._components:
                self.register_component(name, required=False)
            
            self._components[name]['status'] = 'ready'
            self._components[name]['ready_at'] = datetime.now()
            self._components[name]['metadata'] = metadata or {}
            
            init_time = time.time() - self._start_time
            logger.info(f"✅ Component ready: {name} (took {init_time:.1f}s)")
            
            # Check if all required components are ready
            if self.is_system_ready():
                self._trigger_ready_callbacks()
    
    def mark_failed(self, name: str, error: str) -> None:
        """Mark a component as failed"""
        with self._lock:
            if name not in self._components:
                self.register_component(name)
                
            self._components[name]['status'] = 'failed'
            self._components[name]['error'] = error
            
            logger.error(f"❌ Component failed: {name} - {error}")
    
    def is_component_ready(self, name: str) -> bool:
        """Check if a specific component is ready"""
        with self._lock:
            return (name in self._components and 
                    self._components[name]['status'] == 'ready')
    
    def is_system_ready(self) -> bool:
        """Check if all required components are ready"""
        with self._lock:
            for component in self._required_components:
                if component not in self._components:
                    return False
                    
                comp_data = self._components[component]
                
                # Check component status
                if comp_data['status'] != 'ready':
                    return False
                
                # Check dependencies
                for dep in comp_data['dependencies']:
                    if not self.is_component_ready(dep):
                        return False
            
            return True
    
    def get_readiness_report(self) -> Dict:
        """Get detailed readiness status of all components"""
        with self._lock:
            total_time = time.time() - self._start_time
            
            report = {
                'system_ready': self.is_system_ready(),
                'total_components': len(self._components),
                'ready_components': sum(1 for c in self._components.values() 
                                      if c['status'] == 'ready'),
                'failed_components': sum(1 for c in self._components.values() 
                                       if c['status'] == 'failed'),
                'initialization_time': total_time,
                'components': {}
            }
            
            # Add individual component status
            for name, data in self._components.items():
                init_time = None
                if data['ready_at'] and data['registered_at']:
                    delta = data['ready_at'] - data['registered_at']
                    init_time = delta.total_seconds()
                
                report['components'][name] = {
                    'status': data['status'],
                    'required': data['required'],
                    'initialization_time': init_time,
                    'error': data['error'],
                    'dependencies_met': all(
                        self.is_component_ready(dep) 
                        for dep in data['dependencies']
                    )
                }
            
            return report
    
    def wait_for_ready(self, timeout: float = 60.0, poll_interval: float = 0.5) -> bool:
        """
        Wait for system to be ready with timeout.
        Returns True if ready, False if timeout.
        """
        start = time.time()
        
        while time.time() - start < timeout:
            if self.is_system_ready():
                total_time = time.time() - self._start_time
                logger.info(f"🎉 System fully initialized in {total_time:.1f}s")
                return True
                
            # Log waiting status periodically
            if int(time.time() - start) % 5 == 0:
                report = self.get_readiness_report()
                waiting_for = [
                    name for name, data in report['components'].items()
                    if data['required'] and data['status'] != 'ready'
                ]
                if waiting_for:
                    logger.info(f"⏳ Still waiting for: {', '.join(waiting_for)}")
            
            time.sleep(poll_interval)
        
        # Timeout - log what's not ready
        report = self.get_readiness_report()
        not_ready = [
            f"{name} ({data['status']})"
            for name, data in report['components'].items()
            if data['required'] and data['status'] != 'ready'
        ]
        
        logger.warning(f"⏰ Initialization timeout! Not ready: {', '.join(not_ready)}")
        return False
    
    def on_ready(self, callback: Callable) -> None:
        """Register a callback to be called when system is ready"""
        with self._lock:
            self._ready_callbacks.append(callback)
            # If already ready, call immediately
            if self.is_system_ready():
                callback()
    
    def _trigger_ready_callbacks(self) -> None:
        """Trigger all registered ready callbacks"""
        for callback in self._ready_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in ready callback: {e}")
    
    def reset(self) -> None:
        """Reset the registry (mainly for testing)"""
        with self._lock:
            self._components.clear()
            self._ready_callbacks.clear()
            self._start_time = time.time()


# Global registry instance
component_registry = ComponentRegistry.instance()


# Decorator for auto-registration
def track_initialization(component_name: str, 
                        required: bool = True,
                        dependencies: Optional[Set[str]] = None):
    """
    Decorator to automatically track component initialization.
    
    Usage:
        @track_initialization('my_component')
        class MyComponent:
            def __init__(self):
                # ... initialization code ...
                component_registry.mark_ready('my_component')
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            # Register component
            component_registry.register_component(
                component_name, 
                required=required,
                dependencies=dependencies
            )
            
            try:
                # Call original init
                result = original_init(self, *args, **kwargs)
                # Auto-mark ready if no exception
                component_registry.mark_ready(component_name)
                return result
            except Exception as e:
                # Mark failed on exception
                component_registry.mark_failed(component_name, str(e))
                raise
        
        cls.__init__ = new_init
        return cls
    
    return decorator
