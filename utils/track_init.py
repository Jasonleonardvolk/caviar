# tori/utils/track_init.py
from functools import wraps
from .component_registry import ComponentRegistry

def track_initialization(name: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            ComponentRegistry().mark_ready(name)
            return result
        return wrapper
    return decorator
