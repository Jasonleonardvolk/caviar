"""
ELFIN Debug Module

This module provides debugging tools for ELFIN controllers, including:
- Lyapunov stability monitoring
- Barrier function verification
- Unit checking and automatic fixes
- Integration with TORI IDE and VS Code
"""

from alan_backend.elfin.debug.lyapunov_monitor import lyapunov_monitor
from alan_backend.elfin.debug.unit_checker import unit_checker
from alan_backend.elfin.debug.bridge import ElfinDebugBridge

__all__ = [
    'lyapunov_monitor',
    'unit_checker',
    'ElfinDebugBridge'
]
