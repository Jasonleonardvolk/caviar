"""
Soliton API Router - Redirects to Production Implementation
This file now redirects to the production implementation
"""

# Import the production router
from .soliton_production import router

# Re-export for backward compatibility
__all__ = ['router']
