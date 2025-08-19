"""
Core components for the ELFIN stability framework.

This package contains foundational components used throughout the
stability framework, including interaction tracking, event systems,
and other shared utilities.
"""

from .interactions import Interaction, InteractionLog

__all__ = ["Interaction", "InteractionLog"]
