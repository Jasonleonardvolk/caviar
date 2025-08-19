"""
Agent-based components for the ELFIN stability framework.

This package contains agent implementations that wrap verification and
training components with interaction logging, event emission, and other
agent-like capabilities.
"""

from .stability_agent import StabilityAgent, VerificationError

__all__ = ["StabilityAgent", "VerificationError"]
