"""
Abstract Syntax Tree (AST) nodes and utilities for the ELFIN compiler.

This package provides classes for representing ELFIN programs as an AST,
along with utilities for transforming and traversing the AST.
"""

from typing import Dict, List, Any, Optional, Union

# Import these explicitly to ensure they're available to other modules
from .nodes import (
    Node, Program, ImportStmt, HelpersSection, HelperFunction,
    SystemSection, LyapunovSection, BarrierSection, ModeSection,
    PlannerSection, IntegrationSection, Expression, VarRef, Number
)

from .megatron import ELFINMegatron

# Re-export as a single namespace for easier imports
__all__ = [
    'Node', 'Program', 'ImportStmt', 'HelpersSection', 'HelperFunction',
    'SystemSection', 'LyapunovSection', 'BarrierSection', 'ModeSection',
    'PlannerSection', 'IntegrationSection', 'Expression', 'VarRef', 'Number',
    'ELFINMegatron'
]
