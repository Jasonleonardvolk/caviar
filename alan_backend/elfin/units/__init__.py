"""
ELFIN unit annotation and dimensional analysis system.

This module provides support for adding physical dimensions to ELFIN variables 
and parameters, tracking them through expressions, and verifying dimensional
consistency.
"""

# Import core components
from .units import Unit, UnitDimension, UnitTable
from .checker import DimensionChecker, DimensionError
from .unit_expr import (UnitExpr, BaseUnit, MulUnit, DivUnit, PowUnit, 
                        parse_unit_expr)
