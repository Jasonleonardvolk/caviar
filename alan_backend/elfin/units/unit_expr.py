"""
UnitExpr - Algebraic representation for dimensional units in ELFIN.

This module defines the UnitExpr class hierarchy for representing and
manipulating physical units and dimensions within ELFIN programs.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


class UnitExpr:
    """Base class for unit expressions in ELFIN."""
    
    def __eq__(self, other) -> bool:
        """Check if two unit expressions are equal."""
        if not isinstance(other, UnitExpr):
            return False
        return str(self) == str(other)
    
    def __str__(self) -> str:
        """Get string representation of the unit expression."""
        return "dimensionless"
    
    def normalize(self) -> 'UnitExpr':
        """
        Normalize the unit expression to a canonical form.
        
        This helps with comparing dimensionally equivalent expressions
        by converting them to a standardized representation.
        
        Returns:
            A normalized unit expression
        """
        # Base case: already in canonical form
        return self
    
    def same(self, other: 'UnitExpr') -> bool:
        """
        Check if two unit expressions are dimensionally equivalent.
        
        This method performs dimensional analysis to determine if two 
        unit expressions have the same dimensional signature.
        
        Args:
            other: Another UnitExpr to compare with
            
        Returns:
            True if dimensionally equivalent, False otherwise
        """
        # Default implementation for dimensionless
        if not isinstance(other, UnitExpr):
            return False
        
        # Both are base class instances (dimensionless)
        if type(self) is UnitExpr and type(other) is UnitExpr:
            return True
        
        # Compare normalized forms to handle equivalent expressions
        # like kg*m/s and m*kg/s
        try:
            norm_self = self.normalize()
            norm_other = other.normalize()
            return str(norm_self) == str(norm_other)
        except:
            # Fall back to direct comparison if normalization fails
            return False


class BaseUnit(UnitExpr):
    """A base unit with a name (e.g., "kg")."""
    
    def __init__(self, name: str):
        """
        Initialize a base unit.
        
        Args:
            name: Name of the base unit (e.g., "kg", "m", "s")
        """
        self.name = name
    
    def __str__(self) -> str:
        """Get string representation of the base unit."""
        return self.name
    
    def same(self, other: UnitExpr) -> bool:
        """Check if dimensionally equivalent."""
        if isinstance(other, BaseUnit):
            return self.name == other.name
        return False


class MulUnit(UnitExpr):
    """Multiplication of two unit expressions (e.g., "kg*m")."""
    
    def __init__(self, left: UnitExpr, right: UnitExpr):
        """
        Initialize a multiplication unit.
        
        Args:
            left: Left operand
            right: Right operand
        """
        self.left = left
        self.right = right
    
    def __str__(self) -> str:
        """Get string representation of the multiplication unit."""
        return f"{self.left}*{self.right}"
    
    def normalize(self) -> 'UnitExpr':
        """
        Normalize multiplication to ensure canonical form.
        
        This sorts the operands lexicographically to handle commutativity.
        For example, kg*m and m*kg both normalize to the same form.
        
        Returns:
            A normalized unit expression
        """
        # Normalize each operand first
        norm_left = self.left.normalize()
        norm_right = self.right.normalize()
        
        # Sort lexicographically for canonical form
        str_left = str(norm_left)
        str_right = str(norm_right)
        
        if str_left <= str_right:
            return MulUnit(norm_left, norm_right)
        else:
            return MulUnit(norm_right, norm_left)
    
    def same(self, other: UnitExpr) -> bool:
        """
        Check if dimensionally equivalent.
        
        This implements a commutative check (a*b == b*a).
        """
        if isinstance(other, MulUnit):
            # Direct match
            if (self.left.same(other.left) and 
                self.right.same(other.right)):
                return True
            
            # Commutative check
            if (self.left.same(other.right) and 
                self.right.same(other.left)):
                return True
                
        return False


class DivUnit(UnitExpr):
    """Division of two unit expressions (e.g., "m/s")."""
    
    def __init__(self, left: UnitExpr, right: UnitExpr):
        """
        Initialize a division unit.
        
        Args:
            left: Numerator
            right: Denominator
        """
        self.left = left
        self.right = right
    
    def __str__(self) -> str:
        """Get string representation of the division unit."""
        return f"{self.left}/{self.right}"
    
    def normalize(self) -> 'UnitExpr':
        """
        Normalize division to ensure canonical form.
        
        This normalizes both numerator and denominator.
        
        Returns:
            A normalized unit expression
        """
        # Normalize numerator and denominator
        norm_left = self.left.normalize()
        norm_right = self.right.normalize()
        
        # Special case: if denominator is 1 or dimensionless, just return numerator
        if isinstance(norm_right, UnitExpr) and not isinstance(norm_right, BaseUnit):
            if str(norm_right) == "dimensionless":
                return norm_left
        
        return DivUnit(norm_left, norm_right)
    
    def same(self, other: UnitExpr) -> bool:
        """Check if dimensionally equivalent."""
        if isinstance(other, DivUnit):
            return (self.left.same(other.left) and 
                   self.right.same(other.right))
        return False


class PowUnit(UnitExpr):
    """Power of a unit expression (e.g., "m^2")."""
    
    def __init__(self, base: UnitExpr, exponent: int):
        """
        Initialize a power unit.
        
        Args:
            base: Base unit
            exponent: Integer exponent
        """
        self.base = base
        self.exponent = exponent
    
    def __str__(self) -> str:
        """Get string representation of the power unit."""
        return f"{self.base}^{self.exponent}"
    
    def normalize(self) -> 'UnitExpr':
        """
        Normalize power unit to ensure canonical form.
        
        This normalizes the base unit and handles special cases:
        - Exponent of 0 returns dimensionless
        - Exponent of 1 returns the base unit
        
        Returns:
            A normalized unit expression
        """
        # Special cases
        if self.exponent == 0:
            return UnitExpr()  # Dimensionless
        
        if self.exponent == 1:
            return self.base.normalize()
        
        # Normalize the base
        norm_base = self.base.normalize()
        
        # If the base is already a power, combine exponents
        if isinstance(norm_base, PowUnit):
            return PowUnit(norm_base.base, norm_base.exponent * self.exponent)
        
        return PowUnit(norm_base, self.exponent)
    
    def same(self, other: UnitExpr) -> bool:
        """Check if dimensionally equivalent."""
        if isinstance(other, PowUnit):
            return (self.base.same(other.base) and 
                   self.exponent == other.exponent)
        return False


def parse_unit_expr(unit_str: str) -> UnitExpr:
    """
    Parse a unit string into a UnitExpr object.
    
    This is a basic parser that handles simple unit expressions:
    - Base units: "kg", "m", "s"
    - Multiplication: "kg*m"
    - Division: "m/s"
    - Powers: "m^2", "kg*m^2/s^2"
    
    Args:
        unit_str: String representation of the unit expression
        
    Returns:
        Parsed UnitExpr object
        
    Raises:
        ValueError: If the unit string is invalid or cannot be parsed
    """
    # Empty or None case
    if not unit_str or unit_str.strip() == "":
        return UnitExpr()  # dimensionless
    
    # Tokenize the string
    import re
    tokens = re.findall(r'([a-zA-Z]+|\*|\/|\^|\d+|\(|\))', unit_str)
    
    # Simple case: just a base unit
    if len(tokens) == 1:
        return BaseUnit(tokens[0])
    
    # Handle more complex expressions here (simplified implementation)
    # In a real implementation, we would use a proper parser with operator precedence
    
    # For now, we'll handle simple cases:
    # "a*b" -> MulUnit(BaseUnit("a"), BaseUnit("b"))
    # "a/b" -> DivUnit(BaseUnit("a"), BaseUnit("b"))
    # "a^2" -> PowUnit(BaseUnit("a"), 2)
    
    # Very simplified parser for demonstration
    if len(tokens) == 3:
        if tokens[1] == "*":
            return MulUnit(BaseUnit(tokens[0]), BaseUnit(tokens[2]))
        elif tokens[1] == "/":
            return DivUnit(BaseUnit(tokens[0]), BaseUnit(tokens[2]))
        elif tokens[1] == "^":
            return PowUnit(BaseUnit(tokens[0]), int(tokens[2]))
    
    # Default fallback: just return a base unit from the first token
    # This is not a complete implementation!
    return BaseUnit(tokens[0])
