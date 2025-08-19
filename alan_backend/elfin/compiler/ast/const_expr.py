"""
Constant expression node for ELFIN AST.

This module defines a constant expression node that represents compile-time
evaluated expressions with associated unit information.
"""

from typing import Optional, Any
from alan_backend.elfin.compiler.ast.nodes import Expression, Node
from alan_backend.elfin.units.unit_expr import UnitExpr


class ConstExpr(Expression):
    """
    Constant expression node.
    
    This represents a compile-time evaluated expression with a value
    and optional dimensional information.
    """
    
    def __init__(self, value: Any, dim: Optional[UnitExpr] = None):
        """
        Initialize a constant expression.
        
        Args:
            value: The constant value
            dim: Optional dimensional information
        """
        super().__init__()
        self.value = value
        self.dim = dim
    
    def __str__(self) -> str:
        """
        Get string representation of the constant expression.
        
        Returns:
            A string representation
        """
        dim_str = f" [{self.dim}]" if self.dim else ""
        return f"{self.value}{dim_str}"
    
    def clone(self) -> 'ConstExpr':
        """
        Create a clone of this node.
        
        Returns:
            A deep copy of this node
        """
        return ConstExpr(self.value, self.dim)
