"""
Constant Folding Pass for ELFIN.

This module implements a compiler pass that evaluates constant expressions
at compile time and replaces them with pre-computed values with proper
dimension information.
"""

import math
import operator
from functools import lru_cache
from typing import Dict, Any, Optional, List, Tuple, Set, Union

from alan_backend.elfin.compiler.ast.nodes import (
    Node, Expression, BinaryOp, VarRef, Number, FunctionCall
)
from alan_backend.elfin.compiler.ast.const_expr import ConstExpr
from alan_backend.elfin.compiler.passes.dim_checker import Diagnostic
from alan_backend.elfin.units.unit_expr import UnitExpr, BaseUnit, MulUnit, DivUnit, PowUnit


class ConstantFolder:
    """
    Constant Folding Pass.
    
    This pass traverses the AST and evaluates constant expressions, replacing
    them with pre-computed ConstExpr nodes that include appropriate dimensional
    information.
    """
    
    def __init__(self):
        """Initialize the constant folder pass."""
        # Environment mapping variables to their constant values (if known)
        self.env: Dict[str, ConstExpr] = {}
        
        # Track diagnostics
        self.diagnostics: List[Diagnostic] = []
        
        # Binary operations and their implementations
        self.binary_ops = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '**': operator.pow,
            '^': operator.pow,  # ELFIN uses ^ for power
        }
        
        # Math functions and their implementations
        self.math_funcs = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'exp': math.exp,
            'log': math.log,
            'sqrt': math.sqrt,
            'abs': abs,
        }
    
    def process(self, node: Node, env: Optional[Dict[str, ConstExpr]] = None) -> Node:
        """
        Process a node and apply constant folding where possible.
        
        Args:
            node: The node to process
            env: Environment mapping variable names to constant values
            
        Returns:
            The processed node, possibly replaced with a ConstExpr
        """
        # Clear previous diagnostics
        self.diagnostics = []
        
        if env is not None:
            self.env = env
        
        # Different node types need different handling
        if isinstance(node, ConstExpr):
            # Already a constant expression
            return node
        elif isinstance(node, Number):
            # Convert numeric literals to ConstExpr
            return ConstExpr(node.value)
        elif isinstance(node, VarRef):
            # Look up variables in the environment
            if node.name in self.env:
                return self.env[node.name]
            return node
        elif isinstance(node, BinaryOp):
            return self._process_binary_op(node)
        elif isinstance(node, FunctionCall):
            return self._process_function_call(node)
        
        # For other node types, recursively process children
        self._process_children(node)
        return node
    
    def _process_binary_op(self, node: BinaryOp) -> Union[BinaryOp, ConstExpr]:
        """
        Process a binary operation for constant folding.
        
        Args:
            node: The binary operation node
            
        Returns:
            Either the original node or a folded ConstExpr
        """
        # Process operands first
        left = self.process(node.left)
        right = self.process(node.right)
        
        # If both operands are constants, compute the result
        if isinstance(left, ConstExpr) and isinstance(right, ConstExpr):
            op = node.op
            
            # Check if we can handle this operation
            if op in self.binary_ops:
                try:
                    # Compute the value
                    result_value = self.binary_ops[op](left.value, right.value)
                    
                    # Compute the dimension
                    result_dim = self._combine_dimensions(left.dim, right.dim, op)
                    
                    # Create a new constant expression
                    return ConstExpr(result_value, result_dim)
                except (ArithmeticError, ValueError):
                    # Fall back to the original node if computation fails
                    pass
        
        # If we couldn't fold, update the node with processed operands
        node.left = left
        node.right = right
        return node
    
    def _process_function_call(self, node: FunctionCall) -> Union[FunctionCall, ConstExpr]:
        """
        Process a function call for constant folding.
        
        Args:
            node: The function call node
            
        Returns:
            Either the original node or a folded ConstExpr
        """
        # Process arguments first
        processed_args = [self.process(arg) for arg in node.arguments]
        
        # Check if all arguments are constants
        if all(isinstance(arg, ConstExpr) for arg in processed_args) and node.function in self.math_funcs:
            try:
                # Extract values
                arg_values = [arg.value for arg in processed_args]
                
                # Compute the result
                result_value = self.math_funcs[node.function](*arg_values)
                
                # Most math functions preserve dimension in ELFIN
                # For specific functions, we would need custom dimension handling
                # For now, just return the result with no dimension
                return ConstExpr(result_value)
            except (ArithmeticError, ValueError):
                # Fall back to the original node if computation fails
                pass
        
        # If we couldn't fold, update the node with processed arguments
        node.arguments = processed_args
        return node
    
    def _process_children(self, node: Node) -> None:
        """
        Recursively process children of a node.
        
        Args:
            node: The parent node
        """
        # Check all attributes for child nodes
        for attr_name, attr_value in vars(node).items():
            # Skip private attributes
            if attr_name.startswith('_'):
                continue
                
            # Handle lists of nodes
            if isinstance(attr_value, list):
                for i, item in enumerate(attr_value):
                    if isinstance(item, Node):
                        attr_value[i] = self.process(item)
            
            # Handle dictionaries with node values
            elif isinstance(attr_value, dict):
                for key, value in attr_value.items():
                    if isinstance(value, Node):
                        attr_value[key] = self.process(value)
            
            # Handle direct node attributes
            elif isinstance(attr_value, Node):
                setattr(node, attr_name, self.process(attr_value))
    
    @lru_cache(maxsize=128)
    def _combine_dimensions(self, dim1: Optional[UnitExpr], dim2: Optional[UnitExpr], 
                           op: str) -> Optional[UnitExpr]:
        """
        Combine dimensions based on operation.
        
        Args:
            dim1: Left dimension
            dim2: Right dimension
            op: Operation ('+', '-', '*', '/', '^', '**')
            
        Returns:
            Combined dimension or None if dimensionless
        """
        # Addition and subtraction require same dimensions
        if op in ('+', '-'):
            # If either operand has no dimension, result has the other's dimension
            if dim1 is None:
                return dim2
            if dim2 is None:
                return dim1
                
            # For addition/subtraction, dimensions must match
            if not isinstance(dim1, UnitExpr) or not isinstance(dim2, UnitExpr) or not dim1.same(dim2):
                self.diagnostics.append(
                    Diagnostic(
                        message=f"Cannot combine values with incompatible dimensions: [{dim1}] {op} [{dim2}]",
                        code="DIM_MISMATCH",
                        source=Diagnostic.CONST_FOLDER
                    )
                )
            return dim1
        
        # Multiplication combines dimensions
        elif op == '*':
            # If either operand has no dimension, result has the other's dimension
            if dim1 is None:
                return dim2
            if dim2 is None:
                return dim1
                
            # Multiply dimensions
            return MulUnit(dim1, dim2)
        
        # Division divides dimensions
        elif op == '/':
            # Special cases for None dimensions
            if dim1 is None and dim2 is None:
                return None
            if dim1 is None:
                # None / dim2 means 1 / dim2
                return DivUnit(BaseUnit("dimensionless"), dim2)
            if dim2 is None:
                # dim1 / None means dim1 / 1
                return dim1
                
            # Divide dimensions
            return DivUnit(dim1, dim2)
        
        # Power (either ^ or **)
        elif op in ('^', '**'):
            # If base has no dimension, result has no dimension
            if dim1 is None:
                return None
                
            # If exponent is not a simple number, we can't determine dimension
            if not isinstance(dim2, (int, float)) and dim2 is not None:
                # In a real implementation, we might want to issue a warning here
                return None
                
            # If exponent has dimension, that's not valid
            # (can't raise to a dimensional power)
            if dim2 is not None:
                # In a real implementation, we would issue an error here
                return None
                
            # Create a power unit
            return PowUnit(dim1, int(dim2))
        
        # Unknown operator
        return None
