"""
Partial Evaluator for ELFIN expressions.

This module provides a partial evaluator that simplifies expressions at compile time,
particularly focusing on numeric constants with units for improved dimensional analysis.
"""

from typing import Dict, Any, Optional, Union, Set, List, Tuple
import math
import copy

from alan_backend.elfin.compiler.ast.nodes import (
    Node, 
    Expression, 
    BinaryOp, 
    UnaryOp, 
    Call, 
    LiteralFloat, 
    LiteralInt, 
    Identifier,
)
from alan_backend.elfin.units.unit_expr import UnitExpr, BaseUnit, MulUnit, DivUnit, PowUnit


class PartialEvaluator:
    """
    Partial Evaluator for ELFIN expressions.
    
    This class traverses AST expressions and simplifies them where possible,
    particularly focusing on numeric constants and unit expressions.
    """
    
    def __init__(self):
        """Initialize the partial evaluator."""
        # Cache for memoizing evaluated expressions
        self.memo = {}
        
    def evaluate(self, node: Node, env: Optional[Dict[str, Any]] = None) -> Node:
        """
        Evaluate and simplify a node in the AST.
        
        Args:
            node: The AST node to evaluate
            env: Environment mapping variable names to values (optional)
            
        Returns:
            A new, simplified AST node
        """
        if env is None:
            env = {}
            
        # Check for cached result
        node_hash = self._node_hash(node)
        if node_hash in self.memo:
            return self.memo[node_hash]
            
        # Evaluate based on node type
        result = None
        
        if isinstance(node, LiteralInt) or isinstance(node, LiteralFloat):
            # Literals are already in their simplest form
            result = node
        elif isinstance(node, Identifier):
            # If the identifier is in the environment, replace with its value
            if node.name in env:
                result = env[node.name]
            else:
                result = node
        elif isinstance(node, BinaryOp):
            result = self._evaluate_binary_op(node, env)
        elif isinstance(node, UnaryOp):
            result = self._evaluate_unary_op(node, env)
        elif isinstance(node, Call):
            result = self._evaluate_call(node, env)
        else:
            # Default case: can't simplify further
            result = node
            
        # Cache and return the result
        self.memo[node_hash] = result
        return result
        
    def _evaluate_binary_op(self, node: BinaryOp, env: Dict[str, Any]) -> Node:
        """
        Evaluate a binary operation (like +, -, *, /).
        
        Args:
            node: Binary operation node
            env: Environment mapping variable names to values
            
        Returns:
            Simplified node
        """
        # Evaluate operands
        left = self.evaluate(node.left, env)
        right = self.evaluate(node.right, env)
        
        # If both operands are numeric literals, perform the operation
        if (isinstance(left, (LiteralInt, LiteralFloat)) and 
            isinstance(right, (LiteralInt, LiteralFloat))):
            
            # Extract values
            left_val = left.value
            right_val = right.value
            
            # Perform operation
            if node.op == '+':
                result_val = left_val + right_val
            elif node.op == '-':
                result_val = left_val - right_val
            elif node.op == '*':
                result_val = left_val * right_val
            elif node.op == '/':
                if right_val == 0:
                    # Division by zero - return the original node
                    return BinaryOp(left, node.op, right)
                result_val = left_val / right_val
            elif node.op == '**' or node.op == '^':
                result_val = left_val ** right_val
            else:
                # Unknown operator - return the original node
                return BinaryOp(left, node.op, right)
            
            # Create appropriate literal based on result type
            if isinstance(result_val, int):
                return LiteralInt(result_val)
            else:
                return LiteralFloat(result_val)
                
        # If the operation is multiplication or division of units, handle it specially
        if hasattr(left, 'dim') and hasattr(right, 'dim'):
            if node.op == '*' and left.dim and right.dim:
                # Combine the units using multiplication
                new_node = BinaryOp(left, node.op, right)
                new_node.dim = MulUnit(left.dim, right.dim)
                return new_node
            elif node.op == '/' and left.dim and right.dim:
                # Combine the units using division
                new_node = BinaryOp(left, node.op, right)
                new_node.dim = DivUnit(left.dim, right.dim)
                return new_node
        
        # If we couldn't fully evaluate, return a new binary op with the simplified operands
        return BinaryOp(left, node.op, right)
        
    def _evaluate_unary_op(self, node: UnaryOp, env: Dict[str, Any]) -> Node:
        """
        Evaluate a unary operation (like -, +, !).
        
        Args:
            node: Unary operation node
            env: Environment mapping variable names to values
            
        Returns:
            Simplified node
        """
        # Evaluate the operand
        operand = self.evaluate(node.operand, env)
        
        # If the operand is a numeric literal, perform the operation
        if isinstance(operand, (LiteralInt, LiteralFloat)):
            if node.op == '-':
                if isinstance(operand, LiteralInt):
                    return LiteralInt(-operand.value)
                else:
                    return LiteralFloat(-operand.value)
            elif node.op == '+':
                return operand
                
        # If we couldn't fully evaluate, return a new unary op with the simplified operand
        return UnaryOp(node.op, operand)
        
    def _evaluate_call(self, node: Call, env: Dict[str, Any]) -> Node:
        """
        Evaluate a function call.
        
        Args:
            node: Function call node
            env: Environment mapping variable names to values
            
        Returns:
            Simplified node
        """
        # Evaluate the arguments
        args = [self.evaluate(arg, env) for arg in node.args]
        
        # Check if all arguments are literals
        all_literals = all(isinstance(arg, (LiteralInt, LiteralFloat)) for arg in args)
        
        # If the function is a known math function and all args are literals, evaluate it
        if isinstance(node.func, Identifier) and all_literals:
            func_name = node.func.name
            
            # Get the argument values
            arg_values = [arg.value for arg in args]
            
            if func_name == 'sin' and len(args) == 1:
                return LiteralFloat(math.sin(arg_values[0]))
            elif func_name == 'cos' and len(args) == 1:
                return LiteralFloat(math.cos(arg_values[0]))
            elif func_name == 'tan' and len(args) == 1:
                return LiteralFloat(math.tan(arg_values[0]))
            elif func_name == 'exp' and len(args) == 1:
                return LiteralFloat(math.exp(arg_values[0]))
            elif func_name == 'log' and len(args) == 1:
                if arg_values[0] <= 0:
                    # Log of non-positive number - return the original node
                    return Call(node.func, args)
                return LiteralFloat(math.log(arg_values[0]))
            elif func_name == 'sqrt' and len(args) == 1:
                if arg_values[0] < 0:
                    # Sqrt of negative number - return the original node
                    return Call(node.func, args)
                return LiteralFloat(math.sqrt(arg_values[0]))
                
        # If we couldn't fully evaluate, return a new call with the simplified arguments
        return Call(node.func, args)
        
    def _node_hash(self, node: Node) -> str:
        """
        Create a hash string for a node to use as a memoization key.
        
        Args:
            node: The node to hash
            
        Returns:
            A string hash of the node
        """
        if isinstance(node, LiteralInt):
            return f"LiteralInt({node.value})"
        elif isinstance(node, LiteralFloat):
            return f"LiteralFloat({node.value})"
        elif isinstance(node, Identifier):
            return f"Identifier({node.name})"
        elif isinstance(node, BinaryOp):
            left_hash = self._node_hash(node.left)
            right_hash = self._node_hash(node.right)
            return f"BinaryOp({left_hash}, {node.op}, {right_hash})"
        elif isinstance(node, UnaryOp):
            operand_hash = self._node_hash(node.operand)
            return f"UnaryOp({node.op}, {operand_hash})"
        elif isinstance(node, Call):
            func_hash = self._node_hash(node.func)
            args_hash = ",".join(self._node_hash(arg) for arg in node.args)
            return f"Call({func_hash}, [{args_hash}])"
        else:
            # Default hash for other node types
            return f"{type(node).__name__}({id(node)})"
