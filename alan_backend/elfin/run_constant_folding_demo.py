"""
Constant Folding Demo

This script demonstrates the constant folding feature by manually creating
AST nodes and running them through the constant folder pass.
"""

import os
import sys
import math
from pathlib import Path

# Simple AST nodes for the demo
class Node:
    pass

class Expression(Node):
    pass

class Number(Expression):
    def __init__(self, value):
        self.value = value
        
    def __repr__(self):
        return f"Number({self.value})"

class BinaryOp(Expression):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
        
    def __repr__(self):
        return f"BinaryOp({self.left}, '{self.op}', {self.right})"

class ConstExpr(Expression):
    def __init__(self, value, dim=None):
        self.value = value
        self.dim = dim
        
    def __repr__(self):
        dim_str = f" [{self.dim}]" if self.dim else ""
        return f"ConstExpr({self.value}{dim_str})"

# Simplified Unit Expression classes
class UnitExpr:
    def __init__(self, name="dimensionless"):
        self.name = name
        
    def __repr__(self):
        return self.name

class BaseUnit(UnitExpr):
    pass

class MulUnit(UnitExpr):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.name = f"{left}*{right}"
        
    def __repr__(self):
        return self.name

class DivUnit(UnitExpr):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.name = f"{left}/{right}"
        
    def __repr__(self):
        return self.name

class PowUnit(UnitExpr):
    def __init__(self, base, exp):
        self.base = base
        self.exp = exp
        self.name = f"{base}^{exp}"
        
    def __repr__(self):
        return self.name

# Simplified constant folder implementation
class ConstantFolder:
    def __init__(self):
        self.diagnostics = []
        self.binary_ops = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
            '^': lambda a, b: a ** b,
        }
    
    def process(self, node):
        if isinstance(node, ConstExpr):
            return node
        elif isinstance(node, Number):
            return ConstExpr(node.value)
        elif isinstance(node, BinaryOp):
            left = self.process(node.left)
            right = self.process(node.right)
            
            if isinstance(left, ConstExpr) and isinstance(right, ConstExpr):
                op = node.op
                if op in self.binary_ops:
                    try:
                        # Compute the value
                        result_value = self.binary_ops[op](left.value, right.value)
                        
                        # Compute the dimension based on operation
                        result_dim = self._combine_dimensions(left.dim, right.dim, op)
                        
                        return ConstExpr(result_value, result_dim)
                    except Exception as e:
                        print(f"Error during folding: {e}")
            
            node.left = left
            node.right = right
            return node
        
        return node
    
    def _combine_dimensions(self, dim1, dim2, op):
        """Simple dimension combination logic."""
        if op in ('+', '-'):
            # Addition/subtraction requires same dimensions
            if dim1 != dim2:
                self.diagnostics.append(f"Cannot {op} values with dimensions {dim1} and {dim2}")
            return dim1
        
        elif op == '*':
            # Multiplication combines dimensions
            if dim1 is None:
                return dim2
            if dim2 is None:
                return dim1
            return MulUnit(dim1, dim2)
        
        elif op == '/':
            # Division
            if dim1 is None and dim2 is None:
                return None
            if dim1 is None:
                return DivUnit(UnitExpr(), dim2)
            if dim2 is None:
                return dim1
            return DivUnit(dim1, dim2)
        
        elif op == '^':
            # Power
            if dim1 is None:
                return None
            if dim2 is None or isinstance(dim2, (int, float)):
                return PowUnit(dim1, dim2 if dim2 is not None else 1)
            
            # Cannot raise to dimensional power
            self.diagnostics.append(f"Cannot raise {dim1} to power with dimension {dim2}")
            return None
        
        return None

def run_demo():
    print("ELFIN Constant Folding Demo")
    print("===========================\n")
    
    folder = ConstantFolder()
    
    # Basic arithmetic examples
    print("Basic Arithmetic Examples:")
    examples = [
        BinaryOp(Number(2), "+", Number(3)),               # 2 + 3 -> 5
        BinaryOp(Number(10), "-", Number(4)),              # 10 - 4 -> 6
        BinaryOp(Number(3), "*", Number(7)),               # 3 * 7 -> 21
        BinaryOp(Number(20), "/", Number(5)),              # 20 / 5 -> 4
        BinaryOp(Number(2), "^", Number(3)),               # 2 ^ 3 -> 8
    ]
    
    for i, expr in enumerate(examples):
        result = folder.process(expr)
        print(f"  {i+1}. {expr} → {result}")
    
    print("\nNested Expressions:")
    nested = BinaryOp(
        BinaryOp(
            BinaryOp(Number(10), "/", Number(2)),  # 10 / 2 = 5
            "+", 
            Number(3)                              # 5 + 3 = 8
        ),
        "*",
        Number(2)                                  # 8 * 2 = 16
    )
    result = folder.process(nested)
    print(f"  ((10 / 2) + 3) * 2 → {result}")
    
    # Dimensional examples
    print("\nDimensional Examples:")
    
    # Create constants with dimensions
    length = ConstExpr(2.0, BaseUnit("m"))  # 2 meters
    time = ConstExpr(0.5, BaseUnit("s"))    # 0.5 seconds
    
    # g = 9.8 m/s^2
    g = ConstExpr(9.8, DivUnit(BaseUnit("m"), PowUnit(BaseUnit("s"), 2)))
    
    # Examples with dimensions
    dim_examples = [
        # length * 3 -> 6 [m]
        BinaryOp(length, "*", Number(3)),
        
        # length / time -> 4 [m/s]
        BinaryOp(length, "/", time),
        
        # g * time^2 / 2 -> 1.225 [m] (displacement)
        BinaryOp(
            BinaryOp(g, "*", BinaryOp(time, "^", Number(2))),
            "/",
            Number(2)
        ),
        
        # Invalid: length + g (mismatched dimensions)
        BinaryOp(length, "+", g)
    ]
    
    for i, expr in enumerate(dim_examples):
        result = folder.process(expr)
        print(f"  {i+1}. {expr} → {result}")
    
    # Show any diagnostics
    if folder.diagnostics:
        print("\nDiagnostics:")
        for i, diag in enumerate(folder.diagnostics):
            print(f"  {i+1}. WARNING: {diag}")

if __name__ == "__main__":
    run_demo()
