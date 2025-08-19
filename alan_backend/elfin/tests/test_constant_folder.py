"""
Tests for the Constant Folding Pass.

This module tests the constant folding pass to ensure it correctly evaluates
expressions at compile time and tracks dimensional information.
"""

import unittest
import sys
from pathlib import Path

# Add the parent directory to the path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

from alan_backend.elfin.compiler.ast.nodes import (
    BinaryOp, FunctionCall, VarRef, Number, Expression, Node
)
from typing import List, Dict, Optional, Any, Union
from alan_backend.elfin.compiler.ast.const_expr import ConstExpr
from alan_backend.elfin.compiler.passes.constant_folder import ConstantFolder
from alan_backend.elfin.units.unit_expr import BaseUnit, MulUnit, DivUnit, PowUnit


class TestConstantFolder(unittest.TestCase):
    """Test cases for the Constant Folding Pass."""
    
    def setUp(self):
        """Set up for tests."""
        self.folder = ConstantFolder()
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations are folded correctly."""
        # 2 + 3 -> 5
        expr = BinaryOp(Number(2), "+", Number(3))
        result = self.folder.process(expr)
        self.assertIsInstance(result, ConstExpr)
        self.assertEqual(result.value, 5)
        
        # 10 - 4 -> 6
        expr = BinaryOp(Number(10), "-", Number(4))
        result = self.folder.process(expr)
        self.assertIsInstance(result, ConstExpr)
        self.assertEqual(result.value, 6)
        
        # 3 * 7 -> 21
        expr = BinaryOp(Number(3), "*", Number(7))
        result = self.folder.process(expr)
        self.assertIsInstance(result, ConstExpr)
        self.assertEqual(result.value, 21)
        
        # 20 / 5 -> 4
        expr = BinaryOp(Number(20), "/", Number(5))
        result = self.folder.process(expr)
        self.assertIsInstance(result, ConstExpr)
        self.assertEqual(result.value, 4)
        
        # 2 ^ 3 -> 8 (power operation)
        expr = BinaryOp(Number(2), "^", Number(3))
        result = self.folder.process(expr)
        self.assertIsInstance(result, ConstExpr)
        self.assertEqual(result.value, 8)
    
    def test_nested_expressions(self):
        """Test that nested expressions are folded correctly."""
        # (2 + 3) * 4 -> 20
        expr = BinaryOp(
            BinaryOp(Number(2), "+", Number(3)),
            "*",
            Number(4)
        )
        result = self.folder.process(expr)
        self.assertIsInstance(result, ConstExpr)
        self.assertEqual(result.value, 20)
        
        # 2 + (3 * 4) -> 14
        expr = BinaryOp(
            Number(2),
            "+",
            BinaryOp(Number(3), "*", Number(4))
        )
        result = self.folder.process(expr)
        self.assertIsInstance(result, ConstExpr)
        self.assertEqual(result.value, 14)
        
        # ((10 / 2) + 3) * 2 -> 16
        expr = BinaryOp(
            BinaryOp(
                BinaryOp(Number(10), "/", Number(2)),
                "+",
                Number(3)
            ),
            "*",
            Number(2)
        )
        result = self.folder.process(expr)
        self.assertIsInstance(result, ConstExpr)
        self.assertEqual(result.value, 16)
    
    def test_function_calls(self):
        """Test that function calls with constant arguments are folded."""
        # sin(0) -> 0.0
        expr = FunctionCall("sin", [Number(0)])
        result = self.folder.process(expr)
        self.assertIsInstance(result, ConstExpr)
        self.assertAlmostEqual(result.value, 0.0)
        
        # cos(0) -> 1.0
        expr = FunctionCall("cos", [Number(0)])
        result = self.folder.process(expr)
        self.assertIsInstance(result, ConstExpr)
        self.assertAlmostEqual(result.value, 1.0)
        
        # sqrt(16) -> 4.0
        expr = FunctionCall("sqrt", [Number(16)])
        result = self.folder.process(expr)
        self.assertIsInstance(result, ConstExpr)
        self.assertAlmostEqual(result.value, 4.0)
    
    def test_variable_substitution(self):
        """Test that variables with known values are substituted."""
        # Create an environment mapping variables to constant values
        env = {
            "x": ConstExpr(5),
            "y": ConstExpr(3)
        }
        
        # x + 2 -> 7 (with x = 5)
        expr = BinaryOp(VarRef("x"), "+", Number(2))
        result = self.folder.process(expr, env)
        self.assertIsInstance(result, ConstExpr)
        self.assertEqual(result.value, 7)
        
        # x * y -> 15 (with x = 5, y = 3)
        expr = BinaryOp(VarRef("x"), "*", VarRef("y"))
        result = self.folder.process(expr, env)
        self.assertIsInstance(result, ConstExpr)
        self.assertEqual(result.value, 15)
        
        # Unknown variables remain as references
        expr = BinaryOp(VarRef("z"), "+", Number(1))
        result = self.folder.process(expr, env)
        self.assertIsInstance(result, BinaryOp)
        self.assertIsInstance(result.left, VarRef)
        self.assertEqual(result.left.name, "z")
    
    def test_dimension_handling(self):
        """Test that dimensions are correctly handled during folding."""
        # Create constants with dimensions
        length = ConstExpr(2.0, BaseUnit("m"))
        time = ConstExpr(0.5, BaseUnit("s"))
        mass = ConstExpr(3.0, BaseUnit("kg"))
        
        # Environment
        env = {
            "length": length,
            "time": time,
            "mass": mass,
            "g": ConstExpr(9.8, DivUnit(BaseUnit("m"), PowUnit(BaseUnit("s"), 2)))
        }
        
        # Test length * 3 -> 6.0 [m]
        expr = BinaryOp(VarRef("length"), "*", Number(3))
        result = self.folder.process(expr, env)
        self.assertIsInstance(result, ConstExpr)
        self.assertEqual(result.value, 6.0)
        self.assertEqual(str(result.dim), "m")
        
        # Test 2 [m] * 3 [kg] -> 6 [m*kg]
        expr = BinaryOp(VarRef("length"), "*", VarRef("mass"))
        result = self.folder.process(expr, env)
        self.assertIsInstance(result, ConstExpr)
        self.assertEqual(result.value, 6.0)
        self.assertIsInstance(result.dim, MulUnit)
        self.assertEqual(str(result.dim), "m*kg")
        
        # Test length / time -> 4.0 [m/s]
        expr = BinaryOp(VarRef("length"), "/", VarRef("time"))
        result = self.folder.process(expr, env)
        self.assertIsInstance(result, ConstExpr)
        self.assertEqual(result.value, 4.0)
        self.assertIsInstance(result.dim, DivUnit)
        self.assertEqual(str(result.dim), "m/s")
        
        # Test g * time^2 / 2 -> 1.225 [m]
        # This is the displacement formula: d = gt²/2
        expr = BinaryOp(
            BinaryOp(
                VarRef("g"),
                "*",
                BinaryOp(VarRef("time"), "^", Number(2))
            ),
            "/",
            Number(2)
        )
        result = self.folder.process(expr, env)
        self.assertIsInstance(result, ConstExpr)
        self.assertAlmostEqual(result.value, 1.225)
        self.assertEqual(str(result.dim), "m")
    
    def test_dimensional_errors(self):
        """Test handling of dimensional errors."""
        # Create constants with dimensions
        length = ConstExpr(2.0, BaseUnit("m"))
        force = ConstExpr(10.0, MulUnit(BaseUnit("kg"), 
                                       DivUnit(BaseUnit("m"), 
                                              PowUnit(BaseUnit("s"), 2))))
        
        # Environment
        env = {
            "length": length,
            "force": force,
        }
        
        # Adding different dimensions should not fold completely
        # length + force -> error or preserve as BinaryOp
        expr = BinaryOp(VarRef("length"), "+", VarRef("force"))
        result = self.folder.process(expr, env)
        
        # The result should either be a BinaryOp (no folding)
        # or a ConstExpr with some error indication
        # For this test, we'll assume it doesn't fold
        self.assertIsInstance(result, BinaryOp)
        
        # Raising to a dimensional power should not fold
        # length ^ force -> error or preserve as BinaryOp
        expr = BinaryOp(VarRef("length"), "^", VarRef("force"))
        result = self.folder.process(expr, env)
        
        # Should not fold since power has dimension
        self.assertIsInstance(result, BinaryOp)


if __name__ == "__main__":
    unittest.main()
