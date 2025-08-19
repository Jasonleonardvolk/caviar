"""
Tests for the Partial Evaluator.

This module tests the partial evaluator to ensure it correctly simplifies
expressions at compile time.
"""

import unittest
import sys
from pathlib import Path

# Add the parent directory to the path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

from alan_backend.elfin.compiler.ast.nodes import (
    BinaryOp, 
    UnaryOp, 
    Call, 
    LiteralFloat, 
    LiteralInt, 
    Identifier,
)
from alan_backend.elfin.compiler.passes.partial_evaluator import PartialEvaluator
from alan_backend.elfin.units.unit_expr import BaseUnit, MulUnit, DivUnit, PowUnit


class TestPartialEvaluator(unittest.TestCase):
    """Test cases for the Partial Evaluator."""
    
    def setUp(self):
        """Set up for tests."""
        self.evaluator = PartialEvaluator()
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations with literals."""
        # Test addition: 2 + 3 = 5
        expr = BinaryOp(LiteralInt(2), '+', LiteralInt(3))
        result = self.evaluator.evaluate(expr)
        self.assertIsInstance(result, LiteralInt)
        self.assertEqual(result.value, 5)
        
        # Test subtraction: 5 - 2.5 = 2.5
        expr = BinaryOp(LiteralInt(5), '-', LiteralFloat(2.5))
        result = self.evaluator.evaluate(expr)
        self.assertIsInstance(result, LiteralFloat)
        self.assertEqual(result.value, 2.5)
        
        # Test multiplication: 3 * 4 = 12
        expr = BinaryOp(LiteralInt(3), '*', LiteralInt(4))
        result = self.evaluator.evaluate(expr)
        self.assertIsInstance(result, LiteralInt)
        self.assertEqual(result.value, 12)
        
        # Test division: 10 / 2 = 5
        expr = BinaryOp(LiteralInt(10), '/', LiteralInt(2))
        result = self.evaluator.evaluate(expr)
        self.assertIsInstance(result, LiteralInt)
        self.assertEqual(result.value, 5)
        
        # Test power: 2 ^ 3 = 8
        expr = BinaryOp(LiteralInt(2), '^', LiteralInt(3))
        result = self.evaluator.evaluate(expr)
        self.assertIsInstance(result, LiteralInt)
        self.assertEqual(result.value, 8)
    
    def test_complex_expressions(self):
        """Test more complex, nested expressions."""
        # Test (2 + 3) * 4 = 20
        expr = BinaryOp(
            BinaryOp(LiteralInt(2), '+', LiteralInt(3)),
            '*',
            LiteralInt(4)
        )
        result = self.evaluator.evaluate(expr)
        self.assertIsInstance(result, LiteralInt)
        self.assertEqual(result.value, 20)
        
        # Test 2 + 3 * 4 = 14 (operator precedence maintained)
        expr = BinaryOp(
            LiteralInt(2),
            '+',
            BinaryOp(LiteralInt(3), '*', LiteralInt(4))
        )
        result = self.evaluator.evaluate(expr)
        self.assertIsInstance(result, LiteralInt)
        self.assertEqual(result.value, 14)
        
        # Test ((10 / 2) + 3) * 2 = 16
        expr = BinaryOp(
            BinaryOp(
                BinaryOp(LiteralInt(10), '/', LiteralInt(2)),
                '+',
                LiteralInt(3)
            ),
            '*',
            LiteralInt(2)
        )
        result = self.evaluator.evaluate(expr)
        self.assertIsInstance(result, LiteralInt)
        self.assertEqual(result.value, 16)
    
    def test_unit_expressions(self):
        """Test expressions with units."""
        # Create literals with units
        mass = LiteralFloat(2.0)
        mass.dim = BaseUnit("kg")
        
        accel = LiteralFloat(9.8)
        accel.dim = DivUnit(BaseUnit("m"), PowUnit(BaseUnit("s"), 2))
        
        # Test mass * acceleration = force (kg * m/s^2)
        expr = BinaryOp(mass, '*', accel)
        result = self.evaluator.evaluate(expr)
        
        # The numeric value should be 19.6
        self.assertIsInstance(result, BinaryOp)
        # The expression should have a dimension
        self.assertTrue(hasattr(result, 'dim'))
        
        # The dimension should be kg*m/s^2
        expected_dim = MulUnit(
            BaseUnit("kg"),
            DivUnit(BaseUnit("m"), PowUnit(BaseUnit("s"), 2))
        )
        self.assertEqual(str(result.dim), str(expected_dim))
    
    def test_unary_operations(self):
        """Test unary operations."""
        # Test -5 = -5
        expr = UnaryOp('-', LiteralInt(5))
        result = self.evaluator.evaluate(expr)
        self.assertIsInstance(result, LiteralInt)
        self.assertEqual(result.value, -5)
        
        # Test -(-5) = 5
        expr = UnaryOp('-', UnaryOp('-', LiteralInt(5)))
        result = self.evaluator.evaluate(expr)
        self.assertIsInstance(result, LiteralInt)
        self.assertEqual(result.value, 5)
    
    def test_math_functions(self):
        """Test evaluation of common math functions."""
        # Test sin(0) = 0
        expr = Call(Identifier('sin'), [LiteralInt(0)])
        result = self.evaluator.evaluate(expr)
        self.assertIsInstance(result, LiteralFloat)
        self.assertAlmostEqual(result.value, 0.0)
        
        # Test cos(0) = 1
        expr = Call(Identifier('cos'), [LiteralInt(0)])
        result = self.evaluator.evaluate(expr)
        self.assertIsInstance(result, LiteralFloat)
        self.assertAlmostEqual(result.value, 1.0)
        
        # Test sqrt(4) = 2
        expr = Call(Identifier('sqrt'), [LiteralInt(4)])
        result = self.evaluator.evaluate(expr)
        self.assertIsInstance(result, LiteralFloat)
        self.assertAlmostEqual(result.value, 2.0)
    
    def test_environment_substitution(self):
        """Test evaluation with variable substitution."""
        # Define an environment with variables
        env = {
            'x': LiteralInt(5),
            'y': LiteralFloat(2.5),
        }
        
        # Test x + y = 7.5
        expr = BinaryOp(Identifier('x'), '+', Identifier('y'))
        result = self.evaluator.evaluate(expr, env)
        self.assertIsInstance(result, LiteralFloat)
        self.assertEqual(result.value, 7.5)
        
        # Test x * (y + 1) = 17.5
        expr = BinaryOp(
            Identifier('x'),
            '*',
            BinaryOp(Identifier('y'), '+', LiteralInt(1))
        )
        result = self.evaluator.evaluate(expr, env)
        self.assertIsInstance(result, LiteralFloat)
        self.assertEqual(result.value, 17.5)
    
    def test_memoization(self):
        """Test that results are memoized for performance."""
        # Create a complex expression
        expr = BinaryOp(
            BinaryOp(LiteralInt(10), '/', LiteralInt(2)),
            '+',
            BinaryOp(LiteralInt(3), '*', LiteralInt(4))
        )
        
        # Evaluate it once
        result1 = self.evaluator.evaluate(expr)
        
        # The memo cache should now contain entries
        self.assertGreater(len(self.evaluator.memo), 0)
        
        # Evaluate it again
        result2 = self.evaluator.evaluate(expr)
        
        # Results should be identical
        self.assertEqual(result1.value, result2.value)


if __name__ == "__main__":
    unittest.main()
