"""
Unit tests for the UnitExpr class hierarchy.

This module tests the UnitExpr classes and their methods for
dimensional analysis and comparison.
"""

import unittest
import sys
from pathlib import Path

# Add the parent directory to the path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.append(str(parent_dir))

from alan_backend.elfin.units.unit_expr import (
    UnitExpr, BaseUnit, MulUnit, DivUnit, PowUnit, parse_unit_expr
)


class TestUnitExpr(unittest.TestCase):
    """Test cases for the UnitExpr class hierarchy."""
    
    def test_base_unit_equality(self):
        """Test equality between base units."""
        # Same units should be equal
        u1 = BaseUnit("m")
        u2 = BaseUnit("m")
        self.assertEqual(u1, u2)
        self.assertTrue(u1.same(u2))
        
        # Different units should not be equal
        u3 = BaseUnit("kg")
        self.assertNotEqual(u1, u3)
        self.assertFalse(u1.same(u3))
    
    def test_mul_unit_equality(self):
        """Test equality between multiplication units."""
        # Same units should be equal
        u1 = MulUnit(BaseUnit("m"), BaseUnit("kg"))
        u2 = MulUnit(BaseUnit("m"), BaseUnit("kg"))
        self.assertEqual(u1, u2)
        self.assertTrue(u1.same(u2))
        
        # Multiplication is commutative, so these should be equal
        u3 = MulUnit(BaseUnit("kg"), BaseUnit("m"))
        self.assertNotEqual(u1, u3)  # Direct equality uses string repr
        self.assertTrue(u1.same(u3))  # same() checks dimensional equivalence
    
    def test_div_unit_equality(self):
        """Test equality between division units."""
        # Same units should be equal
        u1 = DivUnit(BaseUnit("m"), BaseUnit("s"))
        u2 = DivUnit(BaseUnit("m"), BaseUnit("s"))
        self.assertEqual(u1, u2)
        self.assertTrue(u1.same(u2))
        
        # Division is not commutative, so these should not be equal
        u3 = DivUnit(BaseUnit("s"), BaseUnit("m"))
        self.assertNotEqual(u1, u3)
        self.assertFalse(u1.same(u3))
    
    def test_pow_unit_equality(self):
        """Test equality between power units."""
        # Same units should be equal
        u1 = PowUnit(BaseUnit("m"), 2)
        u2 = PowUnit(BaseUnit("m"), 2)
        self.assertEqual(u1, u2)
        self.assertTrue(u1.same(u2))
        
        # Different exponents should not be equal
        u3 = PowUnit(BaseUnit("m"), 3)
        self.assertNotEqual(u1, u3)
        self.assertFalse(u1.same(u3))
    
    def test_complex_expressions(self):
        """Test more complex unit expressions."""
        # m/s
        u1 = DivUnit(BaseUnit("m"), BaseUnit("s"))
        
        # kg*m/s^2 (force)
        u2 = DivUnit(
            MulUnit(BaseUnit("kg"), BaseUnit("m")),
            PowUnit(BaseUnit("s"), 2)
        )
        
        # These are different dimensions
        self.assertNotEqual(u1, u2)
        self.assertFalse(u1.same(u2))
        
        # (kg*m/s^2) / kg = m/s^2 (acceleration)
        u3 = DivUnit(u2, BaseUnit("kg"))
        
        # m/s^2
        u4 = DivUnit(BaseUnit("m"), PowUnit(BaseUnit("s"), 2))
        
        # These should be dimensionally the same
        self.assertTrue(u3.same(u4))
    
    def test_normalization(self):
        """Test normalization of unit expressions."""
        # Test commutative normalization for multiplication
        u1 = MulUnit(BaseUnit("kg"), BaseUnit("m"))
        u2 = MulUnit(BaseUnit("m"), BaseUnit("kg"))
        
        # These should normalize to the same form
        norm_u1 = u1.normalize()
        norm_u2 = u2.normalize()
        
        # Check that they are equal after normalization
        self.assertEqual(str(norm_u1), str(norm_u2))
        
        # Also check the same() method works with normalization
        self.assertTrue(u1.same(u2))
        
        # Test power handling in normalization
        u3 = PowUnit(BaseUnit("m"), 0)
        self.assertEqual(str(u3.normalize()), "dimensionless")
        
        u4 = PowUnit(BaseUnit("m"), 1)
        self.assertEqual(str(u4.normalize()), "m")
        
        # Test nested powers
        u5 = PowUnit(PowUnit(BaseUnit("m"), 2), 3)
        self.assertEqual(str(u5.normalize()), "m^6")
    
    def test_parse_unit_expr(self):
        """Test parsing unit expressions from strings."""
        # Test basic parsing
        u1 = parse_unit_expr("m")
        self.assertIsInstance(u1, BaseUnit)
        self.assertEqual(u1.name, "m")
        
        # Test parsing with our simplified implementation
        # Note: The actual implementation would need a more robust parser
        u2 = parse_unit_expr("m*kg")
        self.assertIsInstance(u2, MulUnit)
        
        u3 = parse_unit_expr("m/s")
        self.assertIsInstance(u3, DivUnit)
        
        u4 = parse_unit_expr("m^2")
        self.assertIsInstance(u4, PowUnit)


if __name__ == "__main__":
    unittest.main()
