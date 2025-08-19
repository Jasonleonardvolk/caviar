#!/usr/bin/env python3
"""
TORI Production Test Suite
"""
import numpy as np
import unittest
from numpy.testing import assert_allclose

class TestPenroseProduction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """One-time setup"""
        from python.core.exotic_topologies_v2 import build_penrose_laplacian_large
        from python.core.penrose_microkernel_v3_production import configure, clear_cache
        
        # Configure for tests
        configure(rank=14, min_spectral_gap=1e-5)
        
        # Build test Laplacian
        cls.L = build_penrose_laplacian_large(target_nodes=1000)
    
    def test_multiply_accuracy(self):
        """Test that Penrose multiply is accurate"""
        from python.core.penrose_microkernel_v3_production import multiply
        
        n = 256
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        # Reference
        C_ref = A @ B
        
        # Penrose
        C_penrose, info = multiply(A, B, self.L)
        
        # Should be close
        assert_allclose(C_penrose, C_ref, rtol=1e-5, atol=1e-8)
        self.assertIn('spectral_gap', info)
        self.assertNotIn('fallback', info)
    
    def test_spectral_gap_check(self):
        """Test that spectral gap is reasonable"""
        from python.core.penrose_microkernel_v3_production import multiply
        
        A = np.random.rand(128, 128)
        B = np.random.rand(128, 128)
        
        _, info = multiply(A, B, self.L)
        
        gap = info.get('spectral_gap', 0)
        self.assertGreater(gap, 1e-5, "Spectral gap too small")
        self.assertLess(gap, 1.0, "Spectral gap suspiciously large")
    
    def test_size_limits(self):
        """Test graceful handling of oversized matrices"""
        from python.core.penrose_microkernel_v3_production import multiply
        
        # Matrix larger than Laplacian
        n = 2000  # Larger than our 1000-node test Laplacian
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        C, info = multiply(A, B, self.L)
        
        # Should fallback gracefully
        self.assertIn('fallback', info)
        self.assertEqual(info['fallback'], 'size_exceeded')
        assert_allclose(C, A @ B)  # Should still give correct result
    
    def test_rank_validation(self):
        """Test rank configuration validation"""
        from python.core.penrose_microkernel_v3_production import configure
        
        # Valid ranks
        configure(rank=8)
        configure(rank=16)
        configure(rank=32)
        
        # Invalid ranks
        with self.assertRaises(ValueError):
            configure(rank=4)  # Too small
        
        with self.assertRaises(ValueError):
            configure(rank=64)  # Too large
    
    def test_cache_clearing(self):
        """Test cache management"""
        from python.core.penrose_microkernel_v3_production import multiply, clear_cache, get_info
        
        # First multiply to populate cache
        A = np.random.rand(128, 128)
        B = np.random.rand(128, 128)
        multiply(A, B, self.L)
        
        info1 = get_info()
        self.assertTrue(info1['cached'])
        
        # Clear cache
        clear_cache()
        
        info2 = get_info()
        self.assertFalse(info2['cached'])

if __name__ == '__main__':
    unittest.main()
