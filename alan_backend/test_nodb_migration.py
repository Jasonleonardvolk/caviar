#!/usr/bin/env python3
"""
Test script to verify No-DB migration completed successfully
Run this after completing the migration steps
"""

import sys
import os
from pathlib import Path
import numpy as np
import time
import json

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_torus_registry():
    """Test TorusRegistry functionality"""
    print("\n📊 Testing TorusRegistry...")
    
    try:
        from python.core.torus_registry import get_torus_registry, REG_PATH
        
        # Check environment
        state_root = os.getenv("TORI_STATE_ROOT", "/var/lib/tori")
        print(f"  State root: {state_root}")
        print(f"  Registry path: {REG_PATH}")
        
        # Get registry
        registry = get_torus_registry()
        
        # Record some test shapes
        test_vertices = np.random.randn(10, 3)
        shape_id = registry.record_shape(
            vertices=test_vertices,
            betti_numbers=[1.0, 0.0, 0.0],
            coherence_band="local",
            metadata={"test": True}
        )
        
        print(f"  ✅ Recorded shape: {shape_id}")
        
        # Flush and check stats
        registry.flush()
        stats = registry.get_statistics()
        print(f"  ✅ Registry stats: {json.dumps(stats, indent=2)}")
        
        # Verify Parquet file exists
        if REG_PATH.exists():
            size_kb = REG_PATH.stat().st_size / 1024
            print(f"  ✅ Parquet file exists: {REG_PATH} ({size_kb:.1f} KB)")
        else:
            print(f"  ❌ Parquet file not found at {REG_PATH}")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_torus_cells():
    """Test TorusCells functionality"""
    print("\n🧩 Testing TorusCells...")
    
    try:
        from python.core.torus_cells import get_torus_cells, betti0_1
        
        # Get cells instance
        cells = get_torus_cells()
        print(f"  Backend: {cells.backend}")
        
        # Test Betti computation
        test_points = np.random.randn(20, 2)  # 20 points in 2D
        b0, b1 = betti0_1(test_points)
        print(f"  ✅ Betti numbers: b0={b0}, b1={b1}")
        
        # Test betti_update
        b0_updated, b1_updated = cells.betti_update(
            idea_id="test_idea_001",
            vertices=test_points,
            coherence_band="global"
        )
        print(f"  ✅ Updated Betti: b0={b0_updated}, b1={b1_updated}")
        
        # Check protected ideas
        protected = cells.get_protected_ideas()
        print(f"  Protected ideas: {len(protected)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_observer_synthesis():
    """Test ObserverSynthesis functionality"""
    print("\n👁️ Testing ObserverSynthesis...")
    
    try:
        from python.core.observer_synthesis import get_observer_synthesis, emit_token
        
        # Get synthesis instance
        synthesis = get_observer_synthesis()
        
        # Emit some test tokens
        token1 = emit_token({
            "type": "test_spectral",
            "source": "test_script",
            "lambda_max": 0.042,
            "timestamp": time.time()
        })
        print(f"  ✅ Emitted token 1: {token1[:8]}...")
        
        token2 = emit_token({
            "type": "test_curvature",
            "source": "test_script",
            "mean_curvature": 1.5,
            "timestamp": time.time()
        })
        print(f"  ✅ Emitted token 2: {token2[:8]}...")
        
        # Add to context
        synthesis.add_to_context(token1)
        synthesis.add_to_context(token2)
        
        # Get context summary
        context = synthesis.synthesize_context()
        print(f"  ✅ Context summary: {json.dumps(context, indent=2)}")
        
        # Check metrics
        metrics = synthesis.metrics
        print(f"  Tokens generated: {metrics['tokens_generated']}")
        print(f"  Tokens in context: {metrics['tokens_in_context']}")
        
        # Create reasoning prompt
        prompt = synthesis.create_reasoning_prompt()
        if prompt:
            print(f"  ✅ Reasoning prompt created:\n{prompt}")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_origin_sentry_integration():
    """Test OriginSentry with new components"""
    print("\n🔮 Testing OriginSentry integration...")
    
    try:
        from alan_backend.origin_sentry import OriginSentry
        
        # Create instance (should use TorusRegistry internally)
        origin = OriginSentry()
        
        # Test classification
        test_eigenvalues = np.array([0.05, 0.03, 0.02, 0.01, 0.005])
        result = origin.classify(test_eigenvalues, betti_numbers=[1.0, 0.0])
        
        print(f"  ✅ Classification successful:")
        print(f"     Coherence: {result['coherence']}")
        print(f"     Novelty: {result['novelty_score']:.3f}")
        print(f"     Dimension: {result['metrics']['current_dimension']}")
        
        # Check if SpectralDB is using TorusRegistry
        if hasattr(origin.spectral_db, 'registry'):
            print(f"  ✅ SpectralDB using TorusRegistry wrapper")
        else:
            print(f"  ⚠️  SpectralDB may not be using TorusRegistry")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_eigensentry_integration():
    """Test EigenSentry with observer tokens"""
    print("\n🛡️ Testing EigenSentry integration...")
    
    try:
        from alan_backend.eigensentry_guard import CurvatureAwareGuard
        from python.core.observer_synthesis import get_observer_synthesis
        
        # Create instance
        guard = CurvatureAwareGuard()
        
        # Test with some eigenvalues
        test_eigenvalues = np.array([0.02, 0.01, 0.005])
        test_state = np.random.randn(100)
        
        # Get initial token count
        synthesis = get_observer_synthesis()
        initial_tokens = synthesis.metrics['tokens_generated']
        
        # Check eigenvalues (should emit token)
        action = guard.check_eigenvalues(test_eigenvalues, test_state)
        
        print(f"  ✅ Check completed:")
        print(f"     Action: {action['action']}")
        print(f"     Threshold: {action['threshold']:.3f}")
        print(f"     Curvature: {action['curvature']:.3f}")
        
        # Verify token was emitted
        final_tokens = synthesis.metrics['tokens_generated']
        if final_tokens > initial_tokens:
            print(f"  ✅ Observer token emitted ({final_tokens - initial_tokens} new)")
        else:
            print(f"  ⚠️  No observer token emitted")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def check_no_database_imports():
    """Verify no database imports remain"""
    print("\n🔍 Checking for database imports...")
    
    forbidden = ['sqlite3', 'psycopg2', 'sqlalchemy', 'pymongo', 'redis']
    found_imports = []
    
    # Check Python path for alan_backend
    backend_path = Path(__file__).parent
    
    for py_file in backend_path.rglob("*.py"):
        if "backup_pre_nodb" in str(py_file):
            continue
            
        try:
            content = py_file.read_text()
            for db_module in forbidden:
                if f"import {db_module}" in content or f"from {db_module}" in content:
                    found_imports.append((py_file.name, db_module))
        except:
            pass
            
    if found_imports:
        print(f"  ❌ Found {len(found_imports)} database imports:")
        for file, module in found_imports:
            print(f"     {file}: {module}")
        return False
    else:
        print(f"  ✅ No database imports found")
        return True

def main():
    """Run all tests"""
    print("="*60)
    print("🧪 TORI/ALAN No-DB Migration Verification")
    print("="*60)
    
    # Check environment
    state_root = os.getenv("TORI_STATE_ROOT")
    if not state_root:
        print("⚠️  TORI_STATE_ROOT not set. Using default: /var/lib/tori")
        print("   Set with: export TORI_STATE_ROOT=C:\\tori_state")
    else:
        print(f"✅ TORI_STATE_ROOT = {state_root}")
    
    # Run tests
    tests = [
        ("TorusRegistry", test_torus_registry),
        ("TorusCells", test_torus_cells),
        ("ObserverSynthesis", test_observer_synthesis),
        ("OriginSentry Integration", test_origin_sentry_integration),
        ("EigenSentry Integration", test_eigensentry_integration),
        ("No Database Imports", check_no_database_imports),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Migration successful!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
