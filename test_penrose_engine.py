#!/usr/bin/env python3
"""
Test script for Penrose Engine Rust Extension
Tests all major functions to ensure proper installation
"""

import sys
import traceback

def test_penrose_engine():
    """Test all functions of the Penrose Engine"""
    print("=" * 60)
    print("Penrose Engine Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Import
        print("\n1. Testing import...")
        import penrose_engine_rs
        print("   SUCCESS: Module imported")
        
        # Test 2: Initialize engine
        print("\n2. Testing engine initialization...")
        result = penrose_engine_rs.initialize_engine(
            max_threads=4,
            cache_size_mb=512,
            enable_gpu=False,
            precision="float32"
        )
        print(f"   SUCCESS: Engine initialized")
        print(f"   Result: {result}")
        
        # Test 3: Get engine info
        print("\n3. Testing get_engine_info...")
        info = penrose_engine_rs.get_engine_info()
        print(f"   SUCCESS: Got engine info")
        print(f"   Info: {info}")
        
        # Test 4: Compute similarity
        print("\n4. Testing compute_similarity...")
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        similarity = penrose_engine_rs.compute_similarity(v1, v2)
        print(f"   SUCCESS: Computed similarity = {similarity:.6f}")
        
        # Test 5: Batch similarity
        print("\n5. Testing batch_similarity...")
        query = [1.0, 0.0, 0.0]
        corpus = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        similarities = penrose_engine_rs.batch_similarity(query, corpus)
        print(f"   SUCCESS: Computed batch similarities")
        print(f"   Results: {similarities}")
        
        # Test 6: Evolve lattice field
        print("\n6. Testing evolve_lattice_field...")
        lattice = [[1.0, 2.0], [3.0, 4.0]]
        phase_field = [[0.0, 0.5], [1.0, 1.5]]
        curvature_field = [[0.1, 0.2], [0.3, 0.4]]
        evolved = penrose_engine_rs.evolve_lattice_field(
            lattice, phase_field, curvature_field, 0.01
        )
        print(f"   SUCCESS: Evolved lattice field")
        print(f"   Result shape: {len(evolved)}x{len(evolved[0])}")
        
        # Test 7: Compute phase entanglement
        print("\n7. Testing compute_phase_entanglement...")
        soliton_positions = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
        phases = [0.0, 1.57, 3.14]
        entanglement = penrose_engine_rs.compute_phase_entanglement(
            soliton_positions, phases, 0.5
        )
        print(f"   SUCCESS: Computed phase entanglement")
        print(f"   Matrix shape: {len(entanglement)}x{len(entanglement[0])}")
        
        # Test 8: Curvature to phase encode
        print("\n8. Testing curvature_to_phase_encode...")
        curvature = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        for mode in ["direct", "log_phase", "tanh"]:
            result = penrose_engine_rs.curvature_to_phase_encode(curvature, mode)
            print(f"   SUCCESS: Encoded with mode '{mode}'")
            print(f"   Phase shape: {len(result['phase'])}x{len(result['phase'][0])}")
            print(f"   Amplitude shape: {len(result['amplitude'])}x{len(result['amplitude'][0])}")
        
        # Test 9: Shutdown engine
        print("\n9. Testing shutdown_engine...")
        penrose_engine_rs.shutdown_engine()
        print("   SUCCESS: Engine shut down")
        
        # Final verification
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("Penrose Engine is properly installed and working.")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"\nERROR: Failed to import module")
        print(f"Details: {e}")
        print("\nSolution: Run 'build_penrose.ps1' or 'BUILD_PENROSE_ENGINE.bat' first")
        return False
        
    except Exception as e:
        print(f"\nERROR: Test failed")
        print(f"Details: {e}")
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = test_penrose_engine()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
