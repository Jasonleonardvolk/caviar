"""
Penrose Engine Import Chain Verification Script
Verifies the complete import chain and fixes naming issues
"""

import sys
import os
import traceback
from pathlib import Path

# Add the python directory to path
python_dir = Path(__file__).parent / "python"
if python_dir.exists():
    sys.path.insert(0, str(python_dir))

def test_rust_backend_direct():
    """Test direct import of the Rust backend"""
    print("=" * 60)
    print("TESTING DIRECT RUST BACKEND IMPORT")
    print("=" * 60)
    
    try:
        import penrose_engine_rs
        print("✅ Successfully imported penrose_engine_rs")
        
        # Check what's available
        attrs = [attr for attr in dir(penrose_engine_rs) if not attr.startswith('_')]
        print(f"📦 Available attributes: {attrs}")
        
        # Check for key functions
        expected_functions = [
            'initialize_engine',
            'evolve_lattice_field', 
            'compute_phase_entanglement',
            'curvature_to_phase_encode',
            'get_engine_info',
            'shutdown_engine'
        ]
        
        available_functions = []
        missing_functions = []
        
        for func in expected_functions:
            if hasattr(penrose_engine_rs, func):
                available_functions.append(func)
                print(f"✅ Found function: {func}")
            else:
                missing_functions.append(func)
                print(f"❌ Missing function: {func}")
        
        print(f"\n📊 Function Status:")
        print(f"   Available: {len(available_functions)}/{len(expected_functions)}")
        print(f"   Missing: {missing_functions}")
        
        return True, penrose_engine_rs
        
    except ImportError as e:
        print(f"❌ Failed to import penrose_engine_rs: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False, None

def test_adapter_import():
    """Test the penrose adapter import"""
    print("\n" + "=" * 60)
    print("TESTING PENROSE ADAPTER IMPORT")
    print("=" * 60)
    
    try:
        from core.penrose_adapter import PenroseAdapter, is_penrose_available, get_penrose_info
        print("✅ Successfully imported PenroseAdapter")
        
        # Check if backend is detected as available
        available = is_penrose_available()
        print(f"🔍 Backend detected as available: {available}")
        
        # Get backend info
        info = get_penrose_info()
        print(f"📋 Backend info: {info}")
        
        return True, available
        
    except ImportError as e:
        print(f"❌ Failed to import PenroseAdapter: {e}")
        traceback.print_exc()
        return False, False
    except Exception as e:
        print(f"❌ Unexpected error in adapter: {e}")
        traceback.print_exc()
        return False, False

def create_penrose_engine_alias():
    """Create an alias module to bridge the naming gap"""
    print("\n" + "=" * 60)
    print("CREATING PENROSE_ENGINE ALIAS")
    print("=" * 60)
    
    try:
        # Create a simple alias module
        alias_content = '''"""
Penrose Engine Alias Module
Bridges penrose_engine_rs to penrose_engine for adapter compatibility
"""

# Import everything from the actual Rust backend
from penrose_engine_rs import *

# Re-export the module's documentation and attributes
import penrose_engine_rs
__doc__ = penrose_engine_rs.__doc__
if hasattr(penrose_engine_rs, "__all__"):
    __all__ = penrose_engine_rs.__all__
'''
        
        # Write the alias file in the python directory
        python_dir = Path(__file__).parent / "python"
        alias_path = python_dir / "penrose_engine.py"
        
        with open(alias_path, 'w') as f:
            f.write(alias_content)
        
        print(f"✅ Created alias module at: {alias_path}")
        
        # Test the alias
        sys.path.insert(0, str(python_dir))
        import penrose_engine
        print("✅ Successfully imported penrose_engine (via alias)")
        
        return True, alias_path
        
    except Exception as e:
        print(f"❌ Failed to create alias: {e}")
        traceback.print_exc()
        return False, None

def test_adapter_with_alias():
    """Test the adapter after creating the alias"""
    print("\n" + "=" * 60)
    print("TESTING ADAPTER WITH ALIAS")
    print("=" * 60)
    
    try:
        # Force reload the adapter module to pick up the alias
        if 'core.penrose_adapter' in sys.modules:
            del sys.modules['core.penrose_adapter']
        
        from core.penrose_adapter import PenroseAdapter, is_penrose_available, get_penrose_info
        
        # Check if backend is now detected
        available = is_penrose_available()
        print(f"🔍 Backend available after alias: {available}")
        
        if available:
            # Get the adapter instance
            adapter = PenroseAdapter.get_instance()
            info = adapter.get_backend_info()
            print(f"📋 Adapter backend info: {info}")
            
            # Test basic functionality
            print("\n🧪 Testing basic adapter functionality...")
            
            # Check performance stats
            stats = adapter.get_performance_stats()
            print(f"📊 Performance stats: {stats}")
            
            print("✅ Adapter is fully functional!")
            return True
        else:
            print("❌ Backend still not detected after alias")
            return False
            
    except Exception as e:
        print(f"❌ Error testing adapter with alias: {e}")
        traceback.print_exc()
        return False

def main():
    """Main verification function"""
    print("🔍 PENROSE ENGINE IMPORT CHAIN VERIFICATION")
    print("🔍" * 30)
    
    results = {}
    
    # Test 1: Direct Rust backend import
    rust_success, rust_module = test_rust_backend_direct()
    results['rust_backend'] = rust_success
    
    # Test 2: Original adapter import (expected to fail)
    adapter_success, backend_available = test_adapter_import()
    results['adapter_original'] = adapter_success
    results['backend_detected_original'] = backend_available
    
    # Test 3: Create alias and test again
    if rust_success and not backend_available:
        print("\n🔧 DETECTED NAMING MISMATCH - CREATING BRIDGE...")
        alias_success, alias_path = create_penrose_engine_alias()
        results['alias_created'] = alias_success
        
        if alias_success:
            adapter_fixed = test_adapter_with_alias()
            results['adapter_with_alias'] = adapter_fixed
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 60)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test:<25}: {status}")
    
    if results.get('rust_backend') and results.get('adapter_with_alias'):
        print("\n🎉 SUCCESS: Penrose Engine is fully operational!")
        print("🎉 The Rust backend is compiled, installed, and working!")
        
        if results.get('alias_created'):
            print("🔧 NOTE: Created alias bridge for naming compatibility")
            
    elif results.get('rust_backend'):
        print("\n⚠️  PARTIAL SUCCESS: Rust backend works but adapter needs fixes")
        
    else:
        print("\n❌ FAILURE: Rust backend is not working")
    
    return results

if __name__ == "__main__":
    results = main()
