"""
Diagnose why soliton API is returning 500 errors
"""
import sys
import os
from pathlib import Path

# Ensure we can import from the right locations
base_path = Path(__file__).parent
sys.path.insert(0, str(base_path))

print("🔍 SOLITON API DIAGNOSTIC")
print("=" * 60)

# 1. Check if the module exists
soliton_module_path = base_path / "mcp_metacognitive" / "core" / "soliton_memory.py"
print(f"\n1. Checking soliton_memory.py existence:")
print(f"   Path: {soliton_module_path}")
print(f"   Exists: {soliton_module_path.exists()}")

# 2. Try to import it
print("\n2. Testing import:")
try:
    from mcp_metacognitive.core import soliton_memory
    print("   ✅ Import successful!")
    
    # Check what functions are available
    print("\n3. Available functions:")
    for name in dir(soliton_memory):
        if not name.startswith('_') and callable(getattr(soliton_memory, name)):
            print(f"   - {name}")
            
    # Check specific required functions
    print("\n4. Required API functions:")
    required = ['initialize_user', 'get_user_stats', 'store_memory', 'recall_memories']
    for func_name in required:
        exists = hasattr(soliton_memory, func_name)
        print(f"   - {func_name}: {'✅ Found' if exists else '❌ Missing'}")
        
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    print("\n   Trying to diagnose why...")
    
    # Try importing step by step
    try:
        import mcp_metacognitive
        print("   ✅ Can import mcp_metacognitive")
    except ImportError as e2:
        print(f"   ❌ Cannot import mcp_metacognitive: {e2}")
        
    try:
        from mcp_metacognitive import core
        print("   ✅ Can import mcp_metacognitive.core")
    except ImportError as e3:
        print(f"   ❌ Cannot import mcp_metacognitive.core: {e3}")

# 5. Check the API route
print("\n5. Checking API route:")
api_route_path = base_path / "api" / "routes" / "soliton.py"
print(f"   Path: {api_route_path}")
print(f"   Exists: {api_route_path.exists()}")

# 6. Python path
print("\n6. Current Python path:")
for i, p in enumerate(sys.path[:5]):
    print(f"   [{i}] {p}")

print("\n📋 DIAGNOSIS:")
print("The 500 errors are likely because:")
print("1. The soliton_memory module can't be imported in the API context")
print("2. OR the required async functions are missing/not exported")
print("3. OR there's a circular import or other module loading issue")

print("\n🔧 TO FIX:")
print("1. Ensure PYTHONPATH includes the base directory")
print("2. Make sure __init__.py files exist in mcp_metacognitive and mcp_metacognitive/core")
print("3. Verify the async functions are defined at module level (not inside if __name__ == '__main__')")
