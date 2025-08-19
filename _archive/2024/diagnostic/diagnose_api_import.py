#!/usr/bin/env python3
"""
Diagnostic script to test if the full API can be imported
"""

import sys
import traceback
from pathlib import Path

def diagnose_api_import():
    """Diagnose issues with importing the full API"""
    
    print("🔍 Diagnosing Full API Import...\n")
    
    # Check directory structure
    script_dir = Path(__file__).parent
    prajna_dir = script_dir / "prajna"
    api_dir = prajna_dir / "api"
    
    print("📁 Directory Structure:")
    print(f"   Script dir: {script_dir}")
    print(f"   Prajna dir exists: {prajna_dir.exists()}")
    print(f"   API dir exists: {api_dir.exists()}")
    
    # Check for __init__.py files
    print("\n📄 Init Files:")
    prajna_init = prajna_dir / "__init__.py"
    api_init = api_dir / "__init__.py"
    print(f"   prajna/__init__.py exists: {prajna_init.exists()}")
    print(f"   prajna/api/__init__.py exists: {api_init.exists()}")
    
    # Check for prajna_api.py
    api_file = api_dir / "prajna_api.py"
    print(f"\n🐍 API Module:")
    print(f"   prajna/api/prajna_api.py exists: {api_file.exists()}")
    
    if api_file.exists():
        # Check file size
        size = api_file.stat().st_size
        print(f"   File size: {size} bytes")
        if size < 50:
            print("   ⚠️ Warning: File seems too small, might be empty or minimal")
    
    # Check PYTHONPATH
    print(f"\n🔧 Python Path:")
    print(f"   Current dir in path: {str(script_dir) in sys.path}")
    print(f"   sys.path[0]: {sys.path[0]}")
    
    # Try to import
    print("\n🚀 Attempting Import...")
    try:
        # Add current directory to path if not there
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
            print("   Added current directory to sys.path")
        
        # Try import
        from prajna.api.prajna_api import app
        print("   ✅ SUCCESS! Full API imported successfully")
        print(f"   App type: {type(app)}")
        print(f"   App module: {app.__module__ if hasattr(app, '__module__') else 'N/A'}")
        
        # Check for expected attributes
        if hasattr(app, 'routes'):
            print(f"   Number of routes: {len(app.routes)}")
            
            # List some routes
            print("\n   📍 Sample routes:")
            for i, route in enumerate(app.routes[:10]):  # Show first 10 routes
                if hasattr(route, 'path'):
                    print(f"      {i+1}. {route.path}")
                    
        # Check for expected endpoints
        print("\n   🔍 Checking for expected endpoints...")
        expected_endpoints = [
            "/api/v1/concepts",
            "/api/v1/concept-mesh/status",
            "/api/v1/soliton/init",
            "/api/v1/soliton/stats/{user}",
            "/api/v1/soliton/embed"
        ]
        
        if hasattr(app, 'routes'):
            route_paths = [route.path for route in app.routes if hasattr(route, 'path')]
            for endpoint in expected_endpoints:
                if any(endpoint in path for path in route_paths):
                    print(f"      ✅ Found: {endpoint}")
                else:
                    print(f"      ❌ Missing: {endpoint}")
            
    except ImportError as e:
        print(f"   ❌ ImportError: {e}")
        print(f"\n   Full traceback:")
        traceback.print_exc()
        
        # Try step-by-step import to find exact failure point
        print("\n   🔍 Step-by-step import test:")
        try:
            import prajna
            print("   ✅ import prajna - OK")
        except ImportError as e:
            print(f"   ❌ import prajna - FAILED: {e}")
            return
            
        try:
            import prajna.api
            print("   ✅ import prajna.api - OK")
        except ImportError as e:
            print(f"   ❌ import prajna.api - FAILED: {e}")
            return
            
        try:
            import prajna.api.prajna_api
            print("   ✅ import prajna.api.prajna_api - OK")
            print("   ⚠️ Module imported but 'app' not found")
            print("   Check that prajna_api.py contains: app = FastAPI()")
        except ImportError as e:
            print(f"   ❌ import prajna.api.prajna_api - FAILED: {e}")
            return
    
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        traceback.print_exc()
        
    print("\n💡 Next Steps:")
    print("   1. Run: python fix_prajna_init_files.py")
    print("   2. Ensure prajna/api/prajna_api.py exists with FastAPI app")
    print("   3. Run enhanced_launcher.py --api full")

if __name__ == "__main__":
    diagnose_api_import()
