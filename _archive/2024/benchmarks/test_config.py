#!/usr/bin/env python3
"""
🔧 CONFIG TEST - Verify configuration loading works
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_config_loading():
    """Test that configuration loads without errors"""
    
    print("🔍 TESTING CONFIG LOADING")
    print("=" * 40)
    
    try:
        # Test import
        from alan_backend.config import cfg
        print("✅ Config import: SUCCESS")
        
        # Test basic structure
        print(f"✅ Config type: {type(cfg)}")
        print(f"✅ Config keys: {list(cfg.keys())}")
        
        # Test key values
        extraction_enabled = cfg.get("extraction", {}).get("enabled", False)
        koopman_enabled = cfg.get("koopman_enabled", False)
        max_concepts = cfg.get("output", {}).get("max_concepts_returned", 0)
        
        print(f"✅ Extraction enabled: {extraction_enabled}")
        print(f"✅ Koopman enabled: {koopman_enabled}")
        print(f"✅ Max concepts: {max_concepts}")
        
        # Test method configurations
        methods = cfg.get("methods", {})
        print(f"✅ Available methods: {list(methods.keys())}")
        
        for method, config in methods.items():
            enabled = config.get("enabled", False)
            print(f"   - {method}: {'enabled' if enabled else 'disabled'}")
        
        print("\n🎉 CONFIG TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ CONFIG TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_loading()
    
    if success:
        print("\n🚀 Ready for extraction pipeline!")
        print("Your imports should now work:")
        print("   from alan_backend.config import cfg")
    else:
        print("\n🔧 Need to debug configuration setup")
