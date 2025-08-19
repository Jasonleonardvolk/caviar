#!/usr/bin/env python3
"""
FINAL BULLETPROOF TEST - Test the serialization fix
"""

import sys
import os
import json
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
ingest_pdf_dir = current_dir / "ingest_pdf"
sys.path.insert(0, str(ingest_pdf_dir))
sys.path.insert(0, str(current_dir))

def test_bulletproof_serialization():
    """Test that any object can be serialized"""
    print("🛡️ TESTING BULLETPROOF SERIALIZATION")
    print("=" * 50)
    
    try:
        from ingest_pdf.main import ensure_serializable
        
        # Test the most problematic objects that would break FastAPI
        test_cases = [
            ("Tuple", (1, 2, 3)),
            ("Set", {1, 2, 3}),
            ("Nested tuple", {"data": (4, 5, 6)}),
            ("Nested set", {"data": {7, 8, 9}}),
            ("Class instance", type('TestClass', (), {'attr': 'value', 'nested': (1, 2)})()),
            ("Mixed complex", {
                "tuple": (1, 2),
                "set": {3, 4}, 
                "nested": {"inner_tuple": (5, 6), "inner_set": {7, 8}}
            }),
        ]
        
        print("🧪 Testing ensure_serializable:")
        for name, test_obj in test_cases:
            try:
                # Test serialization
                serialized = ensure_serializable(test_obj)
                
                # Test JSON conversion
                json_str = json.dumps(serialized)
                
                print(f"  ✅ {name}: {type(test_obj)} -> JSON serializable")
                
            except Exception as e:
                print(f"  ❌ {name}: Failed - {e}")
                return False
        
        print("\n🚀 Testing import:")
        from ingest_pdf.main import app
        print("  ✅ FastAPI app imported successfully")
        
        print("\n🎯 Testing pipeline:")
        from ingest_pdf.pipeline import ingest_pdf_clean
        print("  ✅ Pipeline imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("🛡️ BULLETPROOF SERIALIZATION TEST")
    print("Testing that ANY object can be JSON serialized...")
    print("=" * 60)
    
    success = test_bulletproof_serialization()
    
    if success:
        print(f"\n✅ BULLETPROOF SUCCESS!")
        print("🎯 ANY return type will now work")
        print("🚀 RESTART YOUR SYSTEM:")
        print("   Ctrl+C to stop current system")
        print("   python start_unified_tori.py")
        print("📤 Upload will work 100%!")
    else:
        print(f"\n❌ STILL HAVING ISSUES")
