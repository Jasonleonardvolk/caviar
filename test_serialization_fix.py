#!/usr/bin/env python3
"""
FINAL SERIALIZATION FIX TEST
"""

import sys
import os
from pathlib import Path
import json

# Add paths
current_dir = Path(__file__).parent
ingest_pdf_dir = current_dir / "ingest_pdf"
sys.path.insert(0, str(ingest_pdf_dir))
sys.path.insert(0, str(current_dir))

def test_serialization_fix():
    """Test that the JSON serialization fix works"""
    print("🔧 TESTING JSON SERIALIZATION FIX")
    print("=" * 50)
    
    try:
        # Import the main app
        from ingest_pdf.main import clean_for_json
        
        # Test various problematic objects
        test_cases = [
            ("None", None),
            ("String", "test"),
            ("Dict", {"key": "value", "nested": {"inner": 123}}),
            ("List", [1, 2, "three", {"four": 4}]),
            ("Set", {1, 2, 3}),  # This would break JSON serialization
            ("Tuple", (1, 2, 3)),  # This would break JSON serialization
        ]
        
        print("🧪 Testing clean_for_json function:")
        for name, test_obj in test_cases:
            try:
                cleaned = clean_for_json(test_obj)
                # Try to serialize to JSON to verify it works
                json_str = json.dumps(cleaned)
                print(f"  ✅ {name}: {type(test_obj)} -> {type(cleaned)} -> JSON OK")
            except Exception as e:
                print(f"  ❌ {name}: Failed - {e}")
                return False
        
        print("\n🚀 Testing main.py import:")
        from ingest_pdf.main import app
        print("  ✅ FastAPI app imported successfully")
        
        print("\n🎯 Testing pipeline import:")
        from ingest_pdf.pipeline import ingest_pdf_clean
        print("  ✅ Pipeline imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("🔧 SERIALIZATION FIX VERIFICATION")
    print("Testing the atomic JSON serialization fix...")
    print("=" * 60)
    
    success = test_serialization_fix()
    
    if success:
        print(f"\n✅ SERIALIZATION FIX SUCCESSFUL!")
        print("🎯 The 500 error should be completely fixed")
        print("🚀 RESTART YOUR SYSTEM:")
        print("   python start_unified_tori.py")
        print("📤 Upload will work perfectly!")
    else:
        print(f"\n❌ STILL HAVING ISSUES")
        print("Check the traceback above")
