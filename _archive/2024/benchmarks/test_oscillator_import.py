#!/usr/bin/env python3
"""
🔧 OSCILLATOR IMPORT TEST - Verify the missing import is fixed
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_oscillator_import():
    """Test that oscillator_update can be imported without errors"""
    
    print("🔍 TESTING OSCILLATOR IMPORT")
    print("=" * 40)
    
    try:
        # Test the specific import that was failing
        from alan_backend.banksy import oscillator_update
        print("✅ oscillator_update import: SUCCESS")
        
        # Test that it's actually the function we expect
        print(f"✅ Function type: {type(oscillator_update)}")
        print(f"✅ Function name: {oscillator_update.__name__}")
        
        # Test other imports that clustering.py might need
        from alan_backend.banksy import step
        print("✅ step import: SUCCESS")
        
        # Verify they're the same function (alias)
        if oscillator_update is step:
            print("✅ oscillator_update is correctly aliased to step")
        else:
            print("⚠️ oscillator_update and step are different functions")
        
        print("\n🎉 OSCILLATOR IMPORT TEST PASSED!")
        print("Your Python API server should now start successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ IMPORT FAILED: {e}")
        print("\n🔧 Import error details:")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_oscillator_import()
    
    if success:
        print("\n🚀 READY TO START THE API SERVER:")
        print('   uvicorn ingest_pdf.main:app --port 8002 --reload')
        print("\n🎯 Expected result:")
        print("   ✅ Server starts without ImportError")
        print("   ✅ API endpoints become available")
        print("   ✅ Upload will finally work end-to-end!")
    else:
        print("\n❌ IMPORT STILL FAILING")
        print("   Need to debug the import path further")
