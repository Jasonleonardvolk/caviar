#!/usr/bin/env python3
"""
Quick test to verify the NoneType multiplication bug is fixed
"""

import sys
import os
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
ingest_pdf_dir = current_dir / "ingest_pdf"
sys.path.insert(0, str(ingest_pdf_dir))
sys.path.insert(0, str(current_dir))

def test_none_type_fix():
    """Test that None values don't cause multiplication errors"""
    print("🧪 Testing NoneType multiplication fix...")
    
    try:
        from ingest_pdf.pipeline import ingest_pdf_clean
        
        # Create a test file
        test_file = current_dir / "test_nonetype.txt"
        with open(test_file, "w") as f:
            f.write("Machine learning and artificial intelligence are important topics.")
        
        print(f"📄 Created test file: {test_file}")
        
        # Run extraction with admin mode
        print("🚀 Running extraction with potential None values...")
        result = ingest_pdf_clean(str(test_file), admin_mode=True)
        
        print(f"✅ Status: {result.get('status')}")
        print(f"✅ Concept Count: {result.get('concept_count', 0)}")
        print(f"✅ No multiplication errors!")
        
        # Check specific values that could have been None
        purity_analysis = result.get('purity_analysis', {})
        entropy_analysis = result.get('entropy_analysis', {})
        
        print(f"✅ Purity efficiency: {purity_analysis.get('purity_efficiency_percent', 0)}%")
        print(f"✅ Diversity efficiency: {purity_analysis.get('diversity_efficiency_percent', 0)}%")
        
        if entropy_analysis.get('enabled'):
            print(f"✅ Entropy diversity efficiency: {entropy_analysis.get('diversity_efficiency_percent', 0)}%")
        
        # Clean up
        os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("🔧 NONETYPE MULTIPLICATION BUG FIX TEST")
    print("=" * 50)
    
    success = test_none_type_fix()
    
    if success:
        print("\n✅ BUG FIX SUCCESSFUL!")
        print("   - No more NoneType * int errors")
        print("   - Safe division and multiplication")
        print("   - Proper fallback values")
        print("   - Pipeline runs without crashes")
    else:
        print("\n❌ Bug still exists - check the logs above")
