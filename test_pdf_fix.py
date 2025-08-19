#!/usr/bin/env python3
"""
TEST THE PDF/TEXT FILE FIX
"""

import sys
import os
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
ingest_pdf_dir = current_dir / "ingest_pdf"
sys.path.insert(0, str(ingest_pdf_dir))
sys.path.insert(0, str(current_dir))

def test_pdf_fix():
    """Test that the PDF/text file issue is fixed"""
    print("🔧 TESTING PDF/TEXT FILE FIX")
    print("=" * 40)
    
    try:
        from ingest_pdf.pipeline import ingest_pdf_clean
        
        # Test 1: Text file (should work now)
        test_txt = current_dir / "test_fix.txt"
        with open(test_txt, "w") as f:
            f.write("""
            A Rulebook for Arguments
            
            Abstract
            This document explores the fundamentals of logical reasoning and argumentation.
            
            Introduction
            Argumentation is a crucial skill in academic writing and critical thinking.
            Logic and reasoning form the backbone of effective arguments.
            
            Chapter 1: Basic Principles
            Arguments consist of premises and conclusions. The premises should 
            support the conclusion through valid logical reasoning.
            """)
        
        print(f"📄 Created test text file: {test_txt}")
        
        # Run extraction
        print("🚀 Testing text file extraction...")
        result = ingest_pdf_clean(str(test_txt), admin_mode=True)
        
        print(f"✅ Status: {result.get('status')}")
        print(f"✅ Concept count: {result.get('concept_count', 0)}")
        
        if result.get('concept_count', 0) > 0:
            concepts = result.get('concept_names', [])[:5]
            print(f"✅ Sample concepts: {', '.join(concepts)}")
        
        # Clean up
        os.remove(test_txt)
        
        if result.get('status') == 'success' and result.get('concept_count', 0) > 0:
            print("🎉 FIX SUCCESSFUL!")
            return True
        else:
            print("❌ Still having issues")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_pdf_fix()
    
    if success:
        print(f"\n✅ THE 500 ERROR SHOULD BE FIXED!")
        print("🚀 Restart your system:")
        print("   python start_unified_tori.py")
        print("📤 Upload should work now!")
    else:
        print(f"\n❌ Still having issues. Check the logs above.")
