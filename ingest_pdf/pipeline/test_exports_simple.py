#!/usr/bin/env python3
"""
Test script to verify exports are correctly configured.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_exports():
    """Test that all important functions are exported correctly"""
    print("Testing Module Exports")
    print("-" * 40)
    
    # Test 1: Import from pipeline package
    print("\n1. Testing pipeline package exports:")
    from ingest_pdf import pipeline
    
    exports_to_check = [
        'run_sync',
        'await_sync', 
        'ProgressTracker',
        'ingest_pdf_clean',
        'preload_concept_database',
        'get_db'  # The key export we're checking
    ]
    
    all_passed = True
    for export in exports_to_check:
        if hasattr(pipeline, export):
            print(f"✅ {export} is available via 'from ingest_pdf import pipeline; pipeline.{export}'")
        else:
            print(f"❌ {export} is NOT available")
            all_passed = False
    
    # Test 2: Check __all__ in pipeline module
    print("\n2. Checking pipeline.__all__:")
    if hasattr(pipeline, '__all__'):
        print(f"pipeline.__all__ contains {len(pipeline.__all__)} exports:")
        for item in ['get_db', 'ProgressTracker', 'preload_concept_database']:
            if item in pipeline.__all__:
                print(f"  ✅ '{item}' in __all__")
            else:
                print(f"  ❌ '{item}' NOT in __all__")
                all_passed = False
    
    # Test 3: Direct import from pipeline.pipeline
    print("\n3. Testing direct imports from pipeline.pipeline:")
    from ingest_pdf.pipeline import pipeline as pipeline_module
    
    if hasattr(pipeline_module, '__all__'):
        print(f"pipeline.pipeline.__all__ = {pipeline_module.__all__}")
        if 'get_db' in pipeline_module.__all__:
            print("  ✅ 'get_db' in pipeline.py __all__")
        else:
            print("  ❌ 'get_db' NOT in pipeline.py __all__")
            all_passed = False
    
    # Test 4: Actual function availability
    print("\n4. Testing actual function calls:")
    try:
        # Test get_db
        db = pipeline.get_db()
        print(f"✅ get_db() works - returned ConceptDB with {len(db.storage)} concepts")
        
        # Test ProgressTracker
        progress = pipeline.ProgressTracker(total=10)
        print(f"✅ ProgressTracker works - created tracker")
        
    except Exception as e:
        print(f"❌ Function call failed: {e}")
        all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("=" * 60)
    print("EXPORT VERIFICATION TEST")
    print("=" * 60)
    
    if test_exports():
        print("\n✅ ALL EXPORTS CONFIGURED CORRECTLY!")
    else:
        print("\n❌ Some exports are missing or misconfigured")
        sys.exit(1)
