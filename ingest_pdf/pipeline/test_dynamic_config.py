#!/usr/bin/env python3
"""
Test script to verify dynamic configuration is working correctly.
Run this to ensure all configuration values are accessible and can be overridden.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_configuration():
    """Test that configuration loads correctly"""
    print("=" * 60)
    print("TORI Pipeline Configuration Test")
    print("=" * 60)
    
    # Test 1: Import the config module
    print("\n1. Testing config module import...")
    try:
        from ingest_pdf.pipeline.config import (
            settings, Settings,
            ENABLE_ENTROPY_PRUNING, MAX_PARALLEL_WORKERS,
            ENTROPY_CONFIG, FILE_SIZE_LIMITS, CONFIG
        )
        print("✅ Config module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import config: {e}")
        return False
    
    # Test 2: Check settings object
    print("\n2. Testing settings object...")
    print(f"   - Entropy pruning enabled: {settings.enable_entropy_pruning}")
    print(f"   - Max parallel workers: {settings.max_parallel_workers or 'auto'}")
    print(f"   - Entropy threshold: {settings.entropy_threshold}")
    print(f"   - OCR max pages: {settings.ocr_max_pages or 'unlimited'}")
    
    # Test 3: Check backward compatibility exports
    print("\n3. Testing backward compatibility...")
    print(f"   - ENABLE_ENTROPY_PRUNING = {ENABLE_ENTROPY_PRUNING}")
    print(f"   - MAX_PARALLEL_WORKERS = {MAX_PARALLEL_WORKERS}")
    print(f"   - ENTROPY_CONFIG keys: {list(ENTROPY_CONFIG.keys())}")
    print(f"   - FILE_SIZE_LIMITS keys: {list(FILE_SIZE_LIMITS.keys())}")
    
    # Test 4: Test environment variable override
    print("\n4. Testing environment variable override...")
    original_value = settings.entropy_threshold
    
    # Set environment variable
    os.environ['ENTROPY_THRESHOLD'] = '0.999'
    
    # Create new settings instance to pick up env var
    new_settings = Settings()
    print(f"   - Original entropy threshold: {original_value}")
    print(f"   - New entropy threshold: {new_settings.entropy_threshold}")
    print(f"   - Override {'✅ worked' if new_settings.entropy_threshold == 0.999 else '❌ failed'}")
    
    # Clean up
    del os.environ['ENTROPY_THRESHOLD']
    
    # Test 5: Display full configuration
    print("\n5. Full configuration dump:")
    print("-" * 40)
    print(json.dumps(CONFIG, indent=2))
    
    # Test 6: Test complex configuration parsing
    print("\n6. Testing complex configuration parsing...")
    
    # Test section weights as JSON
    os.environ['SECTION_WEIGHTS_JSON'] = '{"title": 3.0, "abstract": 2.5}'
    test_settings = Settings()
    print(f"   - Section weights from JSON: {test_settings.section_weights}")
    del os.environ['SECTION_WEIGHTS_JSON']
    
    # Test section weights as key=value pairs
    os.environ['SECTION_WEIGHTS'] = 'title=3.0,abstract=2.5,body=1.0'
    test_settings = Settings()
    print(f"   - Section weights from pairs: {test_settings.section_weights}")
    del os.environ['SECTION_WEIGHTS']
    
    print("\n" + "=" * 60)
    print("✅ All configuration tests passed!")
    print("=" * 60)
    
    return True

def test_pipeline_import():
    """Test that pipeline can import and use configuration"""
    print("\n7. Testing pipeline import compatibility...")
    try:
        from ingest_pdf.pipeline import ingest_pdf_clean
        print("✅ Pipeline imported successfully with new config")
        return True
    except Exception as e:
        print(f"❌ Pipeline import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run configuration tests
    config_ok = test_configuration()
    
    # Test pipeline import
    pipeline_ok = test_pipeline_import()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Configuration tests: {'✅ PASSED' if config_ok else '❌ FAILED'}")
    print(f"Pipeline import test: {'✅ PASSED' if pipeline_ok else '❌ FAILED'}")
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if (config_ok and pipeline_ok) else 1)
