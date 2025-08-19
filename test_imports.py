#!/usr/bin/env python3
"""
Test the multimodal pipeline imports and basic functionality.
Run this to verify the PYTHONPATH fix worked.
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent))

print("Testing multimodal pipeline imports...")
print(f"Python path includes: {sys.path[0]}")

try:
    # Test core imports
    print("\n1. Testing router import...")
    from ingest_pdf.pipeline.router import ingest_file, set_hologram_bus
    print("✅ Router imported successfully!")
    
    print("\n2. Testing holographic bus import...")
    from ingest_pdf.pipeline.holographic_bus import get_event_bus, get_display_api
    print("✅ Holographic bus imported successfully!")
    
    print("\n3. Testing handler imports through router...")
    from ingest_pdf.pipeline import router
    print(f"✅ Router has {len(router.HANDLERS)} MIME type handlers")
    print(f"✅ Router has {len(router.EXTENSION_HANDLERS)} extension handlers")
    
    # List available handlers
    print("\n4. Available MIME type handlers:")
    for mime_type in router.HANDLERS.keys():
        print(f"   - {mime_type}")
    
    print("\n5. Available extension handlers:")
    for ext in router.EXTENSION_HANDLERS.keys():
        print(f"   - {ext}")
    
    # Test if audio/video are connected
    print("\n6. Checking audio/video integration:")
    if "audio/" in router.HANDLERS:
        print("✅ Audio handler connected")
    else:
        print("❌ Audio handler NOT connected")
        
    if "video/" in router.HANDLERS:
        print("✅ Video handler connected")
    else:
        print("❌ Video handler NOT connected")
    
    print("\n✅ ALL IMPORTS SUCCESSFUL! The multimodal pipeline is ready to use.")
    print("\nTo run the server:")
    print("  python unified_pipeline_example.py")
    
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're in the project root directory")
    print("2. Check that ingest_bus folder exists (renamed from ingest-bus)")
    print("3. Verify all handler files exist in their expected locations")
    sys.exit(1)
