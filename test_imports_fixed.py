#!/usr/bin/env python3
"""
Test script to verify the import paths are fixed after adding __init__.py files
"""

import sys
from pathlib import Path

# Add project root to path (same as the band-aid fix)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing import paths after fix...")
print(f"Python path includes: {project_root}")
print("-" * 50)

# Test 1: Import the router
try:
    from ingest_pdf.pipeline.router import ingest_file
    print("✅ SUCCESS: Can import ingest_file from router")
except ImportError as e:
    print(f"❌ FAILED: Cannot import from router: {e}")

# Test 2: Import audio handler directly
try:
    from ingest_bus.audio import ingest_audio
    print("✅ SUCCESS: Can import ingest_audio from ingest_bus.audio")
except ImportError as e:
    print(f"❌ FAILED: Cannot import audio handler: {e}")

# Test 3: Import video handler directly  
try:
    from ingest_bus.video import ingest_video
    print("✅ SUCCESS: Can import ingest_video from ingest_bus.video")
except ImportError as e:
    print(f"❌ FAILED: Cannot import video handler: {e}")

# Test 4: Check if handle functions exist
try:
    from ingest_bus.audio.ingest_audio import handle as audio_handle
    from ingest_bus.video.ingest_video import handle as video_handle
    print("✅ SUCCESS: Both handle functions exist")
except ImportError as e:
    print(f"❌ FAILED: Cannot import handle functions: {e}")

print("-" * 50)
print("Import test complete!")
