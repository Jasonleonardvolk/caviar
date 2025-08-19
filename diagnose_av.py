#!/usr/bin/env python3
"""
Diagnose av import issues
"""

import sys
import os

print("AV Module Diagnostic")
print("=" * 60)

# Check if av.py exists
av_file = os.path.join(os.path.dirname(__file__), 'av.py')
print(f"\nav.py exists: {os.path.exists(av_file)}")

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try importing av
print("\nAttempting to import av...")
try:
    import av
    print(f"SUCCESS: av imported from {av.__file__}")
    print(f"av.__version__ = {getattr(av, '__version__', 'NOT SET')}")
    
    # Check for critical attributes
    checks = [
        ("av.logging", hasattr(av, 'logging')),
        ("av.video", hasattr(av, 'video')),
        ("av.video.frame", hasattr(av.video, 'frame') if hasattr(av, 'video') else False),
        ("av.video.frame.VideoFrame", hasattr(av.video.frame, 'VideoFrame') if hasattr(av.video, 'frame') if hasattr(av, 'video') else False else False),
        ("VideoFrame.pict_type", hasattr(av.video.frame.VideoFrame, 'pict_type') if hasattr(av.video.frame, 'VideoFrame') if hasattr(av.video, 'frame') if hasattr(av, 'video') else False else False else False),
    ]
    
    print("\nAttribute checks:")
    for name, exists in checks:
        status = "EXISTS" if exists else "MISSING"
        print(f"  {name}: {status}")
    
    # Try the specific check that torchvision does
    print("\nTorchvision compatibility check:")
    try:
        if hasattr(av.video.frame.VideoFrame, 'pict_type'):
            print("  SUCCESS: av.video.frame.VideoFrame.pict_type exists")
        else:
            print("  FAILED: pict_type attribute missing")
    except Exception as e:
        print(f"  FAILED: {e}")
        
except Exception as e:
    print(f"FAILED to import av: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnosis complete.")
