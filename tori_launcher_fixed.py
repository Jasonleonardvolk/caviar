#!/usr/bin/env python3
"""
TORI Launcher with Complete AV Fix
This applies the complete av mock before starting any TORI components
"""

# Apply complete AV fix BEFORE any imports
import sys
import types
from importlib.machinery import ModuleSpec

print("Applying complete AV compatibility fix...")

# Create complete av mock
av = types.ModuleType('av')
av.__spec__ = ModuleSpec("av", None)
av.__file__ = "<mock>"
av.__version__ = "10.0.0"

# av.logging
av.logging = types.ModuleType('av.logging')
av.logging.ERROR = 0
av.logging.WARNING = 1
av.logging.INFO = 2
av.logging.DEBUG = 3
av.logging.set_level = lambda x: None

# av.video with VideoFrame
av.video = types.ModuleType('av.video')
av.video.frame = types.ModuleType('av.video.frame')

class VideoFrame:
    pict_type = 'I'  # Critical attribute that torchvision checks
    def __init__(self):
        self.width = 0
        self.height = 0
        self.format = None

av.video.frame.VideoFrame = VideoFrame

# av.audio
av.audio = types.ModuleType('av.audio')
av.audio.frame = types.ModuleType('av.audio.frame')

class AudioFrame:
    def __init__(self):
        self.samples = 0
        self.rate = 44100

av.audio.frame.AudioFrame = AudioFrame

# Other modules
av.container = types.ModuleType('av.container')
av.container.Container = type('Container', (), {})
av.codec = types.ModuleType('av.codec')
av.codec.CodecContext = type('CodecContext', (), {})
av.filter = types.ModuleType('av.filter')
av.filter.Graph = type('Graph', (), {})
av.stream = types.ModuleType('av.stream')
av.stream.Stream = type('Stream', (), {})

# Register everything
sys.modules['av'] = av
sys.modules['av.logging'] = av.logging
sys.modules['av.video'] = av.video
sys.modules['av.video.frame'] = av.video.frame
sys.modules['av.audio'] = av.audio
sys.modules['av.audio.frame'] = av.audio.frame
sys.modules['av.container'] = av.container
sys.modules['av.codec'] = av.codec
sys.modules['av.filter'] = av.filter
sys.modules['av.stream'] = av.stream

print("AV compatibility fix applied successfully!")

# Now we can safely import and run TORI
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the launcher
try:
    print("\nStarting TORI enhanced launcher...")
    from enhanced_launcher import main
    main()
except Exception as e:
    print(f"\nError starting launcher: {e}")
    print("\nTrying alternative: isolated_startup...")
    try:
        import isolated_startup
        isolated_startup.main() if hasattr(isolated_startup, 'main') else None
    except Exception as e2:
        print(f"Error with isolated_startup: {e2}")
        import traceback
        traceback.print_exc()
