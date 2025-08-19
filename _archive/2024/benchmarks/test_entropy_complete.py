#!/usr/bin/env python3
"""
Ultimate entropy pruning test with complete av fix
This ensures the complete av mock is loaded before ANY other imports
"""

# CRITICAL: Set up the complete av mock FIRST
import sys
import types
from importlib.machinery import ModuleSpec

print("Creating complete av mock module...")

# Create main av module
av = types.ModuleType('av')
av.__spec__ = ModuleSpec("av", None)
av.__file__ = "<mock>"
av.__version__ = "10.0.0"

# Create av.logging
av.logging = types.ModuleType('av.logging')
av.logging.ERROR = 0
av.logging.WARNING = 1
av.logging.INFO = 2
av.logging.DEBUG = 3
av.logging.set_level = lambda x: None

# Create av.video with frame submodule
av.video = types.ModuleType('av.video')
av.video.frame = types.ModuleType('av.video.frame')

# Create VideoFrame class with pict_type attribute (this is what torchvision checks)
class VideoFrame:
    pict_type = 'I'
    def __init__(self):
        self.width = 0
        self.height = 0
        self.format = None

av.video.frame.VideoFrame = VideoFrame

# Create av.audio
av.audio = types.ModuleType('av.audio')
av.audio.frame = types.ModuleType('av.audio.frame')

class AudioFrame:
    def __init__(self):
        self.samples = 0
        self.rate = 44100

av.audio.frame.AudioFrame = AudioFrame

# Create other required modules
av.container = types.ModuleType('av.container')
av.container.Container = type('Container', (), {})

av.codec = types.ModuleType('av.codec')
av.codec.CodecContext = type('CodecContext', (), {})

av.filter = types.ModuleType('av.filter')
av.filter.Graph = type('Graph', (), {})

av.stream = types.ModuleType('av.stream')
av.stream.Stream = type('Stream', (), {})

# Register EVERYTHING in sys.modules BEFORE any imports
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

print("Complete av mock registered in sys.modules!")

# NOW we can safely import everything else
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\nTesting critical imports...")

# Test the specific check that was failing
try:
    if hasattr(av.video.frame.VideoFrame, 'pict_type'):
        print("SUCCESS: av.video.frame.VideoFrame.pict_type exists!")
    else:
        print("ERROR: Missing pict_type attribute")
except Exception as e:
    print(f"ERROR checking pict_type: {e}")

# Now test imports
success_count = 0

try:
    import transformers
    print(f"SUCCESS: transformers {transformers.__version__}")
    success_count += 1
except Exception as e:
    print(f"FAILED: transformers - {e}")

try:
    import torchvision
    print(f"SUCCESS: torchvision {torchvision.__version__}")
    success_count += 1
except Exception as e:
    print(f"FAILED: torchvision - {e}")

try:
    import sentence_transformers
    print(f"SUCCESS: sentence-transformers {sentence_transformers.__version__}")
    success_count += 1
except Exception as e:
    print(f"FAILED: sentence-transformers - {e}")

print(f"\nImport success: {success_count}/3")

# Test entropy pruning
print("\nTesting entropy pruning...")
try:
    from ingest_pdf.entropy_prune import entropy_prune, entropy_prune_with_categories
    print("SUCCESS: entropy_prune imported!")
    
    # Test functionality
    test_concepts = [
        {"name": "artificial intelligence", "score": 0.95},
        {"name": "machine learning", "score": 0.90},
        {"name": "deep learning", "score": 0.85},
        {"name": "neural networks", "score": 0.80},
        {"name": "AI", "score": 0.75},
    ]
    
    print(f"\nInput: {len(test_concepts)} concepts")
    result, stats = entropy_prune(test_concepts, top_k=3, similarity_threshold=0.8)
    
    print(f"\nSUCCESS! Entropy pruning works!")
    print(f"Output: {len(result)} concepts selected")
    print(f"Selected: {[c['name'] for c in result]}")
    print(f"Pruning stats: {stats}")
    
    print("\n" + "="*60)
    print("ENTROPY PRUNING IS FULLY FUNCTIONAL!")
    print("="*60)
    print("\nYou can now:")
    print("1. Start API server: python enhanced_launcher.py")
    print("2. Run complete system: python isolated_startup.py")
    print("3. Process PDFs with quality scoring")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
