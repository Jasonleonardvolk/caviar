#!/usr/bin/env python3
"""
Enhanced launcher with AV fix applied first
"""

# Apply AV fix BEFORE any imports
import sys
import types
from importlib.machinery import ModuleSpec

# Create the mock av module
av = types.ModuleType('av')
av.__spec__ = ModuleSpec("av", None)
av.__file__ = "<mock>"
av.__version__ = "10.0.0"
av.logging = types.ModuleType('av.logging')
av.logging.ERROR = 0
av.logging.WARNING = 1
av.logging.INFO = 2
av.logging.DEBUG = 3
av.logging.set_level = lambda x: None

# Register in sys.modules
sys.modules['av'] = av
sys.modules['av.logging'] = av.logging

print("AV compatibility fix applied")

# Now import and run the enhanced launcher
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_launcher import main
    print("Starting TORI enhanced launcher...")
    main()
except Exception as e:
    print(f"Error starting launcher: {e}")
    import traceback
    traceback.print_exc()
