"""
Test DLL loading for intent_router
"""

import os
import sys
import ctypes
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    # Get the path to the DLL
    dll_path = Path(r"{PROJECT_ROOT}\concept_mesh\intent_router\target\release\intent_router.dll").absolute()
    
    print(f"Looking for DLL at: {dll_path}")
    print(f"DLL exists: {os.path.exists(dll_path)}")
    
    try:
        # Try to load the DLL
        lib = ctypes.cdll.LoadLibrary(str(dll_path))
        print("Successfully loaded the DLL!")
        
        # Try to get function pointers
        function_names = ["route_intent", "add_routing_context", "get_routing_contexts", "clear_routing_contexts"]
        for name in function_names:
            try:
                func = getattr(lib, name, None)
                print(f"Function '{name}' found: {func is not None}")
            except Exception as e:
                print(f"Error finding function '{name}': {e}")
        
    except Exception as e:
        print(f"Error loading DLL: {e}")
    
    # Print Python version and architecture info
    print(f"\nPython version: {sys.version}")
    print(f"Python is {'64' if sys.maxsize > 2**32 else '32'}-bit")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"Running in virtual environment: {in_venv}")
    
    # Try alternative loading methods
    try:
        from ctypes import WinDLL
        lib2 = WinDLL(str(dll_path))
        print("Successfully loaded the DLL using WinDLL!")
    except Exception as e:
        print(f"Error loading DLL with WinDLL: {e}")
    
    # Check system architecture
    import platform
    print(f"System architecture: {platform.architecture()}")
    print(f"Platform: {platform.platform()}")

if __name__ == "__main__":
    main()
