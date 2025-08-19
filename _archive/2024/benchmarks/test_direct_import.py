"""
Test direct import of the intent_router module
"""

import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    # Get the path to the DLL directory
    dll_dir = Path(r"{PROJECT_ROOT}\concept_mesh\intent_router\target\release").absolute()
    print(f"DLL directory: {dll_dir}")
    
    # Add to system path
    os.environ['PATH'] = str(dll_dir) + os.pathsep + os.environ.get('PATH', '')
    
    # Add to Python's module search path
    sys.path.insert(0, str(dll_dir))
    
    # Try direct import
    try:
        print("Attempting to import intent_router...")
        import intent_router
        print("Successfully imported intent_router module!")
        
        # Check available functions
        print("\nAvailable functions:")
        for name in dir(intent_router):
            if not name.startswith('_'):
                print(f"- {name}")
        
        # Try calling a function
        if hasattr(intent_router, 'is_available'):
            result = intent_router.is_available()
            print(f"\nResult of is_available(): {result}")
        
        if hasattr(intent_router, 'route_intent'):
            result = intent_router.route_intent("explain", "What is intent routing?")
            print(f"\nResult of route_intent(): {result}")
        
    except ImportError as e:
        print(f"ImportError: {e}")
    except Exception as e:
        print(f"Error: {e}")
    
    # List files in the directory
    print("\nFiles in DLL directory:")
    for file in os.listdir(str(dll_dir)):
        print(f"- {file}")

if __name__ == "__main__":
    main()
