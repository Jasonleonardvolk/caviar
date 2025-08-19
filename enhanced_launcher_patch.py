from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#!/usr/bin/env python3
"""
Patch for enhanced_launcher.py to use quick API server instead of heavy prajna API
"""

import os
import sys

def apply_patch():
    """Apply the patch to enhanced_launcher.py"""
    launcher_path = r"{PROJECT_ROOT}\enhanced_launcher.py"
    
    # Read the current launcher
    with open(launcher_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup the original
    backup_path = launcher_path + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Created backup at: {backup_path}")
    
    # Replace the import
    content = content.replace(
        "from prajna.api.prajna_api import prajna_app",
        "from quick_api_server import app as prajna_app"
    )
    
    # Also add the quick_api_server to the imports section if needed
    if "import quick_api_server" not in content:
        # Find the imports section and add it
        import_section_end = content.find("# Global variables")
        if import_section_end > 0:
            content = content[:import_section_end] + "import quick_api_server\n" + content[import_section_end:]
    
    # Write the patched version
    with open(launcher_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Patch applied successfully!")
    print("The API server will now use the lightweight quick_api_server.py")
    print("\nTo revert, run: copy enhanced_launcher.py.backup enhanced_launcher.py")

if __name__ == "__main__":
    apply_patch()
