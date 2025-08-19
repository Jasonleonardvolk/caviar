#!/usr/bin/env python3
"""
Quick integration script to add audio/visual capabilities to Prajna
"""

import sys
from pathlib import Path

def integrate_audio_visual():
    """Add import and initialization to prajna_api.py"""
    
    # Path to prajna_api.py
    api_path = Path("prajna/api/prajna_api.py")
    
    if not api_path.exists():
        print(f"❌ Cannot find {api_path}")
        return False
    
    # Read current content
    with open(api_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already integrated
    if "audio_visual" in content:
        print("✅ Audio/Visual already integrated")
        return True
    
    # Find where to add the import (after other imports)
    import_marker = "from sse_starlette.sse import EventSourceResponse"
    import_pos = content.find(import_marker)
    
    if import_pos == -1:
        print("❌ Cannot find import marker")
        return False
    
    # Find end of line
    line_end = content.find("\n", import_pos)
    
    # Add audio/visual import
    av_import = """

# Import Audio/Visual enhancements
try:
    from prajna.api.audio_visual import create_audio_visual_endpoints, avatar_state
    AUDIO_VISUAL_AVAILABLE = True
    print("[Prajna] Audio/Visual module loaded successfully!")
except ImportError as e:
    print(f"[Prajna] Audio/Visual not available: {e}")
    AUDIO_VISUAL_AVAILABLE = False
    create_audio_visual_endpoints = None"""
    
    # Insert import
    new_content = content[:line_end] + av_import + content[line_end:]
    
    # Find where app is created (to add endpoints)
    app_marker = 'app = FastAPI('
    app_pos = new_content.find(app_marker)
    
    if app_pos == -1:
        print("❌ Cannot find FastAPI app creation")
        return False
    
    # Find the end of app creation (closing parenthesis)
    paren_count = 1
    pos = app_pos + len(app_marker)
    while paren_count > 0 and pos < len(new_content):
        if new_content[pos] == '(':
            paren_count += 1
        elif new_content[pos] == ')':
            paren_count -= 1
        pos += 1
    
    # Add audio/visual initialization after app creation
    av_init = """

# Initialize Audio/Visual endpoints
if AUDIO_VISUAL_AVAILABLE and create_audio_visual_endpoints:
    app = create_audio_visual_endpoints(app)
    print("[Prajna] ✅ Audio/Visual endpoints initialized")"""
    
    # Insert initialization
    new_content = new_content[:pos] + av_init + new_content[pos:]
    
    # Write updated content
    with open(api_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Audio/Visual integration complete!")
    return True

def create_requirements():
    """Create requirements file for audio/visual dependencies"""
    
    requirements = """# Audio/Visual Requirements for Prajna
openai-whisper>=20230314
edge-tts>=6.1.9
opencv-python>=4.8.0
transformers>=4.35.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
numpy>=1.24.0
"""
    
    with open("prajna/requirements_av.txt", 'w') as f:
        f.write(requirements)
    
    print("✅ Created requirements_av.txt")

def create_install_script():
    """Create installation script for audio/visual dependencies"""
    
    script = """#!/usr/bin/env python3
'''
Install Audio/Visual dependencies for Prajna
'''

import subprocess
import sys

def install_packages():
    packages = [
        "openai-whisper",
        "edge-tts", 
        "opencv-python",
        "transformers",
        "torch",
        "torchvision",
        "Pillow",
        "numpy"
    ]
    
    print("🎭 Installing Audio/Visual packages for Prajna...")
    
    for package in packages:
        print(f"\\n📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed")
        except Exception as e:
            print(f"⚠️ Failed to install {package}: {e}")
            print("   You may need to install it manually")
    
    print("\\n✨ Installation complete!")
    print("\\nNote: For GPU support with PyTorch, you may need to install the CUDA version:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    install_packages()
"""
    
    script_path = Path("prajna/install_av_deps.py")
    with open(script_path, 'w') as f:
        f.write(script)
    
    # Make executable on Unix
    import stat
    import os
    if hasattr(os, 'chmod'):
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)
    
    print("✅ Created install_av_deps.py")

def main():
    print("🎭 Integrating Audio/Visual capabilities into Prajna\n")
    
    # Integrate into API
    if not integrate_audio_visual():
        print("\n❌ Integration failed!")
        return
    
    # Create requirements file
    create_requirements()
    
    # Create install script
    create_install_script()
    
    print("\n✨ Integration complete!")
    print("\nNext steps:")
    print("1. Install dependencies: python prajna/install_av_deps.py")
    print("2. Restart Prajna API")
    print("3. Test with: python test_prajna_audio_visual.py")
    print("\nNew endpoints:")
    print("- POST /api/answer/audio - Audio transcription + Prajna response")
    print("- POST /api/answer/video - Video analysis + Prajna response")
    print("- POST /api/answer/image - Image analysis + Prajna response")
    print("- GET /api/avatar/state - Current avatar state")
    print("- WS /api/avatar/updates - Real-time avatar updates")
    print("- POST /api/tts/generate - Text-to-speech generation")

if __name__ == "__main__":
    main()
