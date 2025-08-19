#!/usr/bin/env python3
"""Fix the photoMorphPipeline import in holographicRenderer.ts"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_import():
    file_path = Path(r"{PROJECT_ROOT}\frontend\lib\holographicRenderer.ts")
    
    if not file_path.exists():
        print("❌ holographicRenderer.ts not found")
        return
    
    content = file_path.read_text(encoding='utf-8')
    
    # Fix the import path
    old_import = "from './photoMorphPipeline'"
    new_import = "from '../../tori_ui_svelte/src/lib/webgpu/photoMorphPipeline'"
    
    if old_import in content:
        content = content.replace(old_import, new_import)
        file_path.write_text(content, encoding='utf-8')
        print(f"✅ Fixed import in holographicRenderer.ts")
        print(f"   Changed: {old_import}")
        print(f"   To: {new_import}")
    else:
        print("Import pattern not found, checking for variations...")
        # Check what the actual import looks like
        if 'photoMorphPipeline' in content:
            print("Found photoMorphPipeline reference, please check the file manually")

if __name__ == "__main__":
    fix_import()
