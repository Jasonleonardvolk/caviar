#!/usr/bin/env python3
"""Fix RealGhostEngine imports to use the correct paths"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_imports():
    print("🔧 Fixing RealGhostEngine imports to use correct paths")
    print("=" * 60)
    
    # Files to fix
    files_to_fix = [
        "realGhostEngine.js",
        "realGhostEngine_v2.js"
    ]
    
    base_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib")
    
    for filename in files_to_fix:
        file_path = base_path / filename
        
        if not file_path.exists():
            print(f"❌ {filename} not found")
            continue
            
        print(f"\n📝 Fixing {filename}...")
        
        content = file_path.read_text(encoding='utf-8')
        
        # Fix the imports
        replacements = [
            # The holographicEngine import is wrong - it should be SpectralHologramEngine from holographicEngine.ts
            ("import { SpectralHologramEngine } from './holographicEngine';",
             "import { SpectralHologramEngine } from '../../../frontend/lib/holographicEngine.js';"),
             
            ("import { ToriHolographicRenderer } from './holographicRenderer';",
             "import { ToriHolographicRenderer } from '../../../frontend/lib/holographicRenderer.js';"),
             
            ("import { FFTCompute } from './webgpu/fftCompute';",
             "import { FFTCompute } from '../../../frontend/lib/webgpu/fftCompute.js';"),
             
            ("import { HologramPropagation } from './webgpu/hologramPropagation';",
             "import { HologramPropagation } from '../../../frontend/lib/webgpu/hologramPropagation.js';"),
             
            ("import { QuiltGenerator } from './webgpu/quiltGenerator';",
             "import { QuiltGenerator } from '../../../frontend/lib/webgpu/quiltGenerator.js';"),
        ]
        
        original_content = content
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                print(f"  ✅ Fixed import: {old.split('from')[1].strip()}")
        
        if content != original_content:
            # Backup original
            backup_path = file_path.with_suffix('.js.backup')
            file_path.rename(backup_path)
            print(f"  📦 Backed up to {backup_path.name}")
            
            # Write fixed version
            file_path.write_text(content, encoding='utf-8')
            print(f"  ✅ Saved fixed {filename}")
        else:
            print(f"  ℹ️ No changes needed for {filename}")
    
    print("\n✅ Import fixes complete!")
    print("\n🎯 Next steps:")
    print("1. The imports now point to the ACTUAL TypeScript files in frontend/lib/")
    print("2. The .js extension will work with TypeScript files in Vite/SvelteKit")
    print("3. Restart the app and the imports should resolve correctly!")

if __name__ == "__main__":
    fix_imports()
