#!/usr/bin/env python3
"""Fix the tsconfig.json issue in frontend folder"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json

def fix_tsconfig():
    print("🔧 FIXING TSCONFIG.JSON")
    print("=" * 50)
    
    # Fix the frontend/tsconfig.json
    frontend_tsconfig = Path(r"{PROJECT_ROOT}\frontend\tsconfig.json")
    
    if frontend_tsconfig.exists():
        print("📝 Fixing frontend/tsconfig.json...")
        
        # Create a standalone tsconfig that doesn't extend SvelteKit
        new_config = {
            "compilerOptions": {
                "target": "ES2020",
                "module": "ESNext",
                "lib": ["ES2020", "DOM", "DOM.Iterable"],
                "allowJs": True,
                "checkJs": True,
                "esModuleInterop": True,
                "forceConsistentCasingInFileNames": True,
                "resolveJsonModule": True,
                "skipLibCheck": True,
                "sourceMap": True,
                "strict": True,
                "moduleResolution": "bundler",
                "types": ["@webgpu/types"],
                "paths": {
                    "$lib/*": ["../tori_ui_svelte/src/lib/*"]
                }
            },
            "include": ["**/*.ts", "**/*.js"],
            "exclude": ["node_modules", "dist"]
        }
        
        # Backup original
        backup = frontend_tsconfig.with_suffix('.json.backup')
        frontend_tsconfig.rename(backup)
        print(f"📦 Backed up to {backup.name}")
        
        # Write new config
        with open(frontend_tsconfig, 'w') as f:
            json.dump(new_config, f, indent=2)
        
        print("✅ Fixed frontend/tsconfig.json")
    
    # Also check if we need to create .svelte-kit/tsconfig.json
    svelte_kit_dir = Path(r"{PROJECT_ROOT}\frontend\.svelte-kit")
    if not svelte_kit_dir.exists():
        print("\n📁 Creating .svelte-kit directory...")
        svelte_kit_dir.mkdir(parents=True)
        
        # Create a basic tsconfig for SvelteKit
        sk_tsconfig = svelte_kit_dir / "tsconfig.json"
        sk_config = {
            "compilerOptions": {
                "paths": {
                    "$lib": ["../src/lib"],
                    "$lib/*": ["../src/lib/*"]
                },
                "rootDirs": ["..", "./types"],
                "importsNotUsedAsValues": "error",
                "isolatedModules": True,
                "preserveValueImports": True,
                "lib": ["esnext", "DOM", "DOM.Iterable"],
                "moduleResolution": "node",
                "module": "esnext",
                "target": "esnext"
            },
            "include": [
                "ambient.d.ts",
                "./types/**/$types.d.ts",
                "../vite.config.ts",
                "../src/**/*.js",
                "../src/**/*.ts",
                "../src/**/*.svelte",
                "../tests/**/*.js",
                "../tests/**/*.ts",
                "../tests/**/*.svelte"
            ],
            "exclude": ["../node_modules/**", "./**"]
        }
        
        with open(sk_tsconfig, 'w') as f:
            json.dump(sk_config, f, indent=2)
        
        print("✅ Created .svelte-kit/tsconfig.json")
    
    print("\n✅ TSCONFIG FIXED!")
    print("\n🎯 Restart Vite and the error should be gone!")

if __name__ == "__main__":
    fix_tsconfig()
