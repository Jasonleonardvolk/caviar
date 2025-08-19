import os
import shutil
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("🔥 NUCLEAR CACHE CLEAR FOR TORI")
print("=" * 50)

base_dir = Path(r"{PROJECT_ROOT}")
frontend_dir = base_dir / "tori_ui_svelte"

# 1. Clear all Svelte/Vite caches
caches_to_clear = [
    frontend_dir / ".svelte-kit",
    frontend_dir / "node_modules" / ".vite", 
    frontend_dir / ".vite",
    frontend_dir / "dist",
    frontend_dir / "build"
]

for cache_dir in caches_to_clear:
    if cache_dir.exists():
        print(f"🗑️ Deleting {cache_dir.name}...")
        try:
            shutil.rmtree(cache_dir)
            print(f"✅ Deleted {cache_dir}")
        except Exception as e:
            print(f"❌ Failed to delete {cache_dir}: {e}")

# 2. Also clear any temp files
temp_patterns = [
    "*.tmp",
    "*.cache",
    ".parcel-cache"
]

for pattern in temp_patterns:
    for temp_file in frontend_dir.glob(pattern):
        try:
            temp_file.unlink()
            print(f"✅ Deleted {temp_file.name}")
        except:
            pass

print("\n✅ All caches cleared!")
print("\nNEXT STEPS:")
print("1. Close ALL browser tabs with localhost:5173")
print("2. Stop the dev server (Ctrl+C)")
print("3. Restart with: python enhanced_launcher.py")
print("4. Open in INCOGNITO/PRIVATE browsing mode")
print("5. Press F12 and check console for errors")
