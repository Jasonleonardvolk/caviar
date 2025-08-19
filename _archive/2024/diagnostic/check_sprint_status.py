from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os
import sys
import subprocess
import json
from datetime import datetime

print("=" * 60)
print("TORI PROJECT STATUS CHECK")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
print()

# Check current directory
current_dir = os.getcwd()
expected_dir = r"{PROJECT_ROOT}"
if current_dir.lower() != expected_dir.lower():
    print(f"⚠️  Wrong directory! Please run from: {expected_dir}")
    print(f"   Current: {current_dir}")
    sys.exit(1)

print("✅ Correct directory")
print()

# Check git status
print("📊 Git Status:")
try:
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.stdout:
        print(f"   Uncommitted changes: {len(result.stdout.splitlines())} files")
    else:
        print("   ✅ Working directory clean")
except:
    print("   ❌ Git not available")

# Check if tag exists
try:
    result = subprocess.run(["git", "tag", "-l", "v0.11.0-hotfix"], capture_output=True, text=True)
    if result.stdout.strip():
        print("   ✅ Tag v0.11.0-hotfix exists")
    else:
        print("   ⚠️  Tag v0.11.0-hotfix not created yet")
except:
    pass

print()

# Check critical files
print("📁 Critical Files:")
critical_files = [
    ("Sprint prep script", "quick_sprint_prep.bat"),
    ("Cleanup script", "cleanup_repository.bat"),
    (".gitignore", ".gitignore"),
    ("Requirements lock", "requirements.lock"),
    ("CI workflow", ".github/workflows/build-concept-mesh.yml"),
    ("Soliton fix", "api/routes/soliton.py"),
    ("Frontend guard", "tori_ui_svelte/src/lib/services/solitonMemory.ts"),
    ("Albert directory", "albert/"),
]

all_exist = True
for name, path in critical_files:
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"   {status} {name}: {path}")
    if not exists:
        all_exist = False

print()

# Check for files that should NOT exist (cleaned up)
print("🧹 Cleanup Status:")
bad_files = [
    ("Log files", "logs/"),
    ("PyCache", "__pycache__/"),
    ("Egg info", "alan_core.egg-info/"),
    ("Duplicate scripts", "GREMLIN_HUNTER_MASTER_FIXED.ps1"),
]

cleanup_needed = False
for name, path in bad_files:
    exists = os.path.exists(path)
    if exists:
        print(f"   ⚠️  {name} still exists: {path}")
        cleanup_needed = True
    else:
        print(f"   ✅ {name} removed")

print()

# Summary
print("=" * 60)
print("SUMMARY:")
print("=" * 60)

if all_exist and not cleanup_needed:
    print("✅ All systems ready for Albert sprint!")
    print()
    print("Next steps:")
    print("1. Run: RUN_THIS_NOW.bat")
    print("2. Update README.md with your GitHub username/repo")
    print("3. Push to GitHub:")
    print("   git push origin main")
    print("   git push origin v0.11.0-hotfix")
else:
    print("⚠️  Some preparation still needed:")
    if cleanup_needed:
        print("   - Run cleanup_repository.bat to remove temporary files")
    if not all_exist:
        print("   - Some critical files are missing")
    print()
    print("Run: RUN_THIS_NOW.bat to fix everything automatically!")

print()
print("🚀 Ready to launch Albert sprint after preparation!")
