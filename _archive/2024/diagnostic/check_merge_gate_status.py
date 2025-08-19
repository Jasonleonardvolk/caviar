import os
import subprocess
import re
from datetime import datetime

def check_item(name, check_func):
    """Run a check and print status"""
    try:
        status, details = check_func()
        icon = "✅" if status else "⚠️"
        print(f"{icon} {name}")
        if details:
            print(f"   {details}")
    except Exception as e:
        print(f"❌ {name}: {str(e)}")

def check_scripts_archive():
    """Check scripts_archive status"""
    if not os.path.exists("scripts_archive"):
        return True, "Already removed"
    
    # Count files
    file_count = 0
    total_size = 0
    for root, dirs, files in os.walk("scripts_archive"):
        for f in files:
            if f != "README.md":
                file_count += 1
                total_size += os.path.getsize(os.path.join(root, f))
    
    if file_count == 0:
        return True, "Already cleaned"
    else:
        size_kb = total_size / 1024
        return False, f"{file_count} files, {size_kb:.1f}KB - run FINAL_MERGE_GATE.bat"

def check_readme_badge():
    """Check if README has real GitHub path"""
    if not os.path.exists("README.md"):
        return False, "README.md not found"
    
    with open("README.md", 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "USERNAME/REPO" in content:
        return False, "Still has USERNAME/REPO placeholder"
    
    # Look for actual GitHub URL
    match = re.search(r'github\.com/([^/]+)/([^/\)]+)', content)
    if match:
        return True, f"Points to {match.group(1)}/{match.group(2)}"
    else:
        return False, "No GitHub URL found"

def check_git_tag():
    """Check for v0.12.0-pre-albert tag"""
    try:
        result = subprocess.run(
            ["git", "tag", "-l", "v0.12.0-pre-albert"],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            return True, "Tag exists"
        else:
            return False, "Tag not created yet"
    except:
        return False, "Git not available"

def check_concept_mesh_naming():
    """Check if Cargo.toml has been updated"""
    cargo_path = "concept-mesh/Cargo.toml"
    if not os.path.exists(cargo_path):
        return False, "Cargo.toml not found"
    
    with open(cargo_path, 'r') as f:
        content = f.read()
    
    if 'name = "concept_mesh_rs"' in content:
        return True, "Already renamed to concept_mesh_rs"
    elif 'name = "concept-mesh"' in content:
        return False, "Still named concept-mesh, needs update"
    else:
        return False, "Unknown package name"

def check_ci_workflow():
    """Check CI workflow exists"""
    workflow_path = ".github/workflows/build-concept-mesh.yml"
    if os.path.exists(workflow_path):
        return True, "Workflow exists"
    else:
        return False, "Workflow missing"

print("=" * 60)
print("FINAL MERGE GATE STATUS CHECK")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
print()

print("Checking 5 merge gate items:")
print()

check_item("1. Scripts Archive", check_scripts_archive)
check_item("2. README Badge", check_readme_badge)
check_item("3. Git Tag v0.12.0-pre-albert", check_git_tag)
check_item("4. Concept Mesh Naming", check_concept_mesh_naming)
check_item("5. CI Workflow", check_ci_workflow)

print()
print("=" * 60)

# Count issues
import sys
all_checks = [
    check_scripts_archive,
    check_readme_badge,
    check_git_tag,
    check_concept_mesh_naming,
    check_ci_workflow
]

issues = sum(1 for check in all_checks if not check()[0])

if issues == 0:
    print("✅ ALL CHECKS PASSED! Ready to push.")
    print()
    print("Final commands:")
    print("  git push origin main")
    print("  git push origin v0.12.0-pre-albert")
else:
    print(f"⚠️  {issues} items need attention")
    print()
    print("Run: FINAL_MERGE_GATE.bat to fix everything")

print("=" * 60)
