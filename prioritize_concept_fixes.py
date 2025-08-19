#!/usr/bin/env python3
"""
Prioritized Fix List for Non-Canonical Concept Files
Groups files by priority for fixing
"""

import json
from pathlib import Path

# Define file categories by priority
CRITICAL_PATHS = [
    "prajna/api/prajna_api.py",
    "prajna_api.py",
    "prajna/memory/concept_mesh_api.py",
    "prajna/memory/soliton_interface.py",
    "ingest_pdf/cognitive_interface.py",
    "conversations/cognitive_interface.py"
]

BACKUP_PATTERN = "backups/"
TEST_PATTERN = "test_"
FIX_PATTERN = "fix_"
VERIFY_PATTERN = "verify_"
INIT_PATTERN = "init_"

def categorize_files():
    """Read audit results and categorize by priority"""
    
    # Parse the audit output
    files_to_fix = {
        "critical": [],     # Core system files
        "tests": [],        # Test files
        "fixers": [],       # Fix/migration scripts
        "verifiers": [],    # Verification scripts
        "initializers": [], # Init scripts
        "backups": [],      # Backup files
        "other": []         # Everything else
    }
    
    # Sample data from your audit (you'd parse fix_suggestions.txt in practice)
    audit_results = [
        "backups/20250719_165542/prajna_api.py",
        "prajna/api/prajna_api.py",
        "prajna_api.py",
        "prajna/memory/concept_mesh_api.py",
        "prajna/memory/soliton_interface.py",
        "ingest_pdf/cognitive_interface.py",
        "conversations/cognitive_interface.py",
        "test_concept_mesh_3_fixed.py",
        "test_concept_mesh_e2e.py",
        "fix_concept_db_critical.py",
        "fix_concept_mesh_import.py",
        "verify_canonical_concept_mesh.py",
        "init_concept_mesh_data.py",
        "check_concept_counts.py",
        "seed_concept_mesh.py"
    ]
    
    for filepath in audit_results:
        path = Path(filepath)
        filename = path.name
        
        # Categorize
        if any(critical in str(filepath) for critical in CRITICAL_PATHS):
            files_to_fix["critical"].append(filepath)
        elif BACKUP_PATTERN in filepath:
            files_to_fix["backups"].append(filepath)
        elif filename.startswith(TEST_PATTERN):
            files_to_fix["tests"].append(filepath)
        elif filename.startswith(FIX_PATTERN):
            files_to_fix["fixers"].append(filepath)
        elif filename.startswith(VERIFY_PATTERN):
            files_to_fix["verifiers"].append(filepath)
        elif filename.startswith(INIT_PATTERN):
            files_to_fix["initializers"].append(filepath)
        else:
            files_to_fix["other"].append(filepath)
    
    return files_to_fix

def generate_fix_commands(files_to_fix):
    """Generate specific fix commands for each category"""
    
    print("🎯 PRIORITIZED FIX ACTION PLAN")
    print("="*60)
    
    # 1. Critical Files - Fix IMMEDIATELY
    print("\n🚨 PRIORITY 1: CRITICAL SYSTEM FILES (Fix these FIRST!)")
    print("-"*40)
    for file in files_to_fix["critical"]:
        print(f"\n📄 {file}")
        print("   ACTIONS:")
        print("   1. Replace 'soliton_concept_memory.json' → 'data/concept_db.json'")
        print("   2. Replace 'concept_mesh_data.json' → 'data/concept_db.json'")
        print("   3. Update any mesh loading code to use canonical structure")
        print("   4. Test thoroughly after changes")
    
    # 2. Active Scripts
    print("\n⚡ PRIORITY 2: ACTIVE SCRIPTS")
    print("-"*40)
    for file in files_to_fix["other"]:
        if "diagnostic" not in file and "smart_mirror" not in file:
            print(f"• {file}")
    
    # 3. Test Files
    print("\n🧪 PRIORITY 3: TEST FILES")
    print("-"*40)
    for file in files_to_fix["tests"]:
        print(f"• {file}")
    print("   ACTION: Update test data paths to canonical sources")
    
    # 4. Migration/Fix Scripts
    print("\n🔧 PRIORITY 4: MIGRATION/FIX SCRIPTS")
    print("-"*40)
    for file in files_to_fix["fixers"] + files_to_fix["initializers"]:
        print(f"• {file}")
    print("   ACTION: These may become obsolete after migration")
    
    # 5. Backups - Can ignore
    print("\n📦 PRIORITY 5: BACKUPS (Can ignore or delete)")
    print("-"*40)
    print(f"   Found {len(files_to_fix['backups'])} backup files")
    print("   ACTION: No changes needed - these are historical")

def create_sed_commands():
    """Create sed commands for bulk replacement"""
    
    print("\n\n🔄 BULK REPLACEMENT COMMANDS")
    print("="*60)
    print("\n# For Unix/Linux/Mac (use sed):")
    print("# WARNING: Test these on a single file first!")
    print("")
    print("# Replace soliton_concept_memory.json references:")
    print("find . -name '*.py' -type f -exec sed -i.bak 's/soliton_concept_memory\\.json/data\\/concept_db.json/g' {} +")
    print("")
    print("# Replace concept_mesh_data.json references:")
    print("find . -name '*.py' -type f -exec sed -i.bak 's/concept_mesh_data\\.json/data\\/concept_db.json/g' {} +")
    print("")
    print("\n# For Windows PowerShell:")
    print("# Replace in all Python files:")
    print("Get-ChildItem -Path . -Filter *.py -Recurse | ForEach-Object {")
    print("    (Get-Content $_.FullName) -replace 'soliton_concept_memory\\.json', 'data/concept_db.json' | Set-Content $_.FullName")
    print("}")

def main():
    files_to_fix = categorize_files()
    generate_fix_commands(files_to_fix)
    create_sed_commands()
    
    print("\n\n📋 RECOMMENDED WORKFLOW")
    print("="*60)
    print("1. ✅ Run migrate_concepts.py FIRST")
    print("2. 🔧 Fix CRITICAL files manually (review each change)")
    print("3. 🧪 Update test files to use canonical paths")
    print("4. 🗑️ Delete or archive obsolete fix/init scripts")
    print("5. 🔄 Run audit_concept_usage.py again to verify")
    print("6. ✅ Test the system thoroughly")
    
    print("\n💡 QUICK WINS:")
    print("• Start with prajna/api/prajna_api.py - it's the main API")
    print("• Fix prajna/memory/concept_mesh_api.py - core mesh loader")
    print("• Update ingest_pdf/cognitive_interface.py - PDF pipeline")
    print("• Then tackle the test files")

if __name__ == "__main__":
    main()
