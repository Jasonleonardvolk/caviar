#!/usr/bin/env python3
"""
Check and fix the duplicate soliton endpoints in prajna_api.py
"""

from pathlib import Path
import re
import time

# Read prajna_api.py with UTF-8 encoding
prajna_file = Path("prajna/api/prajna_api.py")
if not prajna_file.exists():
    print("❌ ERROR: prajna/api/prajna_api.py not found!")
    exit(1)

print("📖 Reading prajna_api.py...")
original_content = prajna_file.read_text(encoding='utf-8')

# Create backup with timestamp
backup_name = f"prajna_api_{int(time.time())}.py.backup"
backup_path = Path("backups") / backup_name
backup_path.parent.mkdir(exist_ok=True)
backup_path.write_text(original_content, encoding='utf-8')
print(f"✅ Created backup: {backup_path}")

# Check if we have duplicate soliton endpoints
if '@app.post("/api/soliton/init")' in original_content:
    print("\n⚠️ FOUND duplicate soliton endpoints!")
    
    # Count how many lines we'll remove
    lines = original_content.split('\n')
    in_soliton_section = False
    duplicate_lines = 0
    
    for i, line in enumerate(lines):
        if '@app.post("/api/soliton/init")' in line:
            in_soliton_section = True
        elif in_soliton_section and "# --- Multi-Tenant Endpoints ---" in line:
            break
        
        if in_soliton_section:
            duplicate_lines += 1
    
    print(f"   Will remove approximately {duplicate_lines} lines of duplicate code")
    
    # Now do the actual fix
    # Find the section between SOLITON MEMORY ENDPOINTS and Multi-Tenant Endpoints
    pattern = r'(# --- SOLITON MEMORY ENDPOINTS ---.*?)(@app\.post\("/api/soliton/init"\).*?)(# --- Multi-Tenant Endpoints ---)'
    
    replacement = r'\1# NOTE: All soliton endpoints are provided by the imported router from api.routes.soliton\n# The duplicate endpoint definitions have been removed to fix the field name mismatch.\n# Router expects "user_id", frontend sends "user_id" - they now match!\n\n\3'
    
    new_content = re.sub(pattern, replacement, original_content, flags=re.DOTALL)
    
    if new_content != original_content:
        # Write the fixed content
        prajna_file.write_text(new_content, encoding='utf-8')
        print("\n✅ SUCCESS! Removed duplicate soliton endpoints")
        print("\n🎯 Field name mismatch FIXED:")
        print("   - Frontend sends: user_id ✓")
        print("   - Router expects: user_id ✓")
        print("   - No more conflicts!")
    else:
        print("\n❌ Could not find the pattern to replace. Manual intervention needed.")
else:
    print("\n✅ No duplicate soliton endpoints found!")
    print("   The file may have already been fixed.")

print("\n📝 Next steps:")
print("   1. Stop the API server (Ctrl+C)")
print("   2. Restart with: poetry run python enhanced_launcher.py")
print("   3. Test with: .\\test_soliton_powershell.ps1")
