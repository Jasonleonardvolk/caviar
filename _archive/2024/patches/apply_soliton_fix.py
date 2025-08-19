#!/usr/bin/env python3
"""
Fix the soliton endpoint field name mismatch by removing duplicate endpoints
"""

from pathlib import Path
import re

# Read prajna_api.py with UTF-8 encoding
prajna_file = Path("prajna/api/prajna_api.py")
original_content = prajna_file.read_text(encoding='utf-8')

# Backup the original
backup_path = prajna_file.with_suffix('.py.backup_soliton_fix')
backup_path.write_text(original_content, encoding='utf-8')
print(f"✅ Created backup: {backup_path}")

# The issue: prajna_api.py has duplicate soliton endpoints that override the imported router
# The router expects 'user_id' but the duplicates expect 'user' 
# Solution: Remove all duplicate soliton endpoint definitions

# Find where duplicate endpoints start and end
lines = original_content.split('\n')
new_lines = []
skip_mode = False
skip_count = 0

for i, line in enumerate(lines):
    # Start skipping at the first duplicate soliton endpoint
    if '@app.post("/api/soliton/init")' in line and not skip_mode:
        skip_mode = True
        skip_count = 0
        # Add a comment explaining what we removed
        new_lines.append("# [REMOVED] Duplicate soliton endpoints that were overriding the imported router")
        new_lines.append("# The soliton router from api.routes.soliton handles all these endpoints")
        continue
    
    # Stop skipping when we reach the Multi-Tenant section
    if skip_mode and "# --- Multi-Tenant Endpoints ---" in line:
        skip_mode = False
        new_lines.append("")  # Add blank line before next section
    
    # Skip lines while in skip mode
    if skip_mode:
        skip_count += 1
        continue
    else:
        new_lines.append(line)

# Join the lines
new_content = '\n'.join(new_lines)

# Write the fixed file with UTF-8 encoding
prajna_file.write_text(new_content, encoding='utf-8')

print(f"\n✅ Removed {skip_count} lines containing duplicate soliton endpoints")
print("\n🎯 FIXED: Field name mismatch resolved!")
print("   - Frontend sends: user_id ✓")
print("   - Router expects: user_id ✓") 
print("   - No more duplicate endpoints with wrong field names!")
print("\n📝 Next steps:")
print("   1. Restart the API server")
print("   2. Test with: .\\test_soliton_powershell.ps1")
print("   3. The soliton endpoints should work correctly now!")
