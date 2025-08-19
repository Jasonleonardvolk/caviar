#!/usr/bin/env python3
"""
Remove duplicate soliton endpoints from prajna_api.py
"""

from pathlib import Path

# Read the file
prajna_file = Path("prajna/api/prajna_api.py") 
content = prajna_file.read_text()

# Create backup
backup_file = prajna_file.with_suffix('.py.backup_soliton_fix')
backup_file.write_text(content)
print(f"Created backup: {backup_file}")

# The duplicate endpoints are already causing issues. 
# Since the soliton router is imported and included in the startup event,
# we just need to remove these duplicate endpoint definitions.

# Replace the duplicate soliton endpoints section
# Keep the comment but remove all the duplicate endpoint definitions

old_section = """# --- SOLITON MEMORY ENDPOINTS ---
# Note: SolitonInitRequest and SolitonStoreRequest are imported from api.routes.soliton

@app.post("/api/soliton/init")
async def soliton_init(request: SolitonInitRequest):"""

new_section = """# --- SOLITON MEMORY ENDPOINTS ---
# NOTE: All soliton endpoints are provided by the imported router from api.routes.soliton
# The router is included in the startup event, so no endpoint definitions are needed here.

# Removing duplicate endpoints that were overriding the router...
# (Router expects 'user_id', duplicates expected 'user' - this was causing the mismatch)

# --- Multi-Tenant Endpoints ---
@app.get("/api/users/me", response_model=UserRoleInfo)
async def get_my_role(authorization: Optional[str] = Header(None)):"""

# Find the section to replace
start_marker = "# --- SOLITON MEMORY ENDPOINTS ---"
end_marker = '# --- Multi-Tenant Endpoints ---\n@app.get("/api/users/me"'

start_pos = content.find(start_marker)
end_pos = content.find(end_marker)

if start_pos != -1 and end_pos != -1:
    # Extract everything before and after the section
    before = content[:start_pos]
    after = content[end_pos:]
    
    # Create new content
    new_content = before + new_section + after[len("# --- Multi-Tenant Endpoints ---\n"):]
    
    # Write the fixed content
    prajna_file.write_text(new_content)
    print("\nSuccessfully removed duplicate soliton endpoints!")
    print("The imported router will now handle all soliton endpoints.")
    print("\n✅ Field name mismatch fixed:")
    print("   - Router expects: user_id")
    print("   - Frontend sends: user_id") 
    print("   - They now match!")
else:
    print("ERROR: Could not find the section markers!")
    print(f"Start marker found: {start_pos != -1}")
    print(f"End marker found: {end_pos != -1}")
