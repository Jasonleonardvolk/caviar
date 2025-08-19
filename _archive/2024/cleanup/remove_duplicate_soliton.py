"""Script to remove all duplicate soliton endpoints from prajna_api.py"""

import re

# Read the file
with open('prajna_api.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the start of soliton endpoints section
start_marker = "# --- SOLITON MEMORY ENDPOINTS ---"
end_marker = "# --- Multi-Tenant Endpoints ---"

# Find positions
start_pos = content.find(start_marker)
end_pos = content.find(end_marker)

if start_pos != -1 and end_pos != -1:
    # Replace the entire section with just a comment
    new_section = """# --- SOLITON MEMORY ENDPOINTS ---
# Soliton endpoints are now provided by the imported soliton_router

"""
    
    # Reconstruct the file
    new_content = content[:start_pos] + new_section + content[end_pos:]
    
    # Write back
    with open('prajna_api.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Removed duplicate soliton endpoints")
else:
    print("❌ Could not find section markers")
