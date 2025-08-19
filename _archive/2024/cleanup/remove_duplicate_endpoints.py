#!/usr/bin/env python3
"""
Remove duplicate soliton endpoints from prajna_api.py
These are overriding the imported router and causing the field name mismatch
"""

import re
from pathlib import Path

# Read prajna_api.py
prajna_file = Path("prajna/api/prajna_api.py")
content = prajna_file.read_text()

# Create backup
backup_file = prajna_file.with_suffix('.py.backup_before_duplicate_removal')
backup_file.write_text(content)
print(f"Created backup: {backup_file}")

# Find and comment out the duplicate soliton endpoints
# They start after "# --- SOLITON MEMORY ENDPOINTS ---" and end before "# --- Multi-Tenant Endpoints ---"

# Split content into lines
lines = content.split('\n')
new_lines = []
in_soliton_section = False
in_endpoint_function = False
indent_level = 0

for i, line in enumerate(lines):
    # Check if we're entering the soliton section
    if line.strip() == "# --- SOLITON MEMORY ENDPOINTS ---":
        in_soliton_section = True
        new_lines.append(line)
        new_lines.append("# NOTE: These duplicate endpoints have been commented out")
        new_lines.append("# The soliton router is imported and provides all these endpoints")
        continue
    
    # Check if we're leaving the soliton section
    if in_soliton_section and line.strip().startswith("# ---") and "SOLITON" not in line:
        in_soliton_section = False
    
    # If we're in the soliton section
    if in_soliton_section:
        # Check if this is a route decorator
        if re.match(r'^@app\.(get|post|put|delete)\(', line.strip()):
            in_endpoint_function = True
            indent_level = len(line) - len(line.lstrip())
            new_lines.append("# " + line)  # Comment out the decorator
        elif in_endpoint_function:
            # Check if we're still in the function (based on indentation)
            current_indent = len(line) - len(line.lstrip())
            if line.strip() and current_indent <= indent_level:
                # We've left the function
                in_endpoint_function = False
                new_lines.append(line)  # Don't comment this line
            else:
                # Still in the function, comment it out
                if line.strip():  # Only comment non-empty lines
                    new_lines.append("# " + line)
                else:
                    new_lines.append(line)  # Keep empty lines as-is
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

# Join lines back together
new_content = '\n'.join(new_lines)

# Write the fixed content
prajna_file.write_text(new_content)
print("\nFixed prajna_api.py - commented out duplicate soliton endpoints")
print("\nThe imported soliton router will now be used!")
print("This fixes the field name mismatch: router expects 'user_id', frontend sends 'user_id'")
