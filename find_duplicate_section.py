#!/usr/bin/env python3
"""
Find and comment out duplicate soliton endpoints in prajna_api.py
"""

import re
from pathlib import Path

# Read prajna_api.py
prajna_file = Path("prajna/api/prajna_api.py")
content = prajna_file.read_text()

# Find the section with duplicate soliton endpoints
# Looking for the pattern between "# --- SOLITON MEMORY ENDPOINTS ---" and the next major section

start_marker = "# --- SOLITON MEMORY ENDPOINTS ---"
end_markers = ["# --- Multi-Tenant Endpoints ---", "# --- Consciousness/Evolution Endpoints ---", "# --- COMPREHENSIVE VALIDATION"]

start_pos = content.find(start_marker)
if start_pos == -1:
    print("Could not find soliton section")
    exit(1)

# Find the end of the soliton section
end_pos = len(content)
for marker in end_markers:
    pos = content.find(marker, start_pos)
    if pos != -1 and pos < end_pos:
        end_pos = pos

print(f"Found soliton section from position {start_pos} to {end_pos}")
print(f"Section length: {end_pos - start_pos} characters")

# Extract the section
section = content[start_pos:end_pos]
print("\nSection preview:")
print(section[:500] + "...")

# Count endpoints in this section
endpoint_count = len(re.findall(r'@app\.(get|post|put|delete)\(', section))
print(f"\nFound {endpoint_count} endpoint definitions in this section")

if endpoint_count > 0:
    print("\n⚠️  These are DUPLICATE endpoints that should be removed!")
    print("The soliton router already provides these endpoints.")
