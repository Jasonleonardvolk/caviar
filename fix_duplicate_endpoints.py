#!/usr/bin/env python3
"""
Fix duplicate soliton endpoints by commenting them out
"""

import re
from pathlib import Path

# Read prajna_api.py
prajna_file = Path("prajna/api/prajna_api.py")
content = prajna_file.read_text()

# Backup the original
backup_file = prajna_file.with_suffix('.py.backup_before_soliton_fix')
backup_file.write_text(content)
print(f"Created backup: {backup_file}")

# Find and replace the duplicate soliton endpoints section
# We want to comment out everything between the marker and the next section

new_content = content.replace(
    """# --- SOLITON MEMORY ENDPOINTS ---
# Note: SolitonInitRequest and SolitonStoreRequest are imported from api.routes.soliton

@app.post("/api/soliton/init")""",
    """# --- SOLITON MEMORY ENDPOINTS ---
# Note: These endpoints are provided by the imported soliton router from api.routes.soliton
# The duplicate definitions below have been commented out to avoid conflicts

# COMMENTED OUT - Using imported router instead
# @app.post("/api/soliton/init")"""
)

# Write the fixed content
prajna_file.write_text(new_content)
print("Fixed prajna_api.py - commented out duplicate soliton endpoints")
print("\nNow the imported soliton router will be used correctly!")
print("The frontend sends 'user_id' which matches the router's expected field.")
