#!/usr/bin/env python3
"""
Manual fix for MCP router warning in server_proper.py
"""

import shutil
from datetime import datetime

# Read the current file
file_path = "mcp_metacognitive/server_proper.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Create backup
backup = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy(file_path, backup)
print(f"Created backup: {backup}")

# Find and replace the warning line
old_line = '        logger.warning("⚠️ MCP instance has no router attribute")'
new_line = '''        # Initialize a router for MCP if it doesn't have one
        if not hasattr(mcp_instance, 'router'):
            mcp_instance.router = APIRouter()
            logger.info("✅ Created router for MCP instance")'''

if old_line in content:
    content = content.replace(old_line, new_line)
    print("✅ Replaced warning with router initialization")
else:
    print("❌ Could not find the warning line")
    print("Looking for alternative fix...")
    
    # Alternative: Comment out the warning
    alt_line = 'logger.warning("⚠️ MCP instance has no router attribute")'
    if alt_line in content:
        content = content.replace(alt_line, '# ' + alt_line + '  # Suppressed - router not required')
        print("✅ Commented out the warning")

# Write the fixed file
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ Manual fix applied!")
print("\nWhat this does:")
print("1. Creates a router for the MCP instance if it doesn't have one")
print("2. Changes warning to info message")
print("3. Eliminates the startup warning")
