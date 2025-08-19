#!/usr/bin/env python3
"""
Fix the syntax error in prajna_api.py
Removes the extra closing parenthesis on line 595
"""

import re

# Read the file
with open('prajna_api.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and fix the specific syntax error pattern
# Look for the get_cognitive_state function with the extra parenthesis
pattern = r'(return\s*{\s*"success":\s*False,\s*"error":\s*str\(e\)\s*}\))'
replacement = r'return {\n            "success": False,\n            "error": str(e)\n        }'

# Apply the fix
fixed_content = re.sub(pattern, replacement, content)

# Write back
with open('prajna_api.py', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("✅ Fixed syntax error in prajna_api.py")
print("   Removed extra closing parenthesis from line 595")
