#!/usr/bin/env python3
"""Create a fixed version of prajna_api.py"""

# Read the original file
with open('prajna_api.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Count the occurrences of the problematic pattern
import re

# Find the pattern where we have the extra parenthesis
pattern = r'(return\s*{\s*"success":\s*False,\s*"error":\s*str\(e\)\s*}\))'

matches = list(re.finditer(pattern, content, re.MULTILINE | re.DOTALL))
print(f"Found {len(matches)} matches for the problematic pattern")

if matches:
    for i, match in enumerate(matches):
        # Get line number
        line_num = content[:match.start()].count('\n') + 1
        print(f"\nMatch {i+1} at line ~{line_num}:")
        print(f"Text: {match.group()}")
        
# Fix the issue by removing the extra parenthesis
fixed_content = re.sub(
    r'(return\s*{\s*"success":\s*False,\s*"error":\s*str\(e\)\s*}\))',
    r'return {\n            "success": False,\n            "error": str(e)\n        }',
    content
)

# Save the fixed version
with open('prajna_api_fixed.py', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("\n✅ Created prajna_api_fixed.py with the fix applied")
print("   Check the file and if it looks good, rename it to prajna_api.py")
