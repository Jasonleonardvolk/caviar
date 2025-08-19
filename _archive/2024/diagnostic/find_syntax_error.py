#!/usr/bin/env python3
"""Script to find and fix syntax error in prajna_api.py"""

import re

# Read the file
with open('prajna_api.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all occurrences of @app.post("/intent")
intent_matches = list(re.finditer(r'@app\.post\("/intent"\)', content))
print(f"Found {len(intent_matches)} @app.post('/intent') decorators")

# Find all process_intent definitions
process_matches = list(re.finditer(r'async def process_intent', content))
print(f"Found {len(process_matches)} process_intent definitions")

# Check for incomplete function around line 439
lines = content.split('\n')
for i in range(430, min(450, len(lines))):
    if i == 438:
        print(f"\nContext around line {i+1}:")
        for j in range(max(0, i-5), min(len(lines), i+5)):
            marker = " <-- ERROR LINE" if j == 438 else ""
            print(f"{j+1:4d}: {lines[j]}{marker}")
        break

# Look for unclosed parentheses or incomplete statements
print("\nChecking for common syntax issues...")
open_parens = 0
open_brackets = 0
open_braces = 0

for i, line in enumerate(lines[:440]):
    open_parens += line.count('(') - line.count(')')
    open_brackets += line.count('[') - line.count(']')
    open_braces += line.count('{') - line.count('}')
    
    if i >= 430 and any([open_parens != 0, open_brackets != 0, open_braces != 0]):
        print(f"Line {i+1}: parens={open_parens}, brackets={open_brackets}, braces={open_braces}")
