#!/usr/bin/env python3
"""Find line 595 and surrounding context in prajna_api.py"""

with open('prajna_api.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines in file: {len(lines)}")
print("\nLines 590-600:")
print("-" * 60)

for i in range(max(0, 589), min(len(lines), 600)):
    line_num = i + 1
    line = lines[i].rstrip()
    marker = " <-- LINE 595" if line_num == 595 else ""
    print(f"{line_num:4d}: {line}{marker}")

# Also search for unmatched parentheses
print("\n\nSearching for lines with only ')' or '}':")
print("-" * 60)

for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped in [')', '}', '})']:
        print(f"{i+1:4d}: {line.rstrip()}")
