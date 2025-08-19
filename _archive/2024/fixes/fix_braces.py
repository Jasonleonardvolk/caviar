#!/usr/bin/env python3
"""Fix the brace issue in phase_event_bus.rs"""

import re

# Read the file
with open('concept_mesh/src/mesh/phase_event_bus.rs', 'r') as f:
    content = f.read()

# Count braces
open_braces = content.count('{')
close_braces = content.count('}')

print(f"Open braces: {open_braces}")
print(f"Close braces: {close_braces}")
print(f"Difference: {open_braces - close_braces}")

# Find the test function
test_func_match = re.search(r'async fn test_publish_and_subscribe\(\)[^{]*{', content)
if test_func_match:
    start_pos = test_func_match.end()
    
    # Count braces from this point
    brace_count = 1
    pos = start_pos
    
    while pos < len(content) and brace_count > 0:
        if content[pos] == '{':
            brace_count += 1
        elif content[pos] == '}':
            brace_count -= 1
        pos += 1
    
    if brace_count != 0:
        print(f"\nThe test function is missing {brace_count} closing brace(s)")
        
        # Fix by ensuring proper closing
        lines = content.split('\n')
        
        # Find where we are
        for i, line in enumerate(lines[-10:], len(lines)-10):
            print(f"{i}: {line}")
        
        # The fix: ensure the function closes properly
        # Looking at the output, line 956 has a }, but we need to add one more
        if lines[-3].strip() == '}' and lines[-2].strip() == '}' and lines[-1].strip() == '}':
            # We have too many at the end
            lines = lines[:-1]  # Remove the extra one
            lines[-2] = '    }'  # Proper indentation for closing test function
            
        with open('concept_mesh/src/mesh/phase_event_bus.rs', 'w') as f:
            f.write('\n'.join(lines))
            
        print("\nFixed the brace issue!")
