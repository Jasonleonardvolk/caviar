#!/usr/bin/env python3
"""Fix all brace issues in phase_event_bus.rs"""

# Read the file
with open('concept_mesh/src/mesh/phase_event_bus.rs', 'r') as f:
    lines = f.readlines()

# Find the test module start
test_mod_line = None
for i, line in enumerate(lines):
    if '#[cfg(test)]' in line:
        test_mod_line = i
        break

if test_mod_line is None:
    print("ERROR: Could not find test module!")
    exit(1)

print(f"Test module starts at line {test_mod_line + 1}")

# Find where the test function ends
test_func_start = None
for i in range(test_mod_line, len(lines)):
    if 'async fn test_publish_and_subscribe()' in lines[i]:
        test_func_start = i
        break

print(f"Test function starts at line {test_func_start + 1}")

# Show the last 10 lines
print("\nLast 10 lines of file:")
for i, line in enumerate(lines[-10:]):
    print(f"{len(lines)-10+i+1}: {line.rstrip()}")

# Fix: ensure proper closing
# The test function needs to close with }
# The test module needs to close with }
# Total: 2 closing braces at the end

# Remove any trailing empty lines and fix the ending
while lines and lines[-1].strip() == '':
    lines.pop()

# Check current ending
if len(lines) >= 3:
    last_lines = [lines[-3].rstrip(), lines[-2].rstrip(), lines[-1].rstrip()]
    print(f"\nCurrent ending:")
    for i, line in enumerate(last_lines):
        print(f"  {i}: '{line}'")

# Ensure proper ending:
# The assert line, then close test function, then close test module
if 'assert_eq!' in lines[-1]:
    # Need to add closing braces
    lines.append('    }\n')  # Close test function
    lines.append('}\n')      # Close test module
elif lines[-1].strip() == '}' and lines[-2].strip() == '}':
    # Already has two closing braces, just fix indentation
    lines[-2] = '    }\n'  # Test function close (indented)
    lines[-1] = '}\n'      # Test module close (not indented)
else:
    # Find the last assert and fix from there
    for i in range(len(lines)-1, -1, -1):
        if 'assert_eq!' in lines[i]:
            # Keep everything up to and including this line
            lines = lines[:i+1]
            lines.append('    }\n')  # Close test function
            lines.append('}\n')      # Close test module
            break

# Write the fixed file
with open('concept_mesh/src/mesh/phase_event_bus.rs', 'w') as f:
    f.writelines(lines)

print("\nFixed! File now ends with:")
print(f"  -2: '{lines[-2].rstrip()}'")
print(f"  -1: '{lines[-1].rstrip()}'")
