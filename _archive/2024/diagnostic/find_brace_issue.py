#!/usr/bin/env python3
"""Find the exact unmatched brace in +page.svelte"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def find_unmatched_brace():
    file_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+page.svelte")
    
    if not file_path.exists():
        print("❌ File not found!")
        return
    
    content = file_path.read_text(encoding='utf-8')
    lines = content.split('\n')
    
    print(f"📄 Analyzing {len(lines)} lines...")
    
    # Track brace depth
    depth = 0
    brace_stack = []  # Stack to track opening braces
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Count braces in this line
        for j, char in enumerate(line):
            if char == '{':
                depth += 1
                brace_stack.append({
                    'line': line_num,
                    'col': j,
                    'context': line.strip()[:50]
                })
            elif char == '}':
                depth -= 1
                if brace_stack:
                    brace_stack.pop()
                else:
                    print(f"\n❌ EXTRA CLOSING BRACE at line {line_num}, column {j}")
                    print(f"   Line: {line.strip()}")
                    print(f"   Context (lines {max(1, line_num-2)} to {min(len(lines), line_num+2)}):")
                    for k in range(max(0, i-2), min(len(lines), i+3)):
                        prefix = ">>>" if k == i else "   "
                        print(f"   {prefix} {k+1}: {lines[k].rstrip()}")
                    return line_num
    
    if depth > 0:
        print(f"\n❌ {depth} UNCLOSED OPENING BRACE(S)")
        print("\nLast few unclosed braces:")
        for brace in brace_stack[-3:]:
            print(f"   Line {brace['line']}: {brace['context']}")
    elif depth < 0:
        print(f"\n❌ {abs(depth)} EXTRA CLOSING BRACE(S)")
    else:
        print("\n✅ All braces are matched!")
    
    return None

def fix_unmatched_brace():
    file_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+page.svelte")
    
    # Find the problematic line
    problem_line = find_unmatched_brace()
    
    if problem_line is None:
        print("\nNo specific line to fix found.")
        return
    
    print(f"\n🔧 Attempting to fix line {problem_line}...")
    
    # Read the file
    content = file_path.read_text(encoding='utf-8')
    lines = content.split('\n')
    
    # Remove the problematic line if it's just a lone closing brace
    if problem_line > 0 and problem_line <= len(lines):
        if lines[problem_line - 1].strip() == '}':
            print(f"❌ Removing lone closing brace at line {problem_line}")
            lines.pop(problem_line - 1)
            
            # Write back
            new_content = '\n'.join(lines)
            file_path.write_text(new_content, encoding='utf-8')
            print("✅ Fixed! File saved.")
            
            # Verify
            print("\n🔍 Verifying fix...")
            find_unmatched_brace()

if __name__ == "__main__":
    fix_unmatched_brace()
