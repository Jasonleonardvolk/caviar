#!/usr/bin/env python3
"""Fix the unclosed toggleHologramVideo function"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_toggleHologramVideo():
    file_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+page.svelte")
    
    print("🔧 Fixing unclosed toggleHologramVideo function...")
    
    if not file_path.exists():
        print("❌ File not found!")
        return False
    
    # Read the file
    content = file_path.read_text(encoding='utf-8')
    lines = content.split('\n')
    
    # Find the toggleHologramVideo function
    function_start = -1
    for i, line in enumerate(lines):
        if 'function toggleHologramVideo()' in line:
            function_start = i
            print(f"📍 Found toggleHologramVideo at line {i+1}")
            break
    
    if function_start == -1:
        print("❌ Could not find toggleHologramVideo function")
        return False
    
    # Look for where the function should end
    # It should end before the "Subscribe to prosody stores" comment
    function_end = -1
    brace_count = 0
    inside_function = False
    
    for i in range(function_start, len(lines)):
        line = lines[i]
        
        # Start counting after the opening brace
        if i == function_start:
            brace_count += line.count('{')
            inside_function = True
            continue
        
        if inside_function:
            brace_count += line.count('{')
            brace_count -= line.count('}')
            
            # If we reach balance, function should end
            if brace_count == 0:
                function_end = i
                break
            
            # If we see the Subscribe comment, function should have ended before this
            if 'Subscribe to prosody stores' in line:
                print(f"⚠️ Found 'Subscribe' comment at line {i+1} but function not closed")
                # Insert closing brace before this line
                lines.insert(i, '  }')
                print(f"✅ Inserted closing brace at line {i}")
                function_end = i
                break
    
    # If we didn't find a proper end, add closing brace
    if function_end == -1:
        # Look for the next function or major section
        for i in range(function_start + 1, min(function_start + 50, len(lines))):
            if 'function ' in lines[i] or 'onMount(' in lines[i] or '$:' in lines[i]:
                lines.insert(i, '  }')
                print(f"✅ Inserted closing brace before line {i+1}")
                break
    
    # Write the fixed content
    new_content = '\n'.join(lines)
    file_path.write_text(new_content, encoding='utf-8')
    
    # Verify the fix
    open_count = new_content.count('{')
    close_count = new_content.count('}')
    
    print(f"\n📊 Final brace count: {{ = {open_count}, }} = {close_count}")
    
    if open_count == close_count:
        print("✅ Braces are now balanced!")
        return True
    else:
        print(f"⚠️ Still unbalanced by {open_count - close_count}")
        return False

if __name__ == "__main__":
    if fix_toggleHologramVideo():
        print("\n🎉 Success! The syntax error is fixed.")
        print("\nNow run:")
        print("cd C:\\Users\\jason\\Desktop\\tori\\kha")
        print("poetry run python enhanced_launcher.py")
    else:
        print("\n⚠️ Manual intervention may be needed.")
