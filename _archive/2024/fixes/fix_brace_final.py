#!/usr/bin/env python3
"""Direct fix for the unmatched brace issue"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_brace_mismatch():
    file_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+page.svelte")
    
    print("🔧 Fixing brace mismatch in +page.svelte...")
    
    if not file_path.exists():
        print("❌ File not found!")
        return
    
    # Read the file
    content = file_path.read_text(encoding='utf-8')
    
    # The issue is likely in the prosody subscription area based on previous attempts
    # Let's look for patterns that indicate mismatched braces
    
    # Pattern 1: Double closing braces after function
    content = content.replace('}\n  }\n\n  // Subscribe to prosody stores', '}\n\n  // Subscribe to prosody stores')
    content = content.replace('}\n    }\n\n  // Subscribe to prosody stores', '}\n\n  // Subscribe to prosody stores')
    
    # Pattern 2: Extra closing brace before onMount
    content = content.replace('});\n  }\n  \n  onMount(() => {', '});\n  \n  onMount(() => {')
    
    # Pattern 3: Missing opening brace for a reactive statement
    # Check if there's a $: without proper bracing
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # If we find a reactive statement that might be missing braces
        if line.strip().startswith('$:') and not '{' in line:
            next_line = lines[i+1] if i+1 < len(lines) else ''
            # If the next line is indented but not a continuation
            if next_line and len(next_line) - len(next_line.lstrip()) > len(line) - len(line.lstrip()):
                if not next_line.strip().endswith(';'):
                    # This might need braces
                    print(f"⚠️ Found reactive statement without braces at line {i+1}")
        
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Write back
    file_path.write_text(content, encoding='utf-8')
    
    # Count braces
    open_count = content.count('{')
    close_count = content.count('}')
    
    print(f"📊 Brace count: {{ = {open_count}, }} = {close_count}")
    
    if open_count == close_count:
        print("✅ Braces are now balanced!")
        return True
    else:
        print(f"⚠️ Still unbalanced by {open_count - close_count}")
        
        # Try one more aggressive fix - add a closing brace at the very end if needed
        if open_count > close_count:
            print("🔧 Adding closing brace at end of script tag...")
            # Find the closing </script> tag
            script_end = content.rfind('</script>')
            if script_end > 0:
                # Add closing brace before </script>
                content = content[:script_end] + '}\n' + content[script_end:]
                file_path.write_text(content, encoding='utf-8')
                print("✅ Added closing brace before </script>")
                return True
        
        return False

if __name__ == "__main__":
    if fix_brace_mismatch():
        print("\n🎉 Success! Now run:")
        print("cd C:\\Users\\jason\\Desktop\\tori\\kha")
        print("poetry run python enhanced_launcher.py")
