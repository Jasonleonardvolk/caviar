#!/usr/bin/env python3
"""Complete fix for +page.svelte syntax issues"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import re

def fix_svelte_file():
    file_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+page.svelte")
    
    print("🔧 Fixing +page.svelte syntax issues...")
    
    if not file_path.exists():
        print("❌ File not found!")
        return False
    
    # Read the file
    content = file_path.read_text(encoding='utf-8')
    original_length = len(content)
    
    print(f"📄 Original file size: {original_length} characters")
    
    # Fix 1: Remove any duplicate prosody subscribe blocks
    # Look for duplicate patterns
    pattern = r'(// .* Subscribe to prosody stores.*?)\1+'
    content = re.sub(pattern, r'\1', content, flags=re.DOTALL)
    
    # Fix 2: Check for orphaned closing braces
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Skip orphaned closing braces that are outside of any function
        if i > 440 and i < 450 and line.strip() == '}':
            # Check if this is an orphaned brace by looking at context
            prev_line = lines[i-1].strip() if i > 0 else ""
            next_line = lines[i+1].strip() if i < len(lines)-1 else ""
            
            # If surrounded by comments or empty lines, it's likely orphaned
            if (prev_line.startswith('//') or prev_line == '') and \
               (next_line.startswith('//') or next_line == ''):
                print(f"❌ Removing orphaned closing brace at line {i+1}")
                continue
        
        fixed_lines.append(line)
    
    # Rejoin the lines
    content = '\n'.join(fixed_lines)
    
    # Fix 3: Remove any duplicate localStorage.setItem outside of functions
    # This pattern catches orphaned code blocks
    pattern = r'\n\s*if\s*\(browser\)\s*{\s*localStorage\.setItem\([^}]+}\s*\n\s*}\s*\n\s*//\s*.*Subscribe'
    if re.search(pattern, content):
        print("❌ Found orphaned localStorage code block, removing...")
        content = re.sub(pattern, '\n  // Subscribe', content)
    
    # Fix 4: Ensure proper structure around prosody subscriptions
    # Find the toggleHologramVideo function and ensure it's properly closed
    toggle_func_pattern = r'(function toggleHologramVideo\(\) {[^}]+})\s*}\s*\n\s*//.*Subscribe'
    match = re.search(toggle_func_pattern, content, re.DOTALL)
    if match:
        print("❌ Found extra closing brace after toggleHologramVideo")
        content = re.sub(toggle_func_pattern, r'\1\n\n  // Subscribe', content, flags=re.DOTALL)
    
    # Fix 5: Set Enola as default persona
    content = content.replace(
        "let currentPersona: Persona = {\n    id: 'scholar',",
        "let currentPersona: Persona = {\n    id: 'enola',"
    )
    content = content.replace(
        "name: 'Scholar',\n    description: 'Analytical and knowledge-focused',",
        "name: 'Enola',\n    description: 'Investigative and analytical consciousness',"
    )
    
    # Write the fixed content
    file_path.write_text(content, encoding='utf-8')
    
    print(f"✅ File fixed! New size: {len(content)} characters")
    print(f"📊 Changed {abs(len(content) - original_length)} characters")
    
    # Verify brace balance
    open_braces = content.count('{')
    close_braces = content.count('}')
    print(f"\n📊 Final brace count: {{ = {open_braces}, }} = {close_braces}")
    
    if open_braces == close_braces:
        print("✅ Braces are balanced!")
        return True
    else:
        print(f"⚠️ Warning: Braces still unbalanced by {open_braces - close_braces}")
        return False

if __name__ == "__main__":
    success = fix_svelte_file()
    if success:
        print("\n🎉 Success! Now run:")
        print("cd C:\\Users\\jason\\Desktop\\tori\\kha")
        print("poetry run python enhanced_launcher.py")
    else:
        print("\n⚠️ Fix may have failed. Please check the file manually.")
