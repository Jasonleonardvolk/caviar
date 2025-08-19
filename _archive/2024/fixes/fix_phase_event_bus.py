#!/usr/bin/env python3
"""
Fix the unclosed delimiter in phase_event_bus.rs
"""

import os
from pathlib import Path

print("FIXING UNCLOSED DELIMITER IN PHASE_EVENT_BUS.RS")
print("=" * 60)

# Navigate to the file
phase_event_bus = Path("concept_mesh/src/mesh/phase_event_bus.rs")

if not phase_event_bus.exists():
    print(f"[ERROR] File not found: {phase_event_bus}")
    exit(1)

print(f"\n1. Reading {phase_event_bus}")
content = phase_event_bus.read_text(encoding='utf-8')

# Backup first
backup = phase_event_bus.with_suffix('.rs.backup_delimiter')
backup.write_text(content, encoding='utf-8')
print(f"   [OK] Created backup: {backup.name}")

# Count braces
open_braces = content.count('{')
close_braces = content.count('}')
missing = open_braces - close_braces

print(f"\n2. Brace analysis:")
print(f"   Open braces:  {open_braces}")
print(f"   Close braces: {close_braces}")
print(f"   Missing:      {missing}")

if missing > 0:
    print(f"\n3. Adding {missing} closing brace(s)")
    
    # Find the test module and add cfg attribute if not present
    lines = content.split('\n')
    new_lines = []
    in_test_module = False
    test_module_line = -1
    
    for i, line in enumerate(lines):
        # Look for mod tests
        if 'mod tests {' in line and not any(x in lines[max(0, i-3):i] for x in ['#[cfg(test)]', '#[test]']):
            print(f"   Found test module at line {i+1} without #[cfg(test)]")
            new_lines.append('#[cfg(test)]')
            test_module_line = i
            
        new_lines.append(line)
    
    # Add missing closing braces at the end
    new_lines.append('}' * missing)
    
    # Write the fixed content
    new_content = '\n'.join(new_lines)
    phase_event_bus.write_text(new_content, encoding='utf-8')
    print("   [OK] Added missing braces and #[cfg(test)] attribute")
else:
    print("\n3. No missing braces detected!")

# Verify the fix with cargo check
print("\n4. Verifying fix with cargo check:")
print("-" * 40)

import subprocess
os.chdir("concept_mesh")

check_result = subprocess.run(
    ["cargo", "check", "--release"],
    capture_output=True,
    text=True
)

if check_result.returncode == 0:
    print("   [OK] Cargo check passed! All syntax errors fixed!")
    
    # Now try to build
    print("\n5. Building the wheel:")
    print("-" * 40)
    
    build_result = subprocess.run(
        [os.sys.executable, "-m", "maturin", "build", "--release"],
        capture_output=False  # Show output
    )
    
    if build_result.returncode == 0:
        print("\n   [OK] Build succeeded!")
        
        # Install the wheel
        wheel_pattern = "target/wheels/concept_mesh_rs-*.whl"
        from pathlib import Path
        wheels = list(Path(".").glob(wheel_pattern))
        
        if wheels:
            wheel = wheels[0]
            print(f"\n6. Installing wheel: {wheel.name}")
            
            install_result = subprocess.run(
                [os.sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel)],
                capture_output=True,
                text=True
            )
            
            if install_result.returncode == 0:
                print("   [OK] Wheel installed successfully!")
                
                # Test import
                os.chdir("..")
                test_result = subprocess.run(
                    [os.sys.executable, "-c", "import concept_mesh_rs; print('[OK] Import works!')"],
                    capture_output=True,
                    text=True
                )
                print(f"\n7. Import test: {test_result.stdout.strip()}")
            else:
                print(f"   [ERROR] Install failed: {install_result.stderr}")
else:
    print("   [ERROR] Still has syntax errors:")
    print(check_result.stderr[:500])

os.chdir("..")

print("\n" + "=" * 60)
print("DONE! The wheel should now be built and installed.")
print("\nStart the server with:")
print("   python enhanced_launcher.py")
