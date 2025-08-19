#!/usr/bin/env python3
"""
NUCLEAR DIAGNOSTIC - Find all import/path issues
"""

import os
import sys
import subprocess
import traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

output_file = r"{PROJECT_ROOT}\omg.txt"

def write_output(text):
    """Write to both console and file"""
    print(text)
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(text + '\n')

# Clear output file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('')

write_output("=== TORI MCP NUCLEAR DIAGNOSTIC ===")
write_output(f"Working Directory: {os.getcwd()}")
write_output("")

# SECTION 1: Python Environment
write_output("=== SECTION 1: PYTHON ENVIRONMENT ===")
write_output(f"Python Version: {sys.version}")
write_output(f"Python Executable: {sys.executable}")
write_output(f"Virtual Environment: {'VIRTUAL_ENV' in os.environ}")
write_output("")

# SECTION 2: File System Check
write_output("=== SECTION 2: FILE SYSTEM REALITY CHECK ===")
critical_files = [
    "mcp_metacognitive/__init__.py",
    "mcp_metacognitive/core/__init__.py",
    "mcp_metacognitive/core/soliton_memory.py",
    "mcp_metacognitive/tools/__init__.py",
    "mcp_metacognitive/tools/soliton_memory_tools.py",
    "mcp_metacognitive/server.py",
    "setup.py",
    "pyproject.toml"
]

for file in critical_files:
    exists = os.path.exists(file)
    status = "✅ EXISTS" if exists else "❌ MISSING"
    write_output(f"{status} : {file}")
write_output("")

# SECTION 3: Python Path
write_output("=== SECTION 3: PYTHON PATH ANALYSIS ===")
write_output("sys.path entries:")
for i, p in enumerate(sys.path):
    write_output(f"  [{i}] {p}")

if os.getcwd() in sys.path:
    write_output(f"\n✅ Current directory IS in sys.path")
else:
    write_output(f"\n❌ Current directory NOT in sys.path")
write_output("")

# SECTION 4: Import Tests
write_output("=== SECTION 4: IMPORT TESTS ===")
tests = [
    ('mcp_metacognitive', 'Base package'),
    ('mcp_metacognitive.core', 'Core subpackage'),
    ('mcp_metacognitive.core.soliton_memory', 'Soliton memory module'),
    ('mcp_metacognitive.tools', 'Tools subpackage'),
    ('mcp_metacognitive.tools.soliton_memory_tools', 'Soliton tools module'),
    ('core.soliton_memory', 'BAD IMPORT - should fail'),
]

for module_name, description in tests:
    try:
        mod = __import__(module_name)
        write_output(f'✅ {module_name}: SUCCESS - {description}')
        if hasattr(mod, '__file__'):
            write_output(f'   Location: {mod.__file__}')
    except ImportError as e:
        write_output(f'❌ {module_name}: FAILED - {description}')
        write_output(f'   Error: {e}')
    except Exception as e:
        write_output(f'💥 {module_name}: UNEXPECTED ERROR - {description}')
        write_output(f'   Error: {type(e).__name__}: {e}')
write_output("")

# SECTION 5: Module Contents
write_output("=== SECTION 5: MODULE CONTENTS CHECK ===")
try:
    import mcp_metacognitive.core.soliton_memory as sm
    write_output('✅ Successfully imported soliton_memory')
    write_output(f'   Module file: {sm.__file__}')
    write_output('   Available attributes:')
    for attr in dir(sm):
        if not attr.startswith('_'):
            write_output(f'     - {attr}')
except Exception as e:
    write_output(f'❌ Failed to import soliton_memory: {e}')
    traceback.print_exc()
write_output("")

# SECTION 6: Test exact failing import
write_output("=== SECTION 6: TESTING EXACT FAILING IMPORT ===")
try:
    from core.soliton_memory import VaultStatus, ContentType
    write_output('❌ BAD NEWS: "from core.soliton_memory" actually worked!')
except ImportError as e:
    write_output('✅ GOOD: "from core.soliton_memory" failed as expected')
    write_output(f'   Error: {e}')

try:
    from mcp_metacognitive.core.soliton_memory import VaultStatus, ContentType
    write_output('✅ GOOD: Correct import works!')
except ImportError as e:
    write_output('❌ BAD: Even the correct import failed!')
    write_output(f'   Error: {e}')
write_output("")

# SECTION 7: Check package installation
write_output("=== SECTION 7: PACKAGE INSTALLATION STATUS ===")
result = subprocess.run(['pip', 'show', 'tori-kha'], capture_output=True, text=True)
if result.returncode == 0:
    write_output("✅ Package 'tori-kha' is installed")
    write_output(result.stdout)
else:
    write_output("❌ Package 'tori-kha' is NOT installed")
    write_output("   Run: pip install -e .")
write_output("")

# SECTION 8: Check import line
write_output("=== SECTION 8: ACTUAL IMPORT LINE CHECK ===")
tools_file = Path("mcp_metacognitive/tools/soliton_memory_tools.py")
if tools_file.exists():
    with open(tools_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if 'from' in line and 'soliton_memory' in line and 'import' in line:
                write_output(f"Line {line_num}: {line.strip()}")
                break
write_output("")

# SECTION 9: Smoking gun
write_output("=== SECTION 9: THE SMOKING GUN ===")
write_output("Checking for rogue 'core' in sys.path:")
for path in sys.path:
    core_path = os.path.join(path, 'core')
    if os.path.exists(core_path) and os.path.isdir(core_path):
        write_output(f'❌ FOUND: {core_path}')
        if os.path.exists(os.path.join(core_path, 'soliton_memory.py')):
            write_output('   ⚠️  Contains soliton_memory.py!')

write_output(f"\nPYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')}")
write_output("")

write_output("=== DIAGNOSTIC COMPLETE ===")
print(f"\n📄 Results saved to: {output_file}")

# Open the file
os.startfile(output_file)
