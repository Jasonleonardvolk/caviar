#!/usr/bin/env pwsh
# NUCLEAR DIAGNOSTIC - Find and destroy all import/path issues

$outputFile = "C:\Users\jason\Desktop\tori\kha\omg.txt"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Start fresh
"" | Out-File $outputFile
"=== TORI MCP NUCLEAR DIAGNOSTIC ===" | Out-File $outputFile -Append
"Timestamp: $timestamp" | Out-File $outputFile -Append
"This will find EVERY issue and crush it`n" | Out-File $outputFile -Append

Write-Host "`n🔥 NUCLEAR DIAGNOSTIC STARTING..." -ForegroundColor Red
Write-Host "Output: $outputFile" -ForegroundColor Yellow

# Navigate to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath
"Working Directory: $(Get-Location)" | Out-File $outputFile -Append

# SECTION 1: Python Environment
"`n=== SECTION 1: PYTHON ENVIRONMENT ===" | Out-File $outputFile -Append
$pythonVersion = python --version 2>&1
"Python Version: $pythonVersion" | Out-File $outputFile -Append
$pythonPath = python -c "import sys; print(sys.executable)"
"Python Executable: $pythonPath" | Out-File $outputFile -Append

# Check virtual environment using a temp file
@"
import sys
import os
venv_active = 'VIRTUAL_ENV' in os.environ or hasattr(sys, 'real_prefix') or sys.prefix != sys.base_prefix
print(venv_active)
"@ | Out-File -FilePath "temp_venv_check.py" -Encoding UTF8
$venvActive = python temp_venv_check.py
Remove-Item "temp_venv_check.py"
"Virtual Environment Active: $venvActive" | Out-File $outputFile -Append

# SECTION 2: File System Reality Check
"`n=== SECTION 2: FILE SYSTEM REALITY CHECK ===" | Out-File $outputFile -Append

# Check critical files exist
$criticalFiles = @(
    "mcp_metacognitive\__init__.py",
    "mcp_metacognitive\core\__init__.py", 
    "mcp_metacognitive\core\soliton_memory.py",
    "mcp_metacognitive\tools\__init__.py",
    "mcp_metacognitive\tools\soliton_memory_tools.py",
    "mcp_metacognitive\server.py",
    "setup.py",
    "pyproject.toml"
)

foreach ($file in $criticalFiles) {
    $exists = Test-Path $file
    $status = if ($exists) { "✅ EXISTS" } else { "❌ MISSING" }
    "$status : $file" | Out-File $outputFile -Append
}

# SECTION 3: Python Path Analysis
"`n=== SECTION 3: PYTHON PATH ANALYSIS ===" | Out-File $outputFile -Append
@"
import sys
import os
print('sys.path entries:')
for i, p in enumerate(sys.path):
    print(f'  [{i}] {p}')
    
cwd = os.getcwd()
if cwd in sys.path:
    print(f'\n✅ Current directory IS in sys.path: {cwd}')
else:
    print(f'\n❌ Current directory NOT in sys.path: {cwd}')
"@ | Out-File -FilePath "temp_path_check.py" -Encoding UTF8
python temp_path_check.py | Out-File $outputFile -Append
Remove-Item "temp_path_check.py"

# SECTION 4: Import Tests - The Nuclear Option
"`n=== SECTION 4: IMPORT TESTS - NUCLEAR OPTION ===" | Out-File $outputFile -Append
"`nTest 1: Direct Import Attempts" | Out-File $outputFile -Append

@"
import sys
import traceback

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
        print(f'✅ {module_name}: SUCCESS - {description}')
        if hasattr(mod, '__file__'):
            print(f'   Location: {mod.__file__}')
    except ImportError as e:
        print(f'❌ {module_name}: FAILED - {description}')
        print(f'   Error: {e}')
    except Exception as e:
        print(f'💥 {module_name}: UNEXPECTED ERROR - {description}')
        print(f'   Error: {type(e).__name__}: {e}')
"@ | Out-File -FilePath "temp_import_test.py" -Encoding UTF8
python temp_import_test.py | Out-File $outputFile -Append
Remove-Item "temp_import_test.py"

# Test 2: Check what's actually in the modules
"`nTest 2: Module Contents Check" | Out-File $outputFile -Append
@"
try:
    import mcp_metacognitive.core.soliton_memory as sm
    print('✅ Successfully imported soliton_memory')
    print(f'   Module file: {sm.__file__}')
    print('   Available attributes:')
    for attr in dir(sm):
        if not attr.startswith('_'):
            print(f'     - {attr}')
except Exception as e:
    print(f'❌ Failed to import soliton_memory: {e}')
    import traceback
    traceback.print_exc()
"@ | Out-File -FilePath "temp_module_check.py" -Encoding UTF8
python temp_module_check.py | Out-File $outputFile -Append
Remove-Item "temp_module_check.py"

# Test 3: Check the actual import line that's failing
"`nTest 3: Testing The Exact Failing Import" | Out-File $outputFile -Append
@"
import os
import sys

print('Current working directory:', os.getcwd())
print('__name__ when running directly:', __name__)

try:
    from core.soliton_memory import VaultStatus, ContentType
    print('❌ BAD NEWS: "from core.soliton_memory" actually worked!')
    print('   This means something is adding "core" to sys.path')
except ImportError as e:
    print('✅ GOOD: "from core.soliton_memory" failed as expected')
    print(f'   Error: {e}')

try:
    from mcp_metacognitive.core.soliton_memory import VaultStatus, ContentType
    print('✅ GOOD: Correct import works!')
    print(f'   VaultStatus: {VaultStatus}')
    print(f'   ContentType: {ContentType}')
except ImportError as e:
    print('❌ BAD: Even the correct import failed!')
    print(f'   Error: {e}')
"@ | Out-File -FilePath "temp_failing_test.py" -Encoding UTF8
python temp_failing_test.py | Out-File $outputFile -Append
Remove-Item "temp_failing_test.py"

# SECTION 5: Package Installation Status
"`n=== SECTION 5: PACKAGE INSTALLATION STATUS ===" | Out-File $outputFile -Append
pip show tori-kha 2>&1 | Out-File $outputFile -Append
if ($LASTEXITCODE -ne 0) {
    "❌ Package 'tori-kha' is NOT installed via pip" | Out-File $outputFile -Append
    "   Run: pip install -e ." | Out-File $outputFile -Append
} else {
    "✅ Package 'tori-kha' is installed" | Out-File $outputFile -Append
}

# SECTION 6: Check the actual import in soliton_memory_tools.py
"`n=== SECTION 6: ACTUAL IMPORT LINE CHECK ===" | Out-File $outputFile -Append
if (Test-Path "mcp_metacognitive\tools\soliton_memory_tools.py") {
    $importLine = Select-String -Path "mcp_metacognitive\tools\soliton_memory_tools.py" -Pattern "from.*soliton_memory.*import" | Select-Object -First 1
    "Import line in soliton_memory_tools.py:" | Out-File $outputFile -Append
    "  $($importLine.Line.Trim())" | Out-File $outputFile -Append
}

# SECTION 7: Running server.py different ways
"`n=== SECTION 7: SERVER STARTUP TESTS ===" | Out-File $outputFile -Append

# Test 1: Wrong way (direct) - capture first few lines of output
"`nTest 1: Running server.py directly (WRONG WAY):" | Out-File $outputFile -Append
$proc1 = Start-Process -FilePath "python" -ArgumentList "mcp_metacognitive/server.py" -NoNewWindow -RedirectStandardError "temp_error1.txt" -RedirectStandardOutput "temp_output1.txt" -PassThru
Start-Sleep -Milliseconds 500
if (!$proc1.HasExited) { Stop-Process -Id $proc1.Id -Force }
if (Test-Path "temp_error1.txt") {
    Get-Content "temp_error1.txt" | Select-Object -First 10 | Out-File $outputFile -Append
    Remove-Item "temp_error1.txt"
}
if (Test-Path "temp_output1.txt") {
    Get-Content "temp_output1.txt" | Select-Object -First 10 | Out-File $outputFile -Append
    Remove-Item "temp_output1.txt"
}

# Test 2: Right way (module)
"`nTest 2: Running as module (RIGHT WAY):" | Out-File $outputFile -Append  
$proc2 = Start-Process -FilePath "python" -ArgumentList "-m", "mcp_metacognitive.server" -NoNewWindow -RedirectStandardError "temp_error2.txt" -RedirectStandardOutput "temp_output2.txt" -PassThru
Start-Sleep -Milliseconds 500
if (!$proc2.HasExited) { Stop-Process -Id $proc2.Id -Force }
if (Test-Path "temp_error2.txt") {
    Get-Content "temp_error2.txt" | Select-Object -First 10 | Out-File $outputFile -Append
    Remove-Item "temp_error2.txt"
}
if (Test-Path "temp_output2.txt") {
    Get-Content "temp_output2.txt" | Select-Object -First 10 | Out-File $outputFile -Append
    Remove-Item "temp_output2.txt"
}

# SECTION 8: The Smoking Gun
"`n=== SECTION 8: THE SMOKING GUN ===" | Out-File $outputFile -Append
@"
import os
import sys

print('Checking for rogue "core" in sys.path:')
for path in sys.path:
    core_path = os.path.join(path, 'core')
    if os.path.exists(core_path) and os.path.isdir(core_path):
        print(f'❌ FOUND: {core_path}')
        if os.path.exists(os.path.join(core_path, 'soliton_memory.py')):
            print('   ⚠️  Contains soliton_memory.py!')
    
print('\nPYTHONPATH:', os.environ.get('PYTHONPATH', 'NOT SET'))
"@ | Out-File -FilePath "temp_smoking_gun.py" -Encoding UTF8
python temp_smoking_gun.py | Out-File $outputFile -Append
Remove-Item "temp_smoking_gun.py"

# FINAL SUMMARY
"`n=== FINAL DIAGNOSIS ===" | Out-File $outputFile -Append
"Check the above output for:" | Out-File $outputFile -Append
"1. Any ❌ marks indicating missing files or failed imports" | Out-File $outputFile -Append
"2. The exact import line in soliton_memory_tools.py" | Out-File $outputFile -Append  
"3. Whether 'tori-kha' package is installed" | Out-File $outputFile -Append
"4. Any rogue 'core' directories in sys.path" | Out-File $outputFile -Append

# Show results
Write-Host "`n✅ Diagnostic complete!" -ForegroundColor Green
Write-Host "📄 Results saved to: $outputFile" -ForegroundColor Yellow
Write-Host "`n🔥 Opening results..." -ForegroundColor Red

# Open the file
notepad $outputFile
