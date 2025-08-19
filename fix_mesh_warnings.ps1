<#
===============================================================================
fix_mesh_warnings.ps1
-------------------------------------------------------------------------------
Creates a proper Python package for concept_mesh, installs it *editable* into
the ALANPY311 virtual-env, and verifies the module can be imported from *any*
working directory (including mcp_metacognitive/).
===============================================================================
#>

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# 1. Locate the TORI venv that enhanced_launcher.py uses
# ---------------------------------------------------------------------------
$venv = "C:\ALANPY311"
$python = Join-Path $venv "python.exe"
$pip    = Join-Path $venv "Scripts\pip.exe"

if (-not (Test-Path $python)) {
    Write-Error "❌ Python interpreter not found at $python"
}

# ---------------------------------------------------------------------------
# 2. Ensure concept_mesh looks like a real package
# ---------------------------------------------------------------------------
$meshRoot = Join-Path $PSScriptRoot "concept_mesh"
$initFile = Join-Path $meshRoot "__init__.py"

if (-not (Test-Path $meshRoot)) {
    Write-Error "❌ Folder $meshRoot does not exist – aborting."
}

if (-not (Test-Path $initFile)) {
    "`n# concept_mesh package root" | Out-File -Encoding utf8 $initFile
    Write-Host "✅ Created missing __init__.py"
} else {
    Write-Host "ℹ️  __init__.py already present"
}

# ---------------------------------------------------------------------------
# 3. Editable-install the package into the venv
# ---------------------------------------------------------------------------
Write-Host "🔧 Installing concept_mesh in editable mode..."
& $pip install --quiet -e $meshRoot
Write-Host "✅ Editable install complete"

# ---------------------------------------------------------------------------
# 4. Smoke-test: import concept_mesh from *mcp_metacognitive* folder
# ---------------------------------------------------------------------------
$mcpDir   = Join-Path $PSScriptRoot "mcp_metacognitive"
$pyScript = @'
import importlib.util, os, json
spec = importlib.util.find_spec("concept_mesh")
print(json.dumps({
    "cwd": os.getcwd(),
    "spec_found": spec is not None,
    "spec_origin": spec.origin if spec else None
}))
'@   # <-- DO NOT delete this line: closes the here-string

Write-Host ">>> Verifying import from mcp_metacognitive/ ..."
Push-Location $mcpDir

# Run Python, feed the script via STDIN (“-”), capture JSON
$resultJson = $pyScript | & $python -
Pop-Location

$result = $resultJson | ConvertFrom-Json
if (-not $result.spec_found) {
    Write-Error "ERROR: concept_mesh still NOT importable!"
    exit 1
} else {
    Write-Host ("SUCCESS: concept_mesh found at {0}" -f $result.spec_origin)
}

Write-Host ""
Write-Host "All done!  Launch TORI normally ->  python enhanced_launcher.py"
Write-Host "You should no longer see the 'Concept mesh library not available' warnings."
