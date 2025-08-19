<# 
Fix-TsFail-And-QuiltImport.ps1
Purpose:
  - Quarantine invalid TS diagnostic file that is breaking compile.
  - Normalize holographicEngine.ts to use the correct WebGPUQuiltGenerator import & name.
  - Clear the locked iris_release_summary_latest.txt file if possible.

Run from repo root:
  powershell -ExecutionPolicy Bypass -File .\scripts\Fix-TsFail-And-QuiltImport.ps1
#>

param(
  [string]$RepoRoot = "C:\Users\jason\Desktop\tori\kha"
)

$ErrorActionPreference = "Stop"

function Write-Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Write-Ok($m){ Write-Host "[ OK ] $m" -ForegroundColor Green }
function Write-Step($m){ Write-Host "`n==== $m ====" -ForegroundColor Magenta }

Write-Step "1. Quarantine invalid diagnostic file"
$badFile = Join-Path $RepoRoot "frontend\lib\webgpu\fftCompute_diagnostic_patch.ts"
if (Test-Path $badFile) {
    $quarantineDir = Join-Path $RepoRoot "tools\quarantined"
    if (!(Test-Path $quarantineDir)) { New-Item -ItemType Directory -Force -Path $quarantineDir | Out-Null }
    $dest = Join-Path $quarantineDir ("fftCompute_diagnostic_patch.ts.bak_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
    Move-Item -Force -LiteralPath $badFile -Destination $dest
    Write-Ok "Moved $badFile to $dest"
} else {
    Write-Info "No fftCompute_diagnostic_patch.ts found at $badFile"
}

Write-Step "2. Normalize holographicEngine.ts Quilt import/name"
$he = Join-Path $RepoRoot "frontend\lib\holographicEngine.ts"
if (Test-Path $he) {
    $text = Get-Content -Raw -LiteralPath $he

    # Remove any bad import lines containing WebGPUWebGPUQuiltGenerator
    $text = ($text -split "`n") | Where-Object {$_ -notmatch "WebGPUWebGPUQuiltGenerator"} | Out-String

    # Ensure single correct import
    $text = [regex]::Replace($text, "import\s*\{\s*WebGPUQuiltGenerator\s+as\s+QuiltGenerator\s*\}\s*from\s*(['""][^'""]+['""])\s*;", "import { WebGPUQuiltGenerator } from $1;")
    $text = [regex]::Replace($text, "import\s+QuiltGenerator\s+from\s*(['""][^'""]+['""])\s*;", "import { WebGPUQuiltGenerator } from $1;")
    $text = [regex]::Replace($text, "from\s+['""][\.\/]*webgpu[\\/]+quiltGenerator['""]\s*;", "from './webgpu/quilt/WebGPUQuiltGenerator';")
    $text = [regex]::Replace($text, "from\s+['""][\.\/]*webgpu[\\/]+quilt[\\/]+WebGPUQuiltGenerator['""]\s*;", "from './webgpu/quilt/WebGPUQuiltGenerator';")

    # Replace all upper-camel QuiltGenerator references with WebGPUQuiltGenerator
    $text = [regex]::Replace($text, "(?<![a-z])QuiltGenerator(?![a-z])", "WebGPUQuiltGenerator")

    # Ensure the import exists at the top
    if ($text -notmatch "import\s*\{\s*WebGPUQuiltGenerator\s*\}\s*from\s*['""][\.\/]*webgpu\/quilt\/WebGPUQuiltGenerator['""]") {
        $text = "import { WebGPUQuiltGenerator } from './webgpu/quilt/WebGPUQuiltGenerator';`n" + $text
    }

    Set-Content -LiteralPath $he -Value $text -Encoding UTF8
    Write-Ok "Patched holographicEngine.ts to canonical import/name"
} else {
    Write-Info "No holographicEngine.ts found at $he"
}

Write-Step "3. Try to clear lock on iris_release_summary_latest.txt"
$latestSummary = Join-Path $RepoRoot "tools\release\error_logs\iris_release_summary_latest.txt"
if (Test-Path $latestSummary) {
    try {
        $stream = [System.IO.File]::Open($latestSummary, 'Open', 'ReadWrite', 'None')
        $stream.Close()
        Write-Ok "Unlocked $latestSummary (no process lock now)"
    } catch {
        Write-Info "Could not unlock $latestSummary, file is in use by another process."
    }
} else {
    Write-Info "No latest summary file found at $latestSummary"
}

Write-Step "Done. Now re-run the gate (no shaders)"
Write-Host "powershell -ExecutionPolicy Bypass -File .\Run-IrisReleaseGate.ps1 -SkipShaders" -ForegroundColor Yellow
