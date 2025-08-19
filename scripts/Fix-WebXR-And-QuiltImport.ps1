<# 
Fix-WebXR-And-QuiltImport.ps1
Purpose:
  - Install @types/webxr to satisfy TS compiler.
  - Normalize holographicEngine.ts to use correct WebGPUQuiltGenerator import & name.

Run from repo root:
  powershell -ExecutionPolicy Bypass -File .\scripts\Fix-WebXR-And-QuiltImport.ps1
#>

param(
  [string]$RepoRoot = "C:\Users\jason\Desktop\tori\kha"
)

$ErrorActionPreference = "Stop"

function Write-Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Write-Ok($m){ Write-Host "[ OK ] $m" -ForegroundColor Green }
function Write-Step($m){ Write-Host "`n==== $m ====" -ForegroundColor Magenta }

Write-Step "1. Install @types/webxr"
try {
    if (Test-Path (Join-Path $RepoRoot "package.json")) {
        Set-Location $RepoRoot
        Write-Info "Running: pnpm add -D @types/webxr"
        & pnpm add -D @types/webxr
        if ($LASTEXITCODE -eq 0) { Write-Ok "@types/webxr installed" } else { throw "pnpm failed" }
    } else {
        Write-Info "package.json not found at $RepoRoot — skipping type install"
    }
} catch {
    Write-Info "Could not install @types/webxr automatically. Please run: pnpm add -D @types/webxr"
}

Write-Step "2. Normalize holographicEngine.ts Quilt import/name"
$he = Join-Path $RepoRoot "frontend\lib\holographicEngine.ts"
if (Test-Path $he) {
    $text = Get-Content -Raw -LiteralPath $he

    # Remove any alias imports
    $text = [regex]::Replace($text, "import\s*\{\s*WebGPUQuiltGenerator\s+as\s+QuiltGenerator\s*\}\s*from\s*(['""][^'""]+['""])\s*;", "import { WebGPUQuiltGenerator } from $1;")
    $text = [regex]::Replace($text, "import\s+QuiltGenerator\s+from\s*(['""][^'""]+['""])\s*;", "import { WebGPUQuiltGenerator } from $1;")

    # Fix the path to canonical location
    $text = [regex]::Replace($text, "from\s+['""][\.\/]*webgpu[\\/]+quiltGenerator['""]\s*;", "from './webgpu/quilt/WebGPUQuiltGenerator';")
    $text = [regex]::Replace($text, "from\s+['""][\.\/]*webgpu[\\/]+quilt[\\/]+WebGPUQuiltGenerator['""]\s*;", "from './webgpu/quilt/WebGPUQuiltGenerator';")

    # Replace all capitalized QuiltGenerator references with WebGPUQuiltGenerator
    $text = [regex]::Replace($text, "(?<![a-z])QuiltGenerator(?![a-z])", "WebGPUQuiltGenerator")

    # Ensure import exists
    if ($text -notmatch "import\s*\{\s*WebGPUQuiltGenerator\s*\}\s*from\s*['""][\.\/]*webgpu\/quilt\/WebGPUQuiltGenerator['""]") {
        $text = "import { WebGPUQuiltGenerator } from './webgpu/quilt/WebGPUQuiltGenerator';`n" + $text
    }

    Set-Content -LiteralPath $he -Value $text -Encoding UTF8
    Write-Ok "Patched holographicEngine.ts to canonical import/name"
} else {
    Write-Info "No holographicEngine.ts found at $he"
}

Write-Step "Done. Now re-run the gate (no shaders)"
Write-Host "powershell -ExecutionPolicy Bypass -File .\Run-IrisReleaseGate.ps1 -SkipShaders" -ForegroundColor Yellow
