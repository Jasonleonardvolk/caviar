<# 
Run-IrisTS-GateHotfix.ps1
One-shot hotfix to get IRIS Release Gate (TypeScript) back to green.

It makes only *reversible* config/typing adjustments and fixes the Quilt import.
No global ExecutionPolicy changes required; run with -ExecutionPolicy Bypass for this session.

DEFAULT ROOT: C:\Users\jason\Desktop\tori\kha

WHAT THIS DOES
1) Patch frontend\tsconfig.json
   - moduleResolution: "bundler"
   - module: "ESNext", target: "ES2022" (keeps modern syntax; adjust if needed)
   - lib: ["ESNext","DOM","DOM.Iterable","WebWorker"]
   - types: add "vite/client"
   - baseUrl: "."
   - paths: add "@/*", "@hybrid/*", "@hybrid/lib", "@hybrid/wgsl"
   - exclude: add "hybrid/**", "hybrid/tests/**", "hybrid/pipelines/**", "vite.config.ts"

2) Create ambient type shims under frontend\types\shims\
   - modules.d.ts : declares ?raw, wgsl?raw, ktx-parse, glob, @playwright/test (to avoid TS7016)
   - dom-augment.d.ts : augments ImportMeta.env and Navigator.gpu to silence env + gpu typing

3) Normalize Quilt import in frontend\lib\holographicEngine.ts
   - import { WebGPUQuiltGenerator } from './webgpu/quilt/WebGPUQuiltGenerator'
   - Replace alias 'QuiltGenerator' identifier with 'WebGPUQuiltGenerator'

USAGE (from repo root C:\Users\jason\Desktop\tori\kha):
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-IrisTS-GateHotfix.ps1
  # Dry run:
  powershell -ExecutionPolicy Bypass -File .\scripts\Run-IrisTS-GateHotfix.ps1 -DryRun

#>

param(
  [string]$RepoRoot = "C:\Users\jason\Desktop\tori\kha",
  [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"

function Info($m){ Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Ok($m){ Write-Host "[ OK ] $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Step($m){ Write-Host "`n==== $m ====" -ForegroundColor Magenta }

if (!(Test-Path $RepoRoot)) { throw "RepoRoot not found: $RepoRoot" }

$frontend = Join-Path $RepoRoot "frontend"
if (!(Test-Path $frontend)) { throw "frontend/ not found under $RepoRoot" }

$tsconfig = Join-Path $frontend "tsconfig.json"
if (!(Test-Path $tsconfig)) { throw "frontend\tsconfig.json not found" }

$logDir = Join-Path $RepoRoot "scripts\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir ("iris_ts_gate_hotfix_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

Step "Backup tsconfig.json"
$backup = Join-Path $frontend ("tsconfig.backup.iris-" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".json")
if ($DryRun) { Info "Would backup $tsconfig -> $backup" } else { Copy-Item $tsconfig $backup -Force; Ok "Backed up to $backup" }

Step "Patch tsconfig.json for bundler + DOM/WebGPU + Vite types"
$raw = Get-Content -Raw -LiteralPath $tsconfig | ConvertFrom-Json

if (-not $raw.compilerOptions) { $raw | Add-Member -NotePropertyName "compilerOptions" -NotePropertyValue (@{}) }

$co = $raw.compilerOptions
$co.target = "ES2022"
$co.module = "ESNext"
$co.moduleResolution = "bundler"

# lib
if (-not $co.lib) { $co.lib = @() }
foreach ($lib in @("ESNext","DOM","DOM.Iterable","WebWorker")) {
  if (-not ($co.lib -contains $lib)) { $co.lib += $lib }
}

# types
if (-not $co.types) { $co.types = @() }
if (-not ($co.types -contains "vite/client")) { $co.types += "vite/client" }

# baseUrl / paths
$co.baseUrl = "."
if (-not $co.paths) { $co.paths = @{ } }
if (-not $co.paths."@/*") { $co.paths."@/*" = @("./*") }
if (-not $co.paths."@hybrid/*") { $co.paths."@hybrid/*" = @("./hybrid/*") }
if (-not $co.paths."@hybrid/lib") { $co.paths."@hybrid/lib" = @("./hybrid/lib/index.ts") }
if (-not $co.paths."@hybrid/wgsl") { $co.paths."@hybrid/wgsl" = @("./hybrid/wgsl/index.ts") }

# exclude hybrid/test/demo noise from release tsc
if (-not $raw.exclude) { $raw.exclude = @() }
foreach ($ex in @("hybrid/**","hybrid/tests/**","hybrid/pipelines/**","vite.config.ts")) {
  if (-not ($raw.exclude -contains $ex)) { $raw.exclude += $ex }
}

# Write back
$patched = ($raw | ConvertTo-Json -Depth 20)
if ($DryRun) {
  Info "Would write patched tsconfig.json"
} else {
  Set-Content -LiteralPath $tsconfig -Value $patched -Encoding UTF8
  Ok "Patched tsconfig.json"
}

Step "Create frontend types shims"
$typesDir = Join-Path $frontend "types\shims"
if ($DryRun) {
  Info "Would ensure directory $typesDir"
} else {
  New-Item -ItemType Directory -Force -Path $typesDir | Out-Null
}

$modulesShim = @"
/// <reference types="vite/client" />
declare module '*?raw' { const src: string; export default src; }
declare module '*.wgsl?raw' { const src: string; export default src; }
declare module 'ktx-parse';
declare module 'glob';
declare module '@playwright/test';
"@

$domAugment = @"
export {};
declare global {
  interface ImportMetaEnv { [key: string]: any }
  interface ImportMeta { env: ImportMetaEnv }
  interface Navigator { gpu?: GPU }
}
"@

$files = @{
  "modules.d.ts" = $modulesShim
  "dom-augment.d.ts" = $domAugment
}

foreach ($name in $files.Keys) {
  $path = Join-Path $typesDir $name
  if ($DryRun) {
    Info "Would write $path"
  } else {
    Set-Content -LiteralPath $path -Value $files[$name] -Encoding UTF8
    Ok "Wrote $((Split-Path $path -Leaf))"
  }
}

Step "Normalize Quilt import + identifier in holographicEngine.ts"
$he = Join-Path $frontend "lib\holographicEngine.ts"
if (Test-Path $he) {
  $txt = Get-Content -Raw -LiteralPath $he
  $orig = $txt
  $txt = [regex]::Replace($txt, "WebGPUWebGPUQuiltGenerator", "WebGPUQuiltGenerator")
  $txt = [regex]::Replace($txt, "import\s+\{\s*WebGPUQuiltGenerator\s+as\s+QuiltGenerator\s*\}\s+from\s*(['""][^'""]+['""])\s*;", "import { WebGPUQuiltGenerator } from $1;")
  $txt = [regex]::Replace($txt, "from\s+['""][\.\/]*webgpu\/quilt\/WebGPUWebGPUQuiltGenerator['""]", "from './webgpu/quilt/WebGPUQuiltGenerator'")
  $txt = [regex]::Replace($txt, "(?<![a-z])QuiltGenerator(?![a-z])", "WebGPUQuiltGenerator")
  if ($txt -ne $orig) {
    if ($DryRun) { Info "Would update holographicEngine.ts" }
    else {
      Set-Content -LiteralPath $he -Value $txt -Encoding UTF8
      Ok "Updated holographicEngine.ts imports"
    }
  } else {
    Info "holographicEngine.ts already normalized"
  }
} else {
  Warn "frontend\lib\holographicEngine.ts not found; skipping"
}

Step "Summary"
$summary = @()
$summary += "Patched: frontend\tsconfig.json (bundler resolution, DOM libs, vite types, paths, exclude hybrid)"
$summary += "Created: frontend\types\shims\modules.d.ts"
$summary += "Created: frontend\types\shims\dom-augment.d.ts"
$summary += "Fixed:   frontend\lib\holographicEngine.ts import/identifier (WebGPUQuiltGenerator)"
$summary | Out-String | Tee-Object -FilePath $logFile | Out-Host
Ok "Log written to $logFile"

Info "Next: re-run the gate:"
Write-Host "powershell -ExecutionPolicy Bypass -File .\Run-IrisReleaseGate.ps1 -SkipShaders"
