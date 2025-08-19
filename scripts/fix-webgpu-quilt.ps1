<# 
fix-webgpu-quilt.ps1
One-shot fixer for the WebGPU quilt generator naming/path convention.

Default repo root: C:\Users\jason\Desktop\tori\kha
- Moves:   frontend\lib\webgpu\quiltGenerator.ts
         → frontend\lib\webgpu\quilt\WebGPUQuiltGenerator.ts
- Normalizes imports to named { WebGPUQuiltGenerator } (no aliasing).
- Fixes import paths to 'webgpu/quilt/WebGPUQuiltGenerator' (forward slashes).
- Ensures the target file has both a named and default export.
- Compiles TypeScript to validate.

USAGE (PowerShell):
  Set-ExecutionPolicy -Scope Process Bypass
  .\scripts\fix-webgpu-quilt.ps1
  # or with options:
  .\scripts\fix-webgpu-quilt.ps1 -RepoRoot "C:\Users\jason\Desktop\tori\kha" -DryRun

PARAMS:
  -RepoRoot <string>  : repo root (default: C:\Users\jason\Desktop\tori\kha)
  -DryRun             : show planned changes only
  -NoGit              : use Move-Item instead of git mv
#>

param(
  [string]$RepoRoot = "C:\Users\jason\Desktop\tori\kha",
  [switch]$DryRun = $false,
  [switch]$NoGit = $false
)

$ErrorActionPreference = "Stop"

function Write-Info($msg)  { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg)  { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Ok($msg)    { Write-Host "[ OK ] $msg"  -ForegroundColor Green }
function Write-Step($msg)  { Write-Host "`n==== $msg ====" -ForegroundColor Magenta }

# Normalize paths
$srcRel = "frontend\lib\webgpu\quiltGenerator.ts"
$dstRel = "frontend\lib\webgpu\quilt\WebGPUQuiltGenerator.ts"

if (!(Test-Path $RepoRoot)) { throw "RepoRoot not found: $RepoRoot" }
Set-Location $RepoRoot

$logDir = Join-Path $RepoRoot "scripts\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir ("fix-webgpu-quilt_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
$changedFiles = New-Object System.Collections.Generic.List[string]

Write-Step "Detect source file"
$srcPath = Join-Path $RepoRoot $srcRel
if (!(Test-Path $srcPath)) {
  Write-Warn "Expected $srcRel not found. Searching for quiltGenerator.ts under frontend\lib\webgpu ..."
  $candidate = Get-ChildItem -Path (Join-Path $RepoRoot "frontend\lib\webgpu") -Recurse -Filter "quiltGenerator.ts" -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($candidate) {
    $srcPath = $candidate.FullName
    $srcRel  = $srcPath.Substring($RepoRoot.Length + 1)
    Write-Info "Found candidate at: $srcRel"
  } else {
    throw "quiltGenerator.ts not found anywhere under frontend\lib\webgpu"
  }
} else {
  Write-Ok "Found $srcRel"
}

Write-Step "Prepare destination folder"
$dstPath = Join-Path $RepoRoot $dstRel
$dstDir  = Split-Path $dstPath -Parent
if (!(Test-Path $dstDir)) {
  if ($DryRun) { Write-Info "Would create directory: $dstDir" }
  else { New-Item -ItemType Directory -Force -Path $dstDir | Out-Null }
}

Write-Step "Move file to canonical location/name"
if ($srcPath -ieq $dstPath) {
  Write-Info "Already at canonical path: $dstRel"
} else {
  if ($DryRun) {
    Write-Info "Would move: $srcRel  ->  $dstRel"
  } else {
    if ((Test-Path ".git") -and -not $NoGit) {
      & git mv -f -- "$srcRel" "$dstRel"
      if ($LASTEXITCODE -ne 0) {
        Write-Warn "git mv failed; falling back to Move-Item"
        Move-Item -Force -LiteralPath $srcPath -Destination $dstPath
      }
    } else {
      Move-Item -Force -LiteralPath $srcPath -Destination $dstPath
    }
    Write-Ok "Moved to $dstRel"
  }
}

Write-Step "Ensure named + default export in target file"
if (-not $DryRun) {
  $content = Get-Content -Raw -LiteralPath $dstPath
  $orig = $content
  # Ensure the class name is WebGPUQuiltGenerator
  $content = [regex]::Replace($content, "class\s+QuiltGenerator", "class WebGPUQuiltGenerator")
  # Ensure 'export class WebGPUQuiltGenerator'
  $content = [regex]::Replace($content, "(?m)^\s*class\s+WebGPUQuiltGenerator", "export class WebGPUQuiltGenerator")
  if ($content -notmatch "export\s+default\s+WebGPUQuiltGenerator") {
    $content += "`nexport default WebGPUQuiltGenerator;`n"
  }
  if ($content -ne $orig) {
    Set-Content -LiteralPath $dstPath -Value $content -Encoding UTF8
    $changedFiles.Add($dstRel) | Out-Null
    Write-Ok "Normalized exports in $dstRel"
  } else {
    Write-Info "Exports already OK in $dstRel"
  }
} else {
  Write-Info "Would ensure 'export class WebGPUQuiltGenerator' and default export in $dstRel"
}

Write-Step "Rewrite imports/usages across frontend"
$scanRoot = Join-Path $RepoRoot "frontend"
$tsFiles = Get-ChildItem -Path $scanRoot -Recurse -Include *.ts,*.tsx -File

$replacements = @(
  @{ Name="PathFix"; Pattern="webgpu[\\/]+quiltGenerator"; Replace="webgpu/quilt/WebGPUQuiltGenerator" },
  @{ Name="AliasImport"; Pattern="import\s*\{\s*WebGPUQuiltGenerator\s+as\s+QuiltGenerator\s*\}\s*from\s*['""][^'""]+['""]\s*;"; Replace="import { WebGPUQuiltGenerator } from '$0';"; IsSpecial=$true },
  @{ Name="DefaultToNamed"; Pattern="import\s+QuiltGenerator\s+from\s*(['""][^'""]+['""])\s*;"; Replace="import { WebGPUQuiltGenerator } from $1;" },
  @{ Name="TypeUses"; Pattern="\bQuiltGenerator\b"; Replace="WebGPUQuiltGenerator" } # capitalized identifier only
)

$modifiedCount = 0
foreach ($f in $tsFiles) {
  $text = Get-Content -Raw -LiteralPath $f.FullName
  $orig = $text

  # 1) Fix path segments first
  $text = [regex]::Replace($text, $replacements[0].Pattern, $replacements[0].Replace)

  # 2) Convert default import to named form
  $text = [regex]::Replace($text, $replacements[2].Pattern, $replacements[2].Replace)

  # 3) Remove aliasing { WebGPUQuiltGenerator as QuiltGenerator }
  # Because the path is already fixed, we can safely collapse alias
  $text = [regex]::Replace($text, "import\s*\{\s*WebGPUQuiltGenerator\s+as\s+QuiltGenerator\s*\}\s*from\s*(['""][^'""]+['""])\s*;", "import { WebGPUQuiltGenerator } from $1;")

  # 4) Replace identifier uses (types / constructors). Avoid touching lower-camel 'quiltGenerator'
  $text = [regex]::Replace($text, "(?<![a-z])QuiltGenerator(?![a-z])", "WebGPUQuiltGenerator")

  if ($text -ne $orig) {
    if ($DryRun) {
      Write-Info "Would update: $($f.FullName.Substring($RepoRoot.Length + 1))"
    } else {
      Set-Content -LiteralPath $f.FullName -Value $text -Encoding UTF8
      $changedFiles.Add($f.FullName.Substring($RepoRoot.Length + 1)) | Out-Null
      $modifiedCount++
    }
  }
}

if (-not $DryRun) {
  Write-Ok "Updated $modifiedCount file(s)."
} else {
  Write-Info "Dry-run: would update $modifiedCount file(s)."
}

Write-Step "Optionally compile to verify"
if ($DryRun) {
  Write-Info "Skipping compile in DryRun mode."
} else {
  $compiled = $false
  try {
    Write-Info "Running: pnpm -w tsc -p frontend\tsconfig.json"
    & pnpm -w tsc -p "frontend\tsconfig.json"
    if ($LASTEXITCODE -eq 0) { $compiled = $true }
  } catch { }

  if (-not $compiled) {
    try {
      Write-Warn "pnpm tsc failed or missing. Trying: npx tsc -p frontend\tsconfig.json"
      & npx tsc -p "frontend\tsconfig.json"
      if ($LASTEXITCODE -eq 0) { $compiled = $true }
    } catch { }
  }

  if ($compiled) { Write-Ok "TypeScript compile passed." }
  else { Write-Warn "TypeScript compile failed. Open the changed files and fix errors." }
}

Write-Step "Summary"
"Moved: $srcRel -> $dstRel" | Out-File -Append $logFile -Encoding UTF8
"Changed files:" | Out-File -Append $logFile
$changedFiles | Out-File -Append $logFile
Write-Ok "Log written to: $logFile"

Write-Info "Tip: Open the changed file list in VS Code Quick Open:"
Write-Host "code -g `"$dstRel`""
foreach ($f in $changedFiles) {
  Write-Host "code -g `"$f`""
}
