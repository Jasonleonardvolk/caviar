# Fix-WGSL-Phase2.ps1
# Mechanical fixes for common WGSL validation errors

param(
    [switch]$Apply = $false
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $PSCommandPath
$repoRoot = Resolve-Path (Join-Path $here "..\..")
$timestamp = Get-Date -Format "yyyy-MM-ddTHH-mm-ss"

Write-Host "Phase-2 WGSL Mechanical Fixes" -ForegroundColor Cyan
Write-Host "Repository root: $repoRoot"

$targetPaths = @(
    "frontend\lib\webgpu\shaders",
    "frontend\public\hybrid\wgsl"
)

$fixCount = 0

foreach ($targetPath in $targetPaths) {
    $fullPath = Join-Path $repoRoot $targetPath
    if (-not (Test-Path $fullPath)) { continue }
    
    $files = Get-ChildItem -Path $fullPath -Filter *.wgsl -Recurse
    
    foreach ($file in $files) {
        $content = Get-Content $file.FullName -Raw
        $original = $content
        
        # Fix 1: textureLoad missing mip level
        $content = $content -replace 'textureLoad\(([^,]+),\s*([^,)]+)\)', 'textureLoad($1, $2, 0)'
        
        # Fix 2: Swizzle assignments
        $content = $content -replace '\.rgb\s*=\s*([^;]+);', ' = vec4($1, $0.a);'
        
        # Fix 3: Workgroup size > 256
        $content = $content -replace '@workgroup_size\((\d+)\)', {
            param($m)
            $size = [int]$m.Groups[1].Value
            if ($size -gt 256) {
                '@workgroup_size(256)'
            } else {
                $m.Value
            }
        }
        
        # Fix 4: Remove leading BOM
        if ($content.StartsWith([char]0xFEFF)) {
            $content = $content.Substring(1)
        }
        
        # Fix 5: Remove leading braces
        $content = $content -replace '^\s*\{', ''
        
        if ($content -ne $original) {
            if ($Apply) {
                Set-Content -Path $file.FullName -Value $content -Encoding UTF8
                Write-Host "  FIXED: $($file.Name)" -ForegroundColor Green
            } else {
                Write-Host "  WOULD FIX: $($file.Name)" -ForegroundColor Yellow
            }
            $fixCount++
        }
    }
}

Write-Host ""
Write-Host "Phase-2 Summary: $fixCount files need fixes" -ForegroundColor Magenta
if (-not $Apply) {
    Write-Host "Run with -Apply to fix" -ForegroundColor Cyan
}
