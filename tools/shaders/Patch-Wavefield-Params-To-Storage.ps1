# Patch-Wavefield-Params-To-Storage.ps1
# Purpose: Convert wavefield_params from uniform to storage buffer to fix stride issues
# This avoids the 16-byte alignment requirement for arrays in uniform buffers

param(
    [switch]$Apply = $false,
    [switch]$IncludeOscData = $false
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $PSCommandPath
$repoRoot = Resolve-Path (Join-Path $here "..\..")
$timestamp = Get-Date -Format "yyyy-MM-ddTHH-mm-ss"

Write-Host "Patching wavefield_params from uniform to storage buffer..."
Write-Host "Repository root: $repoRoot"

# Files to patch
$targetFiles = @(
    "frontend\lib\webgpu\shaders\wavefieldEncoder.wgsl",
    "frontend\public\hybrid\wgsl\wavefieldEncoder.wgsl"
)

$patchCount = 0

foreach ($relPath in $targetFiles) {
    $fullPath = Join-Path $repoRoot $relPath
    
    if (-not (Test-Path $fullPath)) {
        Write-Warning "File not found: $fullPath"
        continue
    }
    
    Write-Host "Processing: $relPath"
    
    # Backup original
    $backupPath = "$fullPath.bak_$timestamp"
    Copy-Item $fullPath $backupPath -Force
    
    # Read content
    $content = Get-Content $fullPath -Raw
    $originalContent = $content
    
    # Patch wavefield_params
    $content = $content -replace '@group\(0\)\s+@binding\(0\)\s+var<uniform>\s+wavefield_params:', '@group(0) @binding(0) var<storage, read> wavefield_params:'
    
    # Optionally patch osc_data too
    if ($IncludeOscData) {
        $content = $content -replace '@group\(1\)\s+@binding\(0\)\s+var<uniform>\s+osc_data:', '@group(1) @binding(0) var<storage, read> osc_data:'
    }
    
    if ($content -ne $originalContent) {
        if ($Apply) {
            Set-Content -Path $fullPath -Value $content -Encoding UTF8
            Write-Host "  [PATCHED] $relPath"
            Write-Host "  Backup saved to: $backupPath"
        } else {
            Write-Host "  [DRY RUN] Would patch: $relPath"
        }
        $patchCount++
    } else {
        Write-Host "  [SKIP] Already patched or not found: $relPath"
    }
}

Write-Host ""
Write-Host "==============================================="
Write-Host "Summary: $patchCount files need patching"

if (-not $Apply) {
    Write-Host ""
    Write-Host "This was a DRY RUN. To apply changes, run:"
    Write-Host "  .\Patch-Wavefield-Params-To-Storage.ps1 -Apply"
    Write-Host ""
    Write-Host "To also fix osc_data arrays, add -IncludeOscData:"
    Write-Host "  .\Patch-Wavefield-Params-To-Storage.ps1 -Apply -IncludeOscData"
} else {
    Write-Host "Patches applied successfully!"
    Write-Host ""
    Write-Host "IMPORTANT: Update your TypeScript/JavaScript WebGPU setup:"
    Write-Host "1. Change buffer creation:"
    Write-Host "   GPUBufferUsage.UNIFORM -> GPUBufferUsage.STORAGE"
    Write-Host "2. Change bind group layout:"
    Write-Host "   buffer: { type: 'uniform' } -> buffer: { type: 'read-only-storage' }"
}
