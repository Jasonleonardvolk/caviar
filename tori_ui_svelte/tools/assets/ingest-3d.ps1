# ingest-3d.ps1
# Optimize 3D models for AR usage in iRis

param(
    [Parameter(Mandatory=$true)]
    [string]$InPath,   # Path to raw .gltf/.glb file
    
    [Parameter(Mandatory=$true)]
    [string]$OutDir,   # Output directory for optimized model
    
    [ValidateSet('draco', 'meshopt')]
    [string]$Mode = 'draco',  # Compression mode
    
    [switch]$Verbose,
    [switch]$SkipBackup
)

$ErrorActionPreference = "Stop"

# Validate input file exists
if (!(Test-Path $InPath)) {
    Write-Error "Input file not found: $InPath"
    exit 1
}

# Create output directory if it doesn't exist
if (!(Test-Path $OutDir)) {
    Write-Host "Creating output directory: $OutDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

# Determine output filename
$ext = [System.IO.Path]::GetExtension($InPath).ToLower()
$baseName = [System.IO.Path]::GetFileNameWithoutExtension($InPath)
$outFile = if ($Mode -eq 'meshopt') { 
    "model.meshopt.glb" 
} else { 
    "model.draco.glb" 
}
$outPath = Join-Path $OutDir $outFile

# Backup existing file if present
if ((Test-Path $outPath) -and !$SkipBackup) {
    $backupPath = Join-Path $OutDir "model.backup.$(Get-Date -Format 'yyyyMMdd-HHmmss').glb"
    Write-Host "Backing up existing file to: $backupPath" -ForegroundColor Yellow
    Copy-Item $outPath $backupPath
}

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "      iRis 3D Asset Optimization" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Input:  $InPath" -ForegroundColor White
Write-Host "Output: $outPath" -ForegroundColor White
Write-Host "Mode:   $Mode compression" -ForegroundColor White
Write-Host ""

# Check if Node.js is available
$nodeVersion = node --version 2>$null
if (!$nodeVersion) {
    Write-Error "Node.js is not installed or not in PATH"
    exit 1
}
Write-Host "Node.js version: $nodeVersion" -ForegroundColor Gray

# Check if optimization script exists
$optimizeScript = Join-Path (Split-Path -Parent $PSScriptRoot) "assets\optimize-gltf.mjs"
if (!(Test-Path $optimizeScript)) {
    Write-Error "Optimization script not found: $optimizeScript"
    Write-Host "Please ensure optimize-gltf.mjs is in tools/assets/" -ForegroundColor Red
    exit 1
}

# Run optimization
Write-Host "`nOptimizing model..." -ForegroundColor Yellow
$startTime = Get-Date

try {
    if ($Verbose) {
        node $optimizeScript $InPath $outPath $Mode
    } else {
        $output = node $optimizeScript $InPath $outPath $Mode 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Optimization failed: $output"
            exit 1
        }
        Write-Host $output
    }
    
    $duration = (Get-Date) - $startTime
    Write-Host "`n[OK] Optimization completed in $([math]::Round($duration.TotalSeconds, 2)) seconds" -ForegroundColor Green
    
    # Show file sizes
    $inSize = (Get-Item $InPath).Length
    $outSize = (Get-Item $outPath).Length
    $reduction = [math]::Round((1 - $outSize/$inSize) * 100, 1)
    
    Write-Host ""
    Write-Host "File size comparison:" -ForegroundColor Cyan
    Write-Host "  Original:  $([math]::Round($inSize/1MB, 2)) MB" -ForegroundColor White
    Write-Host "  Optimized: $([math]::Round($outSize/1MB, 2)) MB" -ForegroundColor White
    Write-Host "  Reduction: ${reduction}%" -ForegroundColor Green
    
    # Create metadata file
    $metaPath = Join-Path $OutDir "model.meta.json"
    $metadata = @{
        source = $InPath
        output = $outFile
        mode = $Mode
        timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        originalSize = $inSize
        optimizedSize = $outSize
        reduction = "${reduction}%"
    }
    $metadata | ConvertTo-Json -Depth 10 | Out-File $metaPath -Encoding UTF8
    Write-Host "`nMetadata saved to: $metaPath" -ForegroundColor Gray
    
} catch {
    Write-Error "Optimization failed: $_"
    exit 1
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "         Optimization Complete!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Test the model at: http://localhost:3000/assets" -ForegroundColor White
Write-Host "  2. Verify AR functionality on mobile device" -ForegroundColor White
Write-Host "  3. Update ASSET_MANIFEST.json if needed" -ForegroundColor White
Write-Host ""