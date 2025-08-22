#!/usr/bin/env pwsh
# Process-ARModels.ps1
# Helper script to process downloaded 3D models

param(
    [Parameter(Mandatory=$false)]
    [switch]$CheckOnly
)

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "   AR Models Processing Helper" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = "D:\Dev\kha\tori_ui_svelte"
Set-Location $projectRoot

# Model configurations
$models = @(
    @{
        Name = "Sneaker"
        Category = "sneakers"
        Id = "cgtrader_generic_sneaker"
        RawPath = "assets\3d\luxury\sneakers\free\cgtrader_generic_sneaker\raw"
        OptPath = "assets\3d\luxury\sneakers\free\cgtrader_generic_sneaker\opt"
        Compression = "draco"
        SourceUrl = "https://www.cgtrader.com/free-3d-models/sports/equipment/sneaker"
    },
    @{
        Name = "Watch"
        Category = "watches"
        Id = "sketchfab_lowpoly_watch"
        RawPath = "assets\3d\luxury\watches\free\sketchfab_lowpoly_watch\raw"
        OptPath = "assets\3d\luxury\watches\free\sketchfab_lowpoly_watch\opt"
        Compression = "meshopt"
        SourceUrl = "https://sketchfab.com/3d-models/low-poly-watch-85e3fe10d5e9411fbcd409f7eab6524f"
    },
    @{
        Name = "Perfume"
        Category = "perfume"
        Id = "sketchfab_perfume_bottle"
        RawPath = "assets\3d\luxury\perfume\free\sketchfab_perfume_bottle\raw"
        OptPath = "assets\3d\luxury\perfume\free\sketchfab_perfume_bottle\opt"
        Compression = "draco"
        SourceUrl = "https://sketchfab.com/3d-models/3d-perfume-bottle-for-product-visualizations-6e58cd19fb4f49829411c7f97df14f6e"
    }
)

if ($CheckOnly) {
    Write-Host "Checking model status..." -ForegroundColor Yellow
    Write-Host ""
    
    foreach ($model in $models) {
        $rawDir = Join-Path $projectRoot $model.RawPath
        $optDir = Join-Path $projectRoot $model.OptPath
        
        $rawFiles = if (Test-Path $rawDir) { Get-ChildItem $rawDir -File } else { @() }
        $optFiles = if (Test-Path $optDir) { Get-ChildItem $optDir -File } else { @() }
        
        Write-Host "$($model.Name):" -ForegroundColor Cyan
        
        if ($rawFiles.Count -gt 0) {
            Write-Host "  [OK] Raw files: $($rawFiles.Count) file(s)" -ForegroundColor Green
            foreach ($file in $rawFiles) {
                Write-Host "       - $($file.Name)" -ForegroundColor Gray
            }
        } else {
            Write-Host "  [X] No raw files found" -ForegroundColor Red
            Write-Host "      Download from: $($model.SourceUrl)" -ForegroundColor Yellow
        }
        
        if ($optFiles.Count -gt 0) {
            Write-Host "  [OK] Optimized files: $($optFiles.Count) file(s)" -ForegroundColor Green
            foreach ($file in $optFiles) {
                $size = [math]::Round($file.Length / 1MB, 2)
                Write-Host "       - $($file.Name) (${size}MB)" -ForegroundColor Gray
            }
        } else {
            Write-Host "  [!] No optimized files" -ForegroundColor Yellow
        }
        Write-Host ""
    }
    
    Write-Host "Run without -CheckOnly to process available models" -ForegroundColor Cyan
    exit 0
}

# Process models
Write-Host "Processing available models..." -ForegroundColor Yellow
Write-Host ""

$processed = 0
$skipped = 0

foreach ($model in $models) {
    Write-Host "Processing $($model.Name)..." -ForegroundColor Cyan
    
    $rawDir = Join-Path $projectRoot $model.RawPath
    $optDir = Join-Path $projectRoot $model.OptPath
    
    # Check for raw files
    if (-not (Test-Path $rawDir)) {
        Write-Host "  [SKIP] No raw directory" -ForegroundColor Yellow
        $skipped++
        continue
    }
    
    $gltfFiles = Get-ChildItem $rawDir -Filter "*.gltf" -File
    $glbFiles = Get-ChildItem $rawDir -Filter "*.glb" -File
    $modelFiles = $gltfFiles + $glbFiles
    
    if ($modelFiles.Count -eq 0) {
        Write-Host "  [SKIP] No GLTF/GLB files in raw directory" -ForegroundColor Yellow
        Write-Host "         Download from: $($model.SourceUrl)" -ForegroundColor Gray
        $skipped++
        continue
    }
    
    # Create opt directory if needed
    if (-not (Test-Path $optDir)) {
        New-Item -ItemType Directory -Path $optDir -Force | Out-Null
    }
    
    # Process each model file
    foreach ($file in $modelFiles) {
        $inputPath = $file.FullName
        $outputName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
        
        Write-Host "  Processing: $($file.Name)" -ForegroundColor White
        Write-Host "  Compression: $($model.Compression)" -ForegroundColor Gray
        
        # Run optimization
        $command = "pwsh tools\assets\ingest-3d.ps1 `"$inputPath`" `"$optDir`" $($model.Compression)"
        Write-Host "  Running: $command" -ForegroundColor DarkGray
        
        Invoke-Expression $command
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [OK] Optimized successfully" -ForegroundColor Green
            $processed++
        } else {
            Write-Host "  [ERROR] Optimization failed" -ForegroundColor Red
        }
    }
    Write-Host ""
}

# Summary
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "  Processed: $processed model(s)" -ForegroundColor Green
Write-Host "  Skipped: $skipped model(s)" -ForegroundColor Yellow

if ($skipped -gt 0) {
    Write-Host ""
    Write-Host "Download missing models from:" -ForegroundColor Yellow
    foreach ($model in $models) {
        $rawDir = Join-Path $projectRoot $model.RawPath
        $hasFiles = (Test-Path $rawDir) -and ((Get-ChildItem $rawDir -File).Count -gt 0)
        
        if (-not $hasFiles) {
            Write-Host "  $($model.Name): $($model.SourceUrl)" -ForegroundColor Gray
        }
    }
}

Write-Host ""
Write-Host "Next: Run .\test-ar-assets.bat to verify" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan