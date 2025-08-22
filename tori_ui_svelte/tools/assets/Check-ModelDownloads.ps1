#!/usr/bin/env pwsh
# Check-ModelDownloads.ps1
# Verify that model files have been placed correctly after downloading

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "   Checking Model Download Status" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = "D:\Dev\kha\tori_ui_svelte"

# Define expected paths
$models = @(
    @{
        Name = "Sneaker"
        RawPath = "assets\3d\luxury\sneakers\free\cgtrader_generic_sneaker\raw"
        ExpectedFiles = @("*.gltf", "*.glb", "*.fbx", "*.obj")
        DownloadUrl = "https://www.cgtrader.com/free-3d-models/sports/equipment/sneaker"
    },
    @{
        Name = "Watch"
        RawPath = "assets\3d\luxury\watches\free\sketchfab_lowpoly_watch\raw"
        ExpectedFiles = @("*.gltf", "*.glb", "*.fbx")
        DownloadUrl = "https://sketchfab.com/3d-models/low-poly-watch-85e3fe10d5e9411fbcd409f7eab6524f"
    },
    @{
        Name = "Perfume"
        RawPath = "assets\3d\luxury\perfume\free\sketchfab_perfume_bottle\raw"
        ExpectedFiles = @("*.gltf", "*.glb", "*.fbx")
        DownloadUrl = "https://sketchfab.com/3d-models/3d-perfume-bottle-for-product-visualizations-6e58cd19fb4f49829411c7f97df14f6e"
    }
)

$allReady = $true
$foundModels = 0

foreach ($model in $models) {
    $fullPath = Join-Path $projectRoot $model.RawPath
    Write-Host "$($model.Name) Model:" -ForegroundColor Yellow
    Write-Host "  Path: $($model.RawPath)" -ForegroundColor Gray
    
    if (-not (Test-Path $fullPath)) {
        Write-Host "  [X] Directory not found!" -ForegroundColor Red
        Write-Host "      Create it or run setup-ar-assets.bat" -ForegroundColor Gray
        $allReady = $false
        continue
    }
    
    # Check for model files
    $modelFiles = @()
    foreach ($pattern in $model.ExpectedFiles) {
        $modelFiles += Get-ChildItem -Path $fullPath -Filter $pattern -ErrorAction SilentlyContinue
    }
    
    if ($modelFiles.Count -eq 0) {
        Write-Host "  [X] No model files found!" -ForegroundColor Red
        Write-Host "      Download from: $($model.DownloadUrl)" -ForegroundColor Gray
        Write-Host "      Expected: .gltf, .glb, .fbx, or .obj files" -ForegroundColor Gray
        $allReady = $false
    } else {
        Write-Host "  [OK] Found $($modelFiles.Count) model file(s):" -ForegroundColor Green
        foreach ($file in $modelFiles) {
            $size = [math]::Round($file.Length / 1KB, 2)
            Write-Host "      - $($file.Name) (${size}KB)" -ForegroundColor Gray
        }
        $foundModels++
        
        # Check for textures
        $textureFiles = Get-ChildItem -Path $fullPath -Include "*.png","*.jpg","*.jpeg" -ErrorAction SilentlyContinue
        if ($textureFiles.Count -gt 0) {
            Write-Host "  [OK] Found $($textureFiles.Count) texture file(s)" -ForegroundColor Green
        }
        
        # Check for .bin files (GLTF binary data)
        $binFiles = Get-ChildItem -Path $fullPath -Filter "*.bin" -ErrorAction SilentlyContinue
        if ($binFiles.Count -gt 0) {
            Write-Host "  [OK] Found $($binFiles.Count) binary data file(s)" -ForegroundColor Green
        }
    }
    Write-Host ""
}

# Summary
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Summary:" -ForegroundColor Yellow

if ($allReady) {
    Write-Host "  [OK] All models downloaded and ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Run: .\process-ar-models.bat" -ForegroundColor White
    Write-Host "  2. Test: .\test-ar-assets.bat" -ForegroundColor White
    Write-Host "  3. View: http://localhost:3000/assets" -ForegroundColor White
} else {
    Write-Host "  Models found: $foundModels/3" -ForegroundColor Yellow
    Write-Host "  Status: NOT READY" -ForegroundColor Red
    Write-Host ""
    Write-Host "To download missing models:" -ForegroundColor Yellow
    Write-Host "  Run: .\open-download-pages.bat" -ForegroundColor White
    Write-Host ""
    Write-Host "After downloading:" -ForegroundColor Yellow
    Write-Host "  1. Extract ZIP files" -ForegroundColor White
    Write-Host "  2. Place contents in /raw directories shown above" -ForegroundColor White
    Write-Host "  3. Run this script again to verify" -ForegroundColor White
}

Write-Host "=====================================" -ForegroundColor Cyan