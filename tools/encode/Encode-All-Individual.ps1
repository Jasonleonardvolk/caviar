# Encode-All-Individual.ps1
# Run each encoding command separately
# This script encodes all three videos one by one

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     Encoding WOW Pack Videos                  " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to the repo root
Set-Location "D:\Dev\kha"

# Check if videos exist
$video1 = ".\content\wowpack\input\holo_flux_loop.mov"
$video2 = ".\content\wowpack\input\mach_lightfield.mov"
$video3 = ".\content\wowpack\input\kinetic_logo_parade.mov"

$allExist = $true

if (Test-Path $video1) {
    Write-Host "[OK] Found: holo_flux_loop.mov" -ForegroundColor Green
} else {
    Write-Host "[X] Missing: holo_flux_loop.mov" -ForegroundColor Red
    $allExist = $false
}

if (Test-Path $video2) {
    Write-Host "[OK] Found: mach_lightfield.mov" -ForegroundColor Green
} else {
    Write-Host "[X] Missing: mach_lightfield.mov" -ForegroundColor Red
    $allExist = $false
}

if (Test-Path $video3) {
    Write-Host "[OK] Found: kinetic_logo_parade.mov" -ForegroundColor Green
} else {
    Write-Host "[X] Missing: kinetic_logo_parade.mov" -ForegroundColor Red
    $allExist = $false
}

if (-not $allExist) {
    Write-Host ""
    Write-Host "Some videos are missing. Please add them to:" -ForegroundColor Yellow
    Write-Host "D:\Dev\kha\content\wowpack\input\" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

Write-Host ""
Write-Host "All videos found! Starting encoding..." -ForegroundColor Green
Write-Host ""

# Encode video 1: holo_flux_loop with HLS
Write-Host "[1/3] Encoding holo_flux_loop..." -ForegroundColor Cyan
& .\tools\encode\Build-WowPack.ps1 `
    -Basename holo_flux_loop `
    -Input $video1 `
    -Framerate 60 `
    -DoSDR `
    -MakeHLS

Write-Host ""
Write-Host "[2/3] Encoding mach_lightfield..." -ForegroundColor Cyan
& .\tools\encode\Build-WowPack.ps1 `
    -Basename mach_lightfield `
    -Input $video2 `
    -Framerate 60 `
    -DoSDR

Write-Host ""
Write-Host "[3/3] Encoding kinetic_logo_parade..." -ForegroundColor Cyan
& .\tools\encode\Build-WowPack.ps1 `
    -Basename kinetic_logo_parade `
    -Input $video3 `
    -Framerate 60 `
    -DoSDR

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "All encoding complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Run verification
Write-Host "Running verification..." -ForegroundColor Yellow
& .\tools\release\Verify-WowPack.ps1

Write-Host ""
Write-Host "Test your clips at:" -ForegroundColor Cyan
Write-Host "  http://localhost:3000/hologram?clip=holo_flux_loop" -ForegroundColor White
Write-Host "  http://localhost:3000/hologram?clip=mach_lightfield" -ForegroundColor White
Write-Host "  http://localhost:3000/hologram?clip=kinetic_logo_parade" -ForegroundColor White
Write-Host ""
