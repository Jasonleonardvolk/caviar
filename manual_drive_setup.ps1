# QUICK FIX: Manual Google Drive Path Setup
# Use this if automatic detection fails

Write-Host "`n===== GOOGLE DRIVE PATH SETUP =====" -ForegroundColor Cyan
Write-Host "Let's find your Google Drive folder manually." -ForegroundColor Yellow

Write-Host "`nIn File Explorer, navigate to your Google Drive folder" -ForegroundColor White
Write-Host "It might be at one of these locations:" -ForegroundColor Gray
Write-Host "  - G:\My Drive" -ForegroundColor Gray
Write-Host "  - C:\Users\$env:USERNAME\My Drive" -ForegroundColor Gray
Write-Host "  - C:\Users\$env:USERNAME\Google Drive" -ForegroundColor Gray

Write-Host "`nPaste the FULL PATH to your Google Drive folder:" -ForegroundColor Yellow
Write-Host "(Example: G:\My Drive)" -ForegroundColor Gray
$drivePath = Read-Host "Drive Path"

# Clean up the path
$drivePath = $drivePath.Trim('"').Trim()

if (Test-Path $drivePath) {
    Write-Host "✓ Drive path exists!" -ForegroundColor Green
    
    # Look for kha folder
    $possibleKhaPaths = @(
        "$drivePath\My Laptop\kha",
        "$drivePath\kha",
        "$drivePath\Computers\My Laptop\kha"
    )
    
    $foundKha = $null
    foreach ($testPath in $possibleKhaPaths) {
        if (Test-Path $testPath) {
            $foundKha = $testPath
            Write-Host "✓ Found kha folder at: $foundKha" -ForegroundColor Green
            break
        }
    }
    
    if (-not $foundKha) {
        Write-Host "`nCouldn't find kha folder automatically." -ForegroundColor Yellow
        Write-Host "Please enter the FULL PATH to your kha folder in Google Drive:" -ForegroundColor Yellow
        $foundKha = Read-Host "Kha Path"
        $foundKha = $foundKha.Trim('"').Trim()
    }
    
    if (Test-Path $foundKha) {
        Write-Host "✓ Kha path verified!" -ForegroundColor Green
        
        # Save configuration
        $config = @{
            GoogleDrivePath = $drivePath
            KhaInDrive = $foundKha
            MyLaptopPath = if ($foundKha -match "My Laptop") { Split-Path $foundKha -Parent } else { $drivePath }
            DetectedOn = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        }
        
        $config | ConvertTo-Json | Out-File -FilePath ".\drive_config.json"
        Write-Host "`n✓ Configuration saved!" -ForegroundColor Green
        
        Write-Host "`n===== NEXT STEPS =====" -ForegroundColor Cyan
        Write-Host "1. Run: .\update_sync_scripts.ps1" -ForegroundColor White
        Write-Host "   This will update all sync scripts with your path" -ForegroundColor Gray
        Write-Host "`n2. Then run: .\quick_8day_status.ps1" -ForegroundColor White
        Write-Host "   To see what needs syncing" -ForegroundColor Gray
        
    } else {
        Write-Host "ERROR: Kha path doesn't exist at: $foundKha" -ForegroundColor Red
    }
} else {
    Write-Host "ERROR: Drive path doesn't exist at: $drivePath" -ForegroundColor Red
    Write-Host "Please check the path and try again." -ForegroundColor Yellow
}

Write-Host "`nPress any key to exit..."
$null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")