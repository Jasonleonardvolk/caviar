# Fix PowerShell Profile Script
Write-Host "Fixing PowerShell profile..." -ForegroundColor Cyan

# Create the corrected function
$correctFunction = @'
function t {
    Set-Location 'C:\Users\jason\Desktop\tori\kha'
    
    # Check if Poetry environment exists
    $poetryCheck = poetry env info --path 2>$null
    if ($poetryCheck) {
        # Activate the Poetry environment
        $activatePath = "$poetryCheck\Scripts\Activate.ps1"
        if (Test-Path $activatePath) {
            & $activatePath
            Write-Host "✓ Activated Poetry environment" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Could not activate environment, use 'poetry run' prefix for commands" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠️ Virtual environment not found. Run 'poetry install' first." -ForegroundColor Yellow
    }
    
    Write-Host "📍 Current directory: C:\Users\jason\Desktop\tori\kha" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Ready to run:" -ForegroundColor Green
    Write-Host "  poetry run python enhanced_launcher.py --api full --require-penrose --enable-hologram --hologram-audio" -ForegroundColor White
}
'@

# Backup current profile
$profilePath = $PROFILE
$backupPath = "$profilePath.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
if (Test-Path $profilePath) {
    Copy-Item $profilePath $backupPath
    Write-Host "Backed up profile to: $backupPath" -ForegroundColor Yellow
}

# Read current profile and remove broken function
$content = Get-Content $profilePath -Raw
$content = $content -replace '(?ms)function t \{[^}]*\}[^}]*\}', ''  # Remove malformed function

# Write corrected profile
$content + "`n$correctFunction`n" | Set-Content $profilePath

Write-Host "`n✅ Profile fixed!" -ForegroundColor Green
Write-Host "Reloading profile..." -ForegroundColor Yellow

# Reload profile
. $PROFILE

Write-Host "✅ Done! Try typing 't' now!" -ForegroundColor Green
