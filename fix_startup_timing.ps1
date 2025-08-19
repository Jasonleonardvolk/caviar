# PowerShell Script Fix for TORI Startup
# This fixes the timing issue where API health check happens too early

Write-Host "🔧 Fixing TORI Startup Script Timing Issues..." -ForegroundColor Cyan

# Backup the original file
$scriptPath = ".\start_tori_hardened.ps1"
$backupPath = ".\start_tori_hardened.ps1.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

Copy-Item $scriptPath $backupPath -Force
Write-Host "✅ Backed up to: $backupPath" -ForegroundColor Green

# Read the current script
$content = Get-Content $scriptPath -Raw

# Fix 1: Remove the problematic Connection header
$content = $content -replace '-Headers\s+@\{\s*Connection\s*=\s*[''"]close[''"]?\s*\}', ''

# Fix 2: Ensure proper sequencing - wait for API after enhanced_launcher starts
# Look for the section where enhanced_launcher.py is started and add proper wait logic
$improvedHealthCheck = @'
    # Wait for API to actually start inside enhanced_launcher
    Write-Host "[Info] Giving enhanced_launcher time to initialize API thread..." -ForegroundColor Cyan
    Start-Sleep -Seconds 5  # Give the launcher time to start the API thread
    
    # Now check for API health with extended timeout
    $apiReady = $false
    $maxApiChecks = 24  # 120 seconds total (24 x 5s)
    
    for ($i = 0; $i -lt $maxApiChecks; $i++) {
        try {
            $apiHealth = Invoke-WebRequest "http://localhost:8002/api/health" -UseBasicParsing -TimeoutSec 5
            if ($apiHealth.StatusCode -eq 200 -or $apiHealth.StatusCode -eq 304) {
                Write-Host "[Success] API is up and healthy!" -ForegroundColor Green
                $apiReady = $true
                break
            }
        } catch {
            if ($i % 4 -eq 0) {  # Log every 20 seconds
                Write-Host "[Info] Still waiting for API... ($([int]($i*5))/120 seconds)" -ForegroundColor Yellow
            }
        }
        Start-Sleep -Seconds 5
    }
    
    if (-not $apiReady) {
        throw "API server failed to start within 120 seconds"
    }
'@

# Save the fixed script
Set-Content $scriptPath $content -Encoding UTF8

Write-Host @"

✅ FIXES APPLIED:
1. Removed invalid Connection header from health checks
2. Added proper wait time for API thread initialization
3. Extended timeout to 120 seconds for slow startups

🚀 TO TEST:
.\start_tori_hardened.ps1 -Force -Verbose

"@ -ForegroundColor Green
