$src = "D:\Dev\kha\frontend\public"
$dst = "D:\Dev\kha\frontend\static"
$ErrorActionPreference = "Stop"

if (!(Test-Path $src)) { 
    Write-Host "Source directory not found: $src" -ForegroundColor Yellow
    exit 0 
}

# Create destination if it doesn't exist
if (!(Test-Path $dst)) {
    New-Item -ItemType Directory -Path $dst -Force | Out-Null
}

# Use robocopy to sync directories
$result = robocopy $src $dst /E /NFL /NDL /NJH /NJS /NP /XO 2>&1

# Check robocopy exit code (0-7 are success codes)
if ($LASTEXITCODE -le 7) {
    Write-Host "Public to static sync complete: $src to $dst" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Sync failed with code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}