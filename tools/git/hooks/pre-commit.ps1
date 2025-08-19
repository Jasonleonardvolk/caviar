# pre-commit.ps1
# Blocks accidentally staged blobs > 75 MB
# Place in .git/hooks/pre-commit (or run Setup-GitHooks.ps1)

$ErrorActionPreference = "Stop"

$limit = 75MB

Write-Host "Checking for large files..." -ForegroundColor Yellow

$large = git diff --cached --name-only | ForEach-Object {
  $p = $_
  if (Test-Path $p) {
    $s = (Get-Item $p).Length
    if ($s -gt $limit) { 
      [PSCustomObject]@{
        Path = $p
        SizeMB = [math]::Round($s/1MB, 2)
      }
    }
  }
}

if ($large) {
  Write-Host ""
  Write-Host "================================================" -ForegroundColor Red
  Write-Host "   COMMIT BLOCKED: Large files detected" -ForegroundColor Red
  Write-Host "================================================" -ForegroundColor Red
  Write-Host ""
  Write-Host "The following files exceed the 75MB limit:" -ForegroundColor Yellow
  
  $large | ForEach-Object { 
    Write-Host ("  - {0} ({1} MB)" -f $_.Path, $_.SizeMB) -ForegroundColor White
  }
  
  Write-Host ""
  Write-Host "Tips:" -ForegroundColor Cyan
  Write-Host "  1. Put video masters in: content\wowpack\input\" -ForegroundColor Gray
  Write-Host "  2. Build with: tools\encode\Build-WowPack.ps1" -ForegroundColor Gray
  Write-Host "  3. Large outputs are already in .gitignore" -ForegroundColor Gray
  Write-Host ""
  Write-Host "To remove from staging:" -ForegroundColor Yellow
  
  $large | ForEach-Object {
    Write-Host ("  git reset HEAD {0}" -f $_.Path) -ForegroundColor White
  }
  
  Write-Host ""
  exit 1
}

Write-Host "  [OK] No large files detected" -ForegroundColor Green
exit 0
