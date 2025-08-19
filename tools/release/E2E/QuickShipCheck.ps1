# Quick validation for shipping readiness - minimal output version
param(
  [string]$RepoRoot = "D:\Dev\kha",
  [switch]$Detailed
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$checks = @()
$pass = 0
$fail = 0

function Check {
  param([string]$Name, [scriptblock]$Test, [bool]$Critical = $true)
  
  $result = & $Test
  $checks += @{Name=$Name; Pass=$result; Critical=$Critical}
  
  if ($result) {
    if ($Detailed) { Write-Host "[OK]" -ForegroundColor Green -NoNewline; Write-Host " $Name" }
    $script:pass++
  } else {
    Write-Host "[FAIL]" -ForegroundColor Red -NoNewline
    Write-Host " $Name" -ForegroundColor Yellow
    $script:fail++
  }
}

Write-Host "`nQUICK SHIP CHECK" -ForegroundColor Cyan
Write-Host "================" -ForegroundColor Cyan

# Critical checks only
Check "Node.js" { $null -ne (node -v 2>$null) }
Check "TypeScript" { (npx tsc --noEmit 2>&1 | Select-String "error").Count -eq 0 }
Check "Package.json" { Test-Path "package.json" }
Check "Node modules" { Test-Path "node_modules" }
Check "Svelte files" { (Get-ChildItem -Path "tori_ui_svelte\src" -Filter "*.svelte" -Recurse).Count -gt 0 }
Check "QuiltGenerator" { Test-Path "tools\quilt\WebGPU\QuiltGenerator.ts" } $false
Check "Build script" { (Get-Content "package.json" | ConvertFrom-Json).scripts.build -ne $null }

# Try quick build test
Write-Host "`nBuild Test..." -ForegroundColor Yellow -NoNewline
$buildTest = $false
try {
  $output = npm run build -- --dry-run 2>&1
  if ($LASTEXITCODE -eq 0) { 
    $buildTest = $true 
    Write-Host " OK" -ForegroundColor Green
  } else {
    Write-Host " FAIL" -ForegroundColor Red
  }
} catch {
  Write-Host " FAIL" -ForegroundColor Red
}

Write-Host "`n" + ("=" * 40) -ForegroundColor Cyan
if ($fail -eq 0 -and $buildTest) {
  Write-Host "READY TO SHIP: YES" -ForegroundColor Green -BackgroundColor DarkGreen
  Write-Host "All critical checks passed!" -ForegroundColor Green
  exit 0
} else {
  Write-Host "READY TO SHIP: NO" -ForegroundColor Red -BackgroundColor DarkRed
  Write-Host "$fail critical issues found" -ForegroundColor Red
  exit 1
}
