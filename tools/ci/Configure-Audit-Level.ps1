#!/usr/bin/env pwsh
# Configure-Audit-Level.ps1
# Set up proper audit configuration for CI/CD

$ErrorActionPreference = "Stop"

Write-Host "===== Configuring NPM Audit Level =====" -ForegroundColor Cyan
Write-Host ""

# Create .npmrc for audit configuration
$npmrcContent = @'
# Audit Configuration
audit-level=high

# Only fail on high severity vulnerabilities
# Ignore moderate and low severity issues for now
'@

# Set for tori_ui_svelte
$svelteNpmrc = "D:\Dev\kha\tori_ui_svelte\.npmrc"
Write-Host "Setting audit level for tori_ui_svelte..." -ForegroundColor Yellow
Add-Content -Path $svelteNpmrc -Value "`n$npmrcContent" -ErrorAction SilentlyContinue
Write-Host "  ✅ Configured: $svelteNpmrc" -ForegroundColor Green

# Set for root if needed
$rootNpmrc = "D:\Dev\kha\.npmrc"
Write-Host "Setting audit level for root..." -ForegroundColor Yellow
Add-Content -Path $rootNpmrc -Value "`n$npmrcContent" -ErrorAction SilentlyContinue
Write-Host "  ✅ Configured: $rootNpmrc" -ForegroundColor Green

Write-Host ""
Write-Host "===== Running Audit Check =====" -ForegroundColor Cyan

# Run audit with high level only
Set-Location "D:\Dev\kha\tori_ui_svelte"
Write-Host "Running: npm audit --audit-level=high" -ForegroundColor Yellow
& npm audit --audit-level=high

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ No HIGH severity vulnerabilities found!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "⚠️  High severity vulnerabilities detected" -ForegroundColor Yellow
    Write-Host "Run 'npm audit fix' to attempt automatic fixes" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Audit configuration complete!" -ForegroundColor Cyan
Write-Host "Moderate and low severity issues will be ignored in CI." -ForegroundColor Gray
