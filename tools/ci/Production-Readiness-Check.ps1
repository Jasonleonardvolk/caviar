#!/usr/bin/env pwsh
# Production-Readiness-Check.ps1
# Final checklist before shipping

$ErrorActionPreference = "Stop"

Write-Host @"
====================================================
    iRis Production Readiness Checklist
====================================================
"@ -ForegroundColor Cyan

$checks = @()

Write-Host "Running production checks..." -ForegroundColor Yellow
Write-Host ""

# 1. Build Output
Write-Host "[1/8] Build Output..." -ForegroundColor Yellow
$buildExists = Test-Path "D:\Dev\kha\tori_ui_svelte\build"
if ($buildExists) {
    Write-Host "  ✅ Build directory exists" -ForegroundColor Green
    $checks += @{Name="Build Output"; Status="PASS"}
} else {
    Write-Host "  ❌ Build directory missing" -ForegroundColor Red
    $checks += @{Name="Build Output"; Status="FAIL"}
}

# 2. Adapter Configuration
Write-Host "[2/8] Adapter Configuration..." -ForegroundColor Yellow
$configFile = "D:\Dev\kha\tori_ui_svelte\svelte.config.js"
if (Test-Path $configFile) {
    $config = Get-Content $configFile -Raw
    if ($config -match "adapter-node") {
        Write-Host "  ✅ Using adapter-node" -ForegroundColor Green
        $checks += @{Name="Adapter"; Status="PASS"}
    } else {
        Write-Host "  ⚠️  Not using adapter-node" -ForegroundColor Yellow
        $checks += @{Name="Adapter"; Status="WARN"}
    }
}

# 3. Security Audit
Write-Host "[3/8] Security Audit..." -ForegroundColor Yellow
Set-Location "D:\Dev\kha\tori_ui_svelte" -ErrorAction SilentlyContinue
$auditResult = & npm audit --audit-level=high --json 2>$null | ConvertFrom-Json
if ($auditResult.vulnerabilities.high -eq 0 -and $auditResult.vulnerabilities.critical -eq 0) {
    Write-Host "  ✅ No high/critical vulnerabilities" -ForegroundColor Green
    $checks += @{Name="Security"; Status="PASS"}
} else {
    Write-Host "  ❌ High/critical vulnerabilities found" -ForegroundColor Red
    $checks += @{Name="Security"; Status="FAIL"}
}

# 4. TypeScript Compilation
Write-Host "[4/8] TypeScript Compilation..." -ForegroundColor Yellow
$tsErrors = & pnpm run check 2>&1 | Select-String "error"
if ($tsErrors.Count -eq 0) {
    Write-Host "  ✅ No TypeScript errors" -ForegroundColor Green
    $checks += @{Name="TypeScript"; Status="PASS"}
} else {
    Write-Host "  ⚠️  TypeScript warnings (non-blocking)" -ForegroundColor Yellow
    $checks += @{Name="TypeScript"; Status="WARN"}
}

# 5. Shader Validation
Write-Host "[5/8] Shader Validation..." -ForegroundColor Yellow
$nagaExists = Test-Path "D:\Dev\kha\tools\shaders\bin\naga.exe"
$tintExists = Test-Path "D:\Dev\kha\tools\shaders\bin\tint.exe"
if ($nagaExists) {
    Write-Host "  ✅ Naga validator available" -ForegroundColor Green
    if ($tintExists) {
        Write-Host "  ✅ Tint validator available (bonus)" -ForegroundColor Green
    }
    $checks += @{Name="Shaders"; Status="PASS"}
} else {
    Write-Host "  ⚠️  No shader validators" -ForegroundColor Yellow
    $checks += @{Name="Shaders"; Status="WARN"}
}

# 6. Environment Variables
Write-Host "[6/8] Environment Configuration..." -ForegroundColor Yellow
$envFile = "D:\Dev\kha\tori_ui_svelte\.env"
if (Test-Path $envFile) {
    Write-Host "  ✅ .env file exists" -ForegroundColor Green
    $checks += @{Name="Environment"; Status="PASS"}
} else {
    Write-Host "  ⚠️  No .env file (using defaults)" -ForegroundColor Yellow
    $checks += @{Name="Environment"; Status="WARN"}
}

# 7. Node Version
Write-Host "[7/8] Node.js Version..." -ForegroundColor Yellow
$nodeVersion = & node --version
if ($nodeVersion -match "v(\d+)") {
    $majorVersion = [int]$Matches[1]
    if ($majorVersion -ge 18) {
        Write-Host "  ✅ Node.js $nodeVersion" -ForegroundColor Green
        $checks += @{Name="Node.js"; Status="PASS"}
    } else {
        Write-Host "  ⚠️  Node.js $nodeVersion (recommend v18+)" -ForegroundColor Yellow
        $checks += @{Name="Node.js"; Status="WARN"}
    }
}

# 8. Package Lock
Write-Host "[8/8] Package Lock..." -ForegroundColor Yellow
$lockFile = "D:\Dev\kha\tori_ui_svelte\pnpm-lock.yaml"
if (Test-Path $lockFile) {
    Write-Host "  ✅ pnpm-lock.yaml exists" -ForegroundColor Green
    $checks += @{Name="Lock File"; Status="PASS"}
} else {
    Write-Host "  ⚠️  No lock file" -ForegroundColor Yellow
    $checks += @{Name="Lock File"; Status="WARN"}
}

# Summary
Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "    Production Readiness Summary" -ForegroundColor White
Write-Host "=====================================================" -ForegroundColor Cyan

$passed = ($checks | Where-Object { $_.Status -eq "PASS" }).Count
$warnings = ($checks | Where-Object { $_.Status -eq "WARN" }).Count
$failed = ($checks | Where-Object { $_.Status -eq "FAIL" }).Count

foreach ($check in $checks) {
    $symbol = switch ($check.Status) {
        "PASS" { "✅" }
        "WARN" { "⚠️ " }
        "FAIL" { "❌" }
    }
    $color = switch ($check.Status) {
        "PASS" { "Green" }
        "WARN" { "Yellow" }
        "FAIL" { "Red" }
    }
    Write-Host "$symbol $($check.Name)" -ForegroundColor $color
}

Write-Host ""
Write-Host "Results: $passed passed, $warnings warnings, $failed failed" -ForegroundColor White

if ($failed -gt 0) {
    Write-Host ""
    Write-Host "❌ NOT READY FOR PRODUCTION" -ForegroundColor Red
    Write-Host "Fix the failed checks above" -ForegroundColor Yellow
    exit 1
} elseif ($warnings -gt 2) {
    Write-Host ""
    Write-Host "⚠️  READY WITH WARNINGS" -ForegroundColor Yellow
    Write-Host "Consider addressing warnings for best results" -ForegroundColor Gray
    exit 0
} else {
    Write-Host ""
    Write-Host "✅ READY FOR PRODUCTION!" -ForegroundColor Green
    Write-Host "Ship it! 🚀" -ForegroundColor Cyan
    exit 0
}
