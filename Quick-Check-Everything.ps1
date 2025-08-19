# Quick-Check-Everything.ps1
# Ensures all critical components are ready for iRis/TORI/Caviar

Write-Host "`n=== CAVIAR SYSTEM CHECK ===" -ForegroundColor Cyan
Write-Host "Checking all critical components..." -ForegroundColor Yellow

$checks = @{
    "README.md" = "Documentation"
    ".gitattributes" = "Line ending config"
    ".gitignore" = "Git exclusions"
    "tools\git\Commit-EOD.ps1" = "EOD commit tool"
    "tools\release\Verify-Iris.ps1" = "iRis verification"
    "tools\encode\Build-WowPack.ps1" = "Video encoding"
    "tori_ui_svelte\package.json" = "Frontend config"
}

$passed = 0
$failed = 0

foreach ($file in $checks.Keys) {
    if (Test-Path $file) {
        Write-Host "[OK] " -NoNewline -ForegroundColor Green
        Write-Host "$($checks[$file]): $file"
        $passed++
    } else {
        Write-Host "[MISSING] " -NoNewline -ForegroundColor Red
        Write-Host "$($checks[$file]): $file"
        $failed++
    }
}

Write-Host "`n=== GIT STATUS ===" -ForegroundColor Cyan
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "Uncommitted changes detected:" -ForegroundColor Yellow
    Write-Host $gitStatus
    Write-Host "`nRun: .\tools\git\Commit-EOD.ps1" -ForegroundColor Cyan
} else {
    Write-Host "Working directory clean!" -ForegroundColor Green
}

Write-Host "`n=== REPOSITORY INFO ===" -ForegroundColor Cyan
git remote -v | Select-String "origin"
Write-Host "Latest commit:" -ForegroundColor Yellow
git log --oneline -n 1

Write-Host "`n=== SUMMARY ===" -ForegroundColor Cyan
Write-Host "Checks passed: $passed/$($checks.Count)" -ForegroundColor $(if ($failed -eq 0) {'Green'} else {'Yellow'})

if ($failed -eq 0) {
    Write-Host "`nSystem ready! Next steps:" -ForegroundColor Green
    Write-Host "1. Test iRis: .\tools\release\Verify-Iris.ps1 -Environment dev" -ForegroundColor Cyan
    Write-Host "2. Commit changes: .\tools\git\Commit-EOD.ps1" -ForegroundColor Cyan
    Write-Host "3. Check GitHub: https://github.com/Jasonleonardvolk/caviar" -ForegroundColor Cyan
} else {
    Write-Host "`nSome components missing. Review above." -ForegroundColor Yellow
}

Write-Host "`n=== CHECK COMPLETE ===" -ForegroundColor Cyan