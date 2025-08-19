# SCAN-FOR-JUNK.ps1
# Finds more files that are obviously safe to archive

Write-Host "=== SCANNING FOR MORE JUNK ===" -ForegroundColor Yellow

$JunkPatterns = @(
    "*.bak",
    "*.bak.bak", 
    "*.backup",
    "*.old",
    "*.orig",
    "*.tmp",
    "*.temp",
    "*.cache",
    "*.log",
    "*.pyc",
    "*_old.py",
    "*_backup.py",
    "*_copy.py",
    "*_test.py",
    "test_*.py",
    "temp_*.py",
    "tmp_*.py",
    "old_*.py",
    "backup_*.py",
    "copy_*.py",
    "draft_*.py",
    "wip_*.py",
    "broken_*.py",
    "deprecated_*.py",
    "unused_*.py",
    "delete_*.py",
    "remove_*.py",
    "cleanup_*.py",
    "clear_*.py",
    "flush_*.py",
    "kill_*.py",
    "force_*.py",
    "nuclear_*.py",
    "emergency_*.py",
    "urgent_*.py",
    "critical_*.py",
    "fix_*.py",
    "patch_*.py",
    "repair_*.py",
    "correct_*.py",
    "debug_*.py",
    "diagnose_*.py",
    "analyze_*.py",
    "inspect_*.py",
    "benchmark_*.py",
    "stress_*.py",
    "load_*.py",
    "migrate_*.py",
    "migration_*.py",
    "upgrade_*.py",
    "update_*.py",
    "install_*.py",
    "setup_*.py",
    "init_*.py",
    "bootstrap_*.py",
    "create_*.py",
    "generate_*.py",
    "make_*.py",
    "build_*.py",
    "compile_*.py",
    "RUN_*.bat",
    "START_*.bat",
    "LAUNCH_*.bat",
    "BUILD_*.bat",
    "TEST_*.bat",
    "FIX_*.bat",
    "INSTALL_*.bat",
    "SETUP_*.bat"
)

$foundFiles = @{}
$totalCount = 0

foreach ($pattern in $JunkPatterns) {
    $files = Get-ChildItem -Path "." -File -Filter $pattern -ErrorAction SilentlyContinue
    if ($files) {
        $foundFiles[$pattern] = $files
        $totalCount += $files.Count
    }
}

Write-Host "`nFOUND $totalCount POTENTIAL JUNK FILES:" -ForegroundColor Cyan

# Group by pattern type
$groups = @{
    "Backup files" = @("*.bak", "*.bak.bak", "*.backup", "*.old", "*.orig", "*_old.py", "*_backup.py", "*_copy.py", "backup_*.py", "old_*.py", "copy_*.py")
    "Temp files" = @("*.tmp", "*.temp", "*.cache", "*.log", "*.pyc", "temp_*.py", "tmp_*.py")
    "Test files" = @("*_test.py", "test_*.py", "benchmark_*.py", "stress_*.py", "load_*.py", "TEST_*.bat")
    "Fix/Debug files" = @("fix_*.py", "patch_*.py", "repair_*.py", "debug_*.py", "diagnose_*.py", "FIX_*.bat")
    "Emergency files" = @("nuclear_*.py", "emergency_*.py", "urgent_*.py", "critical_*.py", "force_*.py")
    "Cleanup files" = @("cleanup_*.py", "clear_*.py", "flush_*.py", "kill_*.py", "remove_*.py", "delete_*.py")
    "Build/Setup files" = @("create_*.py", "generate_*.py", "make_*.py", "build_*.py", "compile_*.py", "BUILD_*.bat", "INSTALL_*.bat", "SETUP_*.bat", "install_*.py", "setup_*.py", "init_*.py", "bootstrap_*.py")
    "Migration files" = @("migrate_*.py", "migration_*.py", "upgrade_*.py", "update_*.py")
    "Launch scripts" = @("RUN_*.bat", "START_*.bat", "LAUNCH_*.bat")
    "WIP/Draft files" = @("draft_*.py", "wip_*.py", "broken_*.py", "deprecated_*.py", "unused_*.py")
}

foreach ($groupName in $groups.Keys | Sort-Object) {
    $groupPatterns = $groups[$groupName]
    $groupFiles = @()
    
    foreach ($pattern in $groupPatterns) {
        if ($foundFiles.ContainsKey($pattern)) {
            $groupFiles += $foundFiles[$pattern]
        }
    }
    
    if ($groupFiles.Count -gt 0) {
        Write-Host "`n$groupName ($($groupFiles.Count)):" -ForegroundColor Yellow
        $groupFiles | Select-Object -First 5 | ForEach-Object {
            Write-Host "  - $($_.Name)" -ForegroundColor Gray
        }
        if ($groupFiles.Count -gt 5) {
            Write-Host "  ... and $($groupFiles.Count - 5) more" -ForegroundColor DarkGray
        }
    }
}

Write-Host "`n=== RECOMMENDATIONS ===" -ForegroundColor Magenta
Write-Host "Add these patterns to HITLIST-ORGANIZE.ps1 for maximum cleanup!" -ForegroundColor Cyan
Write-Host "Total additional files to clean: $totalCount" -ForegroundColor Yellow
