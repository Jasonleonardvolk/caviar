# AGGRESSIVE-ORGANIZE.ps1
# More aggressive cleanup - gets rid of MORE old stuff

param(
    [switch]$WhatIf,
    [switch]$SkipGitAdd
)

$StartTime = Get-Date
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "        AGGRESSIVE FILE ORGANIZER              " -ForegroundColor Cyan
Write-Host "        Cleaning up MORE old files             " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# MORE aggressive patterns
$AggressiveMoves = @{
    "_archive/2024/benchmarks" = @("benchmark_*.py", "*_test.py", "test_*.py", "*_stress_test.py", "*_benchmark.py")
    "_archive/2024/diagnostic" = @("check_*.py", "diagnose_*.py", "debug_*.py", "diagnostic*.py", "find_*.py", "analyze_*.py")
    "_archive/2024/emergency" = @("emergency_*.py", "emergency_*.ps1", "EMERGENCY_*.ps1", "nuclear_*.py", "NUCLEAR*.ps1", "CRITICAL_*.ps1", "critical_*.py")
    "_archive/2024/fixes" = @("fix_*.py", "fix_*.ps1", "fix_*.cjs", "fix_*.js", "fix_*.bat", "FIX_*.bat", "FIX_*.ps1", "*_fix.py", "*_fixes.py", "*_fixed.py")
    "_archive/2024/install_scripts" = @("install_*.py", "INSTALL_*.bat", "Install*.ps1", "setup_*.py", "SETUP_*.bat", "Setup*.ps1")
    "_archive/2024/patches" = @("apply_*.py", "*_patch.py", "*_patches.py", "patch_*.py", "create_*_patches.py", "*_patch.txt")
    "_archive/2024/run_scripts" = @("run_*.bat", "run_*.py", "RUN_*.bat", "RUN_*.ps1", "START_*.bat", "start_*.py", "LAUNCH_*.bat", "launch_*.py")
    "_archive/2024/build_scripts" = @("build_*.py", "BUILD_*.bat", "Build*.ps1", "compile_*.py", "*_build.py")
    "_archive/2024/create_scripts" = @("create_*.py", "CREATE_*.bat", "generate_*.py", "make_*.py")
    "_archive/2024/migration" = @("migrate_*.py", "migration_*.py", "*_migration.py", "*_migrate.py")
    "_archive/2024/integration" = @("integrate_*.py", "integration_*.py", "*_integration.py", "connect_*.py")
    "_archive/2024/demos" = @("demo_*.py", "*_demo.py", "example_*.py", "*_example.py", "sample_*.py")
    "_archive/2024/monitoring" = @("monitor_*.py", "health_*.py", "*_monitor.py", "*_health.py", "status_*.py")
    "_archive/2024/cleanup" = @("cleanup_*.py", "clean_*.py", "*_cleanup.py", "clear_*.py", "flush_*.py", "remove_*.py")
    "_archive/2024/validation" = @("validate_*.py", "verify_*.py", "*_validation.py", "*_verify.py", "check_*.bat")
    "_archive/2024/final_scripts" = @("final_*.py", "FINAL_*.bat", "*_final.py", "*_complete.py", "complete_*.py")
    "_archive/docs/old_docs" = @("*_COMPLETE.md", "*_SUMMARY.md", "*_STATUS.md", "*_PLAN.md", "*_GUIDE.md", "*_IMPLEMENTATION*.md", "*_ARCHITECTURE*.md", "*_DOCUMENTATION*.md")
    "_archive/docs/reports" = @("*_CHECKLIST.md", "*_report*.md", "*_REPORT*.md", "health_report*.md", "*_ANALYSIS.txt", "*_analysis.md")
    "_archive/configs/old" = @("*.json.backup*", "*.json.bak*", "*.yml.bak", "*.yaml.bak", "*.toml.bak", "*_old.json", "*_backup.json")
    "_cleanup/duplicates" = @("*.bak", "*.bak.bak", "*.backup", "*.original", "*.old", "*_old.py", "*_backup.py", "*.backup_*")
    "_cleanup/temp_files" = @("*.tmp", "*.temp", "*.cache", "*.log", "*.pyc", "*.pyo", "*.swp", "*.swo", "*~", ".DS_Store", "Thumbs.db")
    "_archive/2024/old_mains" = @("main_*.py", "*_main.py", "app_*.py", "*_app.py", "server_*.py", "*_server.py", "api_*.py", "*_api.py")
}

# Files to NEVER move (critical)
$NeverMove = @(
    "main.py",
    "enhanced_launcher.py",
    "p.ps1",
    "README.md",
    "LICENSE",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pyproject.toml",
    "requirements.txt",
    "Makefile",
    "docker-compose.yml",
    ".gitignore",
    ".gitattributes",
    "TARGETED-ORGANIZE.ps1",
    "AGGRESSIVE-ORGANIZE.ps1",
    "UNDO-AGGRESSIVE.ps1"
)

$FilesToMove = @{}
$FileCount = 0
$SkippedCount = 0

Write-Host "`nCreating archive directories..." -ForegroundColor Yellow

# Create all directories
foreach ($folder in $AggressiveMoves.Keys) {
    $targetPath = Join-Path $PSScriptRoot $folder
    if (-not (Test-Path $targetPath)) {
        if (-not $WhatIf) {
            New-Item -ItemType Directory -Force -Path $targetPath | Out-Null
            Write-Host "  Created: $folder" -ForegroundColor Green
        } else {
            Write-Host "  [WhatIf] Would create: $folder" -ForegroundColor Gray
        }
    }
}

Write-Host "`nScanning files aggressively..." -ForegroundColor Yellow

# Process files
Get-ChildItem -Path $PSScriptRoot -File | ForEach-Object {
    $file = $_
    $fileName = $file.Name
    $moved = $false
    
    # Check if in never move list
    if ($fileName -in $NeverMove) {
        Write-Host "  CRITICAL: $fileName (never move)" -ForegroundColor Cyan
        $SkippedCount++
        return
    }
    
    # Check all aggressive patterns
    foreach ($folder in $AggressiveMoves.Keys) {
        $patterns = $AggressiveMoves[$folder]
        foreach ($pattern in $patterns) {
            if ($fileName -like $pattern) {
                if (-not $FilesToMove.ContainsKey($folder)) {
                    $FilesToMove[$folder] = @()
                }
                $FilesToMove[$folder] += $file
                $moved = $true
                $FileCount++
                break
            }
        }
        if ($moved) { break }
    }
    
    if (-not $moved) {
        # Extra aggressive: Move ANY Python file with certain keywords if older than 14 days
        if ($fileName -match '\.(py|ps1|bat|sh)$' -and $file.LastWriteTime -lt (Get-Date).AddDays(-14)) {
            $keywords = @('test', 'debug', 'fix', 'patch', 'temp', 'old', 'backup', 'copy', 'draft', 'wip', 'todo', 'broken', 'deprecated')
            foreach ($keyword in $keywords) {
                if ($fileName -match $keyword) {
                    if (-not $FilesToMove.ContainsKey("_archive/2024/old_scripts")) {
                        New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "_archive/2024/old_scripts") -ErrorAction SilentlyContinue | Out-Null
                        $FilesToMove["_archive/2024/old_scripts"] = @()
                    }
                    $FilesToMove["_archive/2024/old_scripts"] += $file
                    $FileCount++
                    $moved = $true
                    break
                }
            }
        }
    }
    
    if (-not $moved) {
        $SkippedCount++
    }
}

Write-Host "`n=== AGGRESSIVE CLEANUP PLAN ===" -ForegroundColor Magenta
Write-Host "Files to archive: $FileCount" -ForegroundColor Yellow
Write-Host "Files to keep: $SkippedCount" -ForegroundColor Green

if ($FileCount -eq 0) {
    Write-Host "`nNo files to move!" -ForegroundColor Yellow
    exit 0
}

# Show what will be moved - DETAILED VIEW
foreach ($folder in $FilesToMove.Keys | Sort-Object) {
    $files = $FilesToMove[$folder]
    Write-Host "`n-> $folder ($($files.Count) files)" -ForegroundColor Cyan
    
    # Show first 3 files explicitly
    $showCount = [Math]::Min(3, $files.Count)
    for ($i = 0; $i -lt $showCount; $i++) {
        Write-Host "    $($files[$i].Name)" -ForegroundColor Gray
    }
    
    # Show remaining count if more than 3
    if ($files.Count -gt 3) {
        Write-Host "    ... and $($files.Count - 3) more" -ForegroundColor DarkGray
    }
}

if ($WhatIf) {
    Write-Host "`n=== WHAT-IF MODE ===" -ForegroundColor Yellow
    Write-Host "No files moved. Remove -WhatIf to execute." -ForegroundColor Yellow
    Write-Host "Command: .\AGGRESSIVE-ORGANIZE.ps1" -ForegroundColor White
    exit 0
}

# Confirm
Write-Host "`n=== READY FOR AGGRESSIVE CLEANUP ===" -ForegroundColor Red
Write-Host "This will archive $FileCount files!" -ForegroundColor Yellow
$confirm = Read-Host "Continue? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

# Create undo script
$undoScript = @"
# UNDO-AGGRESSIVE.ps1 - Generated $(Get-Date)
Write-Host 'Restoring $FileCount files...' -ForegroundColor Yellow
"@

# Move files
$movedCount = 0
foreach ($folder in $FilesToMove.Keys | Sort-Object) {
    $targetPath = Join-Path $PSScriptRoot $folder
    $files = $FilesToMove[$folder]
    
    Write-Host "`nArchiving to $folder..." -ForegroundColor Cyan
    
    foreach ($file in $files) {
        $destination = Join-Path $targetPath $file.Name
        try {
            Move-Item -Path $file.FullName -Destination $destination -Force
            $movedCount++
            
            if ($movedCount % 50 -eq 0) {
                Write-Host "  ... $movedCount files moved" -ForegroundColor Gray
            }
            
            $undoScript += "Move-Item '$destination' '$($file.FullName)' -Force`n"
            
            if (-not $SkipGitAdd) {
                git add $file.FullName 2>$null
                git add $destination 2>$null
            }
        } catch {
            Write-Host "  Failed: $($file.Name)" -ForegroundColor Red
        }
    }
}

$undoScript += "Write-Host 'Restored $movedCount files!' -ForegroundColor Green"
$undoScript | Out-File -FilePath ".\UNDO-AGGRESSIVE.ps1" -Encoding UTF8

$Duration = (Get-Date) - $StartTime

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "         AGGRESSIVE CLEANUP COMPLETE!          " -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Files archived:    $movedCount" -ForegroundColor White
Write-Host "  Files kept:        $SkippedCount" -ForegroundColor White
Write-Host "  Time:              $([math]::Round($Duration.TotalSeconds, 2)) seconds" -ForegroundColor White
Write-Host ""
Write-Host "  Undo available:    .\UNDO-AGGRESSIVE.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Commit with:       .\p.ps1 'massive cleanup: archived $movedCount old files'" -ForegroundColor Yellow
