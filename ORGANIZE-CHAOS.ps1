# ORGANIZE-CHAOS.ps1
# INTELLIGENT REPO REORGANIZATION - Makes 1,662+ files manageable!
# Run with -WhatIf first to preview changes

param(
    [switch]$WhatIf,
    [switch]$SkipGitAdd
)

$StartTime = Get-Date
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "        REPO CHAOS ORGANIZER                   " -ForegroundColor Cyan
Write-Host "        1,662+ files -> Organized Structure    " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Create organized structure
$Folders = @{
    "_archive/2024/fixes" = "fix_*.py", "fix_*.ps1", "fix_*.cjs", "fix_*.js", "fix_*.bat"
    "_archive/2024/emergency" = "emergency_*.py", "EMERGENCY_*.ps1", "nuclear_*.py", "NUCLEAR*.ps1"
    "_archive/2024/diagnostic" = "diagnose_*.py", "debug_*.py", "diagnostic*.py", "check_*.py"
    "_archive/2024/patches" = "*_patch.py", "*_patches.py", "patch_*.py", "apply_*.py"
    "_archive/2024/benchmarks" = "benchmark_*.py", "*_test.py", "test_*.py", "*_stress_test.py"
    "_archive/2024/create_scripts" = "create_*.py", "generate_*.py", "build_*.py"
    "_archive/2024/install_scripts" = "install_*.py", "INSTALL_*.bat", "Install*.ps1"
    "_archive/2024/start_scripts" = "START_*.bat", "start_*.py", "LAUNCH_*.bat", "launch_*.py"
    "_archive/2024/run_scripts" = "RUN_*.bat", "RUN_*.ps1", "run_*.py"
    "_archive/docs/old_readmes" = "*_COMPLETE.md", "*_SUMMARY.md", "*_STATUS.md", "*_PLAN.md", "*_GUIDE.md"
    "_archive/docs/reports" = "*_REPORT*.md", "*_ANALYSIS.txt", "*_CHECKLIST.md"
    "_cleanup/duplicates" = "*.bak", "*.bak.bak", "*.backup", "*.original"
    "tools/migration" = "migrate_*.py", "migration_*.py", "*_migration.py"
    "tools/integration" = "integrate_*.py", "integration_*.py", "*_integration.py"
    "tools/monitoring" = "monitor_*.py", "health*.py", "*_monitor.py"
    "tools/demo" = "demo_*.py", "*_demo.py"
}

# Files to NEVER move (critical for operation)
$KeepInRoot = @(
    "README.md",
    "LICENSE",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pyproject.toml",
    "requirements.txt",
    "Makefile",
    "Dockerfile*",
    "docker-compose.yml",
    ".gitignore",
    ".gitattributes",
    ".env*",
    "*.toml",
    "*.yml",
    "*.yaml",
    "main.py",
    "enhanced_launcher.py",
    "p.ps1",
    "Quick-Check-Everything.ps1",
    "TODAY_CHANGES.md",
    "AI_QUICK_REFERENCE.md",
    "COLLABORATION_GUIDE.md",
    ".ai-context"
)

# Recent files to keep accessible (last 7 days)
$RecentCutoff = (Get-Date).AddDays(-7)

$FilesToMove = @{}
$FileCount = 0
$SkippedCount = 0

Write-Host "`nAnalyzing files..." -ForegroundColor Yellow

# First, create all target directories
foreach ($folder in $Folders.Keys) {
    $targetPath = Join-Path $PSScriptRoot $folder
    if (-not (Test-Path $targetPath)) {
        if (-not $WhatIf) {
            New-Item -ItemType Directory -Force -Path $targetPath | Out-Null
            Write-Host "Created: $folder" -ForegroundColor Green
        } else {
            Write-Host "[WhatIf] Would create: $folder" -ForegroundColor Gray
        }
    }
}

# Analyze each file in root
Get-ChildItem -Path $PSScriptRoot -File | ForEach-Object {
    $file = $_
    $fileName = $file.Name
    $moved = $false
    
    # Skip if in keep list
    foreach ($pattern in $KeepInRoot) {
        if ($fileName -like $pattern) {
            Write-Host "  KEEP: $fileName (critical file)" -ForegroundColor Cyan
            $SkippedCount++
            return
        }
    }
    
    # Skip if modified recently (unless it's a .bak file)
    if ($file.LastWriteTime -gt $RecentCutoff -and $fileName -notlike "*.bak*") {
        Write-Host "  KEEP: $fileName (modified recently)" -ForegroundColor Yellow
        $SkippedCount++
        return
    }
    
    # Check each folder pattern
    foreach ($folder in $Folders.Keys) {
        $patterns = $Folders[$folder]
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
    
    # If not categorized, check if it's an old file (30+ days)
    if (-not $moved) {
        if ($file.LastWriteTime -lt (Get-Date).AddDays(-30)) {
            $archiveFolder = "_archive/2024/uncategorized"
            if (-not (Test-Path (Join-Path $PSScriptRoot $archiveFolder))) {
                if (-not $WhatIf) {
                    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot $archiveFolder) | Out-Null
                }
            }
            if (-not $FilesToMove.ContainsKey($archiveFolder)) {
                $FilesToMove[$archiveFolder] = @()
            }
            $FilesToMove[$archiveFolder] += $file
            $FileCount++
        } else {
            Write-Host "  KEEP: $fileName (no category, but recent)" -ForegroundColor Gray
            $SkippedCount++
        }
    }
}

Write-Host "`n=== ORGANIZATION PLAN ===" -ForegroundColor Magenta
Write-Host "Files to move: $FileCount" -ForegroundColor Yellow
Write-Host "Files to keep: $SkippedCount" -ForegroundColor Green

if ($FileCount -eq 0) {
    Write-Host "`nNothing to organize! Repo is already clean." -ForegroundColor Green
    exit 0
}

# Show the plan
foreach ($folder in $FilesToMove.Keys | Sort-Object) {
    $files = $FilesToMove[$folder]
    Write-Host "`n-> $folder ($($files.Count) files)" -ForegroundColor Cyan
    if ($files.Count -le 5) {
        $files | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
    } else {
        $files | Select-Object -First 3 | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
        Write-Host "    ... and $($files.Count - 3) more" -ForegroundColor DarkGray
    }
}

if ($WhatIf) {
    Write-Host "`n=== WHAT-IF MODE ===" -ForegroundColor Yellow
    Write-Host "No files were moved. Remove -WhatIf to execute." -ForegroundColor Yellow
    Write-Host "Command: .\ORGANIZE-CHAOS.ps1" -ForegroundColor White
    exit 0
}

# Confirm before moving
Write-Host "`n=== READY TO ORGANIZE ===" -ForegroundColor Red
$confirm = Read-Host "Move $FileCount files? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

# Create undo script
$undoScript = @"
# UNDO-ORGANIZE.ps1 - Generated $(Get-Date)
# Restores files to original locations

Write-Host 'Restoring files to original locations...' -ForegroundColor Yellow

"@

# Move the files
$movedCount = 0
foreach ($folder in $FilesToMove.Keys | Sort-Object) {
    $targetPath = Join-Path $PSScriptRoot $folder
    $files = $FilesToMove[$folder]
    
    Write-Host "`nMoving to $folder..." -ForegroundColor Cyan
    
    foreach ($file in $files) {
        $destination = Join-Path $targetPath $file.Name
        try {
            Move-Item -Path $file.FullName -Destination $destination -Force
            Write-Host "  Moved: $($file.Name)" -ForegroundColor Green
            $movedCount++
            
            # Add to undo script
            $undoScript += "Move-Item '$destination' '$($file.FullName)' -Force`n"
            
            # Git add if not skipped
            if (-not $SkipGitAdd) {
                git add $file.FullName 2>$null
                git add $destination 2>$null
            }
        } catch {
            Write-Host "  Failed: $($file.Name) - $_" -ForegroundColor Red
        }
    }
}

# Save undo script
$undoScript += "`nWrite-Host 'Restore complete!' -ForegroundColor Green"
$undoScript | Out-File -FilePath ".\UNDO-ORGANIZE.ps1" -Encoding UTF8
Write-Host "`nCreated UNDO-ORGANIZE.ps1 (in case you need to revert)" -ForegroundColor Cyan

# Create new README for archive
$archiveReadme = @"
# Archive Directory Structure

## Organization Date: $(Get-Date)

This archive contains files that were moved from the root directory to improve navigation.

### Folder Structure:
- **2024/fixes/** - All fix scripts from development
- **2024/emergency/** - Emergency and nuclear fix scripts  
- **2024/diagnostic/** - Diagnostic and debug tools
- **2024/patches/** - Various patch scripts
- **2024/benchmarks/** - Test and benchmark scripts
- **2024/create_scripts/** - Generation and creation utilities
- **2024/install_scripts/** - Installation scripts
- **2024/start_scripts/** - Launch and start scripts
- **2024/run_scripts/** - Run and execution scripts
- **2024/uncategorized/** - Old files without clear category
- **docs/old_readmes/** - Completed documentation
- **docs/reports/** - Analysis reports and checklists

### To Restore:
Run .\UNDO-ORGANIZE.ps1 from the root directory.

### Files Kept in Root:
- Critical configuration files (package.json, pyproject.toml, etc.)
- Recent files (modified in last 7 days)
- Core scripts (main.py, enhanced_launcher.py, p.ps1)
- Active documentation (README.md, LICENSE, TODAY_CHANGES.md)

Total files organized: $movedCount
"@

$archiveReadme | Out-File -FilePath ".\_archive\README.md" -Encoding UTF8

$Duration = (Get-Date) - $StartTime

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "            ORGANIZATION COMPLETE!              " -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Files moved:      $movedCount" -ForegroundColor White
Write-Host "  Files kept:       $SkippedCount" -ForegroundColor White
Write-Host "  Time taken:       $($Duration.TotalSeconds) seconds" -ForegroundColor White
Write-Host ""
Write-Host "  Undo script:      .\UNDO-ORGANIZE.ps1" -ForegroundColor Cyan
Write-Host "  Archive info:     .\_archive\README.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Next step:        Commit the organized structure" -ForegroundColor Yellow
Write-Host "  Command:          .\p.ps1 'organized: 1662 files into logical folders'" -ForegroundColor Yellow
Write-Host ""

if (-not $SkipGitAdd) {
    Write-Host "Git tracking updated. Ready to commit!" -ForegroundColor Cyan
}
