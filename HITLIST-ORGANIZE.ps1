# HITLIST-ORGANIZE.ps1
# ONLY moves the specific files you want gone - NO .md files, NO main files!

param(
    [switch]$WhatIf,
    [switch]$SkipGitAdd
)

$StartTime = Get-Date
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "          HITLIST FILE ORGANIZER               " -ForegroundColor Cyan
Write-Host "          Only your specified targets          " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# ONLY these specific patterns from your hitlist
$HitlistMoves = @{
    "_archive/2024/validation" = @(
        "check_*.bat",
        "check_*.py",
        "validate_*.py",
        "verify_*.py",
        "VERIFY_*.bat"
    )
    "_archive/2024/old_scripts" = @(
        "APPLY_*.bat",
        "claude_optimization*.ps1",
        "continue_*.bat",
        "continuation_*.py",
        "*_optimization_*.ps1"
    )
    "_archive/2024/diagnostic" = @(
        "find_*.py",
        "find_*.ps1",
        "search_*.py",
        "locate_*.py"
    )
    "_archive/2024/emergency" = @(
        "CRITICAL_*.ps1",
        "CRITICAL_*.py",
        "force_*.py",
        "emergency_*.py"
    )
    "_archive/2024/final_scripts" = @(
        "complete_*.py",
        "final_*.py",
        "final_*.ps1",
        "final_*.cjs",
        "FINAL_*.bat"
    )
    "_archive/2024/migration" = @(
        "migrate_*.py",
        "migration_*.py",
        "update_*.py"
    )
    "_archive/2024/temp_files" = @(
        "temp_*.py",
        "tmp_*.py",
        "*.tmp",
        "*.temp"
    )
    "_archive/2024/test_files" = @(
        "test_*.bat",
        "test_*.py",
        "TEST_*.bat"
    )
    "_archive/2024/build_setup" = @(
        "create_*.py",
        "CREATE_*.bat",
        "generate_*.py",
        "make_*.py",
        "init_*.py"
    )
    "_archive/2024/cleanup" = @(
        "cleanup_*.py",
        "clean_*.py",
        "flush_*.py",
        "kill_*.py",
        "remove_*.py"
    )
    "_archive/2024/backup" = @(
        "*_backup.py",
        "*_original_backup.py",
        "copy_*.py",
        "*.bak",
        "*.backup"
    )
}

# Files to NEVER move
$NeverMove = @(
    "main.py",
    "enhanced_launcher.py",
    "p.ps1",
    "README.md",
    "LICENSE",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "HITLIST-ORGANIZE.ps1",
    "UNDO-HITLIST.ps1"
)

$FilesToMove = @{}
$FileCount = 0
$SkippedCount = 0

Write-Host "`nCreating directories..." -ForegroundColor Yellow

# Create directories
foreach ($folder in $HitlistMoves.Keys) {
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

Write-Host "`nScanning for hitlist files..." -ForegroundColor Yellow

# Process files
Get-ChildItem -Path $PSScriptRoot -File | ForEach-Object {
    $file = $_
    $fileName = $file.Name
    $moved = $false
    
    # Skip if in never move list
    if ($fileName -in $NeverMove) {
        $SkippedCount++
        return
    }
    
    # Skip ALL .md files - we don't want to move docs!
    if ($fileName -like "*.md") {
        $SkippedCount++
        return
    }
    
    # Skip anything that looks like a main/server/api file
    if ($fileName -match "(main|server|api|client|app).*\.py$") {
        $SkippedCount++
        return
    }
    
    # Check hitlist patterns ONLY
    foreach ($folder in $HitlistMoves.Keys) {
        $patterns = $HitlistMoves[$folder]
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
        $SkippedCount++
    }
}

Write-Host "`n=== HITLIST TARGETS ===" -ForegroundColor Magenta
Write-Host "Files to archive: $FileCount" -ForegroundColor Yellow
Write-Host "Files to keep: $SkippedCount" -ForegroundColor Green

if ($FileCount -eq 0) {
    Write-Host "`nNo hitlist files found!" -ForegroundColor Yellow
    exit 0
}

# Show what will be moved
foreach ($folder in $FilesToMove.Keys | Sort-Object) {
    $files = $FilesToMove[$folder]
    Write-Host "`n-> $folder ($($files.Count) files)" -ForegroundColor Cyan
    
    # Show first 3 files
    $showCount = [Math]::Min(3, $files.Count)
    for ($i = 0; $i -lt $showCount; $i++) {
        Write-Host "    $($files[$i].Name)" -ForegroundColor Gray
    }
    
    # Show remaining count
    if ($files.Count -gt 3) {
        Write-Host "    ... and $($files.Count - 3) more" -ForegroundColor DarkGray
    }
}

if ($WhatIf) {
    Write-Host "`n=== WHAT-IF MODE ===" -ForegroundColor Yellow
    Write-Host "No files moved. Remove -WhatIf to execute." -ForegroundColor Yellow
    Write-Host "Command: .\HITLIST-ORGANIZE.ps1" -ForegroundColor White
    Write-Host "`nThis will ONLY move:" -ForegroundColor Cyan
    Write-Host "  - Validation scripts (check_*, validate_*, verify_*)" -ForegroundColor Gray
    Write-Host "  - Old optimization scripts" -ForegroundColor Gray
    Write-Host "  - Find/search diagnostic scripts" -ForegroundColor Gray
    Write-Host "  - Critical/emergency scripts" -ForegroundColor Gray
    Write-Host "  - Final/complete scripts" -ForegroundColor Gray
    Write-Host "`nNO .md files will be moved!" -ForegroundColor Green
    Write-Host "NO main/server/api files will be moved!" -ForegroundColor Green
    exit 0
}

# Confirm
Write-Host "`n=== READY TO EXECUTE HITLIST ===" -ForegroundColor Red
Write-Host "This will ONLY move the specific patterns." -ForegroundColor Yellow
Write-Host "NO documentation files will be touched!" -ForegroundColor Green
$confirm = Read-Host "`nMove $FileCount hitlist files? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

# Create undo script
$undoScript = @"
# UNDO-HITLIST.ps1 - Generated $(Get-Date)
Write-Host 'Restoring $FileCount files...' -ForegroundColor Yellow
"@

# Move files
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
$undoScript | Out-File -FilePath ".\UNDO-HITLIST.ps1" -Encoding UTF8

$Duration = (Get-Date) - $StartTime

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "           HITLIST COMPLETE!                   " -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Files archived:    $movedCount" -ForegroundColor White
Write-Host "  Files kept:        $SkippedCount" -ForegroundColor White
Write-Host "  Time:              $([math]::Round($Duration.TotalSeconds, 2)) seconds" -ForegroundColor White
Write-Host ""
Write-Host "  Undo available:    .\UNDO-HITLIST.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Commit with:       .\p.ps1 'cleaned: moved validation and final scripts'" -ForegroundColor Yellow
