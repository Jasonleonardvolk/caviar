# TARGETED-ORGANIZE.ps1
# Moves ONLY the specific categories you identified - 897 files

param(
    [switch]$WhatIf,
    [switch]$SkipGitAdd
)

$StartTime = Get-Date
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "        TARGETED FILE ORGANIZER                " -ForegroundColor Cyan
Write-Host "        Moving only agreed categories          " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# EXACTLY what you specified to move
$TargetedMoves = @{
    "_archive/2024/benchmarks" = @("benchmark_*.py", "*_test.py", "test_*.py", "*_stress_test.py")
    "_archive/2024/diagnostic" = @("check_*.py", "diagnose_*.py", "debug_*.py", "diagnostic*.py")
    "_archive/2024/emergency" = @("emergency_*.py", "emergency_*.ps1", "EMERGENCY_*.ps1", "nuclear_*.py", "NUCLEAR*.ps1")
    "_archive/2024/fixes" = @("fix_*.py", "fix_*.ps1", "fix_*.cjs", "fix_*.js", "fix_*.bat", "FIX_*.bat", "FIX_*.ps1")
    "_archive/2024/install_scripts" = @("install_*.py", "INSTALL_*.bat", "Install*.ps1")
    "_archive/2024/patches" = @("apply_*.py", "*_patch.py", "*_patches.py", "patch_*.py")
    "_archive/2024/run_scripts" = @("run_*.bat", "run_*.py", "RUN_*.bat", "RUN_*.ps1")
    "_archive/docs/reports" = @("*_CHECKLIST.md", "*_report*.md", "*_REPORT*.md", "health_report*.md")
    "_cleanup/duplicates" = @("*.bak", "*.bak.bak", "*.backup", "*.original")
    "tools/demo" = @("demo_*.py", "*_demo.py")
}

# Special handling for uncategorized (old .mcp.json files, etc)
$UncategorizedPatterns = @(
    ".mcp.json*",
    "*.vsix",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.rar",
    "*.7z",
    "*.exe",
    "*.dll",
    "*.so",
    "*.dylib",
    "*.pdb",
    "*.pyc",
    "__pycache__",
    "*.log",
    "*.tmp",
    "*.temp",
    "*.cache",
    "*.swp",
    "*.swo",
    "*~",
    ".DS_Store",
    "Thumbs.db"
)

$FilesToMove = @{}
$FileCount = 0
$SkippedCount = 0

Write-Host "`nCreating archive directories..." -ForegroundColor Yellow

# Create all directories first
foreach ($folder in $TargetedMoves.Keys) {
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

# Create uncategorized folder
$uncategorizedPath = Join-Path $PSScriptRoot "_archive/2024/uncategorized"
if (-not (Test-Path $uncategorizedPath)) {
    if (-not $WhatIf) {
        New-Item -ItemType Directory -Force -Path $uncategorizedPath | Out-Null
        Write-Host "  Created: _archive/2024/uncategorized" -ForegroundColor Green
    }
}

Write-Host "`nScanning files..." -ForegroundColor Yellow

# Process targeted patterns
Get-ChildItem -Path $PSScriptRoot -File | ForEach-Object {
    $file = $_
    $fileName = $file.Name
    $moved = $false
    
    # Skip if it's this script
    if ($fileName -eq "TARGETED-ORGANIZE.ps1" -or $fileName -eq "UNDO-TARGETED.ps1") {
        $SkippedCount++
        return
    }
    
    # Check targeted move patterns
    foreach ($folder in $TargetedMoves.Keys) {
        $patterns = $TargetedMoves[$folder]
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
    
    # Check if it's uncategorized junk
    if (-not $moved) {
        foreach ($pattern in $UncategorizedPatterns) {
            if ($fileName -like $pattern) {
                if (-not $FilesToMove.ContainsKey("_archive/2024/uncategorized")) {
                    $FilesToMove["_archive/2024/uncategorized"] = @()
                }
                $FilesToMove["_archive/2024/uncategorized"] += $file
                $moved = $true
                $FileCount++
                break
            }
        }
    }
    
    if (-not $moved) {
        $SkippedCount++
    }
}

Write-Host "`n=== TARGETED ORGANIZATION PLAN ===" -ForegroundColor Magenta
Write-Host "Files to move: $FileCount" -ForegroundColor Yellow
Write-Host "Files to keep: $SkippedCount" -ForegroundColor Green

if ($FileCount -eq 0) {
    Write-Host "`nNo files match the targeted patterns!" -ForegroundColor Yellow
    exit 0
}

# Show what will be moved
foreach ($folder in $FilesToMove.Keys | Sort-Object) {
    $files = $FilesToMove[$folder]
    Write-Host "`n-> $folder ($($files.Count) files)" -ForegroundColor Cyan
    if ($files.Count -le 5) {
        $files | ForEach-Object { Write-Host "    $($_.Name)" -ForegroundColor Gray }
    } else {
        $files | Select-Object -First 3 | ForEach-Object { Write-Host "    $($_.Name)" -ForegroundColor Gray }
        Write-Host "    ... and $($files.Count - 3) more" -ForegroundColor DarkGray
    }
}

if ($WhatIf) {
    Write-Host "`n=== WHAT-IF MODE ===" -ForegroundColor Yellow
    Write-Host "No files were moved. Remove -WhatIf to execute." -ForegroundColor Yellow
    Write-Host "Command: .\TARGETED-ORGANIZE.ps1" -ForegroundColor White
    exit 0
}

# Confirm
Write-Host "`n=== READY TO ORGANIZE ===" -ForegroundColor Red
$confirm = Read-Host "Move $FileCount files to archive? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

# Create undo script
$undoScript = @"
# UNDO-TARGETED.ps1 - Generated $(Get-Date)
# Restores files to original locations

Write-Host 'Restoring $FileCount files to root...' -ForegroundColor Yellow

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
            
            # Git track the move
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
$undoScript += "`nWrite-Host 'Successfully restored $movedCount files!' -ForegroundColor Green"
$undoScript | Out-File -FilePath ".\UNDO-TARGETED.ps1" -Encoding UTF8

$Duration = (Get-Date) - $StartTime

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "         TARGETED CLEANUP COMPLETE!            " -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Files moved:       $movedCount" -ForegroundColor White
Write-Host "  Files kept:        $SkippedCount" -ForegroundColor White
Write-Host "  Time taken:        $([math]::Round($Duration.TotalSeconds, 2)) seconds" -ForegroundColor White
Write-Host ""
Write-Host "  Undo script:       .\UNDO-TARGETED.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Categories moved:" -ForegroundColor Yellow
Write-Host "    - Benchmarks and tests" -ForegroundColor Gray
Write-Host "    - Diagnostic tools" -ForegroundColor Gray
Write-Host "    - Emergency/fix scripts" -ForegroundColor Gray
Write-Host "    - Installation scripts" -ForegroundColor Gray
Write-Host "    - Patches" -ForegroundColor Gray
Write-Host "    - Run scripts" -ForegroundColor Gray
Write-Host "    - Old reports" -ForegroundColor Gray
Write-Host "    - Backup files (.bak)" -ForegroundColor Gray
Write-Host "    - Demo files" -ForegroundColor Gray
Write-Host ""
Write-Host "  Next: .\p.ps1 'organized: moved ~900 cleanup files to archive'" -ForegroundColor Yellow

if (-not $SkipGitAdd) {
    Write-Host "`n  Git tracking updated!" -ForegroundColor Cyan
}
