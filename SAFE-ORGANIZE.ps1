# SAFE-ORGANIZE.ps1
# SAFER VERSION - Only moves obvious junk, keeps anything that might be imported

param(
    [switch]$WhatIf,
    [switch]$SkipGitAdd,
    [switch]$Conservative  # Extra safe mode
)

$StartTime = Get-Date
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "        SAFE REPO ORGANIZER                    " -ForegroundColor Cyan
Write-Host "        Conservative cleanup mode              " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# ONLY move these obvious non-critical files
$SafeToMove = @{
    "_archive/2024/fixes" = "fix_*.py", "fix_*.ps1", "fix_*.cjs", "fix_*.js", "fix_*.bat"
    "_archive/2024/emergency" = "emergency_*.py", "EMERGENCY_*.ps1", "nuclear_*.py", "NUCLEAR*.ps1"
    "_archive/2024/diagnostic" = "diagnose_*.py", "debug_*.py", "diagnostic*.py", "check_*.py"
    "_archive/2024/benchmarks" = "benchmark_*.py", "*_test.py", "test_*.py", "*_stress_test.py"
    "_cleanup/duplicates" = "*.bak", "*.bak.bak", "*.backup", "*.original", "*.backup_*"
}

# NEVER MOVE THESE PATTERNS (might be imported/used)
$NeverMove = @(
    "*_integration.py",      # Integration modules
    "*_COMPLETE.md",         # Might be referenced docs
    "*_GUIDE.md",           # Important guides
    "*_DEPLOYMENT*.md",      # Deployment docs
    "dynamic_*.py",         # Dynamic modules
    "advanced_*.py",        # Advanced features
    "*launcher*.py",        # Launcher files
    "*_api.py",            # API files
    "*_server.py",         # Server files
    "*_client.py",         # Client files
    "*_core.py",           # Core modules
    "*_base.py",           # Base classes
    "*_utils.py",          # Utilities
    "*_config.py",         # Configuration
    "*_settings.py",       # Settings
    "*_models.py",         # Data models
    "*_types.py",          # Type definitions
    "*hologram*.py",       # Hologram system
    "*penrose*.py",        # Penrose system
    "*tori*.py",           # TORI system
    "*iris*.py",           # iRis system
    "*soliton*.py",        # Soliton features
    "*metacognition*.py",  # Metacognition
    "*consciousness*.py",  # Consciousness modules
    "*prajna*.py",         # Prajna system
    "*concept*.py",        # Concept mesh
    "*memory*.py",         # Memory systems
    "*cognitive*.py"       # Cognitive modules
)

# Files to ALWAYS KEEP in root
$AlwaysKeepInRoot = @(
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
    "*.ps1",  # Keep all PowerShell scripts accessible
    "*.bat"   # Keep all batch files accessible
)

$FilesToMove = @{}
$FileCount = 0
$SkippedCount = 0
$ProtectedCount = 0

Write-Host "`nAnalyzing files (SAFE MODE)..." -ForegroundColor Yellow

# Create directories
foreach ($folder in $SafeToMove.Keys) {
    $targetPath = Join-Path $PSScriptRoot $folder
    if (-not (Test-Path $targetPath)) {
        if (-not $WhatIf) {
            New-Item -ItemType Directory -Force -Path $targetPath | Out-Null
            Write-Host "Created: $folder" -ForegroundColor Green
        }
    }
}

# Analyze each file
Get-ChildItem -Path $PSScriptRoot -File | ForEach-Object {
    $file = $_
    $fileName = $file.Name
    $shouldMove = $false
    
    # Check if always keep
    foreach ($pattern in $AlwaysKeepInRoot) {
        if ($fileName -like $pattern) {
            Write-Host "  KEEP: $fileName (critical file)" -ForegroundColor Cyan
            $SkippedCount++
            return
        }
    }
    
    # Check if protected pattern
    foreach ($pattern in $NeverMove) {
        if ($fileName -like $pattern) {
            Write-Host "  PROTECTED: $fileName (might be imported)" -ForegroundColor Yellow
            $ProtectedCount++
            return
        }
    }
    
    # Only move if matches safe patterns
    foreach ($folder in $SafeToMove.Keys) {
        $patterns = $SafeToMove[$folder]
        foreach ($pattern in $patterns) {
            if ($fileName -like $pattern) {
                if (-not $FilesToMove.ContainsKey($folder)) {
                    $FilesToMove[$folder] = @()
                }
                $FilesToMove[$folder] += $file
                $shouldMove = $true
                $FileCount++
                break
            }
        }
        if ($shouldMove) { break }
    }
    
    if (-not $shouldMove) {
        Write-Host "  KEEP: $fileName (not in safe-to-move list)" -ForegroundColor Gray
        $SkippedCount++
    }
}

Write-Host "`n=== SAFE ORGANIZATION PLAN ===" -ForegroundColor Magenta
Write-Host "Files to move (obvious cleanup): $FileCount" -ForegroundColor Yellow
Write-Host "Files protected (might be used): $ProtectedCount" -ForegroundColor Cyan
Write-Host "Files kept (other): $SkippedCount" -ForegroundColor Green

if ($FileCount -eq 0) {
    Write-Host "`nNothing safe to move!" -ForegroundColor Green
    exit 0
}

# Show what will be moved
foreach ($folder in $FilesToMove.Keys | Sort-Object) {
    $files = $FilesToMove[$folder]
    Write-Host "`n-> $folder ($($files.Count) files)" -ForegroundColor Cyan
    if ($files.Count -le 10) {
        $files | ForEach-Object { Write-Host "    $($_.Name)" -ForegroundColor Gray }
    } else {
        $files | Select-Object -First 5 | ForEach-Object { Write-Host "    $($_.Name)" -ForegroundColor Gray }
        Write-Host "    ... and $($files.Count - 5) more" -ForegroundColor DarkGray
    }
}

if ($WhatIf) {
    Write-Host "`n=== WHAT-IF MODE ===" -ForegroundColor Yellow
    Write-Host "No files were moved. Remove -WhatIf to execute." -ForegroundColor Yellow
    Write-Host "`nThis SAFE mode only moves:" -ForegroundColor Cyan
    Write-Host "  - fix_* scripts" -ForegroundColor Gray
    Write-Host "  - emergency_* scripts" -ForegroundColor Gray
    Write-Host "  - diagnose_* scripts" -ForegroundColor Gray
    Write-Host "  - benchmark/test scripts" -ForegroundColor Gray
    Write-Host "  - .bak/.backup files" -ForegroundColor Gray
    Write-Host "`nIt KEEPS all:" -ForegroundColor Green
    Write-Host "  - Integration modules" -ForegroundColor Gray
    Write-Host "  - Documentation" -ForegroundColor Gray
    Write-Host "  - System modules" -ForegroundColor Gray
    Write-Host "  - Anything that might be imported" -ForegroundColor Gray
    exit 0
}

# Confirm
Write-Host "`n=== READY TO SAFELY ORGANIZE ===" -ForegroundColor Green
Write-Host "This will ONLY move obvious cleanup files." -ForegroundColor Yellow
Write-Host "All integration, docs, and system files stay." -ForegroundColor Yellow
$confirm = Read-Host "`nMove $FileCount safe files? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

# Create undo script
$undoScript = @"
# UNDO-SAFE-ORGANIZE.ps1 - Generated $(Get-Date)
Write-Host 'Restoring files...' -ForegroundColor Yellow
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
            Write-Host "  Failed: $($file.Name) - $_" -ForegroundColor Red
        }
    }
}

$undoScript += "Write-Host 'Restore complete!' -ForegroundColor Green"
$undoScript | Out-File -FilePath ".\UNDO-SAFE-ORGANIZE.ps1" -Encoding UTF8

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "         SAFE ORGANIZATION COMPLETE!            " -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Files safely moved:  $movedCount" -ForegroundColor White
Write-Host "  Files protected:     $ProtectedCount" -ForegroundColor Cyan
Write-Host "  Files kept:          $SkippedCount" -ForegroundColor White
Write-Host ""
Write-Host "  Undo script: .\UNDO-SAFE-ORGANIZE.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "  This was a SAFE cleanup that only moved:" -ForegroundColor Green
Write-Host "  - Fix scripts" -ForegroundColor Gray
Write-Host "  - Emergency scripts" -ForegroundColor Gray  
Write-Host "  - Diagnostic tools" -ForegroundColor Gray
Write-Host "  - Test files" -ForegroundColor Gray
Write-Host "  - Backup files" -ForegroundColor Gray
Write-Host ""
Write-Host "  All integration modules and docs remain!" -ForegroundColor Cyan
