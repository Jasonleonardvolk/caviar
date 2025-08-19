# PowerShell Setup Script for TORI No-DB Migration v2.2
# Enhanced with environment validation and dry-run mode

param(
    [switch]$DryRun,
    [switch]$SkipDependencies,
    [switch]$Force,
    [string]$StateRoot = "C:\tori_state",
    [string]$MaxTokensPerMin = "200"
)

# Script configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Trap errors
trap {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    Write-Host "   Location: $($_.InvocationInfo.PositionMessage)" -ForegroundColor Red
    exit 1
}

# Helper functions
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-PythonInstallation {
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            return @{
                Installed = $true
                Version = "$major.$minor"
                FullVersion = $pythonVersion
                Suitable = ($major -eq 3 -and $minor -ge 8)
            }
        }
    } catch {
        return @{ Installed = $false }
    }
}

function Test-Environment {
    Write-Host "`n🔍 Environment Validation" -ForegroundColor Cyan
    Write-Host "========================" -ForegroundColor Cyan
    
    $issues = @()
    
    # Check admin rights
    if (-not (Test-Administrator)) {
        Write-Host "⚠️  Not running as Administrator" -ForegroundColor Yellow
        Write-Host "   Some operations may require elevation" -ForegroundColor Yellow
    }
    
    # Check Python
    $python = Test-PythonInstallation
    if ($python.Installed) {
        Write-Host "✅ Python: $($python.FullVersion)" -ForegroundColor Green
        if (-not $python.Suitable) {
            $issues += "Python 3.8+ required (found $($python.Version))"
        }
    } else {
        Write-Host "❌ Python: Not found" -ForegroundColor Red
        $issues += "Python not found in PATH"
    }
    
    # Check pip
    try {
        $pipVersion = pip --version 2>&1
        Write-Host "✅ pip: Found" -ForegroundColor Green
    } catch {
        Write-Host "❌ pip: Not found" -ForegroundColor Red
        $issues += "pip not found"
    }
    
    # Check Git (optional but recommended)
    try {
        $gitVersion = git --version 2>&1
        Write-Host "✅ Git: $gitVersion" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Git: Not found (optional)" -ForegroundColor Yellow
    }
    
    # Check required files
    $requiredFiles = @(
        "master_nodb_fix_v2.py",
        "requirements_nodb.txt"
    )
    
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Host "✅ File: $file" -ForegroundColor Green
        } else {
            Write-Host "❌ File: $file not found" -ForegroundColor Red
            $issues += "Required file missing: $file"
        }
    }
    
    # Check disk space
    $drive = (Get-Location).Drive
    if ($drive) {
        $freeSpace = [math]::Round($drive.Free / 1GB, 2)
        if ($freeSpace -lt 1) {
            Write-Host "⚠️  Disk space: ${freeSpace}GB free (low)" -ForegroundColor Yellow
            $issues += "Low disk space: ${freeSpace}GB available"
        } else {
            Write-Host "✅ Disk space: ${freeSpace}GB free" -ForegroundColor Green
        }
    }
    
    return $issues
}

function Show-Configuration {
    Write-Host "`n📋 Configuration" -ForegroundColor Cyan
    Write-Host "===============" -ForegroundColor Cyan
    Write-Host "State Root:       $StateRoot" -ForegroundColor White
    Write-Host "Max Tokens/Min:   $MaxTokensPerMin" -ForegroundColor White
    Write-Host "Working Dir:      $(Get-Location)" -ForegroundColor White
    Write-Host "Dry Run:          $DryRun" -ForegroundColor White
    Write-Host "Skip Deps:        $SkipDependencies" -ForegroundColor White
}

# Main execution
Write-Host "🚀 TORI No-DB Setup Script v2.2" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "`n🔍 DRY RUN MODE - No changes will be made" -ForegroundColor Yellow
}

# Set working directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($scriptPath) {
    Set-Location $scriptPath
}

# Validate environment
$validationIssues = Test-Environment

if ($validationIssues.Count -gt 0 -and -not $Force) {
    Write-Host "`n❌ Environment validation failed:" -ForegroundColor Red
    foreach ($issue in $validationIssues) {
        Write-Host "   - $issue" -ForegroundColor Red
    }
    Write-Host "`nUse -Force to continue anyway" -ForegroundColor Yellow
    exit 1
}

# Show configuration
Show-Configuration

if ($DryRun) {
    Write-Host "`n📝 DRY RUN: Would perform the following actions:" -ForegroundColor Yellow
}

# 1. Set environment variables
Write-Host "`n📋 Setting environment variables..." -ForegroundColor Yellow
if (-not $DryRun) {
    $env:TORI_STATE_ROOT = $StateRoot
    $env:MAX_TOKENS_PER_MIN = $MaxTokensPerMin
    $env:PYTHONPATH = "$PWD;$PWD\kha"
    
    Write-Host "   TORI_STATE_ROOT: $env:TORI_STATE_ROOT" -ForegroundColor Green
    Write-Host "   MAX_TOKENS_PER_MIN: $env:MAX_TOKENS_PER_MIN" -ForegroundColor Green
    Write-Host "   PYTHONPATH: $env:PYTHONPATH" -ForegroundColor Green
} else {
    Write-Host "   Would set TORI_STATE_ROOT=$StateRoot" -ForegroundColor Gray
    Write-Host "   Would set MAX_TOKENS_PER_MIN=$MaxTokensPerMin" -ForegroundColor Gray
    Write-Host "   Would set PYTHONPATH=$PWD;$PWD\kha" -ForegroundColor Gray
}

# 2. Create state directory
Write-Host "`n📁 Creating state directory..." -ForegroundColor Yellow
if (-not $DryRun) {
    if (-not (Test-Path $StateRoot)) {
        New-Item -ItemType Directory -Force -Path $StateRoot | Out-Null
        Write-Host "   Created: $StateRoot" -ForegroundColor Green
    } else {
        Write-Host "   Already exists: $StateRoot" -ForegroundColor Green
    }
} else {
    if (-not (Test-Path $StateRoot)) {
        Write-Host "   Would create: $StateRoot" -ForegroundColor Gray
    } else {
        Write-Host "   Already exists: $StateRoot" -ForegroundColor Gray
    }
}

# 3. Install dependencies
if (-not $SkipDependencies) {
    Write-Host "`n📦 Installing dependencies..." -ForegroundColor Yellow
    if (Test-Path "requirements_nodb.txt") {
        if (-not $DryRun) {
            try {
                pip install -r requirements_nodb.txt --quiet
                Write-Host "   Dependencies installed" -ForegroundColor Green
            } catch {
                Write-Host "   Warning: Some dependencies failed to install" -ForegroundColor Yellow
                Write-Host "   Error: $_" -ForegroundColor Yellow
            }
        } else {
            Write-Host "   Would run: pip install -r requirements_nodb.txt" -ForegroundColor Gray
        }
    } else {
        Write-Host "   requirements_nodb.txt not found - skipping" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n📦 Skipping dependency installation" -ForegroundColor Yellow
}

# 4. Run master fix script
Write-Host "`n🔧 Running master fix script..." -ForegroundColor Yellow
if (-not $DryRun) {
    if (Test-Path "master_nodb_fix_v2.py") {
        $result = python master_nodb_fix_v2.py 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   Master fix completed successfully" -ForegroundColor Green
        } else {
            Write-Host "   Master fix failed with exit code: $LASTEXITCODE" -ForegroundColor Red
            Write-Host "   Output: $result" -ForegroundColor Red
            if (-not $Force) {
                throw "Master fix script failed"
            }
        }
    } else {
        Write-Host "   master_nodb_fix_v2.py not found" -ForegroundColor Red
    }
} else {
    Write-Host "   Would run: python master_nodb_fix_v2.py" -ForegroundColor Gray
}

# 5. Run validation
Write-Host "`n✅ Running validation..." -ForegroundColor Yellow
if (-not $DryRun) {
    if (Test-Path "alan_backend\validate_nodb_final.py") {
        python alan_backend\validate_nodb_final.py
        if ($LASTEXITCODE -ne 0) {
            Write-Host "   Some validation checks failed - review above" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   Validation script not found" -ForegroundColor Yellow
    }
} else {
    Write-Host "   Would run: python alan_backend\validate_nodb_final.py" -ForegroundColor Gray
}

# 6. Create distribution package
Write-Host "`n📦 Creating distribution package..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$zipName = "tori_nodb_complete_$timestamp.zip"

if (-not $DryRun) {
    # Collect files with proper expansion (excluding backups)
    $files = @()
    
    # Pattern-based file collection
    $patterns = @(
        "alan_backend\*_modified.py",
        "python\core\torus_registry.py",
        "python\core\torus_cells.py",
        "python\core\observer_synthesis.py",
        "python\core\__init__.py",
        "alan_backend\migrate_to_nodb_ast.py",
        "alan_backend\test_nodb_migration.py",
        "alan_backend\validate_nodb_final.py",
        "*.md",
        "requirements_nodb.txt"
    )
    
    foreach ($pattern in $patterns) {
        $files += Get-ChildItem -Path $pattern -Exclude "*.backup" -ErrorAction SilentlyContinue
    }
    
    if ($files.Count -gt 0) {
        # Filter out any remaining backup files
        $files = $files | Where-Object { $_.Name -notlike "*.backup" }
        
        try {
            Compress-Archive -Path $files -DestinationPath $zipName -Force
            Write-Host "   Created: $zipName" -ForegroundColor Green
            Write-Host "   Files: $($files.Count)" -ForegroundColor Green
            Write-Host "   Size: $([math]::Round((Get-Item $zipName).Length / 1KB, 2)) KB" -ForegroundColor Green
        } catch {
            Write-Host "   Failed to create archive: $_" -ForegroundColor Red
        }
    } else {
        Write-Host "   No files found to package" -ForegroundColor Yellow
    }
} else {
    Write-Host "   Would create: $zipName" -ForegroundColor Gray
}

# Summary
Write-Host "`n" -NoNewline
if ($DryRun) {
    Write-Host "📝 DRY RUN COMPLETE" -ForegroundColor Yellow
    Write-Host "==================" -ForegroundColor Yellow
    Write-Host "No changes were made. Remove -DryRun to execute." -ForegroundColor Yellow
} else {
    Write-Host "✨ Setup Complete!" -ForegroundColor Green
    Write-Host "=================" -ForegroundColor Green
    Write-Host "`nTo start the system:" -ForegroundColor Cyan
    Write-Host "  python alan_backend\start_true_metacognition.bat" -ForegroundColor White
    Write-Host "`nFor testing:" -ForegroundColor Cyan
    Write-Host "  pytest alan_backend\test_nodb_migration.py" -ForegroundColor White
}

# Cleanup recommendations
if (-not $DryRun) {
    $backupFiles = Get-ChildItem -Path . -Filter "*.backup" -Recurse -ErrorAction SilentlyContinue
    if ($backupFiles.Count -gt 0) {
        Write-Host "`n💡 Tip: Found $($backupFiles.Count) backup files" -ForegroundColor Yellow
        Write-Host "   To clean up: Get-ChildItem -Filter '*.backup' -Recurse | Remove-Item" -ForegroundColor Yellow
    }
}
