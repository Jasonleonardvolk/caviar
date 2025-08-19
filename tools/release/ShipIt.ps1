# Master shipping orchestrator - runs all validation and build steps
param(
  [string]$RepoRoot = "D:\Dev\kha",
  [switch]$SkipFix,
  [switch]$SkipBuild,
  [switch]$QuickMode,
  [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$scriptDir = Join-Path $RepoRoot "tools\release"

function Write-Banner {
  param([string]$Text, [string]$Color = "Cyan")
  
  $width = 60
  $padding = [math]::Max(0, ($width - $Text.Length - 2) / 2)
  $paddedText = (" " * [math]::Floor($padding)) + $Text + (" " * [math]::Ceiling($padding))
  
  Write-Host ""
  Write-Host ("+" + ("=" * ($width - 2)) + "+") -ForegroundColor $Color
  Write-Host ("|" + $paddedText.PadRight($width - 2) + "|") -ForegroundColor $Color
  Write-Host ("+" + ("=" * ($width - 2)) + "+") -ForegroundColor $Color
  Write-Host ""
}

function Run-Step {
  param(
    [string]$Name,
    [string]$Script,
    [string[]]$Arguments = @(),
    [bool]$Critical = $true
  )
  
  Write-Host ""
  Write-Host ">> " -ForegroundColor Cyan -NoNewline
  Write-Host $Name -ForegroundColor White
  Write-Host ("-" * 50) -ForegroundColor Gray
  
  $scriptPath = Join-Path $scriptDir $Script
  
  if (-not (Test-Path $scriptPath)) {
    Write-Host "  ✗ Script not found: $Script" -ForegroundColor Red
    if ($Critical) {
      Write-Host "  CRITICAL ERROR - Cannot continue" -ForegroundColor Red
      exit 1
    }
    return $false
  }
  
  try {
    & powershell -ExecutionPolicy Bypass -File $scriptPath @Arguments
    
    if ($LASTEXITCODE -eq 0) {
      Write-Host "  [OK] $Name completed successfully" -ForegroundColor Green
      return $true
    } else {
      Write-Host "  [FAIL] $Name failed with exit code $LASTEXITCODE" -ForegroundColor Red
      if ($Critical) {
        Write-Host "  CRITICAL ERROR - Cannot continue" -ForegroundColor Red
        exit 1
      }
      return $false
    }
  } catch {
    Write-Host "  [ERROR] Error running $Name`: $_" -ForegroundColor Red
    if ($Critical) {
      exit 1
    }
    return $false
  }
}

function Prompt-Continue {
  param([string]$Message)
  
  Write-Host ""
  Write-Host $Message -ForegroundColor Yellow
  $response = Read-Host "Continue? (Y/N)"
  
  if ($response -ne 'Y' -and $response -ne 'y') {
    Write-Host "Aborted by user" -ForegroundColor Red
    exit 0
  }
}

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

Write-Banner "IRIS SHIPPING ORCHESTRATOR" "Cyan"

Write-Host "Repository: $RepoRoot" -ForegroundColor Gray
Write-Host "Mode: " -NoNewline
if ($QuickMode) {
  Write-Host "QUICK" -ForegroundColor Yellow
} else {
  Write-Host "FULL" -ForegroundColor Green
}
Write-Host ""

# Step 1: Auto-fix issues
if (-not $SkipFix) {
  $args = @("-RepoRoot", $RepoRoot)
  if ($Force) { $args += "-Force" }
  
  $fixResult = Run-Step -Name "AUTO-FIX COMMON ISSUES" `
    -Script "AutoFixForShipping.ps1" `
    -Arguments $args `
    -Critical $false
    
  if (-not $fixResult) {
    Prompt-Continue "Auto-fix encountered issues. Review and continue?"
  }
} else {
  Write-Host ">> Skipping auto-fix (--SkipFix specified)" -ForegroundColor Gray
}

# Step 2: Quick validation
Write-Host ""
$quickResult = Run-Step -Name "QUICK VALIDATION CHECK" `
  -Script "QuickShipCheck.ps1" `
  -Arguments @("-RepoRoot", $RepoRoot) `
  -Critical $true

if (-not $quickResult) {
  Write-Host ""
  Write-Host "Quick validation failed. Run detailed validation for more info:" -ForegroundColor Yellow
  Write-Host "  .\tools\release\ShipReadyValidation.ps1 -Verbose" -ForegroundColor White
  exit 1
}

# Step 3: Full validation (unless in quick mode)
if (-not $QuickMode) {
  Write-Host ""
  $args = @("-RepoRoot", $RepoRoot)
  if ($SkipBuild) { $args += "-SkipBuild" }
  
  $fullResult = Run-Step -Name "FULL SHIP-READY VALIDATION" `
    -Script "ShipReadyValidation.ps1" `
    -Arguments $args `
    -Critical $true
    
  if (-not $fullResult) {
    Write-Host ""
    Write-Host "Full validation failed. Review the report for details." -ForegroundColor Red
    exit 1
  }
} else {
  Write-Host ">> Skipping full validation (--QuickMode specified)" -ForegroundColor Gray
}

# Step 4: End-to-end verification
if (-not $SkipBuild -and -not $QuickMode) {
  Write-Host ""
  Prompt-Continue "Ready to run full end-to-end verification with build?"
  
  $e2eResult = Run-Step -Name "END-TO-END VERIFICATION" `
    -Script "Verify-EndToEnd.ps1" `
    -Arguments @("-RepoRoot", $RepoRoot) `
    -Critical $false
    
  if (-not $e2eResult) {
    Write-Host ""
    Write-Host "End-to-end verification failed. Check logs in tools\release\reports\" -ForegroundColor Yellow
  }
}

# ============================================================================
# FINAL SUMMARY
# ============================================================================

Write-Banner "SHIPPING STATUS" "Green"

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$commit = git rev-parse --short HEAD 2>$null
$branch = git rev-parse --abbrev-ref HEAD 2>$null

Write-Host "Timestamp: $timestamp" -ForegroundColor Gray
Write-Host "Branch: $branch" -ForegroundColor Gray  
Write-Host "Commit: $commit" -ForegroundColor Gray
Write-Host ""

# Check for final release artifacts
$releaseReady = $false
$releasePath = Join-Path $RepoRoot "releases"

if (Test-Path $releasePath) {
  $latest = Get-ChildItem $releasePath -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  
  if ($latest) {
    $manifest = Join-Path $latest.FullName "manifest.json"
    $dist = Join-Path $latest.FullName "dist"
    
    if ((Test-Path $manifest) -and (Test-Path $dist)) {
      $releaseReady = $true
      Write-Host "[OK] Release artifacts found: $($latest.Name)" -ForegroundColor Green
      
      $fileCount = (Get-ChildItem $dist -Recurse -File | Measure-Object).Count
      Write-Host "  --> $fileCount files in dist folder" -ForegroundColor Gray
    }
  }
}

Write-Host ""
if ($releaseReady) {
  Write-Host "+================================================+" -ForegroundColor Green
  Write-Host "|           READY TO SHIP!                      |" -ForegroundColor Green
  Write-Host "|                                                |" -ForegroundColor Green
  Write-Host "|     All validations passed successfully       |" -ForegroundColor Green
  Write-Host "|         Release artifacts generated           |" -ForegroundColor Green
  Write-Host "+================================================+" -ForegroundColor Green
  
  Write-Host ""
  Write-Host "Release location: $($latest.FullName)" -ForegroundColor Cyan
  
  # Ask to open release folder
  Write-Host ""
  $open = Read-Host "Open release folder? (Y/N)"
  if ($open -eq 'Y' -or $open -eq 'y') {
    Start-Process explorer.exe $latest.FullName
  }
  
  exit 0
} else {
  Write-Host "+================================================+" -ForegroundColor Yellow
  Write-Host "|         NOT READY TO SHIP                     |" -ForegroundColor Yellow
  Write-Host "|                                                |" -ForegroundColor Yellow
  Write-Host "|    Release artifacts not found or incomplete  |" -ForegroundColor Yellow
  Write-Host "+================================================+" -ForegroundColor Yellow
  
  Write-Host ""
  Write-Host "Next steps:" -ForegroundColor Cyan
  Write-Host "1. Review validation reports in tools\release\reports\" -ForegroundColor White
  Write-Host "2. Fix any remaining issues" -ForegroundColor White
  Write-Host "3. Run this script again" -ForegroundColor White
  
  exit 1
}
