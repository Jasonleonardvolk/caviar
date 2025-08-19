<#
Run-SimpleRefactor.ps1
Wrapper for the simple, reliable mass_refactor_simple.py script
#>
param(
  [string]$RepoRoot = "D:\Dev\kha",
  [string]$OldRoot = "C:\Users\jason\Desktop\tori\kha",
  [switch]$DryRun = $false,
  [switch]$Resume = $false,
  [string]$BackupDir = "",
  [string]$TextToken = "IRIS_ROOT"
)

$Script = Join-Path $RepoRoot "tools\refactor\mass_refactor_simple.py"
if (!(Test-Path $Script)) {
  Write-Host "Error: mass_refactor_simple.py not found at $Script" -ForegroundColor Red
  exit 2
}

$scriptArgs = @(
  $Script,
  "--root", $RepoRoot,
  "--old", $OldRoot,
  "--text-token", $TextToken
)

if ($DryRun) { 
  $scriptArgs += "--dry-run"
  Write-Host "Running in DRY RUN mode - no files will be modified" -ForegroundColor Yellow
}

if ($Resume) { 
  $scriptArgs += "--resume"
  Write-Host "Resuming from previous state..." -ForegroundColor Cyan
}

if ($BackupDir -ne "") { 
  $scriptArgs += @("--backup-dir", $BackupDir)
  Write-Host "Backing up files to: $BackupDir" -ForegroundColor Green
}

Write-Host "`nStarting refactoring..." -ForegroundColor Cyan
Write-Host "Repository: $RepoRoot" -ForegroundColor Gray
Write-Host "Old Path: $OldRoot" -ForegroundColor Gray
Write-Host ""

& python @scriptArgs

if ($LASTEXITCODE -eq 0) {
  Write-Host "`nRefactoring completed successfully!" -ForegroundColor Green
} else {
  Write-Host "`nRefactoring encountered errors. Check the logs in tools\refactor\" -ForegroundColor Red
}
