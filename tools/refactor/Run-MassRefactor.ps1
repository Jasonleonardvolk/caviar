<#
Run-MassRefactor.ps1
Convenience wrapper around mass_path_refactor_v2.py with sane defaults, excludes and resume.
#>
param(
  [string]$RepoRoot = "D:\Dev\kha",
  [string]$OldRoot = "C:\Users\jason\Desktop\tori\kha",
  [int]$Workers = 8,
  [switch]$DryRun = $false,
  [switch]$Resume = $true,
  [string]$BackupDir = ""
)

$Script = Join-Path $RepoRoot "tools\refactor\mass_path_refactor_v2.py"
if (!(Test-Path $Script)) {
  Write-Host "Copy mass_path_refactor_v2.py into tools\refactor first."
  exit 2
}

$refDir = Join-Path $RepoRoot "tools\refactor"
if (!(Test-Path $refDir)) { New-Item -ItemType Directory -Path $refDir | Out-Null }

$excludes = @(
  ".git",".hg",".svn",".venv","venv","env","node_modules","dist","build","out",".cache",".pytest_cache","target",".idea",".vscode","tools\dawn","tools/dawn"
) -join ","

$scriptArgs = @(
  $Script,
  "--root", $RepoRoot,
  "--old", $OldRoot,
  "--include-exts", ".py,.ts,.tsx,.js,.svelte,.wgsl,.txt,.json,.md,.yaml,.yml",
  "--exclude-dirs", $excludes,
  "--max-bytes", "2000000",
  "--workers", $Workers
)

if ($DryRun) { 
  $scriptArgs += @("--dry-run", "--plan", "tools/refactor/refactor_plan.csv") 
}
if ($Resume) { 
  $scriptArgs += @("--resume") 
}
if ($BackupDir -ne "") { 
  $scriptArgs += @("--backup-dir", $BackupDir) 
}

Write-Host "Running: python" ($scriptArgs -join " ")
& python @scriptArgs
