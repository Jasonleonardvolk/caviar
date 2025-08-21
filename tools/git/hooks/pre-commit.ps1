# blocks venv/node_modules/large files from being committed
$ErrorActionPreference = "Stop"
$blocked = @('(^|/)(\.?venv|node_modules|logs|reports)/')
$maxMB   = 50

$staged = git diff --cached --name-only --diff-filter=ACMR
$fail = @()

foreach ($f in $staged) {
  foreach ($pat in $blocked) {
    if ($f -match $pat) { $fail += "blocked path: $f"; break }
  }
  if (Test-Path $f) {
    $lenMB = [math]::Round((Get-Item $f).Length / 1MB, 2)
    if ($lenMB -gt $maxMB) { $fail += "oversize ($lenMB MB): $f" }
  }
}

if ($fail.Count -gt 0) {
  Write-Error ("Pre-commit blocked:`n" + ($fail -join "`n"))
  exit 1
}
exit 0