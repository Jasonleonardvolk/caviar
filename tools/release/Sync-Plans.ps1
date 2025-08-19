$ErrorActionPreference = "Stop"
$src = "D:\Dev\kha\config\plans.json"
$dst = "D:\Dev\kha\frontend\static\config\plans.json"

if (!(Test-Path $src)) { throw "Missing $src" }
$srcHash = (Get-FileHash -Algorithm SHA256 $src).Hash

$need = $true
if (Test-Path $dst) {
  $dstHash = (Get-FileHash -Algorithm SHA256 $dst).Hash
  if ($srcHash -eq $dstHash) { $need = $false }
}
if ($need) {
  New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
  Copy-Item $src $dst -Force
  Write-Host "🔄 plans.json synced → frontend/static/config/plans.json"
} else {
  Write-Host "✔ plans.json already in sync"
}