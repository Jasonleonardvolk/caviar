param(
  [string]$ProjectRoot = "D:\Dev\kha"
)
$ErrorActionPreference = "Stop"
$script = Join-Path $ProjectRoot "tools\release\check-mobile-claims.mjs"
node $script