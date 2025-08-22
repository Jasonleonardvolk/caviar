[CmdletBinding()]
param([int]$Port = 3000)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$base = "http://127.0.0.1:$Port"
$urls = @("$base/","$base/api/health","$base/api/list","$base/api/pdf/stats","$base/api/memory/state")

foreach ($u in $urls) {
  try {
    $r = Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 10
    if ($r.StatusCode -eq 200) { Write-Host "[OK] $u" } else { throw "HTTP $($r.StatusCode)" }
  } catch { Write-Error "[FAIL] $u  ->  $_"; exit 1 }
}
