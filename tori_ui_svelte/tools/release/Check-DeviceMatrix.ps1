param(
  [int]$Port = 3000,
  [string]$Url = "http://localhost:{0}/device/matrix"
)

Write-Host "=== DEVICE MATRIX SANITY CHECK ===" -ForegroundColor Cyan
$u = [string]::Format($Url, $Port)
try {
  $resp = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 5
  $data = $resp.Content | ConvertFrom-Json
} catch {
  Write-Host " [X] Endpoint not reachable at $u" -ForegroundColor Red
  exit 2
}

$hw    = $data.hw
$tier  = $data.tier
$caps  = $data.caps
$min   = $data.minSupportedModel
$ok    = $data.ok

Write-Host (" MinSupportedModel  : {0}" -f $min)
Write-Host (" UA Hardware        : {0}" -f $(if($hw) {$hw} else {"<unknown>"}))
Write-Host (" Resolved Tier      : {0}" -f $tier)
if ($caps) {
  Write-Host (" Caps (maxN/zModes) : {0}/{1}" -f $caps.maxN, $caps.zernikeModes)
  Write-Host (" Server Fallback    : {0}" -f $caps.serverFallback)
}
if ($data.reason) { Write-Host (" Reason             : {0}" -f $data.reason) }

# hard blocks (11/12)
if ($data.ua -match "iPhone11" -or $data.ua -match "iPhone12") {
  Write-Host " [X] BLOCKED: iPhone11/12 not supported." -ForegroundColor Red
  exit 2
}

if (-not $ok -or $tier -eq "UNSUPPORTED") {
  Write-Host " [X] UNSUPPORTED: device not allowed by matrix." -ForegroundColor Red
  exit 2
}

Write-Host " [OK] Matrix allows this device with above caps." -ForegroundColor Green
exit 0