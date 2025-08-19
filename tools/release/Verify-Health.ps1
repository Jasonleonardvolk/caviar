param(
  [string]$Url = "http://127.0.0.1:5173/health",
  [string]$OutDir = "D:\Dev\kha\verification_reports"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
$out = Join-Path $OutDir "health_$stamp.json"

try {
  $r = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 10
  $status = $r.StatusCode
  $body = $r.Content
  Set-Content -Path $out -Value $body -Encoding UTF8
  if ($status -eq 200) {
    Write-Host "[OK] Health OK → $out" -ForegroundColor Green
    exit 0
  } else {
    Write-Host "[!] Health returned $status → $out" -ForegroundColor Yellow
    exit 1
  }
} catch {
  Write-Host "[X] Failed to fetch $Url : $_" -ForegroundColor Red
  exit 2
}