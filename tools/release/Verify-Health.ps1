param([string]$ProjectRoot = "D:\Dev\kha")

$ErrorActionPreference="Stop"
function Ok($m){Write-Host "[OK] $m" -f Green}
function Warn($m){Write-Host "[!] $m" -f Yellow}
function Fail($m){Write-Host "[X] $m" -f Red}

$ReportDir = Join-Path $ProjectRoot "verification_reports"; New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
$Report = Join-Path $ReportDir ("verify_health_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

# find dev port
function Up($port){ try { (Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/health/ping" -TimeoutSec 3).StatusCode -eq 200 } catch { $false } }
$ports=@($env:IRIS_DEV_PORT,5173,3000,3310) | ? { $_ }
$port=$null; foreach($p in $ports){ if(Up $p){ $port=$p; break } }

if(-not $port){ 
  Fail "Dev server not reachable"
  $o=@{ok=$false; reason="server_unreachable"}
  $o|ConvertTo-Json|Set-Content $Report
  exit 3 
}

Ok "Dev server on http://127.0.0.1:$port"

# Check health endpoint
try {
  $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 10
  Ok "Health endpoint responded"
  
  $o = @{
    ok = $true
    server = "http://127.0.0.1:$port"
    health = $health
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  }
  
  if ($health.status -eq "healthy") {
    Ok "System is healthy"
  } elseif ($health.status -eq "degraded") {
    Warn "System is degraded - check components"
  } else {
    Fail "System is unhealthy"
    $o.ok = $false
  }
} catch {
  Fail "Health check failed: $_"
  $o = @{
    ok = $false
    server = "http://127.0.0.1:$port"
    error = $_.Exception.Message
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  }
}

$o | ConvertTo-Json -Depth 8 | Set-Content $Report
Ok "Report → $Report"
exit ($o.ok ? 0 : 1)