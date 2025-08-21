param([string]$ProjectRoot = "D:\Dev\kha")

$ErrorActionPreference="Stop"
function Ok($m){Write-Host "[OK] $m" -f Green}
function Warn($m){Write-Host "[!] $m" -f Yellow}
function Fail($m){Write-Host "[X] $m" -f Red}

$ReportDir = Join-Path $ProjectRoot "verification_reports"; New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
$Report = Join-Path $ReportDir ("quick_verify_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

# find dev port
function Up($port){ try { (Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/health/ping" -TimeoutSec 3).StatusCode -eq 200 } catch { $false } }
$ports=@($env:IRIS_DEV_PORT,5173,3000,3310) | Where-Object { $_ }
$port=$null; foreach($p in $ports){ if(Up $p){ $port=$p; break } }

if(-not $port){ 
  Fail "No dev server found running on ports: $($ports -join ', ')"
  Warn "Please start the dev server first with: cd frontend && pnpm dev"
  $o=@{ok=$false; reason="server_not_running"}
  $o|ConvertTo-Json|Set-Content $Report
  exit 1
}

Ok "Dev server found on http://127.0.0.1:$port"

function Ping($path){
  try{ 
    $r=Invoke-WebRequest -UseBasicParsing -Uri ("http://127.0.0.1:{0}{1}" -f $port,$path) -TimeoutSec 10
    @{path=$path; ok=$true; status=$r.StatusCode} 
  }
  catch{ 
    @{path=$path; ok=$false; status=0; error="$($_.Exception.Message)"} 
  }
}

# Test all routes
$routes=@("/hologram","/templates","/pricing")
$res=@()
foreach($r in $routes){ 
  $result = Ping $r
  $res += $result
  if($result.ok){ 
    Ok "$($result.path) → 200" 
  } else { 
    Fail "$($result.path) → failed: $($result.error)" 
  }
}

# Test Export API
Ok "Testing Export API..."
try {
  $exportResult = Invoke-RestMethod -Method POST `
    -Uri "http://127.0.0.1:$port/api/templates/export" `
    -ContentType "application/json" `
    -Body '{"input":"data\\concept_graph.json","layout":"grid","scale":0.12}'
  
  if ($exportResult.ok) {
    Ok "/api/templates/export → 200 (Export ready)"
  } else {
    Warn "/api/templates/export → Response received but not OK"
  }
  $exportStatus = @{ok=$true; response=$exportResult}
} catch {
  Fail "/api/templates/export → Error: $_"
  $exportStatus = @{ok=$false; error="$_"}
}

$allOk = ($res | Where-Object {$_.ok}).Count -eq $routes.Count -and $exportStatus.ok
$o=@{
  ok=$allOk
  server="http://127.0.0.1:$port"
  routes=$res
  export=$exportStatus
  timestamp=Get-Date -Format "yyyy-MM-dd HH:mm:ss"
}

$o|ConvertTo-Json -Depth 8 | Set-Content $Report
Ok "Report saved → $Report"

if ($allOk) { 
  Ok "✅ All checks passed!"
  exit 0 
} else { 
  Warn "⚠️ Some checks failed - see report for details"
  exit 1 
}