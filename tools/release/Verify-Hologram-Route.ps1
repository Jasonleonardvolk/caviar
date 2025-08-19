param([switch]$StartServerIfNeeded = $true, [int]$TimeoutSec = 90, [string]$ProjectRoot = "D:\Dev\kha")

$ErrorActionPreference="Stop"
function Ok($m){Write-Host "[OK] $m" -f Green}
function Warn($m){Write-Host "[!] $m" -f Yellow}
function Fail($m){Write-Host "[X] $m" -f Red}

$Frontend = Join-Path $ProjectRoot "frontend"
$ReportDir = Join-Path $ProjectRoot "verification_reports"; New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
$Report = Join-Path $ReportDir ("verify_hologram_route_{0}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

$Must=@(
"frontend\src\routes\hologram\+page.svelte",
"frontend\src\lib\hologram\engineShim.ts",
"frontend\src\lib\device\capabilities.ts",
"frontend\src\lib\stores\userPlan.ts",
"frontend\src\lib\utils\exportVideo.ts",
"frontend\src\lib\components\HologramRecorder.svelte",
"frontend\src\routes\pricing\+page.svelte",
"frontend\src\lib\components\PricingTable.svelte"
) | ForEach-Object { Join-Path $ProjectRoot $_ }

$missing=@()
foreach($p in $Must){ if(Test-Path $p){ Ok $p } else { Fail "Missing: $p"; $missing += $p } }
if($missing.Count){ $o=@{ok=$false; reason="missing_files"; missing=$missing}; $o|ConvertTo-Json -Depth 8|Set-Content $Report; exit 2 }

# find dev port
function Up($port){ try { (Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/health/ping" -TimeoutSec 3).StatusCode -eq 200 } catch { $false } }
$ports=@($env:IRIS_DEV_PORT,5173,3000,3310) | ? { $_ }
$port=$null; foreach($p in $ports){ if(Up $p){ $port=$p; break } }

if(-not $port -and $StartServerIfNeeded){
  Warn "Starting dev server…"
  # Try to find the right command to use
  $pnpmCmd = Get-Command pnpm.cmd -ErrorAction SilentlyContinue
  if ($pnpmCmd) {
    $p=Start-Process -FilePath "pnpm.cmd" -ArgumentList "dev" -WorkingDirectory $Frontend -WindowStyle Hidden -PassThru
  } else {
    $npmCmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($npmCmd) {
      $p=Start-Process -FilePath "npm.cmd" -ArgumentList "run","dev" -WorkingDirectory $Frontend -WindowStyle Hidden -PassThru
    } else {
      $p=Start-Process -FilePath "npm" -ArgumentList "run","dev" -WorkingDirectory $Frontend -WindowStyle Hidden -PassThru
    }
  }
  $deadline=(Get-Date).AddSeconds($TimeoutSec)
  while(-not $port -and (Get-Date) -lt $deadline){ Start-Sleep -Milliseconds 500; foreach($q in $ports){ if(Up $q){ $port=$q; break } } }
}

if(-not $port){ Fail "Dev server not reachable"; $o=@{ok=$false; reason="server_unreachable"}; $o|ConvertTo-Json|Set-Content $Report; exit 3 }
Ok "Dev server on http://127.0.0.1:$port"

function Ping($path){
  try{ $r=Invoke-WebRequest -UseBasicParsing -Uri ("http://127.0.0.1:{0}{1}" -f $port,$path) -TimeoutSec 10; @{path=$path; ok=$true; status=$r.StatusCode} }
  catch{ @{path=$path; ok=$false; status=0; error="$($_.Exception.Message)"} }
}

$routes=@("/hologram","/pricing")
$res=@(); foreach($r in $routes){ $res += Ping $r }
$res | % { if($_.ok){ Ok "$($_.path) 200" } else { Fail "$($_.path) failed" } }

$o=@{ok=($res|?{$_.ok}).Count -eq $routes.Count; server=("http://127.0.0.1:{0}" -f $port); routes=$res }
$o|ConvertTo-Json -Depth 8 | Set-Content $Report
Ok "Report → $Report"
if ($o.ok) { exit 0 } else { exit 1 }