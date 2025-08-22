# D:\Dev\kha\tori_ui_svelte\tools\release\Verify-EndToEnd.ps1
# Unified verify for iRis (dev | mock | prod) + Penrose
[CmdletBinding()]
param(
  [ValidateSet('dev','mock','prod')]
  [string]$Mode = 'dev',

  # App port: dev->5173, prod->3000 (defaulted below if not supplied)
  [int]$Port,

  # Penrose port
  [int]$PenrosePort = 7401,

  # Bring servers up before verifying
  [switch]$StartServer,

  # Cleanly stop anything we started on exit
  [switch]$StopOnExit
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Paths ---
$ROOT   = "D:\Dev\kha"
$UI     = Join-Path $ROOT "tori_ui_svelte"
$TOOLS  = Join-Path $ROOT "tools\runtime"
$PEN    = Join-Path $ROOT "services\penrose"
$PIDS   = @()

# --- Defaults ---
if (-not $Port) { $Port = ($(if ($Mode -eq 'prod') {3000} else {5173})) }
$BaseUrl      = "http://localhost:$Port"
$PenroseUrl   = "http://127.0.0.1:$PenrosePort"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host (" VERIFY iRis [{0}] on {1}" -f $Mode.ToUpper(), $BaseUrl) -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

function Start-Child {
  param([string]$Cmd, [string]$Cwd)
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = "powershell"
  $psi.Arguments = "-NoLogo -NoProfile -Command $Cmd"
  $psi.WorkingDirectory = $Cwd
  $psi.WindowStyle = 'Minimized'
  $proc = [System.Diagnostics.Process]::Start($psi)
  if ($proc) { $global:PIDS += $proc.Id }
  return $proc
}

function Wait-ForHttp {
  param([string]$Url, [int]$Tries=30, [int]$DelayMs=300)
  for ($i=0; $i -lt $Tries; $i++) {
    try { Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2 | Out-Null; return $true } catch {}
    Start-Sleep -Milliseconds $DelayMs
  }
  return $false
}

function Ensure-Penrose {
  Write-Host "[Penrose] Checking $PenroseUrl/docs ..." -ForegroundColor Yellow
  try { Invoke-WebRequest "$PenroseUrl/docs" -UseBasicParsing -TimeoutSec 2 | Out-Null; return } catch {}
  Write-Host "[Penrose] Not responding. Starting..." -ForegroundColor Yellow
  if (Test-Path (Join-Path $PEN "start-penrose.ps1")) {
    Start-Child ".\start-penrose.ps1" $PEN | Out-Null
  } else {
    $cmd = '$env:PYTHONUNBUFFERED="1"; .\.venv\Scripts\Activate.ps1; uvicorn main:app --host 0.0.0.0 --port ' + $PenrosePort
    Start-Child $cmd $PEN | Out-Null
  }
  if (-not (Wait-ForHttp "$PenroseUrl/docs")) { throw "Penrose did not start on :$PenrosePort" }
  Write-Host "[Penrose] UP on $PenroseUrl" -ForegroundColor Green
}

function Start-Dev {
  # Own dev port then start Vite
  if (Test-Path (Join-Path $TOOLS "Run-Ports.ps1")) {
    & (Join-Path $TOOLS "Run-Ports.ps1") -Ports $Port -Kill | Out-Null
  }
  Start-Child "pnpm dev" $UI | Out-Null
  if (-not (Wait-ForHttp "$BaseUrl/")) { throw "Dev server did not respond on $BaseUrl/" }
  Write-Host "[iRis DEV] UP on $BaseUrl" -ForegroundColor Green
}

function Start-Prod {
  # Own prod port; build; start Node adapter on $Port
  if (Test-Path (Join-Path $TOOLS "Run-Ports.ps1")) {
    & (Join-Path $TOOLS "Run-Ports.ps1") -Ports $Port -Kill | Out-Null
  }
  Write-Host "[iRis PROD] Building..." -ForegroundColor Yellow
  Push-Location $UI
  try {
    pnpm install | Out-Null
    pnpm run build | Out-Null
  } finally {
    Pop-Location
  }
  # Give builds a moment (avoid racing the filesystem watcher)
  Start-Sleep -Seconds 2
  $cmd = '$env:PORT=' + $Port + '; node .\build\index.js'
  Start-Child $cmd $UI | Out-Null
  if (-not (Wait-ForHttp "$BaseUrl/")) { throw "Prod server did not respond on $BaseUrl/" }
  Write-Host "[iRis PROD] UP on $BaseUrl" -ForegroundColor Green
}

function Stop-Started {
  if ($PIDS.Count -eq 0) { return }
  Write-Host "[CLEANUP] Stopping started processes: $($PIDS -join ', ')" -ForegroundColor Yellow
  foreach ($pid in $PIDS) { try { Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue } catch {} }
}

# --- Bring services up if requested ---
if ($StartServer) {
  Ensure-Penrose
  if ($Mode -eq 'dev' -or $Mode -eq 'mock') { Start-Dev } else { Start-Prod }
}

# --- Tests (GET only; no HEAD) ---
$passes = 0; $fails = 0; $warns = 0
function OK($m){ $script:passes++; Write-Host "  OK - $m" -ForegroundColor Green }
function FAIL($m){ $script:fails++; Write-Host "  FAIL - $m" -ForegroundColor Red }
function WARN($m){ $script:warns++; Write-Host "  WARN - $m" -ForegroundColor Yellow }

Write-Host "`n[1] Testing root redirect to /hologram..." -ForegroundColor Cyan
try {
  $root = Invoke-WebRequest "$BaseUrl/" -UseBasicParsing -MaximumRedirection 3
  if ($root.StatusCode -ge 200 -and $root.StatusCode -lt 400) { OK "Root resolves (final: $($root.BaseResponse.ResponseUri))" }
  else { FAIL "Root bad status $($root.StatusCode)" }
} catch { FAIL "Root fetch error: $($_.Exception.Message)" }

Write-Host "`n[2] Testing health endpoint..." -ForegroundColor Cyan
try {
  $h = Invoke-WebRequest "$BaseUrl/api/health" -UseBasicParsing
  if ($h.StatusCode -eq 200 -and $h.Content -match '"ok":\s*true') { OK "Health endpoint OK" } else { FAIL "/api/health unexpected body" }
} catch { FAIL "/api/health error: $($_.Exception.Message)" }

Write-Host "`n[3] Testing Penrose direct access..." -ForegroundColor Cyan
try {
  $d = Invoke-WebRequest "$PenroseUrl/docs" -UseBasicParsing
  if ($d.StatusCode -eq 200) { OK "Penrose docs reachable" } else { FAIL "Penrose docs bad status $($d.StatusCode)" }
} catch { FAIL "Penrose docs error: $($_.Exception.Message)" }

Write-Host "`n[4] Testing Penrose proxy through iRis..." -ForegroundColor Cyan
try {
  $p = Invoke-WebRequest "$BaseUrl/api/penrose/docs" -UseBasicParsing
  if ($p.StatusCode -eq 200) { OK "Penrose proxy /api/penrose/docs OK" } else { FAIL "Proxy bad status $($p.StatusCode)" }
} catch { FAIL "Proxy error: $($_.Exception.Message)" }

Write-Host "`n[5] Testing device matrix..." -ForegroundColor Cyan
try {
  $m = Invoke-WebRequest "$BaseUrl/device/matrix" -UseBasicParsing
  if ($m.StatusCode -eq 200) {
    if ($m.Content -match "UNSUPPORTED") { OK "Device matrix reports UNSUPPORTED (desktop UA)" }
    else { WARN "Device matrix did not show UNSUPPORTED; verify UA or thresholds" }
  } else { FAIL "/device/matrix bad status $($m.StatusCode)" }
} catch { FAIL "/device/matrix error: $($_.Exception.Message)" }

Write-Host "`n[6] Testing HUD page..." -ForegroundColor Cyan
try {
  $hud = Invoke-WebRequest "$BaseUrl/hologram" -UseBasicParsing
  if ($hud.StatusCode -eq 200) { OK "HUD page accessible" } else { FAIL "HUD bad status $($hud.StatusCode)" }
} catch { FAIL "HUD error: $($_.Exception.Message)" }

Write-Host "`n[7] Testing Penrose solve endpoint..." -ForegroundColor Cyan
try {
  $payload = @{ scene="demo"; N=256 } | ConvertTo-Json -Depth 5
  $solve = Invoke-WebRequest "$BaseUrl/api/penrose/solve" -Method POST -ContentType 'application/json' -Body $payload -UseBasicParsing
  if ($solve.StatusCode -eq 200 -and $solve.Content -match '"ok":\s*true') { OK "Penrose /solve OK" } else { WARN "Penrose /solve returned unexpected body" }
} catch { WARN "Penrose /solve error: $($_.Exception.Message)" }

# --- Summary ---
Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "           TEST RESULTS SUMMARY" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ("PASS: {0}   WARN: {1}   FAIL: {2}" -f $passes, $warns, $fails)
if ($fails -gt 0) { $code = 1; $status = "FAIL" } else { $code = 0; $status = "OK" }
Write-Host ("RESULT: {0}" -f $status) -ForegroundColor ($(if ($fails -gt 0) {'Red'} else {'Green'}))

if ($StopOnExit) { Stop-Started }

exit $code