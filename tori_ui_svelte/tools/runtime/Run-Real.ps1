# D:\Dev\kha\tori_ui_svelte\tools\runtime\Run-Real.ps1
# Purpose: Run iRIS SSR with REAL backends (PDF + Memory Vault).
# Usage examples:
#   powershell -ExecutionPolicy Bypass D:\Dev\kha\tori_ui_svelte\tools\runtime\Run-Real.ps1
#   powershell ...\Run-Real.ps1 -Port 3000 -PdfUrl http://127.0.0.1:7401 -VaultUrl http://127.0.0.1:7501 -Open
#   powershell ...\Run-Real.ps1 -Port 3001 -ForceKill

[CmdletBinding()]
param(
  [int]$Port = $(if ($env:PORT -as [int]) { [int]$env:PORT } else { 3000 }),
  [string]$PdfUrl   = $(if ($env:IRIS_PDF_SERVICE_URL) { $env:IRIS_PDF_SERVICE_URL } else { "http://127.0.0.1:7401" }),
  [string]$VaultUrl = $(if ($env:IRIS_MEMORY_VAULT_URL) { $env:IRIS_MEMORY_VAULT_URL } else { "http://127.0.0.1:7501" }),
  [ValidateSet("local","drive","s3")] [string]$StorageType = $(if ($env:IRIS_STORAGE_TYPE) { $env:IRIS_STORAGE_TYPE } else { "local" }),
  [switch]$ForceKill,
  [switch]$Open
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Wait-PortListening {
  param([int]$Port, [int]$TimeoutSec = 20)
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
    $ok = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq 'Listen' }
    if ($ok) { return }
    Start-Sleep -Milliseconds 250
  }
  throw "Server did not start listening on port $Port within $TimeoutSec seconds."
}

$uiRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
$entry  = Join-Path $uiRoot "build\index.js"

if (-not (Test-Path $entry)) {
  throw "Entry not found: $entry. Build first: `n  cd $uiRoot; pnpm install; pnpm run build"
}

# Port guard (only kill if it's clearly a node we own)
$tcp = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
if ($tcp) {
  $pid  = $tcp.OwningProcess
  $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
  $pp   = if ($proc) { $proc.Path } else { "Unknown" }
  if ($ForceKill -and $pp -and ($pp -like "*\node.exe")) {
    Write-Warning "[Run-Real] Port $Port in use by PID $pid ($pp). Stopping..."
    Stop-Process -Id $pid -Force
    Start-Sleep -Milliseconds 300
  } elseif ($tcp) {
    throw "Port $Port in use by PID $pid ($pp). Re-run with -ForceKill or choose another -Port."
  }
}

# Environment for REAL services
$env:IRIS_USE_MOCKS        = "0"
$env:IRIS_PDF_SERVICE_URL  = $PdfUrl
$env:IRIS_MEMORY_VAULT_URL = $VaultUrl
$env:IRIS_STORAGE_TYPE     = $StorageType
$env:PORT                  = "$Port"

Write-Host "[Run-Real] UI root: $uiRoot"
Write-Host "[Run-Real] Entry:   $entry"
Write-Host "[Run-Real] Port:    $Port"
Write-Host "[Run-Real] Mocks:   $($env:IRIS_USE_MOCKS)"
Write-Host "[Run-Real] PDF:     $PdfUrl"
Write-Host "[Run-Real] Vault:   $VaultUrl"

# If -Open, spawn a tiny watcher to launch browser after the socket is listening
if ($Open) {
  Start-Job -Name "Open-$Port" -ScriptBlock {
    param($p)
    $deadline = (Get-Date).AddSeconds(20)
    do {
      $ok = Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue | Where-Object { $_.State -eq 'Listen' }
      if ($ok) { Start-Process "http://localhost:$p/"; break }
      Start-Sleep -Milliseconds 300
    } while ((Get-Date) -lt $deadline)
  } -ArgumentList $Port | Out-Null
}

# Start SSR (foreground to keep logs visible)
pushd $uiRoot
try {
  Write-Host "[Run-Real] Starting SSR..."
  Start-Process -FilePath "node" -ArgumentList "`"$entry`"" -WorkingDirectory $uiRoot -PassThru | Out-Null
  Wait-PortListening -Port $Port -TimeoutSec 20
  Write-Host "[Run-Real] Listening on http://localhost:$Port/"
  # Tail the port state so this console stays informative
  while ($true) { Start-Sleep -Seconds 300 }
} finally {
  popd
}
