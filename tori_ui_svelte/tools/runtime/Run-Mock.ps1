<#  D:\Dev\kha\tori_ui_svelte\tools\runtime\Run-Mock.ps1
    Purpose: Run iRIS SSR locally with mock backends enabled.
    Usage:
      powershell -ExecutionPolicy Bypass D:\Dev\kha\tori_ui_svelte\tools\runtime\Run-Mock.ps1
      # optional flags:
      #   -Port 3000         -> override port
      #   -Open              -> open default browser
      #   -ForceKill         -> kill existing process bound to the port if it's from this repo
#>

[CmdletBinding()]
param(
  [int]$Port = $(if ($env:PORT -as [int]) { [int]$env:PORT } else { 3000 }),
  [switch]$Open,
  [switch]$ForceKill
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Resolve paths - handle empty PSScriptRoot
if ($PSScriptRoot) {
    $uiRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
} else {
    # Fallback to known path
    $uiRoot = "D:\Dev\kha\tori_ui_svelte"
    if (-not (Test-Path $uiRoot)) {
        # Try current directory
        $current = Get-Location
        if ($current.Path -like "*tori_ui_svelte*") {
            $uiRoot = $current.Path
        } else {
            throw "Cannot determine UI root. Please run from D:\Dev\kha\tori_ui_svelte"
        }
    }
}
$entry = Join-Path $uiRoot "build\index.js"

# Env for mock mode
$env:IRIS_USE_MOCKS    = "1"
$env:IRIS_ALLOW_UNAUTH = "1"
$env:IRIS_STORAGE_TYPE = "local"
$env:PORT              = "$Port"

Write-Host "[Run-Mock] UI root: $uiRoot"
Write-Host "[Run-Mock] Entry:    $entry"
Write-Host "[Run-Mock] Port:     $Port"
Write-Host "[Run-Mock] Mocks:    $($env:IRIS_USE_MOCKS)"

if (-not (Test-Path $entry)) {
  throw "Entry not found: $entry. Build first: `n  cd $uiRoot; pnpm install; pnpm run build"
}

# Free the port if requested (only if the holder looks like our local node)
$tcp = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
if ($tcp) {
  $procId  = $tcp.OwningProcess
  $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
  $pp   = $proc?.Path
  if ($ForceKill -and $pp -and ($pp -like "*\node.exe")) {
    Write-Warning "[Run-Mock] Port $Port in use by PID $procId ($pp). Stopping..."
    Stop-Process -Id $procId -Force
    Start-Sleep -Milliseconds 300
  } elseif ($tcp) {
    throw "Port $Port in use by PID $procId ($pp). Re-run with -ForceKill or choose another -Port."
  }
}

# Start server
Write-Host "[Run-Mock] Starting SSR at http://localhost:$Port ..."
pushd $uiRoot
try {
  node "$entry"
} finally {
  popd
}

# Optionally open browser
if ($Open) {
  Start-Process "http://localhost:$Port/"
}
