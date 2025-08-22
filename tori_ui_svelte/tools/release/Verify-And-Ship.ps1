param(
  [ValidateSet('mock','real')][string]$Mode = 'mock',
  [int]$Port = 3000,
  [string]$RcTag = ''
)

$ErrorActionPreference = 'Stop'
Push-Location (Resolve-Path "$PSScriptRoot\..\..")  # -> D:\Dev\kha\tori_ui_svelte

try {
  if ($Mode -eq 'mock') { $env:IRIS_USE_MOCKS = '1' } else { $env:IRIS_USE_MOCKS = '0' }
  $env:PORT = "$Port"

  # Free the port first
  & "$PSScriptRoot\..\runtime\Run-Ports.ps1" -Ports $Port -Kill | Out-Null

  pnpm install
  pnpm run build

  # Verify
  & "$PSScriptRoot\Verify-EndToEnd.ps1" -Mode $Mode -Port $Port -StartServer -StopOnExit

  # Ship (PM2)
  & "$PSScriptRoot\Reset-And-Ship.ps1" -UsePM2

  if ($RcTag) {
    powershell "$PSScriptRoot\..\git\Git-Workflow.ps1" rc "$RcTag" -Message "iRis UI $Mode ship via Verify-And-Ship" -Push
  }

  Write-Host "[OK] Verify -> Ship complete on port $Port ($Mode)." -ForegroundColor Green
}
finally {
  Pop-Location
}
