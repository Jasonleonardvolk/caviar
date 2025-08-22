param(
  [Parameter(Mandatory = $true)][int[]]$Ports,
  [switch]$Kill
)

foreach ($p in $Ports) {
  $procId = $null

  # Primary: Get-NetTCPConnection
  try {
    $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    if ($conns) {
      $procId = ($conns | Where-Object { $_.OwningProcess -gt 0 } |
                 Select-Object -First 1 -ExpandProperty OwningProcess)
    }
  } catch { }

  # Fallback: netstat (handles PID 0 ambiguity)
  if (-not $procId) {
    try {
      $line = netstat -aon | Select-String -Pattern "LISTENING\s+(\d+)" |
             Where-Object { $_ -match "[:\.]$p\s" } | Select-Object -First 1
      if ($line -and $line.Matches.Count -gt 0) {
        $procId = [int]$line.Matches[0].Groups[1].Value
      }
    } catch { }
  }

  if (-not $procId) { Write-Host "[ports] $($p): free"; continue }

  $proc = $null
  try { $proc = Get-Process -Id $procId -ErrorAction Stop } catch { }

  if ($Kill) {
    Write-Host "[ports] $($p): killing PID $procId $($proc.ProcessName)"
    try { Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue } catch { }
    Start-Sleep -Milliseconds 200
    Write-Host "[ports] $($p): freed"
  } else {
    Write-Host "[ports] $($p): PID $procId $($proc.ProcessName)"
  }
}
