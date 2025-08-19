# Inspect-BridgeCommunication.ps1
# Fast fingerprint - who's spawning BridgeCommunication.exe?
# Requires: PowerShell as Admin for full visibility

$ErrorActionPreference = 'SilentlyContinue'

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "BridgeCommunication.exe Process Inspector" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Grab all BridgeCommunication.exe processes and their parents
$procs = Get-CimInstance Win32_Process -Filter "Name='BridgeCommunication.exe'"

if (!$procs) { 
    Write-Host "No BridgeCommunication.exe processes found." -ForegroundColor Yellow
    exit 
}

Write-Host "Found $($procs.Count) BridgeCommunication.exe process(es)" -ForegroundColor Green
Write-Host ""

$ppids = $procs | Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine, ExecutablePath
$parents = @{}

foreach ($p in $ppids) {
    $ppid = [int]$p.ParentProcessId
    if (-not $parents.ContainsKey($ppid)) {
        $parents[$ppid] = Get-CimInstance Win32_Process -Filter "ProcessId=$ppid"
    }
}

# Enrich with parent info and file metadata
$report = foreach ($p in $ppids) {
    $parent = $parents[[int]$p.ParentProcessId]
    $path   = $p.ExecutablePath
    $vi     = if ($path) { (Get-Item $path).VersionInfo } else { $null }
    $sig    = if ($path) { Get-AuthenticodeSignature -FilePath $path } else { $null }
    
    [pscustomobject]@{
        ProcId         = $p.ProcessId
        PPID           = $p.ParentProcessId
        ParentName     = $parent.Name
        ParentCmd      = $parent.CommandLine
        Started        = ([Management.ManagementDateTimeConverter]::ToDateTime($p.CreationDate))
        EXE            = $path
        Product        = $vi.ProductName
        Company        = $vi.CompanyName
        FileDesc       = $vi.FileDescription
        FileVersion    = $vi.ProductVersion
        Signature      = $sig.SignerCertificate.Subject
        SigStatus      = $sig.Status
        CmdLine        = $p.CommandLine
    }
}

Write-Host "DETAILED PROCESS REPORT:" -ForegroundColor Yellow
Write-Host "------------------------" -ForegroundColor Yellow
$report | Sort-Object Started | Format-Table -AutoSize

Write-Host ""
Write-Host "SPAWNER ANALYSIS:" -ForegroundColor Yellow
Write-Host "-----------------" -ForegroundColor Yellow
# Quick aggregation: which parent is the spawner?
$report | Group-Object ParentName | Sort-Object Count -Descending |
    Select-Object Name, Count | Format-Table -AutoSize

# Save a CSV snapshot for the incident log
$log = "D:\Dev\kha\tools\diagnostics\BridgeComm_snapshot_{0:yyyyMMdd_HHmmss}.csv" -f (Get-Date)
$report | Export-Csv -NoTypeInformation -Path $log
Write-Host ""
Write-Host "Snapshot saved: $log" -ForegroundColor Cyan

# Additional analysis
Write-Host ""
Write-Host "MEMORY USAGE:" -ForegroundColor Yellow
Write-Host "-------------" -ForegroundColor Yellow
$memTotal = 0
foreach ($proc in $procs) {
    try {
        $p = Get-Process -Id $proc.ProcessId -ErrorAction SilentlyContinue
        if ($p) {
            $memMB = [math]::Round($p.WorkingSet64 / 1MB, 2)
            Write-Host "PID $($proc.ProcessId): $memMB MB" -ForegroundColor Gray
            $memTotal += $memMB
        }
    } catch {}
}
Write-Host "Total Memory: $memTotal MB" -ForegroundColor Magenta

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Analysis Complete" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan