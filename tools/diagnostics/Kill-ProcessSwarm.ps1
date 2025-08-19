# Kill-ProcessSwarm.ps1
# Generic tool to kill process swarms by name
# Useful for dealing with any runaway process multiplication

param(
    [Parameter(Mandatory=$true)]
    [string]$ProcessName,
    
    [switch]$Force,
    [switch]$WaitForTermination,
    [int]$MaxWaitSeconds = 10,
    [switch]$Verbose
)

# Remove .exe if provided
$ProcessName = $ProcessName -replace '\.exe$', ''

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Process Swarm Killer" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Target Process: $ProcessName" -ForegroundColor Yellow
Write-Host ""

# Get initial count
$processes = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue

if (-not $processes) {
    Write-Host "No processes found with name: $ProcessName" -ForegroundColor Green
    exit 0
}

$initialCount = ($processes | Measure-Object).Count
Write-Host "Found $initialCount process(es) to terminate" -ForegroundColor Red

# Show details if verbose
if ($Verbose) {
    Write-Host ""
    Write-Host "Process Details:" -ForegroundColor Yellow
    $processes | ForEach-Object {
        $memMB = [math]::Round($_.WorkingSet64 / 1MB, 2)
        $cpuTime = $_.TotalProcessorTime
        Write-Host "  PID: $($_.Id) | Memory: $memMB MB | CPU Time: $cpuTime | Start: $($_.StartTime)" -ForegroundColor Gray
    }
    
    # Calculate total memory
    $totalMemMB = [math]::Round(($processes | Measure-Object WorkingSet64 -Sum).Sum / 1MB, 2)
    Write-Host ""
    Write-Host "Total Memory Usage: $totalMemMB MB" -ForegroundColor Magenta
}

# Confirmation unless Force is specified
if (-not $Force) {
    Write-Host ""
    Write-Host "Are you sure you want to terminate all $initialCount $ProcessName process(es)?" -ForegroundColor Yellow
    $confirmation = Read-Host "Type 'YES' to continue, anything else to cancel"
    
    if ($confirmation -ne 'YES') {
        Write-Host "Operation cancelled by user" -ForegroundColor Yellow
        exit 0
    }
}

Write-Host ""
Write-Host "Terminating processes..." -ForegroundColor Yellow

# Kill processes
$killCount = 0
$failCount = 0

foreach ($proc in $processes) {
    try {
        $proc | Stop-Process -Force -ErrorAction Stop
        $killCount++
        if ($Verbose) {
            Write-Host "  Killed PID: $($proc.Id)" -ForegroundColor Green
        }
    } catch {
        $failCount++
        if ($Verbose) {
            Write-Host "  Failed to kill PID: $($proc.Id) - $_" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "Initial termination complete:" -ForegroundColor Cyan
Write-Host "  Killed: $killCount" -ForegroundColor Green
Write-Host "  Failed: $failCount" -ForegroundColor Red

# Wait for termination if requested
if ($WaitForTermination) {
    Write-Host ""
    Write-Host "Waiting for processes to fully terminate..." -ForegroundColor Yellow
    
    $elapsed = 0
    $checkInterval = 1
    
    while ($elapsed -lt $MaxWaitSeconds) {
        Start-Sleep -Seconds $checkInterval
        $elapsed += $checkInterval
        
        $remaining = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
        
        if (-not $remaining) {
            Write-Host "All processes terminated successfully!" -ForegroundColor Green
            break
        } else {
            $remainCount = ($remaining | Measure-Object).Count
            Write-Host "  Still running: $remainCount process(es) (waited $elapsed seconds)" -ForegroundColor Yellow
            
            # Try to kill remaining processes
            if ($elapsed -ge ($MaxWaitSeconds / 2)) {
                Write-Host "  Attempting to force-kill remaining processes..." -ForegroundColor Red
                $remaining | Stop-Process -Force -ErrorAction SilentlyContinue
            }
        }
    }
    
    # Final check
    $finalRemaining = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
    if ($finalRemaining) {
        $finalCount = ($finalRemaining | Measure-Object).Count
        Write-Host ""
        Write-Host "WARNING: $finalCount process(es) could not be terminated" -ForegroundColor Red
        Write-Host "These processes may be:" -ForegroundColor Yellow
        Write-Host "  - System-protected" -ForegroundColor Gray
        Write-Host "  - Required by Windows" -ForegroundColor Gray
        Write-Host "  - Respawning immediately" -ForegroundColor Gray
        
        if ($Verbose) {
            Write-Host ""
            Write-Host "Remaining PIDs:" -ForegroundColor Red
            $finalRemaining | ForEach-Object {
                Write-Host "  PID: $($_.Id)" -ForegroundColor Gray
            }
        }
    }
}

# Check for immediate respawn
Write-Host ""
Write-Host "Checking for respawns..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

$respawned = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
if ($respawned) {
    $respawnCount = ($respawned | Measure-Object).Count
    Write-Host "WARNING: $respawnCount new process(es) have already respawned!" -ForegroundColor Red
    Write-Host ""
    Write-Host "This indicates a persistence mechanism is active." -ForegroundColor Yellow
    Write-Host "Recommended actions:" -ForegroundColor Yellow
    Write-Host "  1. Run Hunt-BridgePersistence.ps1 to find the spawner" -ForegroundColor White
    Write-Host "  2. Disable the parent service/task/application" -ForegroundColor White
    Write-Host "  3. Run Quarantine script for permanent removal" -ForegroundColor White
} else {
    Write-Host "No respawns detected - processes appear to be terminated" -ForegroundColor Green
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Process Swarm Killer Complete" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# Return exit code based on success
if ($failCount -gt 0 -or $respawned) {
    exit 1
} else {
    exit 0
}