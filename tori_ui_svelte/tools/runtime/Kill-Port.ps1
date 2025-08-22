param(
    [int]$Port = 3000
)

Write-Host "Checking for processes on port $Port..." -ForegroundColor Yellow

# Method 1: Using Get-NetTCPConnection
try {
    $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if ($connections) {
        $processIds = $connections.OwningProcess | Select-Object -Unique
        foreach ($procId in $processIds) {
            if ($procId -gt 0) {
                $process = Get-Process -Id $procId -ErrorAction SilentlyContinue
                if ($process) {
                    Write-Host "Found process: $($process.ProcessName) (PID: $procId)" -ForegroundColor Cyan
                    Write-Host "Killing process..." -ForegroundColor Red
                    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                    Write-Host "Process killed successfully" -ForegroundColor Green
                }
            }
        }
    }
} catch {
    Write-Host "Could not use Get-NetTCPConnection, trying netstat..." -ForegroundColor Yellow
}

# Method 2: Using netstat as fallback
$netstatOutput = netstat -ano | Select-String ":$Port.*LISTENING"
if ($netstatOutput) {
    foreach ($line in $netstatOutput) {
        $procId = $line -split '\s+' | Select-Object -Last 1
        if ($procId -match '^\d+$' -and $procId -gt 0) {
            $process = Get-Process -Id $procId -ErrorAction SilentlyContinue
            if ($process) {
                Write-Host "Found process via netstat: $($process.ProcessName) (PID: $procId)" -ForegroundColor Cyan
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                Write-Host "Process killed" -ForegroundColor Green
            }
        }
    }
}

# Kill any node processes in the kha directory
$nodeProcesses = Get-Process node -ErrorAction SilentlyContinue | 
    Where-Object { $_.Path -and ($_.Path -like "*D:\Dev\kha*") }

if ($nodeProcesses) {
    Write-Host "`nFound node processes in D:\Dev\kha:" -ForegroundColor Yellow
    foreach ($proc in $nodeProcesses) {
        Write-Host "  - $($proc.ProcessName) (PID: $($proc.Id))" -ForegroundColor Cyan
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "All node processes killed" -ForegroundColor Green
}

# Wait a moment for ports to be released
Start-Sleep -Seconds 2

# Verify port is free
$stillInUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if (-not $stillInUse) {
    Write-Host "`nPort $Port is now free!" -ForegroundColor Green
} else {
    Write-Host "`nWarning: Port $Port may still be in use. Try running this script again." -ForegroundColor Yellow
}
