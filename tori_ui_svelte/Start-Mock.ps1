# Start-Mock.ps1 - Simple mock server starter
# No complex path resolution - just works

Write-Host "Starting Mock Server..." -ForegroundColor Cyan

# Go to the right directory
Set-Location D:\Dev\kha\tori_ui_svelte

# Check if build exists
if (-not (Test-Path "build\index.js")) {
    Write-Host "Build not found. Building now..." -ForegroundColor Yellow
    pnpm run build
}

# Set environment
$env:IRIS_USE_MOCKS = "1"
$env:IRIS_ALLOW_UNAUTH = "1"
$env:IRIS_STORAGE_TYPE = "local"
$env:PORT = "3000"

# Check if port is in use and kill node processes
$tcp = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue | 
       Where-Object { $_.OwningProcess -gt 0 } | 
       Select-Object -First 1

if ($tcp) {
    $procId = $tcp.OwningProcess
    if ($procId -gt 0) {
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "Port 3000 is in use by $($proc.ProcessName) (PID: $procId)" -ForegroundColor Yellow
            if ($proc.ProcessName -eq "node") {
                Write-Host "Killing existing node process..." -ForegroundColor Yellow
                Stop-Process -Id $procId -Force
                Start-Sleep -Seconds 1
            } else {
                Write-Host "Warning: Port 3000 is used by $($proc.ProcessName), not node." -ForegroundColor Red
                Write-Host "Please stop that process manually or use a different port." -ForegroundColor Yellow
                exit 1
            }
        }
    }
}

# Alternative: Kill all node processes in this directory
$nodeProcs = Get-Process node -ErrorAction SilentlyContinue | 
             Where-Object { $_.Path -and ($_.Path -like "*D:\Dev\kha*") }
if ($nodeProcs) {
    Write-Host "Found node processes in kha directory, killing them..." -ForegroundColor Yellow
    $nodeProcs | Stop-Process -Force
    Start-Sleep -Seconds 1
}

Write-Host ""
Write-Host "Environment:" -ForegroundColor Yellow
Write-Host "  IRIS_USE_MOCKS = 1" -ForegroundColor Gray
Write-Host "  IRIS_ALLOW_UNAUTH = 1" -ForegroundColor Gray
Write-Host "  PORT = 3000" -ForegroundColor Gray
Write-Host ""

# Start server
Write-Host "Starting server on http://localhost:3000" -ForegroundColor Green
node build\index.js
