# Start Backend on Port 8002 - Step by Step
Write-Host "Step 2: Starting Soliton Backend on Port 8002..." -ForegroundColor Green

# Check if port 8002 is available
$portInUse = Get-NetTCPConnection -LocalPort 8002 -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "Port 8002 is already in use. Stopping existing process..." -ForegroundColor Yellow
    $processes = Get-Process | Where-Object {$_.ProcessName -eq "python" -or $_.ProcessName -eq "node"}
    foreach ($proc in $processes) {
        try {
            $proc.Kill()
            Write-Host "Stopped process: $($proc.ProcessName) (ID: $($proc.Id))" -ForegroundColor Yellow
        } catch {
            Write-Host "Could not stop process $($proc.ProcessName)" -ForegroundColor Red
        }
    }
}

Write-Host "Starting enhanced launcher on port 8002..." -ForegroundColor Cyan
# Start the Python enhanced launcher
python enhanced_launcher.py --port 8002
