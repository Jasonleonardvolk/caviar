# Nuclear option - kill ALL node processes
Write-Host "FORCE KILLING ALL NODE PROCESSES" -ForegroundColor Red -BackgroundColor Yellow
Write-Host ""

$nodeProcesses = Get-Process node -ErrorAction SilentlyContinue
if ($nodeProcesses) {
    Write-Host "Found $($nodeProcesses.Count) node process(es):" -ForegroundColor Yellow
    foreach ($proc in $nodeProcesses) {
        Write-Host "  Killing: $($proc.ProcessName) (PID: $($proc.Id))" -ForegroundColor Red
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host ""
    Write-Host "All node processes terminated!" -ForegroundColor Green
} else {
    Write-Host "No node processes found." -ForegroundColor Green
}

Write-Host ""
Write-Host "Checking port 3000..." -ForegroundColor Yellow
$port3000 = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
if (-not $port3000) {
    Write-Host "Port 3000 is FREE!" -ForegroundColor Green
} else {
    Write-Host "Port 3000 still has connections. They should clear soon." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "You can now run: .\tools\runtime\Run-Mock.ps1" -ForegroundColor Cyan
