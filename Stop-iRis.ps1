# Stop-iRis.ps1
# Stops all iRis services cleanly

Write-Host "`nStopping iRis services..." -ForegroundColor Yellow

$processes = @(
    @{Name = "node"; Ports = @(5173, 3000)},
    @{Name = "uvicorn"; Ports = @(7401)},
    @{Name = "python"; Ports = @(7401)}
)

$stopped = 0

foreach ($proc in $processes) {
    $found = Get-Process -Name $proc.Name -ErrorAction SilentlyContinue
    if ($found) {
        $found | Stop-Process -Force
        Write-Host "  Stopped $($proc.Name)" -ForegroundColor Green
        $stopped++
    }
}

# Also check by port
$ports = @(5173, 3000, 7401)
foreach ($port in $ports) {
    $process = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | 
               Select-Object -ExpandProperty OwningProcess -Unique
    if ($process) {
        Stop-Process -Id $process -Force -ErrorAction SilentlyContinue
        Write-Host "  Stopped process on port $port" -ForegroundColor Green
        $stopped++
    }
}

if ($stopped -gt 0) {
    Write-Host "`n✓ iRis services stopped" -ForegroundColor Green
} else {
    Write-Host "`n○ No iRis services were running" -ForegroundColor Gray
}