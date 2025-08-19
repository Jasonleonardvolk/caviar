# Enhanced PowerShell script to completely clean up node processes
Write-Host "=======================================" -ForegroundColor Red
Write-Host "EMERGENCY: KILLING ALL NODE PROCESSES" -ForegroundColor Red  
Write-Host "=======================================" -ForegroundColor Red
Write-Host ""

Write-Host "WARNING: This will terminate ALL node.exe processes on your system!" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to cancel, or any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Write-Host ""

# Function to get all node processes with details
function Get-NodeProcessDetails {
    try {
        $processes = Get-Process -Name "node" -ErrorAction SilentlyContinue
        if ($processes) {
            Write-Host "Current node processes:" -ForegroundColor Cyan
            $processes | ForEach-Object {
                $memoryMB = [math]::Round($_.WorkingSet64 / 1MB, 2)
                $runtime = if ($_.StartTime) { 
                    [math]::Round(((Get-Date) - $_.StartTime).TotalMinutes, 1) 
                } else { 
                    "Unknown" 
                }
                Write-Host "  PID: $($_.Id) | Memory: ${memoryMB}MB | Runtime: ${runtime}min" -ForegroundColor Gray
            }
            return $processes
        } else {
            Write-Host "No node processes found." -ForegroundColor Green
            return @()
        }
    } catch {
        Write-Host "Error getting process details: $($_.Exception.Message)" -ForegroundColor Red
        return @()
    }
}

# Get initial process list
$initialProcesses = Get-NodeProcessDetails
Write-Host ""

if ($initialProcesses.Count -eq 0) {
    Write-Host "✅ No node processes to kill." -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now start your MCP server safely." -ForegroundColor Green
    Read-Host "Press Enter to exit"
    exit 0
}

Write-Host "🔥 Terminating $($initialProcesses.Count) node processes..." -ForegroundColor Red

# First attempt: Graceful termination
foreach ($process in $initialProcesses) {
    try {
        Write-Host "  📤 Sending SIGTERM to PID $($process.Id)..." -ForegroundColor Yellow
        $process.CloseMainWindow() | Out-Null
        Start-Sleep -Milliseconds 500
        
        if (!$process.HasExited) {
            Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
        }
    } catch {
        Write-Host "    ⚠️  Could not gracefully stop PID $($process.Id): $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "⏳ Waiting 3 seconds for graceful termination..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Check what's left
$remainingProcesses = Get-Process -Name "node" -ErrorAction SilentlyContinue

if ($remainingProcesses) {
    Write-Host "💀 Force killing remaining $($remainingProcesses.Count) processes..." -ForegroundColor Red
    
    foreach ($process in $remainingProcesses) {
        try {
            Write-Host "  🔨 Force killing PID $($process.Id)..." -ForegroundColor Red
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Host "    ⚠️  Could not force kill PID $($process.Id): $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    
    Start-Sleep -Seconds 2
}

# Final verification
Write-Host ""
Write-Host "🔍 Final verification..." -ForegroundColor Cyan
$finalProcesses = Get-Process -Name "node" -ErrorAction SilentlyContinue

if ($finalProcesses) {
    Write-Host "⚠️  WARNING: $($finalProcesses.Count) node processes still running:" -ForegroundColor Red
    $finalProcesses | ForEach-Object {
        $memoryMB = [math]::Round($_.WorkingSet64 / 1MB, 2)
        Write-Host "  PID: $($_.Id) | Memory: ${memoryMB}MB" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "🔧 You may need to:" -ForegroundColor Yellow
    Write-Host "  1. Close any applications using Node.js" -ForegroundColor Yellow
    Write-Host "  2. Restart your computer if processes are stuck" -ForegroundColor Yellow
    Write-Host "  3. Check Task Manager for additional details" -ForegroundColor Yellow
} else {
    Write-Host "✅ SUCCESS: All node processes terminated!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🎉 System is clean. You can now:" -ForegroundColor Green
    Write-Host "  • Run: node cleanup-and-start.js" -ForegroundColor Green
    Write-Host "  • Start your MCP server safely" -ForegroundColor Green
    Write-Host "  • Use the new process management tools" -ForegroundColor Green
}

Write-Host ""
Write-Host "📊 Process Summary:" -ForegroundColor Cyan
Write-Host "  Initial processes: $($initialProcesses.Count)" -ForegroundColor Gray
Write-Host "  Final processes: $(if ($finalProcesses) { $finalProcesses.Count } else { 0 })" -ForegroundColor Gray
Write-Host "  Processes killed: $($initialProcesses.Count - $(if ($finalProcesses) { $finalProcesses.Count } else { 0 }))" -ForegroundColor Gray

Write-Host ""
Write-Host "Emergency cleanup completed!" -ForegroundColor Green
Read-Host "Press Enter to exit"
