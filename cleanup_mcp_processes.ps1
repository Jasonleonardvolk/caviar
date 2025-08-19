# PowerShell script to clean up MCP-related node processes
Write-Host "Cleaning up MCP-related node processes..." -ForegroundColor Yellow

# Get all node processes with MCP-related command lines
$mcpProcesses = Get-WmiObject Win32_Process -Filter "name='node.exe'" | Where-Object {
    $_.CommandLine -like "*@modelcontextprotocol*" -or 
    $_.CommandLine -like "*server-filesystem*" -or 
    $_.CommandLine -like "*server-memory*" -or 
    $_.CommandLine -like "*server-sequential-thinking*" -or
    $_.CommandLine -like "*server-github*"
}

if ($mcpProcesses) {
    Write-Host "Found $($mcpProcesses.Count) MCP-related node processes. Terminating..." -ForegroundColor Red
    foreach ($process in $mcpProcesses) {
        Write-Host "Killing process ID: $($process.ProcessId) - $($process.CommandLine)" -ForegroundColor Gray
        try {
            Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Host "Could not kill process $($process.ProcessId): $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    
    # Wait for processes to terminate
    Start-Sleep -Seconds 2
} else {
    Write-Host "No MCP-related node processes found." -ForegroundColor Green
}

# Show remaining node processes
Write-Host "`nRemaining node processes:" -ForegroundColor Cyan
Get-Process -Name "node" -ErrorAction SilentlyContinue | Format-Table Id, ProcessName, StartTime -AutoSize

Write-Host "`nMCP process cleanup completed." -ForegroundColor Green
Write-Host "You can now restart Claude Desktop safely." -ForegroundColor Yellow
