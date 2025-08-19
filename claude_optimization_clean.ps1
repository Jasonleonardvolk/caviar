# Claude Desktop MCP Process Optimization Script
param(
    [int]$MonitorDuration = 60,
    [int]$PollingInterval = 3
)

$banner = "=" * 60
Write-Host "Claude Desktop MCP Optimization Protocol" -ForegroundColor Cyan
Write-Host $banner -ForegroundColor Cyan

# Phase 1: Baseline Analysis
Write-Host "`nPHASE 1: Baseline System Analysis" -ForegroundColor Yellow
$initialProcesses = Get-Process node -ErrorAction SilentlyContinue
$initialCount = ($initialProcesses | Measure-Object).Count
$initialMemory = if ($initialProcesses) { ($initialProcesses | Measure-Object WorkingSet -Sum).Sum / 1MB } else { 0 }

Write-Host "Initial Node.js processes: $initialCount"
Write-Host "Initial memory usage: $([math]::Round($initialMemory, 1))MB"

if ($initialProcesses) {
    Write-Host "Process details:" -ForegroundColor Gray
    $initialProcesses | Format-Table @{Name="PID"; Expression={$_.Id}}, @{Name="Memory(MB)"; Expression={[math]::Round($_.WorkingSet/1MB, 1)}}, @{Name="StartTime"; Expression={$_.StartTime}} -AutoSize
}

# Phase 2: Configuration Validation
Write-Host "`nPHASE 2: Configuration Validation" -ForegroundColor Yellow
$configPath = "$env:APPDATA\Claude\claude_desktop_config.json"

if (Test-Path $configPath) {
    $configContent = Get-Content $configPath -Raw | ConvertFrom-Json
    $mcpTimeout = $configContent.mcpTimeout
    $connectionTimeout = $configContent.connectionTimeout
    $maxReconnects = $configContent.maxReconnectAttempts
    
    Write-Host "Configuration analysis:"
    $mcpStatus = if($mcpTimeout -eq 30000){"OPTIMIZED"}else{"TOXIC"}
    $connStatus = if($connectionTimeout -eq 10000){"OPTIMIZED"}else{"TOXIC"}
    $reconnStatus = if($maxReconnects -eq 3){"OPTIMIZED"}else{"TOXIC"}
    
    Write-Host "  mcpTimeout: $mcpTimeout ms [$mcpStatus]"
    Write-Host "  connectionTimeout: $connectionTimeout ms [$connStatus]"
    Write-Host "  maxReconnectAttempts: $maxReconnects [$reconnStatus]"
    
    if ($mcpTimeout -ne 30000 -or $connectionTimeout -ne 10000) {
        Write-Host "Configuration not optimized - restart may not resolve issues" -ForegroundColor Red
    } else {
        Write-Host "Configuration optimized - proceeding with restart" -ForegroundColor Green
    }
} else {
    Write-Host "Configuration file not found at $configPath" -ForegroundColor Red
    exit 1
}

# Phase 3: Claude Termination
Write-Host "`nPHASE 3: Claude Desktop Termination" -ForegroundColor Yellow
$claudeProcesses = Get-Process -Name "*Claude*" -ErrorAction SilentlyContinue

if ($claudeProcesses) {
    Write-Host "Found Claude processes:"
    $claudeProcesses | Format-Table Name, Id, @{Name="Memory(MB)"; Expression={[math]::Round($_.WorkingSet/1MB, 1)}} -AutoSize
    
    Write-Host "Attempting graceful termination..."
    $claudeProcesses | ForEach-Object { 
        try {
            $_.CloseMainWindow()
            Write-Host "  Sent close signal to PID $($_.Id)"
        } catch {
            Write-Host "  Failed to close PID $($_.Id): $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    
    Write-Host "Waiting for graceful shutdown (10 seconds)..."
    Start-Sleep -Seconds 10
    
    $remainingClaude = Get-Process -Name "*Claude*" -ErrorAction SilentlyContinue
    if ($remainingClaude) {
        Write-Host "Force terminating remaining Claude processes..." -ForegroundColor Yellow
        $remainingClaude | Stop-Process -Force
        Start-Sleep -Seconds 2
    }
} else {
    Write-Host "No Claude processes found - proceeding to Node.js cleanup"
}

# Phase 4: Process Cleanup Analysis
Write-Host "`nPHASE 4: Node.js Process Cleanup Analysis" -ForegroundColor Yellow
Start-Sleep -Seconds 3

$postShutdownProcesses = Get-Process node -ErrorAction SilentlyContinue
$postShutdownCount = ($postShutdownProcesses | Measure-Object).Count
$postShutdownMemory = if ($postShutdownProcesses) { ($postShutdownProcesses | Measure-Object WorkingSet -Sum).Sum / 1MB } else { 0 }

Write-Host "Post-shutdown Node.js processes: $postShutdownCount"
Write-Host "Post-shutdown memory usage: $([math]::Round($postShutdownMemory, 1))MB"

$processReduction = $initialCount - $postShutdownCount
$memoryReduction = $initialMemory - $postShutdownMemory

Write-Host "Cleanup effectiveness:"
if ($initialCount -gt 0) {
    $processPercent = [math]::Round($processReduction/$initialCount*100, 1)
    Write-Host "  Process reduction: $processReduction processes ($processPercent%)"
}
if ($initialMemory -gt 0) {
    $memoryPercent = [math]::Round($memoryReduction/$initialMemory*100, 1)
    Write-Host "  Memory reclaimed: $([math]::Round($memoryReduction, 1))MB ($memoryPercent%)"
}

if ($postShutdownProcesses) {
    Write-Host "Remaining Node.js processes:" -ForegroundColor Yellow
    $postShutdownProcesses | Format-Table @{Name="PID"; Expression={$_.Id}}, @{Name="Memory(MB)"; Expression={[math]::Round($_.WorkingSet/1MB, 1)}}, @{Name="StartTime"; Expression={$_.StartTime}}, @{Name="Age(min)"; Expression={[math]::Round(((Get-Date) - $_.StartTime).TotalMinutes, 1)}} -AutoSize
}

# Phase 5: Application Restart
Write-Host "`nPHASE 5: Claude Desktop Restart" -ForegroundColor Yellow

$claudePaths = @(
    "$env:LOCALAPPDATA\Programs\Claude\Claude.exe",
    "$env:PROGRAMFILES\Claude\Claude.exe",
    "$env:PROGRAMFILES(X86)\Claude\Claude.exe"
)

$claudeExe = $null
foreach ($path in $claudePaths) {
    if (Test-Path $path) {
        $claudeExe = $path
        break
    }
}

if (-not $claudeExe) {
    Write-Host "Claude executable not found in standard locations" -ForegroundColor Red
    Write-Host "Please manually restart Claude Desktop" -ForegroundColor Yellow
    Write-Host "Continuing with monitoring..." -ForegroundColor Gray
} else {
    Write-Host "Starting Claude Desktop from: $claudeExe"
    try {
        Start-Process -FilePath $claudeExe -WindowStyle Normal
        Write-Host "Claude Desktop restart initiated" -ForegroundColor Green
    } catch {
        Write-Host "Failed to start Claude: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Please manually restart Claude Desktop" -ForegroundColor Yellow
    }
}

# Phase 6: Real-time Monitoring
Write-Host "`nPHASE 6: Real-time Performance Monitoring" -ForegroundColor Yellow
Write-Host "Monitoring for $MonitorDuration seconds (polling every $PollingInterval seconds)..."
Write-Host "Time     | Processes | Memory(MB) | Delta | Status" -ForegroundColor Gray
$separator = "-" * 55
Write-Host $separator -ForegroundColor Gray

$monitorStart = Get-Date
$maxProcesses = 0
$maxMemory = 0
$stabilityWindow = @()

while (((Get-Date) - $monitorStart).TotalSeconds -lt $MonitorDuration) {
    $currentProcesses = Get-Process node -ErrorAction SilentlyContinue
    $currentCount = ($currentProcesses | Measure-Object).Count
    $currentMemory = if ($currentProcesses) { ($currentProcesses | Measure-Object WorkingSet -Sum).Sum / 1MB } else { 0 }
    
    $maxProcesses = [math]::Max($maxProcesses, $currentCount)
    $maxMemory = [math]::Max($maxMemory, $currentMemory)
    
    $processDelta = $currentCount - $postShutdownCount
    $memoryDelta = $currentMemory - $postShutdownMemory
    
    $stabilityWindow += $currentCount
    if ($stabilityWindow.Count -gt 10) { $stabilityWindow = $stabilityWindow[-10..-1] }
    
    $stabilityVariance = if ($stabilityWindow.Count -gt 1) {
        $mean = ($stabilityWindow | Measure-Object -Average).Average
        ($stabilityWindow | ForEach-Object { [math]::Pow($_ - $mean, 2) } | Measure-Object -Sum).Sum / ($stabilityWindow.Count - 1)
    } else { 0 }
    
    $status = switch ($true) {
        ($currentCount -gt 5) { "HIGH" }
        ($currentMemory -gt 100) { "MEM" }
        ($stabilityVariance -lt 0.1) { "STABLE" }
        default { "MONITOR" }
    }
    
    $timestamp = (Get-Date).ToString("HH:mm:ss")
    $deltaStr = $processDelta.ToString("+0;-0;0").PadLeft(5)
    Write-Host "$timestamp |     $currentCount     | $([math]::Round($currentMemory, 1).ToString().PadLeft(7)) | $deltaStr | $status"
    
    Start-Sleep -Seconds $PollingInterval
}

# Phase 7: Final Analysis
Write-Host "`nPHASE 7: Performance Analysis & Recommendations" -ForegroundColor Yellow

$finalProcesses = Get-Process node -ErrorAction SilentlyContinue
$finalCount = ($finalProcesses | Measure-Object).Count
$finalMemory = if ($finalProcesses) { ($finalProcesses | Measure-Object WorkingSet -Sum).Sum / 1MB } else { 0 }

Write-Host "`nPerformance Summary:"
Write-Host "  Peak processes during monitoring: $maxProcesses"
Write-Host "  Peak memory during monitoring: $([math]::Round($maxMemory, 1))MB"
Write-Host "  Final process count: $finalCount"
Write-Host "  Final memory usage: $([math]::Round($finalMemory, 1))MB"

$overallProcessReduction = $initialCount - $finalCount
$overallMemoryReduction = $initialMemory - $finalMemory
$processEfficiency = if ($initialCount -gt 0) { [math]::Round($overallProcessReduction / $initialCount * 100, 1) } else { 0 }
$memoryEfficiency = if ($initialMemory -gt 0) { [math]::Round($overallMemoryReduction / $initialMemory * 100, 1) } else { 0 }

Write-Host "`nOptimization Effectiveness:"
Write-Host "  Process reduction: $overallProcessReduction processes ($processEfficiency%)"
Write-Host "  Memory reduction: $([math]::Round($overallMemoryReduction, 1))MB ($memoryEfficiency%)"

$healthScore = 100
if ($finalCount -gt 3) { $healthScore -= 20 }
if ($finalMemory -gt 75) { $healthScore -= 15 }
if ($maxProcesses -gt 5) { $healthScore -= 10 }
if ($stabilityVariance -gt 1) { $healthScore -= 10 }

Write-Host "`nSystem Health Score: $healthScore/100"

$healthStatus = switch ($true) {
    ($healthScore -ge 90) { "EXCELLENT - Configuration optimization successful" }
    ($healthScore -ge 75) { "GOOD - Minor performance concerns" }
    ($healthScore -ge 60) { "FAIR - Consider additional optimization" }
    default { "POOR - Configuration issues persist" }
}

Write-Host "Status: $healthStatus"

Write-Host "`nRecommendations:"
if ($finalCount -le 2 -and $finalMemory -le 50) {
    Write-Host "  System operating within optimal parameters"
    Write-Host "  MCP timeout configuration successful"
    Write-Host "  Process leak resolution achieved"
} else {
    Write-Host "  Consider investigating remaining high-resource processes:"
    if ($finalProcesses) {
        $finalProcesses | Where-Object { $_.WorkingSet/1MB -gt 25 } | ForEach-Object {
            $age = [math]::Round(((Get-Date) - $_.StartTime).TotalMinutes, 1)
            $mem = [math]::Round($_.WorkingSet/1MB, 1)
            Write-Host "    PID $($_.Id): ${mem}MB - Age: $age minutes"
        }
    }
}

Write-Host "`nProtocol Complete - Monitor system for sustained performance" -ForegroundColor Green
Write-Host $banner -ForegroundColor Cyan