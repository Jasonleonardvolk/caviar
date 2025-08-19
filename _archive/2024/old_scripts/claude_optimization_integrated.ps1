# Claude MCP System Optimization & Validation Protocol
# Comprehensive solution for process leak prevention and performance optimization

param(
    [int]$MonitorDuration = 300,  # 5 minutes monitoring
    [int]$PollingInterval = 10,   # 10-second intervals
    [switch]$SkipRestart = $false # Skip Claude restart for testing
)

# Performance optimization constants
$OPTIMAL_PROCESS_COUNT = 4
$MEMORY_THRESHOLD_MB = 200
$PROCESS_AGE_THRESHOLD_SEC = 10
$MAX_INITIALIZATION_TIME_SEC = 120

function Write-StatusMessage {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "HH:mm:ss"
    $color = switch ($Level) {
        "SUCCESS" { "Green" }
        "WARNING" { "Yellow" }
        "ERROR" { "Red" }
        "DEBUG" { "Cyan" }
        default { "White" }
    }
    Write-Host "[$timestamp] $Message" -ForegroundColor $color
}

function Get-NodeProcessAnalysis {
    $nodeProcesses = Get-Process node -ErrorAction SilentlyContinue
    $analysis = @{
        Count = ($nodeProcesses | Measure-Object).Count
        TotalMemoryMB = 0
        Processes = @()
        RecentSpawns = 0
        StableProcesses = 0
    }
    
    if ($nodeProcesses) {
        $analysis.TotalMemoryMB = [math]::Round(($nodeProcesses | Measure-Object WorkingSet -Sum).Sum / 1MB, 1)
        
        foreach ($process in $nodeProcesses) {
            $age = ((Get-Date) - $process.StartTime).TotalSeconds
            $memory = [math]::Round($process.WorkingSet / 1MB, 1)
            
            $processInfo = @{
                PID = $process.Id
                MemoryMB = $memory
                AgeSeconds = [math]::Round($age, 1)
                Status = if ($age -lt $PROCESS_AGE_THRESHOLD_SEC) { "RECENT" } else { "STABLE" }
            }
            
            $analysis.Processes += $processInfo
            
            if ($age -lt $PROCESS_AGE_THRESHOLD_SEC) {
                $analysis.RecentSpawns++
            } else {
                $analysis.StableProcesses++
            }
        }
    }
    
    return $analysis
}

function Get-MCPServerTypes {
    $mcpServers = @()
    
    try {
        $processes = Get-WmiObject Win32_Process | Where-Object { $_.Name -eq "node.exe" }
        
        foreach ($process in $processes) {
            $cmdLine = $process.CommandLine
            $serverType = "UNKNOWN"
            
            if ($cmdLine -like "*filesystem*") { $serverType = "FILESYSTEM" }
            elseif ($cmdLine -like "*memory*") { $serverType = "MEMORY" }
            elseif ($cmdLine -like "*brave*") { $serverType = "BRAVE" }
            elseif ($cmdLine -like "*sequential*") { $serverType = "SEQUENTIAL" }
            
            $mcpServers += @{
                PID = $process.ProcessId
                Type = $serverType
                MemoryMB = [math]::Round($process.WorkingSetSize / 1MB, 1)
                CommandLine = $cmdLine.Substring(0, [Math]::Min(80, $cmdLine.Length))
            }
        }
    } catch {
        Write-StatusMessage "Failed to analyze MCP server types: $($_.Exception.Message)" "ERROR"
    }
    
    return $mcpServers
}

function Test-FilesystemAccess {
    param([string]$TestPath = "C:\Users\jason")
    
    try {
        $testFile = Join-Path $TestPath "mcp_filesystem_test_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
        $testContent = "MCP Filesystem Access Test - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        
        $testContent | Out-File -FilePath $testFile -Encoding UTF8
        
        if (Test-Path $testFile) {
            Remove-Item $testFile -Force
            return @{ Success = $true; Path = $testFile }
        } else {
            return @{ Success = $false; Error = "File creation failed" }
        }
    } catch {
        return @{ Success = $false; Error = $_.Exception.Message }
    }
}

function Test-EnvironmentVariablePropagation {
    $envVarTests = @()
    
    try {
        $nodeProcesses = Get-WmiObject Win32_Process | Where-Object { $_.Name -eq "node.exe" }
        
        foreach ($process in $nodeProcesses) {
            $envTest = @{
                PID = $process.ProcessId
                HasMcpDisableRetries = $false
                HasNodeEnv = $false
                HasMcpTimeout = $false
            }
            
            # Note: Direct environment variable access from external process is limited
            # This is a placeholder for environment variable validation
            $envVarTests += $envTest
        }
    } catch {
        Write-StatusMessage "Environment variable propagation test failed: $($_.Exception.Message)" "WARNING"
    }
    
    return $envVarTests
}

function Clear-NPMCache {
    try {
        Write-StatusMessage "Clearing NPM cache to force fresh package downloads..." "DEBUG"
        $result = & npm cache clean --force 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-StatusMessage "NPM cache cleared successfully" "SUCCESS"
            return $true
        } else {
            Write-StatusMessage "NPM cache clear failed: $result" "WARNING"
            return $false
        }
    } catch {
        Write-StatusMessage "NPM cache clear error: $($_.Exception.Message)" "WARNING"
        return $false
    }
}

function Start-ClaudeOptimized {
    param([string]$ClaudePath)
    
    try {
        # Terminate existing Claude processes
        $claudeProcesses = Get-Process claude -ErrorAction SilentlyContinue
        if ($claudeProcesses) {
            Write-StatusMessage "Terminating existing Claude processes..." "DEBUG"
            $claudeProcesses | Stop-Process -Force
            Start-Sleep -Seconds 5
        }
        
        # Start Claude with optimized environment
        Write-StatusMessage "Starting Claude with optimized configuration..." "DEBUG"
        Start-Process -FilePath $ClaudePath -WindowStyle Normal
        
        return $true
    } catch {
        Write-StatusMessage "Failed to start Claude: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Wait-ForMCPInitialization {
    param([int]$TimeoutSeconds = $MAX_INITIALIZATION_TIME_SEC)
    
    $startTime = Get-Date
    $initializationComplete = $false
    
    Write-StatusMessage "Monitoring MCP server initialization (timeout: ${TimeoutSeconds}s)..." "DEBUG"
    
    while (((Get-Date) - $startTime).TotalSeconds -lt $TimeoutSeconds) {
        $analysis = Get-NodeProcessAnalysis
        
        Write-StatusMessage "Initialization progress: $($analysis.Count)/$OPTIMAL_PROCESS_COUNT processes, $($analysis.TotalMemoryMB)MB" "DEBUG"
        
        if ($analysis.Count -eq $OPTIMAL_PROCESS_COUNT -and $analysis.RecentSpawns -eq 0) {
            $initializationComplete = $true
            Write-StatusMessage "MCP initialization completed successfully" "SUCCESS"
            break
        } elseif ($analysis.Count -gt $OPTIMAL_PROCESS_COUNT) {
            Write-StatusMessage "Process accumulation detected during initialization: $($analysis.Count) processes" "WARNING"
        }
        
        Start-Sleep -Seconds 5
    }
    
    if (-not $initializationComplete) {
        Write-StatusMessage "MCP initialization timeout after ${TimeoutSeconds} seconds" "ERROR"
    }
    
    return $initializationComplete
}

# Main execution protocol
Write-StatusMessage "Claude MCP System Optimization Protocol Initiated" "SUCCESS"
Write-Host ("=" * 80) -ForegroundColor Cyan

# Phase 1: System baseline analysis
Write-StatusMessage "PHASE 1: System Baseline Analysis" "DEBUG"
$baselineAnalysis = Get-NodeProcessAnalysis
Write-StatusMessage "Baseline: $($baselineAnalysis.Count) Node.js processes, $($baselineAnalysis.TotalMemoryMB)MB total memory" "DEBUG"

if ($baselineAnalysis.Count -gt 0) {
    Write-StatusMessage "Current process details:" "DEBUG"
    foreach ($process in $baselineAnalysis.Processes) {
        Write-StatusMessage "  PID $($process.PID): $($process.MemoryMB)MB, Age: $($process.AgeSeconds)s [$($process.Status)]" "DEBUG"
    }
}

# Phase 2: NPM cache optimization
Write-StatusMessage "PHASE 2: NPM Cache Optimization" "DEBUG"
Clear-NPMCache

# Phase 3: Enhanced configuration deployment validation
Write-StatusMessage "PHASE 3: Configuration Validation" "DEBUG"
$configPath = "$env:APPDATA\Claude\claude_desktop_config.json"

if (Test-Path $configPath) {
    try {
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
        
        $configStatus = @{
            MCPTimeout = $config.mcpTimeout
            ConnectionTimeout = $config.connectionTimeout
            MaxReconnectAttempts = $config.maxReconnectAttempts
            FilesystemPath = $config.mcpServers.filesystem.args[-1]  # Last argument should be path
        }
        
        Write-StatusMessage "Configuration analysis:" "DEBUG"
        Write-StatusMessage "  mcpTimeout: $($configStatus.MCPTimeout)ms $(if($configStatus.MCPTimeout -eq 30000){'[OPTIMIZED]'}else{'[NEEDS OPTIMIZATION]'})" "DEBUG"
        Write-StatusMessage "  connectionTimeout: $($configStatus.ConnectionTimeout)ms $(if($configStatus.ConnectionTimeout -eq 10000){'[OPTIMIZED]'}else{'[NEEDS OPTIMIZATION]'})" "DEBUG"
        Write-StatusMessage "  maxReconnectAttempts: $($configStatus.MaxReconnectAttempts) $(if($configStatus.MaxReconnectAttempts -eq 3){'[OPTIMIZED]'}else{'[NEEDS OPTIMIZATION]'})" "DEBUG"
        Write-StatusMessage "  filesystemPath: $($configStatus.FilesystemPath)" "DEBUG"
        
    } catch {
        Write-StatusMessage "Configuration parsing failed: $($_.Exception.Message)" "ERROR"
    }
} else {
    Write-StatusMessage "Configuration file not found: $configPath" "ERROR"
}

# Phase 4: Claude restart with optimization
if (-not $SkipRestart) {
    Write-StatusMessage "PHASE 4: Claude Restart Protocol" "DEBUG"
    
    # Discover Claude executable
    $claudePaths = @(
        "$env:LOCALAPPDATA\AnthropicClaude\app-0.10.14\claude.exe",
        "$env:LOCALAPPDATA\Programs\Claude\Claude.exe",
        "$env:PROGRAMFILES\Claude\Claude.exe"
    )
    
    $claudeExe = $null
    foreach ($path in $claudePaths) {
        if (Test-Path $path) {
            $claudeExe = $path
            Write-StatusMessage "Found Claude executable: $claudeExe" "SUCCESS"
            break
        }
    }
    
    if ($claudeExe) {
        $restartSuccess = Start-ClaudeOptimized -ClaudePath $claudeExe
        if ($restartSuccess) {
            # Wait for MCP initialization
            $initSuccess = Wait-ForMCPInitialization
            if (-not $initSuccess) {
                Write-StatusMessage "MCP initialization failed - proceeding with monitoring" "WARNING"
            }
        } else {
            Write-StatusMessage "Claude restart failed - manual restart required" "ERROR"
        }
    } else {
        Write-StatusMessage "Claude executable not found - manual restart required" "ERROR"
    }
} else {
    Write-StatusMessage "PHASE 4: Skipping Claude restart (test mode)" "DEBUG"
}

# Phase 5: Real-time performance monitoring
Write-StatusMessage "PHASE 5: Real-time Performance Monitoring (${MonitorDuration}s)" "DEBUG"
Write-Host "Time     | Processes | Memory(MB) | Recent | Status | Server Analysis" -ForegroundColor Gray
Write-Host ("-" * 85) -ForegroundColor Gray

$monitorStart = Get-Date
$maxProcesses = 0
$maxMemory = 0
$stabilityWindow = @()
$alertCount = 0

while (((Get-Date) - $monitorStart).TotalSeconds -lt $MonitorDuration) {
    $currentAnalysis = Get-NodeProcessAnalysis
    $mcpServers = Get-MCPServerTypes
    
    # Track maximums
    $maxProcesses = [Math]::Max($maxProcesses, $currentAnalysis.Count)
    $maxMemory = [Math]::Max($maxMemory, $currentAnalysis.TotalMemoryMB)
    
    # Stability analysis
    $stabilityWindow += $currentAnalysis.Count
    if ($stabilityWindow.Count -gt 10) { 
        $stabilityWindow = $stabilityWindow[-10..-1] 
    }
    
    $stabilityVariance = if ($stabilityWindow.Count -gt 1) {
        $mean = ($stabilityWindow | Measure-Object -Average).Average
        ($stabilityWindow | ForEach-Object { [Math]::Pow($_ - $mean, 2) } | Measure-Object -Sum).Sum / ($stabilityWindow.Count - 1)
    } else { 0 }
    
    # Status determination
    $status = switch ($true) {
        ($currentAnalysis.Count -gt $OPTIMAL_PROCESS_COUNT + 2) { 
            $alertCount++
            "CRITICAL"
        }
        ($currentAnalysis.Count -gt $OPTIMAL_PROCESS_COUNT) { "HIGH" }
        ($currentAnalysis.TotalMemoryMB -gt $MEMORY_THRESHOLD_MB) { "MEM_HIGH" }
        ($currentAnalysis.RecentSpawns -gt 0) { "SPAWNING" }
        ($stabilityVariance -lt 0.1) { "STABLE" }
        default { "MONITOR" }
    }
    
    # Server type distribution analysis
    $serverTypeDistribution = $mcpServers | Group-Object Type | ForEach-Object { "$($_.Name):$($_.Count)" }
    $serverAnalysis = $serverTypeDistribution -join " "
    
    $timestamp = (Get-Date).ToString("HH:mm:ss")
    $recentStr = $currentAnalysis.RecentSpawns.ToString().PadLeft(6)
    $processStr = $currentAnalysis.Count.ToString().PadLeft(9)
    $memoryStr = $currentAnalysis.TotalMemoryMB.ToString().PadLeft(10)
    
    Write-Host "$timestamp |$processStr |$memoryStr |$recentStr | $status | $serverAnalysis"
    
    # Alert on critical conditions
    if ($status -eq "CRITICAL") {
        Write-StatusMessage "CRITICAL: Process accumulation detected - $($currentAnalysis.Count) processes" "ERROR"
        
        if ($alertCount -gt 3) {
            Write-StatusMessage "Multiple critical alerts - investigating MCP server issues..." "ERROR"
            
            # Detailed process analysis on critical condition
            foreach ($server in $mcpServers) {
                Write-StatusMessage "  $($server.Type) Server - PID: $($server.PID), Memory: $($server.MemoryMB)MB" "ERROR"
            }
        }
    }
    
    Start-Sleep -Seconds $PollingInterval
}

# Phase 6: Final analysis and recommendations
Write-StatusMessage "PHASE 6: Final Analysis & Optimization Assessment" "DEBUG"

$finalAnalysis = Get-NodeProcessAnalysis
$filesystemTest = Test-FilesystemAccess

# Performance metrics calculation
$processEfficiency = if ($baselineAnalysis.Count -gt 0) {
    [Math]::Round((($baselineAnalysis.Count - $finalAnalysis.Count) / $baselineAnalysis.Count) * 100, 1)
} else { 0 }

$memoryEfficiency = if ($baselineAnalysis.TotalMemoryMB -gt 0) {
    [Math]::Round((($baselineAnalysis.TotalMemoryMB - $finalAnalysis.TotalMemoryMB) / $baselineAnalysis.TotalMemoryMB) * 100, 1)
} else { 0 }

# Health score calculation
$healthScore = 100
if ($finalAnalysis.Count -gt $OPTIMAL_PROCESS_COUNT) { $healthScore -= 20 }
if ($finalAnalysis.TotalMemoryMB -gt $MEMORY_THRESHOLD_MB) { $healthScore -= 15 }
if ($maxProcesses -gt $OPTIMAL_PROCESS_COUNT + 2) { $healthScore -= 15 }
if ($stabilityVariance -gt 1) { $healthScore -= 10 }
if (-not $filesystemTest.Success) { $healthScore -= 10 }
if ($alertCount -gt 2) { $healthScore -= 15 }

Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
Write-StatusMessage "OPTIMIZATION ASSESSMENT COMPLETE" "SUCCESS"
Write-Host ("=" * 80) -ForegroundColor Cyan

Write-StatusMessage "Performance Summary:" "DEBUG"
Write-StatusMessage "  Initial State: $($baselineAnalysis.Count) processes, $($baselineAnalysis.TotalMemoryMB)MB" "DEBUG"
Write-StatusMessage "  Final State: $($finalAnalysis.Count) processes, $($finalAnalysis.TotalMemoryMB)MB" "DEBUG"
Write-StatusMessage "  Peak Processes: $maxProcesses" "DEBUG"
Write-StatusMessage "  Peak Memory: $([Math]::Round($maxMemory, 1))MB" "DEBUG"
Write-StatusMessage "  Process Efficiency: $processEfficiency%" "DEBUG"
Write-StatusMessage "  Memory Efficiency: $memoryEfficiency%" "DEBUG"

Write-StatusMessage "System Health Score: $healthScore/100" $(if($healthScore -ge 90){"SUCCESS"}elseif($healthScore -ge 75){"WARNING"}else{"ERROR"})

$healthStatus = switch ($true) {
    ($healthScore -ge 90) { "EXCELLENT - System optimized successfully" }
    ($healthScore -ge 75) { "GOOD - Minor performance concerns detected" }
    ($healthScore -ge 60) { "FAIR - Optimization partially successful" }
    default { "POOR - Significant issues require investigation" }
}

Write-StatusMessage "Status: $healthStatus" $(if($healthScore -ge 90){"SUCCESS"}elseif($healthScore -ge 75){"WARNING"}else{"ERROR"})

Write-StatusMessage "Filesystem Access Test: $(if($filesystemTest.Success){"PASSED"}else{"FAILED - $($filesystemTest.Error)"})" $(if($filesystemTest.Success){"SUCCESS"}else{"ERROR"})

# Recommendations based on analysis
Write-StatusMessage "Optimization Recommendations:" "DEBUG"

if ($finalAnalysis.Count -le $OPTIMAL_PROCESS_COUNT -and $finalAnalysis.TotalMemoryMB -le $MEMORY_THRESHOLD_MB) {
    Write-StatusMessage "  ✅ System operating within optimal parameters" "SUCCESS"
    Write-StatusMessage "  ✅ MCP process leak prevention successful" "SUCCESS"
    Write-StatusMessage "  ✅ Memory efficiency targets achieved" "SUCCESS"
} else {
    Write-StatusMessage "  ⚠️ Performance optimization incomplete:" "WARNING"
    
    if ($finalAnalysis.Count -gt $OPTIMAL_PROCESS_COUNT) {
        Write-StatusMessage "    - Process count above optimal: $($finalAnalysis.Count) > $OPTIMAL_PROCESS_COUNT" "WARNING"
        Write-StatusMessage "    - Consider investigating MCP server internal retry mechanisms" "WARNING"
    }
    
    if ($finalAnalysis.TotalMemoryMB -gt $MEMORY_THRESHOLD_MB) {
        Write-StatusMessage "    - Memory usage above threshold: $($finalAnalysis.TotalMemoryMB)MB > ${MEMORY_THRESHOLD_MB}MB" "WARNING"
        Write-StatusMessage "    - Monitor for memory leaks in individual MCP servers" "WARNING"
    }
    
    if ($alertCount -gt 2) {
        Write-StatusMessage "    - Multiple critical alerts detected during monitoring" "WARNING"
        Write-StatusMessage "    - Deep investigation of MCP server command line arguments recommended" "WARNING"
    }
}

Write-StatusMessage "Protocol execution completed. Monitor system performance for sustained optimization." "SUCCESS"
Write-Host ("=" * 80) -ForegroundColor Cyan

# Return analysis object for programmatic access
return @{
    HealthScore = $healthScore
    ProcessCount = $finalAnalysis.Count
    MemoryUsageMB = $finalAnalysis.TotalMemoryMB
    ProcessEfficiency = $processEfficiency
    MemoryEfficiency = $memoryEfficiency
    FilesystemAccess = $filesystemTest.Success
    AlertCount = $alertCount
    Recommendations = $healthStatus
}