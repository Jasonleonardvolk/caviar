# Claude MCP System - Master Execution Controller
# Single-point execution for all MCP optimization and monitoring operations

param(
    [ValidateSet("full-optimization", "quick-fix", "monitor", "diagnose", "repair")]
    [string]$Operation = "full-optimization",
    [int]$MonitorTime = 300,  # 5 minutes default monitoring
    [switch]$AutoCleanup = $false,
    [switch]$SkipBackup = $false
)

$ScriptRoot = "C:\Users\jason\Desktop\tori\kha"
$LogFile = Join-Path $ScriptRoot "master_execution_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Write-MasterLog {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    
    $color = switch ($Level) {
        "CRITICAL" { "Red" }
        "ERROR" { "Red" }
        "WARNING" { "Yellow" }
        "SUCCESS" { "Green" }
        "DEBUG" { "Cyan" }
        default { "White" }
    }
    
    Write-Host $logEntry -ForegroundColor $color
    $logEntry | Add-Content -Path $LogFile -Encoding UTF8
}

function Invoke-ScriptWithErrorHandling {
    param(
        [string]$ScriptPath,
        [hashtable]$Parameters = @{},
        [string]$Description
    )
    
    Write-MasterLog "Executing: $Description" "DEBUG"
    Write-MasterLog "Script: $ScriptPath" "DEBUG"
    
    if (-not (Test-Path $ScriptPath)) {
        Write-MasterLog "Script not found: $ScriptPath" "ERROR"
        return $false
    }
    
    try {
        $result = & $ScriptPath @Parameters
        Write-MasterLog "Script execution completed: $Description" "SUCCESS"
        return $result
    } catch {
        Write-MasterLog "Script execution failed: $Description" "ERROR"
        Write-MasterLog "Error: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

Write-MasterLog "Claude MCP Master Execution Controller Started" "SUCCESS"
Write-MasterLog "Operation: $Operation" "INFO"
Write-MasterLog "Log file: $LogFile" "INFO"

switch ($Operation) {
    "full-optimization" {
        Write-MasterLog "FULL OPTIMIZATION PROTOCOL INITIATED" "SUCCESS"
        Write-MasterLog "This will perform complete system optimization with monitoring" "INFO"
        
        # Step 1: Configuration backup and optimization
        if (-not $SkipBackup) {
            Write-MasterLog "Step 1: Configuration backup" "DEBUG"
            Invoke-ScriptWithErrorHandling -ScriptPath (Join-Path $ScriptRoot "mcp_config_toolkit.ps1") -Parameters @{Action="backup"} -Description "Configuration Backup"
        }
        
        # Step 2: System repair and optimization
        Write-MasterLog "Step 2: System repair and optimization" "DEBUG"
        Invoke-ScriptWithErrorHandling -ScriptPath (Join-Path $ScriptRoot "mcp_config_toolkit.ps1") -Parameters @{Action="repair"} -Description "System Repair"
        
        # Step 3: Comprehensive optimization
        Write-MasterLog "Step 3: Comprehensive system optimization" "DEBUG"
        $optimizationResult = Invoke-ScriptWithErrorHandling -ScriptPath (Join-Path $ScriptRoot "claude_optimization_integrated.ps1") -Parameters @{MonitorDuration=$MonitorTime} -Description "Comprehensive Optimization"
        
        if ($optimizationResult) {
            Write-MasterLog "Optimization Health Score: $($optimizationResult.HealthScore)/100" "INFO"
            Write-MasterLog "Process Count: $($optimizationResult.ProcessCount)" "INFO"
            Write-MasterLog "Memory Usage: $($optimizationResult.MemoryUsageMB)MB" "INFO"
        }
        
        # Step 4: Post-optimization monitoring
        Write-MasterLog "Step 4: Post-optimization monitoring" "DEBUG"
        Invoke-ScriptWithErrorHandling -ScriptPath (Join-Path $ScriptRoot "mcp_realtime_monitor.ps1") -Parameters @{MonitorDuration=180; AutoCleanup=$AutoCleanup} -Description "Post-Optimization Monitoring"
    }
    
    "quick-fix" {
        Write-MasterLog "QUICK FIX PROTOCOL INITIATED" "SUCCESS"
        Write-MasterLog "This will perform rapid problem resolution" "INFO"
        
        # Step 1: Emergency process cleanup
        Write-MasterLog "Step 1: Emergency process cleanup" "DEBUG"
        try {
            Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force
            Start-Sleep -Seconds 3
            $remaining = (Get-Process node -ErrorAction SilentlyContinue | Measure-Object).Count
            Write-MasterLog "Process cleanup completed. Remaining processes: $remaining" "SUCCESS"
        } catch {
            Write-MasterLog "Process cleanup failed: $($_.Exception.Message)" "ERROR"
        }
        
        # Step 2: Configuration optimization
        Write-MasterLog "Step 2: Configuration optimization" "DEBUG"
        Invoke-ScriptWithErrorHandling -ScriptPath (Join-Path $ScriptRoot "mcp_config_toolkit.ps1") -Parameters @{Action="optimize"} -Description "Quick Configuration Optimization"
        
        # Step 3: Short monitoring
        Write-MasterLog "Step 3: Validation monitoring" "DEBUG"
        Invoke-ScriptWithErrorHandling -ScriptPath (Join-Path $ScriptRoot "mcp_realtime_monitor.ps1") -Parameters @{MonitorDuration=60; AutoCleanup=$true} -Description "Quick Validation"
    }
    
    "monitor" {
        Write-MasterLog "MONITORING PROTOCOL INITIATED" "SUCCESS"
        Write-MasterLog "This will start continuous system monitoring" "INFO"
        
        Invoke-ScriptWithErrorHandling -ScriptPath (Join-Path $ScriptRoot "mcp_realtime_monitor.ps1") -Parameters @{MonitorDuration=$MonitorTime; AutoCleanup=$AutoCleanup} -Description "System Monitoring"
    }
    
    "diagnose" {
        Write-MasterLog "DIAGNOSTIC PROTOCOL INITIATED" "SUCCESS"
        Write-MasterLog "This will perform comprehensive system diagnostics" "INFO"
        
        Invoke-ScriptWithErrorHandling -ScriptPath (Join-Path $ScriptRoot "mcp_config_toolkit.ps1") -Parameters @{Action="diagnose"} -Description "System Diagnostics"
    }
    
    "repair" {
        Write-MasterLog "REPAIR PROTOCOL INITIATED" "SUCCESS"
        Write-MasterLog "This will perform system repair operations" "INFO"
        
        if (-not $SkipBackup) {
            Invoke-ScriptWithErrorHandling -ScriptPath (Join-Path $ScriptRoot "mcp_config_toolkit.ps1") -Parameters @{Action="backup"} -Description "Pre-Repair Backup"
        }
        
        Invoke-ScriptWithErrorHandling -ScriptPath (Join-Path $ScriptRoot "mcp_config_toolkit.ps1") -Parameters @{Action="repair"} -Description "System Repair"
    }
}

Write-MasterLog "Master execution completed" "SUCCESS"
Write-MasterLog "Full execution log available at: $LogFile" "INFO"

# Final system status
try {
    $nodeCount = (Get-Process node -ErrorAction SilentlyContinue | Measure-Object).Count
    $nodeMemory = if ($nodeCount -gt 0) { 
        [math]::Round((Get-Process node | Measure-Object WorkingSet -Sum).Sum / 1MB, 1) 
    } else { 0 }
    
    Write-MasterLog "Final System Status:" "SUCCESS"
    Write-MasterLog "  Node.js Processes: $nodeCount" "INFO"
    Write-MasterLog "  Node.js Memory: ${nodeMemory}MB" "INFO"
    
    $status = if ($nodeCount -le 4 -and $nodeMemory -le 200) { "OPTIMAL" } 
             elseif ($nodeCount -le 6 -and $nodeMemory -le 300) { "ACCEPTABLE" }
             else { "REQUIRES_ATTENTION" }
    
    Write-MasterLog "  System Status: $status" $(if($status -eq "OPTIMAL"){"SUCCESS"}elseif($status -eq "ACCEPTABLE"){"WARNING"}else{"ERROR"})
} catch {
    Write-MasterLog "Failed to get final system status: $($_.Exception.Message)" "WARNING"
}