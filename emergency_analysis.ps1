# Emergency Process Intervention & Claude Executable Discovery
# Addresses continued process accumulation despite configuration optimization

Write-Host "Emergency Process Analysis & Intervention Protocol" -ForegroundColor Red
Write-Host ("=" * 60) -ForegroundColor Red

# Critical Resource Analysis
Write-Host "`nCRITICAL RESOURCE ANALYSIS:" -ForegroundColor Yellow
$nodeProcesses = Get-Process node -ErrorAction SilentlyContinue
$claudeProcesses = Get-Process claude -ErrorAction SilentlyContinue

$nodeMemory = if ($nodeProcesses) { ($nodeProcesses | Measure-Object WorkingSet -Sum).Sum / 1MB } else { 0 }
$claudeMemory = if ($claudeProcesses) { ($claudeProcesses | Measure-Object WorkingSet -Sum).Sum / 1MB } else { 0 }
$totalMemory = $nodeMemory + $claudeMemory

Write-Host "Node.js Memory Consumption: $([math]::Round($nodeMemory, 1))MB"
Write-Host "Claude Memory Consumption: $([math]::Round($claudeMemory, 1))MB" 
Write-Host "Total System Impact: $([math]::Round($totalMemory, 1))MB"

if ($totalMemory -gt 1000) {
    Write-Host "⚠️ CRITICAL: Memory consumption exceeds 1GB threshold" -ForegroundColor Red
}

# Process Age Analysis for Leak Detection
Write-Host "`nPROCESS AGE ANALYSIS:" -ForegroundColor Yellow
if ($nodeProcesses) {
    $nodeProcesses | ForEach-Object {
        $age = [math]::Round(((Get-Date) - $_.StartTime).TotalMinutes, 1)
        $memory = [math]::Round($_.WorkingSet / 1MB, 1)
        Write-Host "  PID $($_.Id): ${memory}MB, Age: ${age}min $(if($age -gt 10){'[STALE]'}else{'[ACTIVE]'})"
    }
}

# Claude Executable Discovery Protocol
Write-Host "`nCLAUDE EXECUTABLE DISCOVERY:" -ForegroundColor Yellow

# Method 1: Registry-based discovery
try {
    $registryPaths = @(
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*"
    )
    
    $claudeInstall = $null
    foreach ($regPath in $registryPaths) {
        if (Test-Path $regPath) {
            $items = Get-ItemProperty $regPath -ErrorAction SilentlyContinue | Where-Object { 
                $_.DisplayName -like "*Claude*" -and $_.InstallLocation 
            }
            if ($items) {
                $claudeInstall = $items[0].InstallLocation
                break
            }
        }
    }
    
    if ($claudeInstall) {
        $claudeExePath = Join-Path $claudeInstall "Claude.exe"
        if (Test-Path $claudeExePath) {
            Write-Host "✅ Found via registry: $claudeExePath" -ForegroundColor Green
        }
    }
} catch {
    Write-Host "Registry discovery failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Method 2: Process-based discovery
if ($claudeProcesses) {
    try {
        $claudeMainProcess = $claudeProcesses | Sort-Object WorkingSet -Descending | Select-Object -First 1
        $claudePath = $claudeMainProcess.Path
        if ($claudePath -and (Test-Path $claudePath)) {
            Write-Host "✅ Found via running process: $claudePath" -ForegroundColor Green
        }
    } catch {
        Write-Host "Process path discovery failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Method 3: Windows Search indexing
try {
    $searchResult = Get-ChildItem -Path $env:LOCALAPPDATA -Recurse -Name "Claude.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($searchResult) {
        $foundPath = Join-Path $env:LOCALAPPDATA $searchResult
        Write-Host "✅ Found via search: $foundPath" -ForegroundColor Green
    }
} catch {
    Write-Host "Search indexing discovery failed" -ForegroundColor Yellow
}

# Method 4: Alternative installation locations
$alternatePaths = @(
    "$env:USERPROFILE\AppData\Local\Claude\Claude.exe",
    "$env:USERPROFILE\AppData\Roaming\Claude\Claude.exe",
    "$env:LOCALAPPDATA\Claude\app-*\Claude.exe"
)

foreach ($altPath in $alternatePaths) {
    if ($altPath -like "*app-*") {
        # Handle wildcard pattern for versioned directories
        $parentDir = Split-Path $altPath -Parent
        if (Test-Path $parentDir) {
            $appDirs = Get-ChildItem -Path $parentDir -Directory -Name "app-*" -ErrorAction SilentlyContinue
            foreach ($appDir in $appDirs) {
                $testPath = Join-Path (Join-Path $parentDir $appDir) "Claude.exe"
                if (Test-Path $testPath) {
                    Write-Host "✅ Found in versioned directory: $testPath" -ForegroundColor Green
                    break
                }
            }
        }
    } else {
        if (Test-Path $altPath) {
            Write-Host "✅ Found in alternative location: $altPath" -ForegroundColor Green
        }
    }
}

# Emergency Process Cleanup Protocol
Write-Host "`nEMERGENCY PROCESS CLEANUP PROTOCOL:" -ForegroundColor Yellow

# Identify processes by command line (more precise than name matching)
$allProcesses = Get-WmiObject -Class Win32_Process | Where-Object { 
    $_.Name -eq "node.exe" -and $_.CommandLine -like "*mcp*" 
}

if ($allProcesses) {
    Write-Host "Found MCP-related processes:"
    $allProcesses | ForEach-Object {
        $memory = [math]::Round($_.WorkingSetSize / 1MB, 1)
        Write-Host "  PID $($_.ProcessId): ${memory}MB - $($_.CommandLine)"
    }
    
    Write-Host "`nExecute emergency cleanup? (Recommend immediate action)" -ForegroundColor Red
    Write-Host "Command: Get-Process node | Stop-Process -Force" -ForegroundColor Yellow
}

# Memory Pressure Analysis
Write-Host "`nMEMORY PRESSURE ANALYSIS:" -ForegroundColor Yellow
$systemMemory = Get-WmiObject -Class Win32_OperatingSystem
$totalRAM = [math]::Round($systemMemory.TotalVisibleMemorySize / 1MB, 1)
$freeRAM = [math]::Round($systemMemory.FreePhysicalMemory / 1MB, 1)
$usedRAM = $totalRAM - $freeRAM
$memoryPressure = [math]::Round($usedRAM / $totalRAM * 100, 1)

Write-Host "Total System Memory: ${totalRAM}GB"
Write-Host "Free Memory: ${freeRAM}GB"
Write-Host "Memory Pressure: ${memoryPressure}%"

if ($memoryPressure -gt 80) {
    Write-Host "⚠️ CRITICAL: High memory pressure detected" -ForegroundColor Red
    Write-Host "Immediate process cleanup recommended" -ForegroundColor Red
}

# Handle Leak Analysis
Write-Host "`nHANDLE LEAK ANALYSIS:" -ForegroundColor Yellow
if ($nodeProcesses) {
    $nodeProcesses | ForEach-Object {
        $handles = $_.Handles
        $memory = [math]::Round($_.WorkingSet / 1MB, 1)
        $efficiency = if ($memory -gt 0) { [math]::Round($handles / $memory, 1) } else { 0 }
        Write-Host "  PID $($_.Id): $handles handles, ${memory}MB (${efficiency} handles/MB)"
        
        if ($handles -gt 1000) {
            Write-Host "    ⚠️ Excessive handle count detected" -ForegroundColor Yellow
        }
    }
}

# Immediate Action Recommendations
Write-Host "`nIMMEDIATE ACTION RECOMMENDATIONS:" -ForegroundColor Red
Write-Host "1. Execute process cleanup: Get-Process node | Stop-Process -Force"
Write-Host "2. Restart Claude manually from Start Menu or taskbar"
Write-Host "3. Monitor process count for 5 minutes post-restart"
Write-Host "4. If accumulation continues, investigate MCP server internal retry logic"

Write-Host "`nProcess accumulation despite optimized configuration indicates:" -ForegroundColor Yellow
Write-Host "- Internal MCP server retry mechanisms still active"
Write-Host "- Possible handle/socket leak in Node.js runtime"
Write-Host "- Resource cleanup inefficiency in garbage collection"

Write-Host "`nAnalysis Complete - Immediate intervention required" -ForegroundColor Red
Write-Host ("=" * 60) -ForegroundColor Red