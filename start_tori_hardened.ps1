#Requires -Version 5.0
<#
.SYNOPSIS
    Hardened TORI System Startup Script
.DESCRIPTION
    Bulletproof startup script with comprehensive error handling,
    port validation, and prerequisite checking to ensure clean launch every time.
.NOTES
    Version: 2.1
    Author: TORI Team
    Requires: PowerShell 5.0+, Python 3.8+, Node.js 16+
#>

[CmdletBinding()]
param(
    [switch]$SkipBrowser,
    [switch]$SkipHologram,
    [switch]$Force
)

# Set strict error handling
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Console colors
$colors = @{
    Success = "Green"
    Error = "Red"
    Warning = "Yellow"
    Info = "Cyan"
    Step = "Magenta"
}

# Configuration
$script:Config = @{
    ProjectRoot = $PSScriptRoot
    PythonExe = "python"
    RequiredPorts = @{
        API = 8002
        Frontend = 5173
        AudioBridge = 8765
        ConceptMesh = 8766
        MCP = 8100
    }
    Timeouts = @{
        ServiceStart = 90
        PortCheck = 5
        ProcessKill = 10
    }
    LogFile = Join-Path $PSScriptRoot "logs\startup_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
}

# Initialize logging
function Initialize-Logging {
    $logDir = Split-Path $script:Config.LogFile -Parent
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    
    # Start transcript
    Start-Transcript -Path $script:Config.LogFile -Append
    Write-Log "TORI Hardened Startup Script v2.1" "Step"
    Write-Log ("=" * 60) "Step"
}

# Enhanced logging function
function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "Info"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    
    # Write to console with color
    if ($colors.ContainsKey($Level)) {
        Write-Host $logMessage -ForegroundColor $colors[$Level]
    } else {
        Write-Host $logMessage
    }
    
    # Also write to log file
    Add-Content -Path $script:Config.LogFile -Value $logMessage -ErrorAction SilentlyContinue
}

# Check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Validate prerequisites
function Test-Prerequisites {
    Write-Log "Checking prerequisites..." "Step"
    
    $issues = @()
    
    # Check Python
    try {
        $pythonVersion = & $script:Config.PythonExe --version 2>&1
        if ($pythonVersion -match "Python (\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            if ($major -ge 3 -and $minor -ge 8) {
                Write-Log "[OK] Python $major.$minor found" "Success"
            } else {
                $issues += "Python 3.8+ required (found $major.$minor)"
            }
        }
    } catch {
        $issues += "Python not found or not in PATH"
    }
    
    # Check Node.js
    try {
        $nodeVersion = & node --version 2>&1
        if ($nodeVersion -match "v(\d+)\.") {
            $nodeMajor = [int]$matches[1]
            if ($nodeMajor -ge 16) {
                Write-Log "[OK] Node.js $nodeVersion found" "Success"
            } else {
                $issues += "Node.js 16+ required (found $nodeVersion)"
            }
        }
    } catch {
        $issues += "Node.js not found or not in PATH"
    }
    
    # Check npm
    try {
        $npmVersion = & npm --version 2>&1
        Write-Log "[OK] npm $npmVersion found" "Success"
    } catch {
        $issues += "npm not found"
    }
    
    # Check critical files
    $criticalFiles = @(
        "enhanced_launcher.py",
        "audio_hologram_bridge.py", 
        "concept_mesh_hologram_bridge.py",
        "tori_ui_svelte\package.json"
    )
    
    foreach ($file in $criticalFiles) {
        $filePath = Join-Path $script:Config.ProjectRoot $file
        if (-not (Test-Path $filePath)) {
            $issues += "Missing critical file: $file"
        }
    }
    
    # Check if npm dependencies are installed
    $nodeModules = Join-Path $script:Config.ProjectRoot "tori_ui_svelte\node_modules"
    if (-not (Test-Path $nodeModules)) {
        $issues += "Node modules not installed - run: cd tori_ui_svelte && npm install"
    }
    
    # Report issues
    if ($issues.Count -gt 0) {
        Write-Log "[ERROR] Prerequisite check failed:" "Error"
        foreach ($issue in $issues) {
            Write-Log "   - $issue" "Error"
        }
        throw "Prerequisites not met. Please fix the issues above."
    }
    
    Write-Log "[OK] All prerequisites satisfied" "Success"
}

# Enhanced port checking with Test-NetConnection
function Test-PortAvailable {
    param(
        [int]$Port,
        [string]$ServiceName
    )
    
    Write-Log "Checking port ${Port} for ${ServiceName}..." "Info"
    
    try {
        # First, check if anything is listening
        $tcpTest = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
        
        if ($tcpTest.TcpTestSucceeded) {
            Write-Log "[WARNING] Port ${Port} is already in use" "Warning"
            
            # Find what's using it
            $netstat = netstat -ano | Select-String ":${Port}\s+.*LISTENING" | Select-Object -First 1
            if ($netstat) {
                $parts = $netstat.Line -split '\s+'
                $procId = $parts[-1]
                
                try {
                    $process = Get-Process -Id $procId -ErrorAction SilentlyContinue
                    if ($process) {
                        Write-Log "   Process: $($process.ProcessName) (PID: ${procId})" "Warning"
                    }
                } catch {
                    Write-Log "   Process ID: ${procId}" "Warning"
                }
            }
            
            return $false
        } else {
            Write-Log "[OK] Port ${Port} is available" "Success"
            return $true
        }
    } catch {
        # If Test-NetConnection fails, fall back to socket test
        try {
            $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $Port)
            $listener.Start()
            $listener.Stop()
            Write-Log "[OK] Port ${Port} is available" "Success"
            return $true
        } catch {
            Write-Log "[WARNING] Port ${Port} is already in use" "Warning"
            return $false
        }
    }
}

# Kill process on port with timeout
function Stop-ProcessOnPort {
    param(
        [int]$Port,
        [string]$ServiceName
    )
    
    Write-Log "Attempting to free port ${Port} for ${ServiceName}..." "Step"
    
    try {
        # Get connections using PowerShell cmdlet instead of netstat
        $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        
        if (-not $connections) {
            Write-Log "No processes found listening on port ${Port}" "Info"
            return $true
        }
        
        # Force connections to array and get unique PIDs
        $pids = @($connections | Select-Object -ExpandProperty OwningProcess -Unique)
        
        Write-Log "Found $($pids.Count) process(es) using port ${Port}" "Info"
        
        foreach ($procId in $pids) {
            if ($procId -eq 0 -or $procId -eq 4) {
                Write-Log "Skipping system process (PID: ${procId})" "Info"
                continue
            }
            
            try {
                $process = Get-Process -Id $procId -ErrorAction SilentlyContinue
                if ($process) {
                    Write-Log "Stopping process $($process.ProcessName) (PID: ${procId})..." "Info"
                    
                    # Try graceful stop first
                    $process | Stop-Process -ErrorAction SilentlyContinue
                    
                    # Wait for process to exit
                    $timeout = [DateTime]::Now.AddSeconds($script:Config.Timeouts.ProcessKill)
                    while ([DateTime]::Now -lt $timeout -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
                        Start-Sleep -Milliseconds 500
                    }
                    
                    # Force kill if still running
                    if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
                        Write-Log "Force killing process ${procId}..." "Warning"
                        Stop-Process -Id $procId -Force -ErrorAction Stop
                    }
                    
                    Write-Log "[SUCCESS] Process ${procId} stopped" "Success"
                }
            } catch {
                Write-Log "[WARNING] Failed to stop process ${procId}: $_" "Warning"
            }
        }
        
        # Wait a moment for port to be released
        Start-Sleep -Seconds 2
        
        # Verify port is now free
        $stillInUse = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($stillInUse) {
            Write-Log "[ERROR] Port ${Port} is still in use after cleanup attempt" "Error"
            return $false
        }
        
        Write-Log "[SUCCESS] Port ${Port} is now available" "Success"
        return $true
        
    } catch {
        Write-Log "[ERROR] Error freeing port ${Port}: $_" "Error"
        return $false
    }
}

# Ensure all required ports are available
function Ensure-PortsAvailable {
    Write-Log "Ensuring all required ports are available..." "Step"
    
    $portsOk = $true
    
    foreach ($service in $script:Config.RequiredPorts.Keys) {
        $port = $script:Config.RequiredPorts[$service]
        
        if (-not (Test-PortAvailable -Port $port -ServiceName $service)) {
            if ($Force) {
                Write-Log "Force flag set - attempting to free port ${port}" "Warning"
                if (-not (Stop-ProcessOnPort -Port $port -ServiceName $service)) {
                    $portsOk = $false
                    Write-Log "[ERROR] Failed to free port ${port} for ${service}" "Error"
                }
            } else {
                $portsOk = $false
                Write-Log "[ERROR] Port ${port} is busy (use -Force to kill existing processes)" "Error"
            }
        }
    }
    
    if (-not $portsOk) {
        throw "Required ports are not available. Cannot continue."
    }
    
    Write-Log "[OK] All required ports are available" "Success"
}

# Apply system fixes if needed
function Apply-SystemFixes {
    Write-Log "Checking and applying system fixes..." "Step"
    
    # Check if fixes have been applied
    $fixesNeeded = @()
    
    # Check Tailwind CSS fix
    $appCss = Join-Path $script:Config.ProjectRoot "tori_ui_svelte\src\app.css"
    if (Test-Path $appCss) {
        $content = Get-Content $appCss -Raw
        if ($content -notmatch '@reference "tailwindcss"') {
            $fixesNeeded += "Tailwind CSS v4 fix"
        }
    }
    
    # Check shader fix
    $shaderMgr = Join-Path $script:Config.ProjectRoot "frontend\lib\webgpu\shaderConstantManager.ts"
    if (Test-Path $shaderMgr) {
        $content = Get-Content $shaderMgr -Raw
        if ($content -match "'1':\s*'normalizationMode'") {
            $fixesNeeded += "WebGPU shader fix"
        }
    }
    
    # Check bridge fixes
    $audioBridge = Join-Path $script:Config.ProjectRoot "audio_hologram_bridge.py"
    if (Test-Path $audioBridge) {
        $content = Get-Content $audioBridge -Raw
        if ($content -notmatch "SO_REUSEADDR") {
            $fixesNeeded += "Bridge port binding fix"
        }
    }
    
    if ($fixesNeeded.Count -gt 0) {
        Write-Log "[WARNING] System fixes needed:" "Warning"
        foreach ($fix in $fixesNeeded) {
            Write-Log "   - $fix" "Warning"
        }
        
        # Check if fix script exists
        $fixScript = Join-Path $script:Config.ProjectRoot "fix_frontend_and_bridges.py"
        if (Test-Path $fixScript) {
            Write-Log "Running fix script..." "Info"
            
            try {
                $result = & $script:Config.PythonExe $fixScript 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Log "[OK] Fixes applied successfully" "Success"
                } else {
                    throw "Fix script failed with exit code $LASTEXITCODE"
                }
            } catch {
                Write-Log "[ERROR] Failed to apply fixes: $_" "Error"
                throw
            }
        } else {
            Write-Log "[WARNING] Fix script not found - manual fixes may be needed" "Warning"
        }
    } else {
        Write-Log "[OK] All system fixes already applied" "Success"
    }
}

# Start TORI system with monitoring
function Start-TORISystem {
    Write-Log "Starting TORI system..." "Step"
    
    # Start MCP server first
    Write-Log "🚀 Launching MCP server in background..." "Step"
    $mcpScript = Join-Path $script:Config.ProjectRoot "start_mcp_real.py"
    if (-not (Test-Path $mcpScript)) {
        # Fallback to manual fixed if real doesn't exist
        $mcpScript = Join-Path $script:Config.ProjectRoot "start_mcp_manual_fixed.py"
    }
    
    if (Test-Path $mcpScript) {
        Start-Process -NoNewWindow -FilePath $script:Config.PythonExe `
            -ArgumentList $mcpScript `
            -WorkingDirectory $script:Config.ProjectRoot
        
        Write-Log "⏳ Waiting for MCP to respond on 8100/api/system/status..." "Info"
        $mcpReady = $false
        for ($i = 0; $i -lt 12; $i++) {
            try {
                $resp = Invoke-RestMethod "http://localhost:8100/api/system/status" -TimeoutSec 5
                if ($resp.mcp_available) {
                    Write-Log "✅ MCP is live and in FastMCP mode" "Success"
                    $mcpReady = $true
                    break
                } else {
                    Write-Log "⚠️ MCP responding but not in FastMCP mode (attempt $($i+1)/12)" "Warning"
                }
            } catch {
                Write-Log "⚠️ MCP not ready yet ($($i+1)/12)... retrying in 5s" "Warning"
            }
            Start-Sleep -Seconds 5
        }
        
        if (-not $mcpReady) {
            Write-Log "[WARNING] MCP server may not be fully ready, continuing anyway..." "Warning"
        }
    } else {
        Write-Log "[WARNING] MCP startup script not found, skipping MCP" "Warning"
    }
    
    $launcherPath = Join-Path $script:Config.ProjectRoot "enhanced_launcher.py"
    
    # Build command arguments
    $arguments = @()
    if ($SkipBrowser) {
        $arguments += "--no-browser"
    }
    if ($SkipHologram) {
        $arguments += "--no-require-penrose"
    }
    
    # Create process start info
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $script:Config.PythonExe
    $processInfo.Arguments = "$launcherPath $($arguments -join ' ')"
    $processInfo.WorkingDirectory = $script:Config.ProjectRoot
    $processInfo.UseShellExecute = $false
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true
    $processInfo.CreateNoWindow = $false
    
    # Set encoding to UTF-8
    $processInfo.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $processInfo.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    
    try {
        Write-Log "Launching: $($processInfo.FileName) $($processInfo.Arguments)" "Info"
        Write-Log "Working directory: $($processInfo.WorkingDirectory)" "Info"
        
        # Create log files for output
        $outputLog = Join-Path $script:Config.ProjectRoot "logs\enhanced_launcher_output.log"
        $errorLog = Join-Path $script:Config.ProjectRoot "logs\enhanced_launcher_error.log"
        
        # Start the process
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $processInfo
        
        # Set up async output handlers
        $outputHandler = {
            param($sender, $e)
            if ($e.Data) {
                Write-Log "[TORI] $($e.Data)" "Info"
                Add-Content -Path $outputLog -Value "[$(Get-Date -Format 'HH:mm:ss')] $($e.Data)" -Encoding UTF8
            }
        }
        
        $errorHandler = {
            param($sender, $e)
            if ($e.Data) {
                Write-Log "[TORI-ERR] $($e.Data)" "Warning"
                Add-Content -Path $errorLog -Value "[$(Get-Date -Format 'HH:mm:ss')] $($e.Data)" -Encoding UTF8
                
                # Check for critical errors
                if ($e.Data -match "Error|Exception|Failed|ImportError|ModuleNotFoundError|Traceback") {
                    Write-Log "[CRITICAL] Error detected in enhanced_launcher.py output" "Error"
                }
            }
        }
        
        Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action $outputHandler | Out-Null
        Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action $errorHandler | Out-Null
        
        $process.Start() | Out-Null
        $process.BeginOutputReadLine()
        $process.BeginErrorReadLine()
        
        Write-Log "Process started with PID: $($process.Id)" "Info"
        
        # Monitor startup
        $startTime = [DateTime]::Now
        $timeout = $startTime.AddSeconds($script:Config.Timeouts.ServiceStart)
        $apiReady = $false
        $frontendReady = $false
        
        Write-Log "Waiting for services to start..." "Info"
        
        while ([DateTime]::Now -lt $timeout -and (-not $apiReady -or -not $frontendReady)) {
            # Check if process is still running
            if ($process.HasExited) {
                throw "TORI launcher exited unexpectedly with code $($process.ExitCode)"
            }
            
            # Check API
            if (-not $apiReady) {
                try {
                    # Use -UseBasicParsing to avoid IE dependency issues
                    $response = Invoke-WebRequest -Uri "http://localhost:$($script:Config.RequiredPorts.API)/api/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
                    if ($response.StatusCode -eq 200) {
                        $apiReady = $true
                        Write-Log "[OK] API server is ready" "Success"
                    }
                } catch [System.Net.WebException] {
                    # More specific error handling
                    if ($_.Exception.Message -match "Unable to connect") {
                        Write-Verbose "API not ready - connection refused"
                    } else {
                        Write-Verbose "API health check failed: $($_.Exception.Message)"
                    }
                } catch {
                    Write-Verbose "API health check error: $_"
                }
            }
            
            # Check Frontend
            if (-not $frontendReady -and -not $SkipBrowser) {
                try {
                    # Use -UseBasicParsing to avoid IE dependency issues
                    $response = Invoke-WebRequest -Uri "http://localhost:$($script:Config.RequiredPorts.Frontend)/" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
                    if ($response.StatusCode -in @(200, 304)) {
                        $frontendReady = $true
                        Write-Log "[OK] Frontend is ready" "Success"
                    }
                } catch {
                    Write-Verbose "Frontend not ready: $_"
                }
            } elseif ($SkipBrowser) {
                $frontendReady = $true  # Don't wait for frontend if skipped
            }
            
            # Get any output from buffers
            # Output is now handled by event handlers
            
            # Show progress every 5 seconds
            if (([DateTime]::Now - $startTime).TotalSeconds % 5 -lt 0.5) {
                $elapsed = [int]([DateTime]::Now - $startTime).TotalSeconds
                Write-Log "Still waiting for services... ($elapsed/$($script:Config.Timeouts.ServiceStart) seconds)" "Info"
            }
            
            Start-Sleep -Milliseconds 500
        }
        
        # Check final status
        if (-not $apiReady) {
            throw "API server failed to start within $($script:Config.Timeouts.ServiceStart) seconds"
        }
        
        if (-not $frontendReady -and -not $SkipBrowser) {
            Write-Log "[WARNING] Frontend not ready within timeout - may still be building" "Warning"
        }
        
        # Clean up event handlers
        Get-EventSubscriber | Where-Object { $_.SourceObject -eq $process } | Unregister-Event
        
        Write-Log "[OK] TORI system started successfully!" "Success"
        Write-Log "" "Info"
        Write-Log "API: http://localhost:$($script:Config.RequiredPorts.API)" "Info"
        Write-Log "API Docs: http://localhost:$($script:Config.RequiredPorts.API)/docs" "Info"
        
        if (-not $SkipBrowser) {
            Write-Log "Frontend: http://localhost:$($script:Config.RequiredPorts.Frontend)" "Info"
        }
        
        if (-not $SkipHologram) {
            Write-Log "Hologram Services Enabled" "Info"
            Write-Log "Audio Bridge: ws://localhost:$($script:Config.RequiredPorts.AudioBridge)/audio_stream" "Info"
            Write-Log "Concept Bridge: ws://localhost:$($script:Config.RequiredPorts.ConceptMesh)/concepts" "Info"
        }
        
        Write-Log "" "Info"
        Write-Log "Press Ctrl+C to shutdown gracefully" "Info"
        
        # Keep script running and monitor
        try {
            $process.WaitForExit()
            Write-Log "TORI system exited with code $($process.ExitCode)" "Info"
        } catch {
            Write-Log "Script interrupted - TORI may still be running" "Warning"
        }
        
    } catch {
        Write-Log "[ERROR] Failed to start TORI system: $_" "Error"
        throw
    }
}

# Main execution
function Main {
    try {
        Initialize-Logging
        
        # Show startup banner
        Write-Log @"
===============================================================
                  TORI HARDENED STARTUP v2.1                  
                                                              
  Bulletproof startup with comprehensive error handling       
===============================================================
"@ "Step"
        
        # Check if running as admin (optional warning)
        if (-not (Test-Administrator)) {
            Write-Log "[WARNING] Not running as administrator - some operations may fail" "Warning"
        }
        
        # Run all checks and fixes
        Test-Prerequisites
        Ensure-PortsAvailable
        Apply-SystemFixes
        
        # Start the system
        Start-TORISystem
        
    } catch {
        Write-Log "[FATAL ERROR] $_" "Error"
        Write-Log "Stack Trace: $($_.ScriptStackTrace)" "Error"
        
        # Stop transcript if it's running
        try {
            if ((Get-Host).Name -eq 'ConsoleHost') {
                Stop-Transcript | Out-Null
            }
        } catch {
            # Transcript wasn't running, ignore
        }
        
        # Exit with error code
        exit 1
    } finally {
        # Ensure transcript is stopped
        try {
            if ((Get-Host).Name -eq 'ConsoleHost') {
                Stop-Transcript | Out-Null
            }
        } catch {
            # Transcript wasn't running, ignore
        }
    }
}

# Run main
Main
