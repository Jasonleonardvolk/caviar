# Enhanced TORI Launcher PowerShell Script
# Version 3.0 - Production Ready

param(
    [int]$Port = 8002,
    [switch]$ApiOnly,
    [switch]$Debug,
    [int]$UiPort = 3000,
    [int]$McpPort = 6660
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ENHANCED TORI LAUNCHER v3.0" -ForegroundColor Green
Write-Host "  BULLETPROOF EDITION" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

# Set working directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $scriptPath) { $scriptPath = "D:\Dev\kha" }
Set-Location $scriptPath
Write-Host "Working directory: $(Get-Location)" -ForegroundColor Gray

# Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "ERROR: Python not found in PATH" -ForegroundColor Red
    exit 1
}

Write-Host "Python found: $($pythonCmd.Path)" -ForegroundColor Gray
$pythonVersion = python --version 2>&1
Write-Host "Python version: $pythonVersion" -ForegroundColor Gray

# Build launch arguments
$launchArgs = @("enhanced_launcher.py")

if ($Port -ne 8002) {
    $launchArgs += "--port", $Port
}

if ($UiPort -ne 3000) {
    $launchArgs += "--ui-port", $UiPort
}

if ($McpPort -ne 6660) {
    $launchArgs += "--mcp-port", $McpPort
}

if ($ApiOnly) {
    $launchArgs += "--api-only"
}

if ($Debug) {
    $launchArgs += "--debug"
}

Write-Host ""
Write-Host "Launching with arguments: $launchArgs" -ForegroundColor Gray
Write-Host ""

# Function to check if port is available
function Test-Port {
    param([int]$Port)
    $tcpClient = New-Object System.Net.Sockets.TcpClient
    try {
        $tcpClient.Connect("127.0.0.1", $Port)
        $tcpClient.Close()
        return $false  # Port is in use
    } catch {
        return $true   # Port is free
    }
}

# Check main ports
$portsToCheck = @{
    "API" = $Port
    "UI" = $UiPort
    "MCP" = $McpPort
}

$portConflicts = @()
foreach ($service in $portsToCheck.Keys) {
    if (-not (Test-Port $portsToCheck[$service])) {
        $portConflicts += "$service port $($portsToCheck[$service])"
    }
}

if ($portConflicts.Count -gt 0) {
    Write-Host "WARNING: The following ports are already in use:" -ForegroundColor Yellow
    $portConflicts | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    Write-Host ""
    
    $response = Read-Host "Continue anyway? (y/N)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        Write-Host "Aborted." -ForegroundColor Red
        exit 1
    }
}

# Register cleanup handler
$cleanupScript = {
    Write-Host ""
    Write-Host "Shutting down TORI services..." -ForegroundColor Yellow
    
    # Kill any remaining Python processes related to TORI
    Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -like "*uvicorn*" -or
        $_.CommandLine -like "*api.main*" -or
        $_.CommandLine -like "*mcp*" -or
        $_.CommandLine -like "*bridge*"
    } | Stop-Process -Force -ErrorAction SilentlyContinue
    
    # Kill Node processes for UI
    Get-Process node -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -like "*vite*" -or
        $_.CommandLine -like "*tori_ui*"
    } | Stop-Process -Force -ErrorAction SilentlyContinue
    
    Write-Host "Cleanup complete." -ForegroundColor Green
}

Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action $cleanupScript | Out-Null

# Handle Ctrl+C
[Console]::TreatControlCAsInput = $false
$null = [Console]::CancelKeyPress.Add({
    param($sender, $e)
    Write-Host ""
    Write-Host "Ctrl+C detected. Initiating graceful shutdown..." -ForegroundColor Yellow
    $e.Cancel = $true
    & $cleanupScript
    exit 0
})

# Launch the enhanced launcher
try {
    Write-Host "Starting enhanced launcher..." -ForegroundColor Green
    Write-Host ""
    
    # Run Python launcher
    & python $launchArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Launcher exited with code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "ERROR: Failed to launch TORI" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    & $cleanupScript
    exit 1
} finally {
    & $cleanupScript
}

Write-Host ""
Write-Host "TORI launcher has terminated." -ForegroundColor Cyan
