# Watch-And-Bundle-Shaders.ps1
# Watches shader files for changes and automatically rebundles

param(
    [string]$ProjectRoot = (Get-Location).Path,
    [switch]$RunOnce = $false
)

$SHADER_DIR = Join-Path $ProjectRoot "frontend\lib\webgpu\shaders"
$BUNDLE_SCRIPT = Join-Path $ProjectRoot "scripts\bundleShaders.ts"
$OUTPUT_FILE = Join-Path $ProjectRoot "frontend\lib\webgpu\generated\shaderSources.ts"

# Function to run bundler
function Invoke-ShaderBundle {
    Write-Host "`n$(Get-Date -Format 'HH:mm:ss') - Bundling shaders..." -ForegroundColor Cyan
    
    $startTime = Get-Date
    
    try {
        # Run the bundler
        $result = npm run bundle-shaders 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            $duration = (Get-Date) - $startTime
            Write-Host "✅ Bundle complete in $($duration.TotalSeconds.ToString('F2'))s" -ForegroundColor Green
            
            # Show bundle size
            if (Test-Path $OUTPUT_FILE) {
                $size = (Get-Item $OUTPUT_FILE).Length
                Write-Host "   Output size: $([math]::Round($size/1024, 2)) KB" -ForegroundColor Gray
            }
            
            return $true
        } else {
            Write-Host "❌ Bundle failed!" -ForegroundColor Red
            Write-Host $result -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "❌ Bundle error: $_" -ForegroundColor Red
        return $false
    }
}

# Function to validate shader before bundling
function Test-ShaderValidity {
    param($FilePath)
    
    $content = Get-Content $FilePath -Raw
    $issues = @()
    
    # Check for JavaScript syntax
    if ($content -match '\bpath\s*:\s*[''"`]' -and $content -match '\bcontent\s*:\s*[''"`]') {
        $issues += "Contains JavaScript object syntax"
    }
    
    # Check for shader entry point
    if (-not ($content -match '@(compute|vertex|fragment)')) {
        $issues += "Missing shader entry point"
    }
    
    # Check for common WGSL errors
    if ($content -match '^\s*//.*$' -and $content.Length -lt 100) {
        $issues += "Appears to be mostly comments"
    }
    
    return $issues
}

if ($RunOnce) {
    # Just run once and exit
    Write-Host "Running shader bundler once..." -ForegroundColor Cyan
    Invoke-ShaderBundle
    exit
}

# Set up file watcher
Write-Host "=== Shader Auto-Bundler ===" -ForegroundColor Cyan
Write-Host "Watching: $SHADER_DIR" -ForegroundColor White
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

# Initial bundle
Invoke-ShaderBundle

# Create file watcher
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $SHADER_DIR
$watcher.Filter = "*.wgsl"
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

# Define action for file changes
$action = {
    $path = $Event.SourceEventArgs.FullPath
    $changeType = $Event.SourceEventArgs.ChangeType
    $fileName = Split-Path $path -Leaf
    
    Write-Host "`n$(Get-Date -Format 'HH:mm:ss') - $changeType`: $fileName" -ForegroundColor Yellow
    
    # Validate the changed shader
    if ($changeType -ne "Deleted") {
        Start-Sleep -Milliseconds 100  # Let file write complete
        
        $issues = Test-ShaderValidity -FilePath $path
        if ($issues.Count -gt 0) {
            Write-Host "⚠️  Shader validation warnings:" -ForegroundColor Yellow
            $issues | ForEach-Object { Write-Host "   - $_" -ForegroundColor Yellow }
        }
    }
    
    # Debounce - wait a bit in case multiple files are being saved
    Start-Sleep -Milliseconds 500
    
    # Run bundler
    Invoke-ShaderBundle
}

# Register event handlers
Register-ObjectEvent -InputObject $watcher -EventName "Created" -Action $action
Register-ObjectEvent -InputObject $watcher -EventName "Changed" -Action $action
Register-ObjectEvent -InputObject $watcher -EventName "Deleted" -Action $action
Register-ObjectEvent -InputObject $watcher -EventName "Renamed" -Action $action

# Keep script running
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    # Clean up
    $watcher.EnableRaisingEvents = $false
    $watcher.Dispose()
    Write-Host "`nWatcher stopped." -ForegroundColor Yellow
}
