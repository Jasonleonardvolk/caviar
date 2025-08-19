# Consolidate-Shaders.ps1
# Canonical shader cleanup and consolidation script
# This ensures all shaders are in the correct location for bundling

param(
    [string]$ProjectRoot = (Get-Location).Path,
    [switch]$DryRun = $false,
    [switch]$Verbose = $false
)

# === CONFIGURATION ===
$CANONICAL_DIR = Join-Path $ProjectRoot "frontend\lib\webgpu\shaders"
$BACKUP_DIR = Join-Path $ProjectRoot "shader_backups\$(Get-Date -Format 'yyyyMMdd_HHmmss')"

$SEARCH_LOCATIONS = @(
    "frontend\shaders",
    "frontend\shaders\desktop",
    "frontend\lib\webgpu\shaders\fft",
    "..\tori\kha backup\kha\frontend\lib\webgpu\shaders",
    "."
)

$REQUIRED_SHADERS = @(
    "bitReversal.wgsl",
    "butterflyStage.wgsl",
    "normalize.wgsl",
    "fftShift.wgsl",
    "transpose.wgsl",
    "multiViewSynthesis.wgsl",
    "lenticularInterlace.wgsl",
    "avatarShader.wgsl",
    "velocityField.wgsl",
    "wavefieldEncoder.wgsl",
    "wavefieldEncoder_optimized.wgsl",
    "propagation.wgsl"
)

# === FUNCTIONS ===
function Write-Status {
    param($Message, $Type = "Info")
    $colors = @{
        "Info" = "Cyan"
        "Success" = "Green"
        "Warning" = "Yellow"
        "Error" = "Red"
    }
    Write-Host $Message -ForegroundColor $colors[$Type]
}

function Find-ShaderFile {
    param($ShaderName)
    
    $foundFiles = @()
    foreach ($location in $SEARCH_LOCATIONS) {
        $searchPath = Join-Path $ProjectRoot $location
        if (Test-Path $searchPath) {
            $found = Get-ChildItem -Path $searchPath -Filter $ShaderName -Recurse -ErrorAction SilentlyContinue
            $foundFiles += $found
        }
    }
    
    if ($foundFiles.Count -gt 0) {
        # Return the most recently modified version
        return $foundFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    }
    return $null
}

function Validate-ShaderContent {
    param($FilePath)
    
    if (-not (Test-Path $FilePath)) {
        return @{ Valid = $false; Error = "File not found" }
    }
    
    $content = Get-Content $FilePath -Raw
    
    # Check for common issues
    if ($content -match '\bpath\s*:\s*[''"`]' -and $content -match '\bcontent\s*:\s*[''"`]') {
        return @{ Valid = $false; Error = "Contains JavaScript object syntax instead of WGSL" }
    }
    
    if ($content.Length -lt 50) {
        return @{ Valid = $false; Error = "File too small to be valid shader" }
    }
    
    if (-not ($content -match '@(compute|vertex|fragment)')) {
        return @{ Valid = $false; Error = "Missing shader entry point" }
    }
    
    return @{ Valid = $true; Error = $null }
}

# === MAIN SCRIPT ===
Write-Status "=== Shader Consolidation Script ===" "Info"
Write-Status "Canonical directory: $CANONICAL_DIR" "Info"

# Create directories
if (-not $DryRun) {
    if (-not (Test-Path $CANONICAL_DIR)) {
        New-Item -ItemType Directory -Path $CANONICAL_DIR -Force | Out-Null
        Write-Status "Created canonical directory" "Success"
    }
    
    if (-not (Test-Path $BACKUP_DIR)) {
        New-Item -ItemType Directory -Path $BACKUP_DIR -Force | Out-Null
    }
}

# Process each shader
$summary = @{
    Found = 0
    Missing = 0
    Invalid = 0
    Copied = 0
}

foreach ($shaderName in $REQUIRED_SHADERS) {
    Write-Host "`n--- Processing $shaderName ---" -ForegroundColor White
    
    $found = Find-ShaderFile -ShaderName $shaderName
    
    if ($found) {
        $summary.Found++
        
        # Validate content
        $validation = Validate-ShaderContent -FilePath $found.FullName
        
        if ($validation.Valid) {
            $destPath = Join-Path $CANONICAL_DIR $shaderName
            
            if ($DryRun) {
                Write-Status "Would copy: $($found.FullName) → $destPath" "Info"
            } else {
                # Backup existing if present
                if (Test-Path $destPath) {
                    $backupPath = Join-Path $BACKUP_DIR $shaderName
                    Copy-Item -Path $destPath -Destination $backupPath -Force
                    if ($Verbose) {
                        Write-Status "  Backed up existing to: $backupPath" "Info"
                    }
                }
                
                # Copy the shader
                Copy-Item -Path $found.FullName -Destination $destPath -Force
                Write-Status "  ✓ Copied from: $($found.FullName)" "Success"
                Write-Status "  ✓ Size: $((Get-Item $destPath).Length) bytes" "Success"
                $summary.Copied++
            }
        } else {
            $summary.Invalid++
            Write-Status "  ✗ Invalid shader: $($validation.Error)" "Error"
            Write-Status "  Found at: $($found.FullName)" "Error"
        }
    } else {
        $summary.Missing++
        Write-Status "  ✗ Not found in any search location" "Error"
    }
}

# Clean up old/duplicate locations
if (-not $DryRun) {
    Write-Host "`n--- Cleaning up duplicate locations ---" -ForegroundColor White
    
    $duplicateLocations = @(
        "frontend\shaders",
        "frontend\lib\webgpu\shaders\fft"
    )
    
    foreach ($location in $duplicateLocations) {
        $fullPath = Join-Path $ProjectRoot $location
        if (Test-Path $fullPath) {
            $shaderFiles = Get-ChildItem -Path $fullPath -Filter "*.wgsl" -File
            if ($shaderFiles.Count -gt 0) {
                Write-Status "Found $($shaderFiles.Count) shader files in: $location" "Warning"
                if ($Verbose) {
                    $shaderFiles | ForEach-Object {
                        Write-Status "  - $($_.Name)" "Info"
                    }
                }
            }
        }
    }
}

# Summary
Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Status "Found: $($summary.Found) shaders" $(if ($summary.Found -eq $REQUIRED_SHADERS.Count) { "Success" } else { "Warning" })
Write-Status "Missing: $($summary.Missing) shaders" $(if ($summary.Missing -gt 0) { "Error" } else { "Success" })
Write-Status "Invalid: $($summary.Invalid) shaders" $(if ($summary.Invalid -gt 0) { "Error" } else { "Success" })
Write-Status "Copied: $($summary.Copied) shaders" "Info"

if ($summary.Missing -gt 0 -or $summary.Invalid -gt 0) {
    Write-Status "`nSome shaders are missing or invalid!" "Error"
    exit 1
} else {
    Write-Status "`nAll shaders consolidated successfully!" "Success"
    
    if (-not $DryRun) {
        Write-Status "`nNext step: Run bundler to regenerate shaderSources.ts" "Info"
        Write-Status "Command: npm run bundle-shaders" "Info"
    }
}

# Export results for other scripts
$global:ShaderConsolidationResult = @{
    Success = ($summary.Missing -eq 0 -and $summary.Invalid -eq 0)
    CanonicalDir = $CANONICAL_DIR
    Summary = $summary
}
