# Build-WowPack.ps1
# HEVC-HDR10 / AV1 encoding pipeline for wow content

param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile,
    
    [Parameter()]
    [ValidateSet('hevc', 'av1', 'both')]
    [string]$Codec = 'both',
    
    [Parameter()]
    [string]$OutputDir = ".\encoded",
    
    [Parameter()]
    [int]$CRF = 23,
    
    [Parameter()]
    [switch]$HDR10
)

Write-Host "`n=== WowPack Encoder ===" -ForegroundColor Magenta

# Ensure ffmpeg is available
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: ffmpeg not found in PATH" -ForegroundColor Red
    Write-Host "Install with: winget install ffmpeg" -ForegroundColor Yellow
    exit 1
}

# Validate input file
if (-not (Test-Path $InputFile)) {
    Write-Host "ERROR: Input file not found: $InputFile" -ForegroundColor Red
    exit 1
}

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
    Write-Host "Created output directory: $OutputDir" -ForegroundColor Green
}

$inputName = [System.IO.Path]::GetFileNameWithoutExtension($InputFile)
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

function Encode-HEVC {
    param([string]$Input, [string]$Output)
    
    Write-Host "`nEncoding HEVC..." -ForegroundColor Cyan
    
    $hevcParams = @(
        "-i", $Input,
        "-c:v", "libx265",
        "-preset", "slow",
        "-crf", $CRF,
        "-c:a", "copy"
    )
    
    if ($HDR10) {
        $hevcParams += @(
            "-x265-params",
            "hdr10=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:max-cll=1000,400"
        )
    }
    
    $hevcParams += @(
        "-tag:v", "hvc1",
        $Output
    )
    
    & ffmpeg @hevcParams
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "HEVC encoding complete: $Output" -ForegroundColor Green
    } else {
        Write-Host "HEVC encoding failed!" -ForegroundColor Red
    }
}

function Encode-AV1 {
    param([string]$Input, [string]$Output)
    
    Write-Host "`nEncoding AV1..." -ForegroundColor Cyan
    
    $av1Params = @(
        "-i", $Input,
        "-c:v", "libsvtav1",
        "-preset", "4",
        "-crf", $CRF,
        "-c:a", "libopus",
        "-b:a", "128k"
    )
    
    if ($HDR10) {
        $av1Params += @(
            "-svtav1-params",
            "enable-hdr=1:mastering-display=G(0.265,0.690)B(0.150,0.060)R(0.680,0.320)WP(0.3127,0.3290)L(1000,0.01):content-light=1000,400"
        )
    }
    
    $av1Params += $Output
    
    & ffmpeg @av1Params
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "AV1 encoding complete: $Output" -ForegroundColor Green
    } else {
        Write-Host "AV1 encoding failed!" -ForegroundColor Red
    }
}

# Main encoding logic
$results = @()

if ($Codec -eq 'hevc' -or $Codec -eq 'both') {
    $hevcOutput = Join-Path $OutputDir "$($inputName)_hevc_$timestamp.mp4"
    Encode-HEVC -Input $InputFile -Output $hevcOutput
    $results += $hevcOutput
}

if ($Codec -eq 'av1' -or $Codec -eq 'both') {
    $av1Output = Join-Path $OutputDir "$($inputName)_av1_$timestamp.webm"
    Encode-AV1 -Input $InputFile -Output $av1Output
    $results += $av1Output
}

# Display results
Write-Host "`n=== Encoding Complete ===" -ForegroundColor Magenta
Write-Host "Output files:" -ForegroundColor Cyan
foreach ($file in $results) {
    if (Test-Path $file) {
        $size = (Get-Item $file).Length / 1MB
        Write-Host "  - $file ($('{0:N2}' -f $size) MB)" -ForegroundColor Green
    }
}