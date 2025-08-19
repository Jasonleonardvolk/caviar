param(
    [string]$ProjectRoot = "D:\Dev\kha",
    [switch]$SkipDevServer = $false,
    [switch]$GenerateReport = $true
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$reportDir = Join-Path $ProjectRoot "verification_reports"
$auditReport = Join-Path $reportDir "full_audit_$timestamp.json"

New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  FULL MONETIZATION AUDIT                " -ForegroundColor Cyan
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$results = @{
    timestamp = (Get-Date).ToString("o")
    projectRoot = $ProjectRoot
    checks = @{}
}

# Function to check if file exists
function Test-FileExists {
    param([string]$Path)
    $exists = Test-Path $Path
    if ($exists) {
        Write-Host "[✓] $Path" -ForegroundColor Green
    } else {
        Write-Host "[X] $Path" -ForegroundColor Red
    }
    return $exists
}

# Function to test HTTP endpoint
function Test-Endpoint {
    param(
        [string]$Url,
        [string]$Method = "GET",
        [hashtable]$Headers = @{},
        [string]$Body = $null
    )
    
    try {
        $params = @{
            Uri = $Url
            Method = $Method
            UseBasicParsing = $true
            TimeoutSec = 5
        }
        
        if ($Headers.Count -gt 0) {
            $params.Headers = $Headers