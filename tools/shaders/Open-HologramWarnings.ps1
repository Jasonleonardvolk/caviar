# Open-HologramWarnings.ps1
# Opens shader validation warnings in VS Code with direct line navigation
# Usage: .\Open-HologramWarnings.ps1 [-Type <warning-type>] [-File <shader-name>]

param(
    [string]$Type = "all",
    [string]$File = "",
    [switch]$ErrorsOnly,
    [switch]$MarkFixed
)

$reportPath = "C:\Users\jason\Desktop\tori\kha\tools\shaders\reports\hologram_warnings_todo.md"

if (-not (Test-Path $reportPath)) {
    Write-Host "❌ Warning report not found at: $reportPath" -ForegroundColor Red
    Write-Host "Run the validator first to generate the report." -ForegroundColor Yellow
    exit 1
}

Write-Host "🔍 Hologram Shader Warning Navigator" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor DarkGray

# Parse the markdown report
$content = Get-Content $reportPath -Raw
$lines = $content -split "`n"

$warnings = @()
$currentFile = ""

foreach ($line in $lines) {
    if ($line -match "^### `(.+?)`$") {
        $currentFile = $matches[1]
    }
    elseif ($line -match "^- \[(ERROR|WARNING)\]\s+L?(\d+)?\s*—\s*(.+?)`(.+?)`$") {
        $severity = $matches[1]
        $lineNum = $matches[2]
        $message = $matches[3]
        $warningType = $matches[4]
        
        $warnings += [PSCustomObject]@{
            File = $currentFile
            Line = if ($lineNum) { [int]$lineNum } else { 0 }
            Severity = $severity
            Message = $message
            Type = $warningType
            Fixed = $false
        }
    }
}

# Filter warnings
if ($ErrorsOnly) {
    $warnings = $warnings | Where-Object { $_.Severity -eq "ERROR" }
    Write-Host "🎯 Showing ERRORS only" -ForegroundColor Red
}

if ($File) {
    $warnings = $warnings | Where-Object { $_.File -like "*$File*" }
    Write-Host "📁 Filtering for file: $File" -ForegroundColor Yellow
}

if ($Type -ne "all") {
    $warnings = $warnings | Where-Object { $_.Type -eq $Type }
    Write-Host "🏷️ Filtering for type: $Type" -ForegroundColor Yellow
}

# Group by type
$grouped = $warnings | Group-Object -Property Type

Write-Host ""
Write-Host "📊 Summary:" -ForegroundColor Green
foreach ($group in $grouped) {
    $color = if ($group.Group[0].Severity -eq "ERROR") { "Red" } else { "Yellow" }
    Write-Host "  $($group.Name): $($group.Count) issues" -ForegroundColor $color
}

Write-Host ""
Write-Host "🚀 Opening files in VS Code..." -ForegroundColor Cyan
Write-Host ""

# Track opened files
$openedFiles = @{}

foreach ($warning in $warnings) {
    $fullPath = Join-Path "C:\Users\jason\Desktop\tori\kha" $warning.File
    
    if (-not (Test-Path $fullPath)) {
        Write-Host "  ⚠️ File not found: $($warning.File)" -ForegroundColor DarkYellow
        continue
    }
    
    $fileKey = "$($warning.File):$($warning.Line)"
    
    if (-not $openedFiles.ContainsKey($fileKey)) {
        $openedFiles[$fileKey] = $true
        
        # Format for display
        $severityIcon = if ($warning.Severity -eq "ERROR") { "❌" } else { "⚠️" }
        $severityColor = if ($warning.Severity -eq "ERROR") { "Red" } else { "Yellow" }
        
        Write-Host "$severityIcon " -NoNewline -ForegroundColor $severityColor
        Write-Host "$($warning.File):$($warning.Line)" -NoNewline -ForegroundColor White
        Write-Host " - " -NoNewline -ForegroundColor DarkGray
        Write-Host "$($warning.Type)" -ForegroundColor Cyan
        Write-Host "    $($warning.Message)" -ForegroundColor Gray
        
        # Open in VS Code with line number
        if ($warning.Line -gt 0) {
            & code -g "${fullPath}:$($warning.Line)"
        } else {
            & code $fullPath
        }
        
        # Small delay to prevent overwhelming VS Code
        Start-Sleep -Milliseconds 100
    }
}

Write-Host ""
Write-Host "=" * 50 -ForegroundColor DarkGray

# Summary
$errorCount = ($warnings | Where-Object { $_.Severity -eq "ERROR" }).Count
$warningCount = ($warnings | Where-Object { $_.Severity -eq "WARNING" }).Count

if ($errorCount -gt 0) {
    Write-Host "🔥 $errorCount ERRORS need immediate attention!" -ForegroundColor Red
} else {
    Write-Host "✅ No errors found!" -ForegroundColor Green
}

Write-Host "📝 $warningCount warnings to review" -ForegroundColor Yellow

# Quick fix suggestions
Write-Host ""
Write-Host "💡 Quick Fix Tips:" -ForegroundColor Cyan

if ($warnings | Where-Object { $_.Type -eq "NAGA_VALIDATION" }) {
    Write-Host "  • NAGA_VALIDATION: Add missing mip level (4th arg) to textureLoad" -ForegroundColor White
}

if ($warnings | Where-Object { $_.Type -eq "DYNAMIC_INDEXING_BOUNDS" }) {
    Write-Host "  • DYNAMIC_INDEXING: Already using clamp_index_dyn - SAFE to suppress" -ForegroundColor White
}

if ($warnings | Where-Object { $_.Type -eq "VEC3_STORAGE_ALIGNMENT" }) {
    Write-Host "  • VEC3_STORAGE: Check if actually storage buffer (not vertex attribute)" -ForegroundColor White
}

Write-Host ""
Write-Host "🎮 Commands:" -ForegroundColor Magenta
Write-Host "  .\Open-HologramWarnings.ps1 -ErrorsOnly        # Show only errors" -ForegroundColor Gray
Write-Host "  .\Open-HologramWarnings.ps1 -Type NAGA_VALIDATION  # Filter by type" -ForegroundColor Gray
Write-Host "  .\Open-HologramWarnings.ps1 -File butterflyStage   # Filter by file" -ForegroundColor Gray
