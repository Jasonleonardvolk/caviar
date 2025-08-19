# Type-Check Runner Script for TORI Project
# This script runs type-check:all and saves results to a file

param(
    [string]$OutputFile = "type-check-results.txt",
    [switch]$Append = $false,
    [switch]$ShowOutput = $true,
    [switch]$IncludeTimestamp = $true
)

# Generate filename with timestamp if requested
if ($IncludeTimestamp) {
    $timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $extension = [System.IO.Path]::GetExtension($OutputFile)
    $nameWithoutExt = [System.IO.Path]::GetFileNameWithoutExtension($OutputFile)
    $OutputFile = "${nameWithoutExt}_${timestamp}${extension}"
}

Write-Host "Running type-check:all..." -ForegroundColor Cyan
Write-Host "Output will be saved to: $OutputFile" -ForegroundColor Green
Write-Host ""

# Run the type check and capture output
if ($ShowOutput) {
    # Show output on console while saving to file
    pnpm run type-check:all 2>&1 | Tee-Object -FilePath $OutputFile -Append:$Append
} else {
    # Save to file without showing on console
    pnpm run type-check:all 2>&1 | Out-File -FilePath $OutputFile -Append:$Append
}

# Check if file was created and show summary
if (Test-Path $OutputFile) {
    $lineCount = (Get-Content $OutputFile | Measure-Object -Line).Lines
    Write-Host ""
    Write-Host "Type-check complete!" -ForegroundColor Green
    Write-Host "Results saved to: $OutputFile" -ForegroundColor Yellow
    Write-Host "Total lines in output: $lineCount" -ForegroundColor Yellow
    
    # Check for errors in the output
    $errorCount = (Select-String -Path $OutputFile -Pattern "error" -SimpleMatch).Count
    if ($errorCount -gt 0) {
        Write-Host "Found $errorCount error references in output" -ForegroundColor Red
    } else {
        Write-Host "No error references found in output" -ForegroundColor Green
    }
} else {
    Write-Host "Warning: Output file was not created" -ForegroundColor Red
}
