# Test-Direct-Encode.ps1
# Direct encoding test - no parameter passing issues

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     Direct Video Encoding Test                " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Direct paths - no variables, no parameters
$input1 = "D:\Dev\kha\content\wowpack\input\holo_flux_loop.mov"
$output1 = "D:\Dev\kha\content\wowpack\input\holo_flux_loop_test.mp4"

if (Test-Path $input1) {
    Write-Host "Testing basic encoding on: $input1" -ForegroundColor Yellow
    
    # Most basic possible ffmpeg command
    $cmd = "ffmpeg -y -i `"$input1`" -c:v libx264 -preset fast -crf 23 `"$output1`""
    
    Write-Host "Command: $cmd" -ForegroundColor Gray
    cmd /c $cmd
    
    if (Test-Path $output1) {
        Write-Host "SUCCESS! Test encoding worked." -ForegroundColor Green
        Write-Host "Output: $output1" -ForegroundColor Green
        
        # Get file size
        $size = (Get-Item $output1).Length / 1MB
        Write-Host "Size: $([math]::Round($size, 2)) MB" -ForegroundColor Gray
    } else {
        Write-Host "FAILED to create output file" -ForegroundColor Red
    }
} else {
    Write-Host "Input file not found: $input1" -ForegroundColor Red
}
