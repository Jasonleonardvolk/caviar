# Validate the fix
Write-Host "=== Validating propagation.wgsl fix ===" -ForegroundColor Cyan

$file = "C:\Users\jason\Desktop\tori\kha\frontend\shaders\propagation.wgsl"

# Run naga validation
$result = & naga $file 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ SUCCESS! propagation.wgsl is now valid!" -ForegroundColor Green -BackgroundColor DarkGreen
    Write-Host "`nThe @group declaration has been moved to the correct position." -ForegroundColor Green
    Write-Host "It's now BEFORE the @compute directive, not between @compute and fn." -ForegroundColor Green
} else {
    Write-Host "`n❌ Still has errors:" -ForegroundColor Red
    $result | ForEach-Object { Write-Host $_ }
    
    Write-Host "`nShowing the current structure around line 474:" -ForegroundColor Yellow
    $lines = Get-Content $file
    for ($i = 470; $i -lt 480 -and $i -lt $lines.Count; $i++) {
        Write-Host "$($i+1): $($lines[$i])" -ForegroundColor DarkGray
    }
}
