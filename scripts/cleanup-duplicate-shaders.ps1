# Delete duplicate shader files from the wrong location
# These are causing path confusion

$duplicateShaderPath = "C:\Users\jason\Desktop\tori\kha\frontend\lib\webgpu\shaders"
$correctShaderPath = "C:\Users\jason\Desktop\tori\kha\frontend\shaders"

Write-Host "Removing duplicate shader files from: $duplicateShaderPath" -ForegroundColor Yellow

# List files that will be removed
$filesToRemove = Get-ChildItem -Path $duplicateShaderPath -Filter "*.wgsl" -File

foreach ($file in $filesToRemove) {
    $correctFile = Join-Path $correctShaderPath $file.Name
    if (Test-Path $correctFile) {
        Write-Host "  Removing duplicate: $($file.Name)" -ForegroundColor Red
        Remove-Item $file.FullName -Force
    } else {
        Write-Host "  Keeping unique file: $($file.Name)" -ForegroundColor Green
    }
}

# Remove empty directory if no files remain
$remainingFiles = Get-ChildItem -Path $duplicateShaderPath -File
if ($remainingFiles.Count -eq 0) {
    Write-Host "`nRemoving empty directory: $duplicateShaderPath" -ForegroundColor Cyan
    Remove-Item $duplicateShaderPath -Recurse -Force
}

Write-Host "`nDuplicate shader cleanup complete!" -ForegroundColor Green
Write-Host "Correct shader path: $correctShaderPath" -ForegroundColor Yellow
