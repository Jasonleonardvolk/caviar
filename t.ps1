cd C:\Users\jason\Desktop\tori\kha
Write-Host "📍 Now in kha directory" -ForegroundColor Green
$poetryEnv = poetry env info --path 2>$null
if ($poetryEnv) {
    $activateScript = "$poetryEnv\Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        Write-Host "✓ Poetry environment activated" -ForegroundColor Cyan
    }
}