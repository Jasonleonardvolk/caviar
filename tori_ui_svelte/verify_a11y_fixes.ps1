Write-Host "`n🔍 Verifying Accessibility Fixes..." -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

$componentsPath = "src\lib\components"

# Check PersonaPanel.svelte
Write-Host "`nChecking PersonaPanel.svelte..." -ForegroundColor Yellow
$personaPanel = Get-Content "$componentsPath\PersonaPanel.svelte" -Raw

if ($personaPanel -match 'role="dialog"' -and $personaPanel -match 'aria-modal="true"') {
    Write-Host "✅ PersonaPanel: Dialog role properly configured" -ForegroundColor Green
} else {
    Write-Host "❌ PersonaPanel: Missing proper dialog configuration" -ForegroundColor Red
}

# Check HolographicDisplay.svelte
Write-Host "`nChecking HolographicDisplay.svelte..." -ForegroundColor Yellow
$holographic = Get-Content "$componentsPath\HolographicDisplay.svelte" -Raw

if ($holographic -match '<track kind="captions"') {
    Write-Host "✅ HolographicDisplay: Video has caption track" -ForegroundColor Green
} else {
    Write-Host "❌ HolographicDisplay: Video missing caption track" -ForegroundColor Red
}

# Check for caption file
if (Test-Path "static\captions\hologram-video.vtt") {
    Write-Host "✅ Caption file exists" -ForegroundColor Green
} else {
    Write-Host "❌ Caption file missing" -ForegroundColor Red
}

Write-Host "`n📋 Quick A11y Checklist:" -ForegroundColor Cyan
Write-Host "  - Replace <div on:click> with <button> where possible"
Write-Host "  - Add keyboard handlers to all interactive elements"
Write-Host "  - Ensure all form inputs have labels"
Write-Host "  - Add alt text to all images"
Write-Host "  - Test with keyboard navigation (Tab key)"

Write-Host "`n✨ Run 'npm run dev' to see if there are any remaining warnings!" -ForegroundColor Green
