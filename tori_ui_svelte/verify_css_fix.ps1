Write-Host "🔍 Verifying TailwindCSS Fix..." -ForegroundColor Cyan
Write-Host ""

# Navigate to the frontend directory
Set-Location -Path "C:\Users\jason\Desktop\tori\kha\tori_ui_svelte"

# Check if app.css exists
if (Test-Path "src\app.css") {
    Write-Host "✅ app.css found" -ForegroundColor Green
    
    # Check for circular references
    $content = Get-Content "src\app.css" -Raw
    if ($content -match '@apply\s+tori-button(?:\s|;)') {
        Write-Host "❌ Found circular @apply tori-button references!" -ForegroundColor Red
    } else {
        Write-Host "✅ No circular @apply references found" -ForegroundColor Green
    }
} else {
    Write-Host "❌ app.css not found!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🔨 Testing CSS build..." -ForegroundColor Yellow
Write-Host ""

# Try to build CSS with TailwindCSS
$testFile = "test-output.css"
$errorFile = "build-error.log"

# Run tailwindcss build
$process = Start-Process -FilePath "npx" -ArgumentList "tailwindcss", "-i", "./src/app.css", "-o", $testFile -Wait -NoNewWindow -PassThru -RedirectStandardError $errorFile

if ($process.ExitCode -eq 0) {
    Write-Host "✅ CSS build successful!" -ForegroundColor Green