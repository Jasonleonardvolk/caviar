# PowerShell Profile Setup Script for TORI Development
# This adds the 't' shortcut to your PowerShell profile

Write-Host "🔧 Setting up TORI PowerShell shortcut..." -ForegroundColor Cyan

# Get the profile path
$profilePath = $PROFILE.CurrentUserAllHosts
$profileDir = Split-Path $profilePath -Parent

# Create profile directory if it doesn't exist
if (!(Test-Path $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    Write-Host "✅ Created profile directory: $profileDir" -ForegroundColor Green
}

# The TORI shortcut function
$toriFunction = @'

# TORI Development Environment Shortcut
function t {
    Set-Location 'C:\Users\jason\Desktop\tori\kha'
    
    # Check if .venv exists
    if (Test-Path '.\.venv\Scripts\Activate.ps1') {
        & '.\.venv\Scripts\Activate.ps1'
        Write-Host "🐍 TORI dev shell ready!" -ForegroundColor Green
        Write-Host "📂 Location: $(Get-Location)" -ForegroundColor Cyan
        Write-Host "🚀 Run: python enhanced_launcher.py" -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  No .venv found! Creating one..." -ForegroundColor Yellow
        python -m venv .venv
        & '.\.venv\Scripts\Activate.ps1'
        Write-Host "✅ Created and activated new venv!" -ForegroundColor Green
    }
}

# Alias for even faster access
Set-Alias tori t

'@

# Check if function already exists
$existingContent = ""
if (Test-Path $profilePath) {
    $existingContent = Get-Content $profilePath -Raw
}

if ($existingContent -match "function t \{") {
    Write-Host "⚠️  't' function already exists in profile!" -ForegroundColor Yellow
    Write-Host "   Location: $profilePath" -ForegroundColor Gray
    
    $response = Read-Host "Replace existing function? (y/n)"
    if ($response -eq 'y') {
        # Remove old function
        $existingContent = $existingContent -replace '(?ms)# TORI Development Environment Shortcut.*?^}.*?\n', ''
        $existingContent = $existingContent -replace '(?ms)function t \{.*?^}.*?\n', ''
        $existingContent = $existingContent -replace 'Set-Alias tori t.*?\n', ''
        
        # Add new function
        $existingContent = $existingContent.TrimEnd() + "`n`n" + $toriFunction
        $existingContent | Set-Content $profilePath -Encoding UTF8
        Write-Host "✅ Updated 't' function in profile!" -ForegroundColor Green
    } else {
        Write-Host "❌ Skipped updating profile" -ForegroundColor Red
        exit
    }
} else {
    # Add function to profile
    if ($existingContent) {
        $newContent = $existingContent.TrimEnd() + "`n`n" + $toriFunction
    } else {
        $newContent = $toriFunction
    }
    
    $newContent | Set-Content $profilePath -Encoding UTF8
    Write-Host "✅ Added 't' function to profile!" -ForegroundColor Green
}

Write-Host ""
Write-Host "🎉 Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Close this PowerShell window" -ForegroundColor White
Write-Host "   2. Open a NEW PowerShell window" -ForegroundColor White
Write-Host "   3. Type 't' from anywhere to jump into TORI dev!" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Shortcuts added:" -ForegroundColor Cyan
Write-Host "   t     - Navigate to TORI and activate venv" -ForegroundColor White
Write-Host "   tori  - Same as 't' (alias)" -ForegroundColor White
Write-Host ""
Write-Host "Profile location: $profilePath" -ForegroundColor Gray
