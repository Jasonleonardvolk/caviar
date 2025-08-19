# Test Git functionality
Write-Host "Testing Git Setup..." -ForegroundColor Cyan
Write-Host ""

# Check if git is available
$gitVersion = git --version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Git found: $gitVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Git not found in PATH" -ForegroundColor Red
    Write-Host "  Please ensure Git is installed and in your PATH" -ForegroundColor Yellow
    exit 1
}

# Check if in a git repository
if (Test-Path .git) {
    Write-Host "✓ Git repository found" -ForegroundColor Green
    
    # Get current branch
    $branch = git rev-parse --abbrev-ref HEAD 2>$null
    if ($branch) {
        Write-Host "✓ Current branch: $branch" -ForegroundColor Green
    }
    
    # Check remote
    $remote = git remote -v 2>$null
    if ($remote) {
        Write-Host "✓ Remote configured:" -ForegroundColor Green
        $remote | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    } else {
        Write-Host "⚠ No remote configured" -ForegroundColor Yellow
    }
    
    # Check status
    Write-Host ""
    Write-Host "Repository Status:" -ForegroundColor Cyan
    git status --short
    
} else {
    Write-Host "✗ Not in a git repository" -ForegroundColor Red
    Write-Host "  Run this script from D:\Dev\kha" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Git test complete!" -ForegroundColor Green
