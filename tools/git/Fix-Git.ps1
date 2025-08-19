# Quick Git repository fix/initialization script

param(
    [switch]$Force,
    [string]$RemoteUrl
)

Write-Host "Git Repository Quick Fix" -ForegroundColor Cyan
Write-Host ("=" * 40)
Write-Host ""

# Ensure we're in D:\Dev\kha
Set-Location D:\Dev\kha
Write-Host "Working directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# Check current state
if (Test-Path .git) {
    $gitTest = git status 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Repository appears to be working!" -ForegroundColor Green
        git status
        exit 0
    } else {
        Write-Host "Found .git directory but git commands fail" -ForegroundColor Red
        Write-Host "Error: $gitTest" -ForegroundColor Gray
        
        if (-not $Force) {
            Write-Host ""
            Write-Host "Run with -Force to reinitialize the repository" -ForegroundColor Yellow
            exit 1
        }
        
        Write-Host "Removing corrupted .git directory..." -ForegroundColor Yellow
        Remove-Item .git -Recurse -Force
    }
}

# Initialize repository
Write-Host "Initializing git repository..." -ForegroundColor Cyan
git init

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to initialize repository!" -ForegroundColor Red
    exit 1
}

Write-Host "Repository initialized successfully!" -ForegroundColor Green

# Configure git if needed
$userName = git config user.name
if (-not $userName) {
    Write-Host ""
    Write-Host "Setting up git config..." -ForegroundColor Yellow
    git config user.name "Developer"
    git config user.email "developer@local"
    Write-Host "Git config set (update with your actual name/email)" -ForegroundColor Cyan
}

# Add remote if provided
if ($RemoteUrl) {
    Write-Host ""
    Write-Host "Adding remote origin: $RemoteUrl" -ForegroundColor Yellow
    git remote add origin $RemoteUrl
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Remote added successfully!" -ForegroundColor Green
    }
}

# Initial commit
Write-Host ""
$hasFiles = Get-ChildItem -File | Select-Object -First 1
if ($hasFiles) {
    Write-Host "Creating initial commit..." -ForegroundColor Yellow
    git add .
    git commit -m "Initial commit" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Initial commit created!" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Repository ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
if (-not $RemoteUrl) {
    Write-Host "  1. Add remote: git remote add origin <your-repo-url>" -ForegroundColor White
}
Write-Host "  2. Create branch: git checkout -b main" -ForegroundColor White
Write-Host "  3. Push: git push -u origin main" -ForegroundColor White
