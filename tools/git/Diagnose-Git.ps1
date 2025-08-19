# Comprehensive Git diagnostics

Write-Host "=== GIT REPOSITORY DIAGNOSTICS ===" -ForegroundColor Cyan
Write-Host ""

# 1. Current Directory
Write-Host "1. Current Directory:" -ForegroundColor Yellow
Write-Host "   Path: $(Get-Location)" -ForegroundColor White
Write-Host ""

# 2. Check .git directory
Write-Host "2. Checking .git directory:" -ForegroundColor Yellow
$gitDir = Join-Path (Get-Location) ".git"
if (Test-Path $gitDir) {
    Write-Host "   .git directory EXISTS at: $gitDir" -ForegroundColor Green
    
    # Check if it's a file (submodule) or directory
    $gitItem = Get-Item $gitDir -Force
    if ($gitItem.PSIsContainer) {
        Write-Host "   Type: Directory" -ForegroundColor Green
        
        # Check contents
        $gitContents = Get-ChildItem $gitDir -Force -ErrorAction SilentlyContinue
        if ($gitContents) {
            Write-Host "   Contents found: $($gitContents.Count) items" -ForegroundColor Green
            Write-Host "   Key files/folders:" -ForegroundColor Cyan
            $essentials = @("HEAD", "config", "refs", "objects")
            foreach ($item in $essentials) {
                $itemPath = Join-Path $gitDir $item
                if (Test-Path $itemPath) {
                    Write-Host "     [OK] $item" -ForegroundColor Green
                } else {
                    Write-Host "     [MISSING] $item" -ForegroundColor Red
                }
            }
        } else {
            Write-Host "   WARNING: .git directory appears to be empty!" -ForegroundColor Red
        }
    } else {
        Write-Host "   Type: File (might be a git submodule)" -ForegroundColor Yellow
        $content = Get-Content $gitDir
        Write-Host "   Content: $content" -ForegroundColor Gray
    }
} else {
    Write-Host "   .git directory NOT FOUND" -ForegroundColor Red
}
Write-Host ""

# 3. Git command availability
Write-Host "3. Git Command:" -ForegroundColor Yellow
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if ($gitCmd) {
    Write-Host "   Git found at: $($gitCmd.Source)" -ForegroundColor Green
    $version = git --version 2>&1
    Write-Host "   Version: $version" -ForegroundColor Green
} else {
    Write-Host "   Git command NOT FOUND in PATH" -ForegroundColor Red
}
Write-Host ""

# 4. Environment
Write-Host "4. Environment:" -ForegroundColor Yellow
Write-Host "   User: $env:USERNAME" -ForegroundColor White
Write-Host "   Computer: $env:COMPUTERNAME" -ForegroundColor White
Write-Host "   Virtual Env: $(if ($env:VIRTUAL_ENV) { 'YES - ' + $env:VIRTUAL_ENV } else { 'NO' })" -ForegroundColor White
Write-Host ""

# 5. Try git commands with full path
Write-Host "5. Git Repository Test:" -ForegroundColor Yellow
Push-Location D:\Dev\kha
try {
    # Try git status
    $statusResult = git status 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   Git repository VALID" -ForegroundColor Green
        $branch = git rev-parse --abbrev-ref HEAD 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   Current branch: $branch" -ForegroundColor Green
        }
    } else {
        Write-Host "   Git status failed: $statusResult" -ForegroundColor Red
        
        # Try to initialize if needed
        Write-Host ""
        Write-Host "   Attempting to reinitialize..." -ForegroundColor Yellow
        $initResult = git init 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   Repository reinitialized successfully!" -ForegroundColor Green
        } else {
            Write-Host "   Failed to initialize: $initResult" -ForegroundColor Red
        }
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "6. Recommended Actions:" -ForegroundColor Yellow
if (-not (Test-Path $gitDir)) {
    Write-Host "   - Run: git init" -ForegroundColor Cyan
    Write-Host "   - Then: git remote add origin <your-repo-url>" -ForegroundColor Cyan
} elseif ($gitContents.Count -eq 0) {
    Write-Host "   - The .git directory is empty or corrupted" -ForegroundColor Red
    Write-Host "   - Delete .git directory and run: git init" -ForegroundColor Cyan
} else {
    Write-Host "   - Try: cd D:\Dev\kha" -ForegroundColor Cyan
    Write-Host "   - Then: git status" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "=== DIAGNOSTICS COMPLETE ===" -ForegroundColor Cyan
