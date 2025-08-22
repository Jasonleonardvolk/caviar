# Initialize-GitRepo.ps1
# Complete Git repository initialization and setup for iRis/TORI project

param(
    [switch]$SkipInitialCommit,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

# Color output helpers
function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "Git Repository Setup for iRis/TORI" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Yellow

# Step 1: Check if Git is installed
Write-Info "Checking Git installation..."
try {
    $gitVersion = git --version
    Write-Success "Git found: $gitVersion"
} catch {
    Write-Error "Git is not installed or not in PATH"
    exit 1
}

# Step 2: Initialize repository if needed
if (Test-Path ".git") {
    Write-Warning "Repository already initialized"
} else {
    Write-Info "Initializing Git repository..."
    git init
    Write-Success "Repository initialized"
}

# Step 3: Set default branch to main
Write-Info "Setting default branch to 'main'..."
git config init.defaultBranch main
git branch -M main
Write-Success "Default branch set to 'main'"

# Step 4: Configure line ending handling
Write-Info "Configuring line ending handling for Windows..."
git config core.autocrlf true
Write-Success "Line ending configuration set"

# Step 5: Add Git configuration
Write-Info "Setting up Git configuration..."
git config core.longpaths true  # Enable long paths for Windows
git config core.ignorecase false  # Case-sensitive file names
Write-Success "Git configuration complete"

# Step 6: Stage configuration files
Write-Info "Staging configuration files..."
git add .gitignore .gitattributes
Write-Success "Configuration files staged"

# Step 7: Normalize line endings
Write-Info "Normalizing line endings..."
git add --renormalize .
Write-Success "Line endings normalized"

# Step 8: Create initial commit if not skipping
if (-not $SkipInitialCommit) {
    Write-Info "Creating initial alignment commit..."
    git commit -m "chore(repo): align Git naming, ignores, attributes for Windows + SvelteKit"
    Write-Success "Initial commit created"
}

# Step 9: Set up hooks directory
Write-Info "Setting up Git hooks..."
$hooksDir = ".git\hooks"
if (-not (Test-Path $hooksDir)) {
    New-Item -ItemType Directory -Path $hooksDir -Force | Out-Null
}

# Create pre-commit hook
$preCommitHook = @'
#!/bin/sh
# Pre-commit hook for iRis/TORI project

echo "Running pre-commit checks..."

# Run Svelte checks
pnpm exec svelte-check --workspace-root
if [ $? -ne 0 ]; then
    echo "Svelte check failed"
    exit 1
fi

# Run linting
pnpm run lint
if [ $? -ne 0 ]; then
    echo "Linting failed"
    exit 1
fi

echo "Pre-commit checks passed!"
exit 0
'@

$preCommitHook | Out-File -FilePath "$hooksDir\pre-commit" -Encoding ASCII -NoNewline
Write-Success "Pre-commit hook created"

# Create pre-push hook
$prePushHook = @'
#!/bin/sh
# Pre-push hook for iRis/TORI project

echo "Running pre-push checks..."

# Build the project
pnpm run build
if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi

# Run end-to-end verification
powershell -NoProfile -File tools\release\Verify-EndToEnd.ps1
if [ $? -ne 0 ]; then
    echo "End-to-end verification failed"
    exit 1
fi

echo "Pre-push checks passed!"
exit 0
'@

$prePushHook | Out-File -FilePath "$hooksDir\pre-push" -Encoding ASCII -NoNewline
Write-Success "Pre-push hook created"

# Step 10: Create the first tag
Write-Info "Creating initial internal demo tag..."
git tag -a iris-internal-v0.1.0 -m "Internal demo build: unauth + mocks + local uploads"
Write-Success "Tag 'iris-internal-v0.1.0' created"

# Summary
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Git Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`nRepository Configuration:" -ForegroundColor Cyan
Write-Host "  - Default branch: main"
Write-Host "  - Line endings: Normalized for Windows"
Write-Host "  - Hooks: pre-commit and pre-push configured"
Write-Host "  - Initial tag: iris-internal-v0.1.0"

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Review staged changes: git status"
Write-Host "  2. Add remote origin: git remote add origin <your-github-url>"
Write-Host "  3. Push to remote: git push -u origin main --follow-tags"

Write-Host "`nBranch naming convention:" -ForegroundColor Cyan
Write-Host "  - Features: feat/scope-description"
Write-Host "  - Fixes: fix/scope-description"
Write-Host "  - Chores: chore/scope-description"
Write-Host "  - Refactors: refactor/scope-description"

Write-Host ""
