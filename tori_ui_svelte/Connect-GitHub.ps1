# Connect-GitHub.ps1
# Script to connect local Git repository to GitHub

param(
    [string]$GitHubUsername = "Jasonleonardvolk",
    [string]$RepoName = "iRis",
    [switch]$CreateNew,
    [switch]$UseSSH
)

$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "GitHub Repository Connection Setup" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Yellow

# Step 1: Check current Git status
Write-Host "Checking Git status..." -ForegroundColor Cyan
$currentBranch = git branch --show-current
$hasCommits = git log --oneline -1 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] No commits found. Please create an initial commit first." -ForegroundColor Yellow
    Write-Host "Run: git commit -m `"chore(repo): initial Git setup with iRis/TORI conventions`"" -ForegroundColor White
    exit 1
}

Write-Host "[OK] Repository has commits on branch: $currentBranch" -ForegroundColor Green

# Step 2: Check for existing remotes
$existingRemote = git remote get-url origin 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "[INFO] Remote 'origin' already exists: $existingRemote" -ForegroundColor Yellow
    $replace = Read-Host "Replace with new remote? (y/n)"
    if ($replace -eq 'y') {
        git remote remove origin
        Write-Host "[OK] Removed existing remote" -ForegroundColor Green
    } else {
        Write-Host "Using existing remote. Run 'git push -u origin main --follow-tags' to push." -ForegroundColor Cyan
        exit 0
    }
}

# Step 3: Construct GitHub URL
if ($UseSSH) {
    $remoteUrl = "git@github.com:$GitHubUsername/$RepoName.git"
} else {
    $remoteUrl = "https://github.com/$GitHubUsername/$RepoName.git"
}

Write-Host "`n[INFO] GitHub Repository URL:" -ForegroundColor Cyan
Write-Host "  $remoteUrl" -ForegroundColor White

# Step 4: Instructions for creating repository on GitHub
if ($CreateNew) {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "Create Repository on GitHub" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    
    Write-Host "`nOpen your browser and go to:" -ForegroundColor Cyan
    Write-Host "  https://github.com/new" -ForegroundColor White
    
    Write-Host "`nRepository settings:" -ForegroundColor Cyan
    Write-Host "  - Repository name: $RepoName" -ForegroundColor White
    Write-Host "  - Description: iRis - Holographic UI with memory synthesis" -ForegroundColor White
    Write-Host "  - Public or Private: Your choice" -ForegroundColor White
    Write-Host "  - DO NOT initialize with README, .gitignore, or license" -ForegroundColor Yellow
    Write-Host "    (We already have these locally)" -ForegroundColor Yellow
    
    Write-Host "`nPress Enter after creating the repository on GitHub..." -ForegroundColor Green
    Read-Host
}

# Step 5: Add remote origin
Write-Host "`n[INFO] Adding remote origin..." -ForegroundColor Cyan
git remote add origin $remoteUrl

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Remote origin added successfully" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Failed to add remote" -ForegroundColor Red
    exit 1
}

# Step 6: Verify remote
Write-Host "`n[INFO] Verifying remote configuration..." -ForegroundColor Cyan
git remote -v

# Step 7: Push to GitHub
Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "Ready to Push to GitHub" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

Write-Host "`nThis will push:" -ForegroundColor Cyan
Write-Host "  - Branch: main" -ForegroundColor White
Write-Host "  - Tag: iris-internal-v0.1.0" -ForegroundColor White
Write-Host "  - All commits" -ForegroundColor White

$push = Read-Host "`nPush to GitHub now? (y/n)"

if ($push -eq 'y') {
    Write-Host "`n[INFO] Pushing to GitHub..." -ForegroundColor Cyan
    
    # Push main branch with tags
    git push -u origin main --follow-tags
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n========================================" -ForegroundColor Green
        Write-Host "Successfully Pushed to GitHub!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        
        Write-Host "`nYour repository is now live at:" -ForegroundColor Cyan
        Write-Host "  https://github.com/$GitHubUsername/$RepoName" -ForegroundColor White
        
        Write-Host "`nNext steps:" -ForegroundColor Yellow
        Write-Host "  1. Create a feature branch:" -ForegroundColor White
        Write-Host "     .\Git-Workflow.ps1 branch feat renderer `"holographic-modes`"" -ForegroundColor Gray
        Write-Host "  2. Make changes and commit:" -ForegroundColor White
        Write-Host "     .\Git-Workflow.ps1 commit feat renderer `"add holographic display`"" -ForegroundColor Gray
        Write-Host "  3. Push feature branch:" -ForegroundColor White
        Write-Host "     git push -u origin feat/renderer-holographic-modes" -ForegroundColor Gray
        
        Write-Host "`nWorkflow commands:" -ForegroundColor Cyan
        Write-Host "  .\Git-Workflow.ps1 help     - Show all workflow commands" -ForegroundColor White
        Write-Host "  .\Git-Workflow.ps1 status   - Check repository status" -ForegroundColor White
        
    } else {
        Write-Host "`n[ERROR] Push failed. Check your GitHub credentials and repository." -ForegroundColor Red
        Write-Host "If using HTTPS, you may need a Personal Access Token instead of password." -ForegroundColor Yellow
        Write-Host "Create one at: https://github.com/settings/tokens" -ForegroundColor White
    }
} else {
    Write-Host "`n[INFO] Push cancelled. To push later, run:" -ForegroundColor Yellow
    Write-Host "  git push -u origin main --follow-tags" -ForegroundColor White
}

Write-Host ""
