# Create-ReleaseCandidate.ps1
# Creates a release candidate branch and tag

param(
    [Parameter(Mandatory=$true)]
    [string]$Version,  # e.g., "1.0.0-rc.1"
    
    [string]$Message = "Release candidate",
    
    [switch]$Push
)

Write-Host "=== CREATING RELEASE CANDIDATE $Version ===" -ForegroundColor Cyan
Write-Host ""

# Ensure we're in the right directory
if (-not (Test-Path .git)) {
    Set-Location D:\Dev\kha
    if (-not (Test-Path .git)) {
        Write-Error "Not in a git repository"
        exit 1
    }
}

# Create release branch
$branchName = "release/v$Version"
Write-Host "Creating branch: $branchName" -ForegroundColor Yellow

# Ensure we're on latest main
git checkout main 2>$null
if ($LASTEXITCODE -ne 0) {
    git checkout master 2>$null
}
git pull origin

# Create and checkout release branch
git checkout -b $branchName
if ($LASTEXITCODE -ne 0) {
    Write-Host "Branch might already exist, switching to it..." -ForegroundColor Yellow
    git checkout $branchName
}

# Commit any pending changes
$changes = git status --porcelain
if ($changes) {
    Write-Host "Committing pending changes..." -ForegroundColor Yellow
    git add -A
    git commit -m "chore: prepare release candidate $Version - $Message"
}

# Create tag
$tagName = "v$Version"
Write-Host "Creating tag: $tagName" -ForegroundColor Yellow
git tag -a $tagName -m "Release Candidate: $Version - $Message"

if ($Push) {
    Write-Host ""
    Write-Host "Pushing branch and tag to origin..." -ForegroundColor Yellow
    git push -u origin $branchName
    git push origin $tagName
    
    Write-Host ""
    Write-Host "Release candidate $Version created and pushed!" -ForegroundColor Green
    Write-Host "Branch: $branchName" -ForegroundColor Cyan
    Write-Host "Tag: $tagName" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Release candidate $Version created locally!" -ForegroundColor Green
    Write-Host "Branch: $branchName" -ForegroundColor Cyan
    Write-Host "Tag: $tagName" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To push:" -ForegroundColor Yellow
    Write-Host "  git push -u origin $branchName" -ForegroundColor White
    Write-Host "  git push origin $tagName" -ForegroundColor White
}
