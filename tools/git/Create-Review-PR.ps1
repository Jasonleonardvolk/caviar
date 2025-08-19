# Create-Review-PR.ps1
# Creates a Draft PR for AI review (Claude/ChatGPT with GitHub connector)
param(
    [string]$BranchName = "review/$(Get-Date -Format 'yyyyMMdd-HHmm')",
    [string]$Title = "Code Review Request - $(Get-Date -Format 'yyyy-MM-dd')",
    [string]$Body = "Please review the following changes:",
    [switch]$OpenInBrowser
)

Write-Host @"
╔════════════════════════════════════════╗
║     CREATE REVIEW PR FOR AI            ║
║     Makes code visible to Claude/GPT    ║
╚════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Ensure we're in the repo
if (!(Test-Path .git)) {
    Write-Host "ERROR: Not in a git repository!" -ForegroundColor Red
    Write-Host "Run from: D:\Dev\kha" -ForegroundColor Yellow
    exit 1
}

# Check for changes
$changes = git status --porcelain
if (!$changes) {
    Write-Host "No changes to review!" -ForegroundColor Yellow
    exit 0
}

Write-Host "`nCreating review branch: $BranchName" -ForegroundColor Yellow

# Create and switch to review branch
git checkout -b $BranchName

# Add all changes
git add -A

# Commit
git commit -m "review: changes for AI review"

# Push to GitHub
Write-Host "`nPushing to GitHub..." -ForegroundColor Cyan
git push -u origin $BranchName

if ($LASTEXITCODE -ne 0) {
    Write-Host "Push failed!" -ForegroundColor Red
    exit 1
}

# Create PR using GitHub CLI if available
$hasGH = Get-Command gh -ErrorAction SilentlyContinue

if ($hasGH) {
    Write-Host "`nCreating Draft PR..." -ForegroundColor Cyan
    
    $prUrl = gh pr create `
        --title $Title `
        --body $Body `
        --base main `
        --head $BranchName `
        --draft `
        --web:$false
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Draft PR created!" -ForegroundColor Green
        Write-Host "PR URL: $prUrl" -ForegroundColor Cyan
        
        # Extract PR number
        if ($prUrl -match '/pull/(\d+)') {
            $prNumber = $matches[1]
            Write-Host "`n=== SHARE WITH AI ===" -ForegroundColor Magenta
            Write-Host "Tell Claude/ChatGPT:" -ForegroundColor Yellow
            Write-Host "  'Review PR #$prNumber in Jasonleonardvolk/caviar'" -ForegroundColor White
            Write-Host "  or" -ForegroundColor Gray
            Write-Host "  'Check draft PR: $prUrl'" -ForegroundColor White
        }
        
        if ($OpenInBrowser) {
            Start-Process $prUrl
        }
    }
} else {
    # Manual instructions if no GitHub CLI
    $repoUrl = git remote get-url origin
    $compareUrl = $repoUrl -replace '\.git$', ''
    $compareUrl = "$compareUrl/compare/main...$BranchName"
    
    Write-Host "`n✅ Branch pushed!" -ForegroundColor Green
    Write-Host "`nNow create a Draft PR manually:" -ForegroundColor Yellow
    Write-Host "1. Open: $compareUrl" -ForegroundColor White
    Write-Host "2. Click 'Create pull request'" -ForegroundColor White
    Write-Host "3. Mark as 'Draft' (dropdown arrow)" -ForegroundColor White
    Write-Host "4. Share PR number with AI" -ForegroundColor White
    
    if ($OpenInBrowser) {
        Start-Process $compareUrl
    }
}

# Switch back to main
git checkout main

Write-Host "`n=== NEXT STEPS ===" -ForegroundColor Cyan
Write-Host "1. AI reviews the PR" -ForegroundColor Gray
Write-Host "2. Apply suggested changes locally" -ForegroundColor Gray
Write-Host "3. Push updates to the same branch" -ForegroundColor Gray
Write-Host "4. Merge when ready" -ForegroundColor Gray