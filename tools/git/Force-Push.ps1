# Force push despite repository corruption

Write-Host "Attempting to push branch despite repository issues..." -ForegroundColor Yellow
Write-Host ""

# Get current branch
$currentBranch = git rev-parse --abbrev-ref HEAD 2>$null
if ($currentBranch) {
    Write-Host "Current branch: $currentBranch" -ForegroundColor Cyan
} else {
    Write-Host "Could not determine current branch" -ForegroundColor Red
    exit 1
}

# Try to push
Write-Host "Attempting push to origin/$currentBranch..." -ForegroundColor Yellow
$pushResult = git push -u origin $currentBranch 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "Push successful!" -ForegroundColor Green
    Write-Host $pushResult
} else {
    Write-Host "Standard push failed. Trying force push..." -ForegroundColor Yellow
    
    # Try force push (be careful with this!)
    $forcePushResult = git push -u origin $currentBranch --force-with-lease 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Force push successful!" -ForegroundColor Green
        Write-Host $forcePushResult
    } else {
        Write-Host "Push failed:" -ForegroundColor Red
        Write-Host $forcePushResult
        
        Write-Host ""
        Write-Host "Alternative approach - creating a fresh branch:" -ForegroundColor Yellow
        
        # Create a new branch from current state
        $newBranch = "$currentBranch-fixed"
        git checkout -b $newBranch 2>&1 | Out-Null
        
        # Try to push the new branch
        $newPushResult = git push -u origin $newBranch 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Successfully pushed as new branch: $newBranch" -ForegroundColor Green
            Write-Host $newPushResult
        } else {
            Write-Host "All push attempts failed" -ForegroundColor Red
            Write-Host $newPushResult
        }
    }
}

Write-Host ""
Write-Host "Note: Your repository has corruption issues that should be fixed." -ForegroundColor Yellow
Write-Host "Run .\tools\git\Repair-Git.ps1 after pushing to fix the corruption." -ForegroundColor Cyan
