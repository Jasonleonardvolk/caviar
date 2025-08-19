# Commit-EOD.ps1
# One-liner daily push for end-of-day commits

param(
    [Parameter()]
    [string]$Message = "",
    
    [Parameter()]
    [switch]$Push = $true,
    
    [Parameter()]
    [switch]$ShowDiff
)

Write-Host "`n=== End of Day Commit ===" -ForegroundColor Yellow

# Get current directory name for context
$projectName = Split-Path -Leaf (Get-Location)
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$dayOfWeek = (Get-Date).DayOfWeek

# Build commit message
if ([string]::IsNullOrWhiteSpace($Message)) {
    $Message = "EOD: $dayOfWeek checkpoint - $timestamp"
} else {
    $Message = "EOD: $Message - $timestamp"
}

# Show current status
Write-Host "`nChecking repository status..." -ForegroundColor Cyan
$status = git status --short

if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "No changes to commit!" -ForegroundColor Green
    exit 0
}

Write-Host "Changes detected:" -ForegroundColor Yellow
Write-Host $status

# Show diff if requested
if ($ShowDiff) {
    Write-Host "`nShowing diff..." -ForegroundColor Cyan
    git diff --stat
    Write-Host ""
}

# Count changes
$changedFiles = ($status -split "`n").Count
Write-Host "`nFiles affected: $changedFiles" -ForegroundColor Cyan

# Confirm before proceeding
$confirm = Read-Host "`nCommit all changes with message '$Message'? (y/N)"
if ($confirm -ne 'y') {
    Write-Host "Aborted!" -ForegroundColor Red
    exit 0
}

# Stage all changes
Write-Host "`nStaging all changes..." -ForegroundColor Cyan
git add -A

# Commit
Write-Host "Committing..." -ForegroundColor Cyan
git commit -m $Message

if ($LASTEXITCODE -ne 0) {
    Write-Host "Commit failed!" -ForegroundColor Red
    exit 1
}

# Push if requested
if ($Push) {
    Write-Host "`nPushing to remote..." -ForegroundColor Cyan
    git push
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nSuccessfully pushed EOD commit!" -ForegroundColor Green
        
        # Show summary
        $lastCommit = git log -1 --oneline
        Write-Host "`nLast commit: $lastCommit" -ForegroundColor Cyan
        
        # Fun EOD message
        $eodMessages = @(
            "Great work today! Time to rest.",
            "Another productive day in the books!",
            "Code secured. See you tomorrow!",
            "EOD checkpoint saved. Well done!",
            "Daily progress backed up. Nice job!"
        )
        $randomMessage = $eodMessages | Get-Random
        Write-Host "`n$randomMessage" -ForegroundColor Magenta
    } else {
        Write-Host "Push failed! Run 'git push' manually." -ForegroundColor Red
    }
} else {
    Write-Host "`nCommit complete (not pushed)" -ForegroundColor Yellow
    Write-Host "Run 'git push' when ready" -ForegroundColor Cyan
}

Write-Host "`n=== EOD Complete ===" -ForegroundColor Yellow