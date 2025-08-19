param(
    [Parameter(Position=0, Mandatory=$true)]
    [ValidateSet("branch", "commit", "push", "status", "sync")]
    [string]$Action,
    
    [Parameter(Position=1)]
    [string]$Type,
    
    [Parameter(Position=2)]
    [string]$Scope,
    
    [Parameter(Position=3)]
    [string]$Name,
    
    [switch]$Push
)

# Ensure we're in a git repository
if (-not (Test-Path .git)) {
    Write-Error "Not in a git repository. Please run from the repository root."
    exit 1
}

# Color output helpers
function Write-Success($msg) { Write-Host $msg -ForegroundColor Green }
function Write-Info($msg) { Write-Host $msg -ForegroundColor Cyan }
function Write-Warning($msg) { Write-Host $msg -ForegroundColor Yellow }

switch ($Action) {
    "branch" {
        if (-not $Type -or -not $Scope -or -not $Name) {
            Write-Error "Usage: .\Git-Workflow.ps1 branch <type> <scope> <name> [-Push]"
            Write-Info "Example: .\Git-Workflow.ps1 branch feat auth login-flow -Push"
            exit 1
        }
        
        $branchName = "$Type/$Scope-$Name"
        Write-Info "Creating branch: $branchName"
        
        # Ensure we're on main/master and up to date
        git checkout main 2>$null
        if ($LASTEXITCODE -ne 0) {
            git checkout master 2>$null
        }
        
        git pull origin
        
        # Create and checkout new branch
        git checkout -b $branchName
        
        if ($Push) {
            Write-Info "Pushing branch to origin..."
            git push -u origin $branchName
        }
        
        Write-Success "Branch '$branchName' created successfully!"
    }
    
    "commit" {
        if (-not $Type -or -not $Scope) {
            Write-Error "Usage: .\Git-Workflow.ps1 commit <type> <scope> <message>"
            Write-Info "Example: .\Git-Workflow.ps1 commit fix auth 'resolve login timeout'"
            exit 1
        }
        
        $message = if ($Name) { $Name } else { Read-Host "Enter commit message" }
        $fullMessage = "${Type}(${Scope}): $message"
        
        Write-Info "Committing with message: $fullMessage"
        git add -A
        git commit -m $fullMessage
        
        Write-Success "Commit created successfully!"
    }
    
    "push" {
        Write-Info "Pushing to origin..."
        $currentBranch = git rev-parse --abbrev-ref HEAD
        git push origin $currentBranch
        Write-Success "Pushed to origin/$currentBranch"
    }
    
    "status" {
        Write-Info "Repository Status:"
        git status
        Write-Info "`nBranch Info:"
        git branch -vv
        Write-Info "`nRecent commits:"
        git log --oneline -5
    }
    
    "sync" {
        Write-Info "Syncing with upstream..."
        $currentBranch = git rev-parse --abbrev-ref HEAD
        
        # Stash any local changes
        $hasChanges = git status --porcelain
        if ($hasChanges) {
            Write-Warning "Stashing local changes..."
            git stash
        }
        
        # Pull latest changes
        git pull origin $currentBranch
        
        # Restore stashed changes if any
        if ($hasChanges) {
            Write-Info "Restoring stashed changes..."
            git stash pop
        }
        
        Write-Success "Sync complete!"
    }
    
    default {
        Write-Error "Unknown action: $Action"
        Write-Info "Available actions: branch, commit, push, status, sync"
    }
}
