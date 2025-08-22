# Git-Workflow.ps1
# Helper script for Git workflow following iRis/TORI conventions

param(
    [Parameter(Position=0)]
    [ValidateSet("branch", "commit", "tag", "status", "help")]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Type,
    
    [Parameter(Position=2)]
    [string]$Scope,
    
    [Parameter(Position=3)]
    [string]$Description,
    
    [string]$Message,
    [switch]$Push,
    [switch]$List
)

# Branch type definitions
$branchTypes = @{
    "feat" = "New feature"
    "fix" = "Bug fix"
    "chore" = "Maintenance task"
    "docs" = "Documentation"
    "refactor" = "Code refactoring"
    "test" = "Test additions/changes"
    "perf" = "Performance improvements"
    "style" = "Code style changes"
}

# Scope definitions (mapping to directories)
$scopes = @{
    "renderer" = "src\routes\renderer"
    "shaders" = "src\lib\shaders"
    "upload" = "src\routes\upload"
    "api-list" = "src\routes\api\list"
    "api-pdf-stats" = "src\routes\api\pdf\stats"
    "api-memory-state" = "src\routes\api\memory\state"
    "api-health" = "src\routes\api\health"
    "auth" = "src\lib\server\auth.ts"
    "fetch" = "src\lib\server\safeFetch.ts"
    "services" = "src\lib\services"
    "persona" = "src\lib\stores\ghostPersona.ts"
    "elfin" = "src\lib\elfin"
    "config" = "Configuration files (.env, svelte.config.js, etc.)"
    "repo" = "Repository configuration (.gitignore, .gitattributes, etc.)"
}

function Show-Help {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "Git Workflow Helper for iRis/TORI" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    
    Write-Host "`nUsage:" -ForegroundColor Cyan
    Write-Host "  .\Git-Workflow.ps1 branch <type> <scope> <description>"
    Write-Host "  .\Git-Workflow.ps1 commit <type> <scope> <description> [-Message <details>]"
    Write-Host "  .\Git-Workflow.ps1 tag <version> [-Message <description>]"
    Write-Host "  .\Git-Workflow.ps1 status"
    Write-Host "  .\Git-Workflow.ps1 help"
    
    Write-Host "`nBranch Types:" -ForegroundColor Cyan
    foreach ($key in $branchTypes.Keys | Sort-Object) {
        Write-Host ("  {0,-10} - {1}" -f $key, $branchTypes[$key])
    }
    
    Write-Host "`nScopes:" -ForegroundColor Cyan
    foreach ($key in $scopes.Keys | Sort-Object) {
        Write-Host ("  {0,-20} - {1}" -f $key, $scopes[$key])
    }
    
    Write-Host "`nExamples:" -ForegroundColor Green
    Write-Host '  .\Git-Workflow.ps1 branch feat renderer "holographic-modes"'
    Write-Host '  .\Git-Workflow.ps1 commit fix upload "mime-validation" -Message "Add stricter MIME type checking"'
    Write-Host '  .\Git-Workflow.ps1 tag "iris-v1.0.0" -Message "First GA release"'
    
    Write-Host "`nTag Naming Convention:" -ForegroundColor Cyan
    Write-Host "  Internal: iris-internal-v0.1.0"
    Write-Host "  RC:       iris-rc-v1.0.0-rc.1"
    Write-Host "  Release:  iris-v1.0.0"
    Write-Host "  Daily:    iris-cut-2025.08.16"
    Write-Host ""
}

function New-Branch {
    param(
        [string]$BranchType,
        [string]$BranchScope,
        [string]$BranchDescription
    )
    
    if (-not $branchTypes.ContainsKey($BranchType)) {
        Write-Host "Invalid branch type: $BranchType" -ForegroundColor Red
        Write-Host "Valid types: $($branchTypes.Keys -join ', ')" -ForegroundColor Yellow
        return
    }
    
    if (-not $scopes.ContainsKey($BranchScope)) {
        Write-Host "Warning: Unknown scope '$BranchScope'" -ForegroundColor Yellow
        $continue = Read-Host "Continue anyway? (y/n)"
        if ($continue -ne 'y') { return }
    }
    
    $branchName = "$BranchType/$BranchScope-$BranchDescription"
    
    Write-Host "Creating branch: $branchName" -ForegroundColor Cyan
    git switch -c $branchName
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Branch created and switched to: $branchName" -ForegroundColor Green
        
        if ($Push) {
            Write-Host "Pushing branch to origin..." -ForegroundColor Cyan
            git push -u origin $branchName
        }
    }
}

function New-Commit {
    param(
        [string]$CommitType,
        [string]$CommitScope,
        [string]$CommitDescription,
        [string]$CommitMessage
    )
    
    if (-not $branchTypes.ContainsKey($CommitType)) {
        Write-Host "Invalid commit type: $CommitType" -ForegroundColor Red
        Write-Host "Valid types: $($branchTypes.Keys -join ', ')" -ForegroundColor Yellow
        return
    }
    
    if (-not $scopes.ContainsKey($CommitScope)) {
        Write-Host "Warning: Unknown scope '$CommitScope'" -ForegroundColor Yellow
        $continue = Read-Host "Continue anyway? (y/n)"
        if ($continue -ne 'y') { return }
    }
    
    $commitHeader = "$CommitType($CommitScope): $CommitDescription"
    
    if ($CommitMessage) {
        $fullCommitMessage = "$commitHeader`n`n$CommitMessage"
    } else {
        $fullCommitMessage = $commitHeader
    }
    
    Write-Host "Commit message:" -ForegroundColor Cyan
    Write-Host $fullCommitMessage -ForegroundColor White
    Write-Host ""
    
    $confirm = Read-Host "Create commit? (y/n)"
    if ($confirm -eq 'y') {
        git commit -m $fullCommitMessage
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Commit created successfully" -ForegroundColor Green
            
            if ($Push) {
                Write-Host "Pushing to origin..." -ForegroundColor Cyan
                git push
            }
        }
    }
}

function New-Tag {
    param(
        [string]$TagVersion,
        [string]$TagMessage
    )
    
    # Validate tag format
    $validFormats = @(
        "iris-internal-v\d+\.\d+\.\d+",
        "iris-rc-v\d+\.\d+\.\d+-rc\.\d+",
        "iris-v\d+\.\d+\.\d+",
        "iris-cut-\d{4}\.\d{2}\.\d{2}"
    )
    
    $isValid = $false
    foreach ($format in $validFormats) {
        if ($TagVersion -match $format) {
            $isValid = $true
            break
        }
    }
    
    if (-not $isValid) {
        Write-Host "Invalid tag format: $TagVersion" -ForegroundColor Red
        Write-Host "Valid formats:" -ForegroundColor Yellow
        Write-Host "  iris-internal-v0.1.0" -ForegroundColor Gray
        Write-Host "  iris-rc-v1.0.0-rc.1" -ForegroundColor Gray
        Write-Host "  iris-v1.0.0" -ForegroundColor Gray
        Write-Host "  iris-cut-2025.08.16" -ForegroundColor Gray
        return
    }
    
    if (-not $TagMessage) {
        $TagMessage = Read-Host "Enter tag message"
    }
    
    Write-Host "Creating tag: $TagVersion" -ForegroundColor Cyan
    Write-Host "Message: $TagMessage" -ForegroundColor White
    Write-Host ""
    
    $confirm = Read-Host "Create tag? (y/n)"
    if ($confirm -eq 'y') {
        git tag -a $TagVersion -m $TagMessage
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Tag created: $TagVersion" -ForegroundColor Green
            
            if ($Push) {
                Write-Host "Pushing tag to origin..." -ForegroundColor Cyan
                git push origin $TagVersion
            }
        }
    }
}

function Show-Status {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "Repository Status" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    
    # Current branch
    $currentBranch = git branch --show-current
    Write-Host "`nCurrent branch: $currentBranch" -ForegroundColor Cyan
    
    # List branches
    Write-Host "`nLocal branches:" -ForegroundColor Cyan
    git branch
    
    # List recent tags
    Write-Host "`nRecent tags:" -ForegroundColor Cyan
    git tag --sort=-version:refname | Select-Object -First 10
    
    # Working directory status
    Write-Host "`nWorking directory:" -ForegroundColor Cyan
    git status -s
    
    # Recent commits
    Write-Host "`nRecent commits:" -ForegroundColor Cyan
    git log --oneline -10
}

# Main command processing
switch ($Command) {
    "branch" {
        if ($List) {
            git branch -a
        } elseif ($Type -and $Scope -and $Description) {
            New-Branch -BranchType $Type -BranchScope $Scope -BranchDescription $Description
        } else {
            Write-Host "Usage: .\Git-Workflow.ps1 branch <type> <scope> <description>" -ForegroundColor Red
            Write-Host "   or: .\Git-Workflow.ps1 branch -List" -ForegroundColor Red
        }
    }
    
    "commit" {
        if ($Type -and $Scope -and $Description) {
            New-Commit -CommitType $Type -CommitScope $Scope -CommitDescription $Description -CommitMessage $Message
        } else {
            Write-Host "Usage: .\Git-Workflow.ps1 commit <type> <scope> <description> [-Message <details>]" -ForegroundColor Red
        }
    }
    
    "tag" {
        if ($List) {
            git tag --sort=-version:refname
        } elseif ($Type) {
            New-Tag -TagVersion $Type -TagMessage $Description
        } else {
            Write-Host "Usage: .\Git-Workflow.ps1 tag <version> [-Message <description>]" -ForegroundColor Red
            Write-Host "   or: .\Git-Workflow.ps1 tag -List" -ForegroundColor Red
        }
    }
    
    "status" {
        Show-Status
    }
    
    "help" {
        Show-Help
    }
    
    default {
        Show-Help
    }
}
