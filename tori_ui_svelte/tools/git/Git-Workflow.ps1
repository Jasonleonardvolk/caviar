# Git-Workflow.ps1
# iRis Git workflow automation - branching, tagging, and release management
# Usage: .\Git-Workflow.ps1 <command> [parameters]

[CmdletBinding()]
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet('branch', 'internal', 'rc', 'release', 'tag', 'status')]
    [string]$Command,
    
    [Parameter(Position = 1)]
    [string]$Type,
    
    [Parameter(Position = 2)]
    [string]$Scope,
    
    [Parameter(Position = 3)]
    [string]$Name,
    
    [string]$Message = "",
    [switch]$Push,
    [switch]$Force,
    [switch]$SkipTests
)

# Find repository root by walking up to .git
function Find-GitRoot {
    $current = Get-Location
    while ($current) {
        if (Test-Path (Join-Path $current ".git")) {
            return $current.Path
        }
        $parent = Split-Path $current -Parent
        if ($parent -eq $current) { break }
        $current = $parent
    }
    throw "Not in a Git repository (no .git found)"
}

# Normalize branch names
function Format-BranchName {
    param([string]$name)
    return $name -replace '[^a-zA-Z0-9\-_/]', '-' -replace '-+', '-' -replace '^-|-$', ''
}

# Run smoke tests
function Invoke-SmokeTests {
    Write-Host "`n[*] Running pre-push smoke tests..." -ForegroundColor Yellow
    
    $verifyScript = Join-Path $repoRoot "tools\release\verify-setup.ps1"
    
    if (Test-Path $verifyScript) {
        Write-Host "Running verify-setup.ps1..." -ForegroundColor Cyan
        & $verifyScript
        if ($LASTEXITCODE -ne 0) {
            throw "Smoke tests failed! Fix issues before proceeding."
        }
    } else {
        Write-Host "verify-setup.ps1 not found, running minimal HTTP probes..." -ForegroundColor Yellow
        
        $endpoints = @(
            "http://localhost:3000/",
            "http://localhost:3000/renderer",
            "http://localhost:3000/upload",
            "http://localhost:3000/api/list",
            "http://localhost:3000/api/pdf/stats",
            "http://localhost:3000/api/memory/state"
        )
        
        $failed = $false
        foreach ($url in $endpoints) {
            try {
                $response = Invoke-WebRequest -Uri $url -UseBasicParsing -Method GET -TimeoutSec 5
                if ($response.StatusCode -eq 200) {
                    Write-Host "  [OK] $url" -ForegroundColor Green
                } else {
                    Write-Host "  [X] $url (Status: $($response.StatusCode))" -ForegroundColor Red
                    $failed = $true
                }
            } catch {
                Write-Host "  [X] $url (Error: $($_.Exception.Message))" -ForegroundColor Red
                $failed = $true
            }
        }
        
        if ($failed) {
            throw "HTTP endpoint checks failed!"
        }
    }
    
    Write-Host "[OK] All smoke tests passed!" -ForegroundColor Green
}

# Ensure we're on latest origin/main
function Update-FromOrigin {
    Write-Host "[*] Fetching latest from origin..." -ForegroundColor Cyan
    git fetch origin
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to fetch from origin"
    }
}

# Check for uncommitted changes
function Test-CleanWorkingTree {
    $status = git status --porcelain
    if ($status) {
        Write-Host "[!] Working tree has uncommitted changes:" -ForegroundColor Yellow
        Write-Host $status
        if (!$Force) {
            throw "Working tree not clean. Commit or stash changes first (or use -Force)"
        }
        Write-Host "  Proceeding with -Force flag..." -ForegroundColor Yellow
    }
}

# Main execution
try {
    $repoRoot = Find-GitRoot
    Set-Location $repoRoot
    Write-Host "[*] Repository root: $repoRoot" -ForegroundColor Green
    
    # Get current branch and status
    $currentBranch = git rev-parse --abbrev-ref HEAD
    Write-Host "[*] Current branch: $currentBranch" -ForegroundColor Cyan
    
    switch ($Command) {
        'status' {
            Write-Host "`n[*] Repository Status:" -ForegroundColor Yellow
            git status
            Write-Host "`n[*] Recent tags:" -ForegroundColor Yellow
            git tag -l "iris-*" | Select-Object -Last 10 | Sort-Object -Descending
            Write-Host "`n[*] Recent branches:" -ForegroundColor Yellow
            git branch -r --sort=-committerdate | Select-Object -First 10
        }
        
        'branch' {
            if (!$Type) { throw "Branch type required (feat/fix/chore/docs)" }
            if (!$Scope) { throw "Scope required (e.g., renderer, api, storage)" }
            if (!$Name) { throw "Branch name required" }
            
            Test-CleanWorkingTree
            Update-FromOrigin
            
            $branchName = "$Type/$(Format-BranchName "$Scope-$Name")"
            Write-Host "`n[*] Creating branch: $branchName" -ForegroundColor Green
            
            # Create branch from origin/main
            git checkout -b $branchName origin/main
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to create branch $branchName"
            }
            
            if ($Push) {
                Write-Host "[*] Pushing branch to origin..." -ForegroundColor Cyan
                git push -u origin $branchName
                if ($LASTEXITCODE -ne 0) {
                    throw "Failed to push branch"
                }
            }
            
            Write-Host "[OK] Branch created: $branchName" -ForegroundColor Green
        }
        
        'internal' {
            if (!$Type) { throw "Version required (e.g., 0.1.1)" }
            
            Test-CleanWorkingTree
            if (!$SkipTests) { Invoke-SmokeTests }
            
            $tagName = "iris-internal-v$Type"
            $tagMessage = if ($Message) { $Message } else { "Internal release v$Type" }
            
            Write-Host "`n[*] Creating internal tag: $tagName" -ForegroundColor Green
            git tag -a $tagName -m "$tagMessage"
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to create tag $tagName"
            }
            
            if ($Push) {
                Write-Host "[*] Pushing tag to origin..." -ForegroundColor Cyan
                git push origin $tagName
                if ($LASTEXITCODE -ne 0) {
                    throw "Failed to push tag"
                }
            }
            
            Write-Host "[OK] Internal tag created: $tagName" -ForegroundColor Green
        }
        
        'rc' {
            if (!$Type) { throw "RC version required (e.g., 1.0.0-rc.1)" }
            
            Test-CleanWorkingTree
            if (!$SkipTests) { Invoke-SmokeTests }
            
            $tagName = "iris-rc-v$Type"
            $tagMessage = if ($Message) { $Message } else { "Release candidate v$Type" }
            
            Write-Host "`n[*] Creating RC tag: $tagName" -ForegroundColor Green
            git tag -a $tagName -m "$tagMessage"
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to create tag $tagName"
            }
            
            if ($Push) {
                Write-Host "[*] Pushing tag to origin..." -ForegroundColor Cyan
                git push origin $tagName
                if ($LASTEXITCODE -ne 0) {
                    throw "Failed to push tag"
                }
            }
            
            Write-Host "[OK] RC tag created: $tagName" -ForegroundColor Green
        }
        
        'release' {
            if (!$Type) { throw "Release version required (e.g., 1.0.0)" }
            
            Test-CleanWorkingTree
            Update-FromOrigin
            
            # Always run smoke tests for releases
            Invoke-SmokeTests
            
            $tagName = "iris-v$Type"
            $tagMessage = if ($Message) { $Message } else { "iRis release v$Type" }
            
            Write-Host "`n[*] Creating release tag: $tagName" -ForegroundColor Green
            
            # Update package.json version if it exists
            $packageJson = Join-Path $repoRoot "package.json"
            if (Test-Path $packageJson) {
                Write-Host "[*] Updating package.json version to $Type..." -ForegroundColor Cyan
                $content = Get-Content $packageJson -Raw | ConvertFrom-Json
                $content.version = $Type
                $content | ConvertTo-Json -Depth 10 | Set-Content $packageJson
                
                git add package.json
                git commit -m "chore: bump version to $Type for release"
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "  [OK] package.json updated" -ForegroundColor Green
                }
            }
            
            # Create annotated tag
            git tag -a $tagName -m "$tagMessage"
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to create tag $tagName"
            }
            
            if ($Push) {
                Write-Host "[*] Pushing release to origin..." -ForegroundColor Cyan
                git push origin main --tags
                if ($LASTEXITCODE -ne 0) {
                    throw "Failed to push release"
                }
                
                # Create GitHub release notes file
                $releaseNotes = @"
# iRis $Type Release

## Highlights
$tagMessage

## Installation
```bash
git clone https://github.com/yourusername/iris.git
cd iris
git checkout $tagName
pnpm install
pnpm build
```

## Running
```bash
# Development
pnpm dev

# Production
PORT=3000 node build/index.js
```

## Endpoints
- Health: http://localhost:3000/api/health
- Renderer: http://localhost:3000/renderer
- Upload: http://localhost:3000/upload

## Verification
Run smoke tests:
```powershell
.\tools\release\verify-setup.ps1
```

---
Tagged: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
"@
                $releaseNotesFile = Join-Path $repoRoot "RELEASE_NOTES_$Type.md"
                $releaseNotes | Out-File -FilePath $releaseNotesFile -Encoding UTF8
                Write-Host "[*] Release notes created: $releaseNotesFile" -ForegroundColor Green
            }
            
            Write-Host "`n[OK] Release $tagName created successfully!" -ForegroundColor Green
            Write-Host "[*] Next steps:" -ForegroundColor Yellow
            Write-Host "  1. Push with: git push origin main --tags" -ForegroundColor White
            Write-Host "  2. Create GitHub release from tag $tagName" -ForegroundColor White
            Write-Host "  3. Deploy to production" -ForegroundColor White
        }
        
        'tag' {
            # Generic tag creation
            if (!$Type) { throw "Tag name required" }
            
            Test-CleanWorkingTree
            
            $tagName = if ($Type -match '^iris-') { $Type } else { "iris-$Type" }
            $tagMessage = if ($Message) { $Message } else { "Tag $tagName" }
            
            Write-Host "`n[*] Creating tag: $tagName" -ForegroundColor Green
            git tag -a $tagName -m "$tagMessage"
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to create tag $tagName"
            }
            
            if ($Push) {
                Write-Host "[*] Pushing tag to origin..." -ForegroundColor Cyan
                git push origin $tagName
                if ($LASTEXITCODE -ne 0) {
                    throw "Failed to push tag"
                }
            }
            
            Write-Host "[OK] Tag created: $tagName" -ForegroundColor Green
        }
    }
    
} catch {
    Write-Host "`n[X] Error: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`n[OK] Git workflow completed successfully!" -ForegroundColor Green