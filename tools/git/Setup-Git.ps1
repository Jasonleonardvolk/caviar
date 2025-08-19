# Initialize Git repository if needed and run test workflow

param(
    [switch]$Force
)

Write-Host "Git Repository Setup" -ForegroundColor Cyan
Write-Host ("=" * 50)

# Ensure we're in the correct directory
$expectedPath = "D:\Dev\kha"
if ((Get-Location).Path -ne $expectedPath) {
    Write-Host "Changing to repository root: $expectedPath" -ForegroundColor Yellow
    Set-Location $expectedPath
}

# Check if .git exists
if (Test-Path .git) {
    Write-Host "Git repository exists" -ForegroundColor Green
    
    # Verify git command works
    $gitTest = git --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Git command available: $gitTest" -ForegroundColor Green
    } else {
        Write-Host "Git command not working. Trying to fix PATH..." -ForegroundColor Yellow
        
        # Common Git installation paths on Windows
        $gitPaths = @(
            "C:\Program Files\Git\cmd",
            "C:\Program Files (x86)\Git\cmd",
            "C:\Git\cmd",
            "$env:LOCALAPPDATA\Programs\Git\cmd"
        )
        
        foreach ($path in $gitPaths) {
            if (Test-Path $path) {
                Write-Host "  Found Git at: $path" -ForegroundColor Cyan
                $env:PATH = "$path;$env:PATH"
                break
            }
        }
        
        # Test again
        $gitTest = git --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Git now working: $gitTest" -ForegroundColor Green
        } else {
            Write-Host "Still cannot run git. Please check Git installation." -ForegroundColor Red
            exit 1
        }
    }
    
} else {
    if ($Force) {
        Write-Host "No .git directory found. Initializing new repository..." -ForegroundColor Yellow
        git init
        Write-Host "Git repository initialized" -ForegroundColor Green
    } else {
        Write-Host "No .git directory found" -ForegroundColor Red
        Write-Host "  This might be a permissions issue or the .git folder is hidden" -ForegroundColor Yellow
        Write-Host "  Run with -Force to initialize a new repository" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "Testing workflow commands..." -ForegroundColor Cyan
Write-Host ("-" * 30)

# Create test file
$testFile = "WORKFLOW_TEST.txt"
"Git workflow test at $(Get-Date)" | Out-File $testFile

# Add and check status
git add $testFile 2>&1 | Out-Null
$status = git status --short
Write-Host "Status after adding test file:" -ForegroundColor Yellow
Write-Host $status

# Remove test file
Remove-Item $testFile -Force -ErrorAction SilentlyContinue
git rm --cached $testFile 2>&1 | Out-Null

Write-Host ""
Write-Host "Git workflow test complete!" -ForegroundColor Green
Write-Host ""
Write-Host "You can now use:" -ForegroundColor Cyan
Write-Host '  .\tools\git\Git-Workflow.ps1 branch feat test workflow-demo -Push' -ForegroundColor White
Write-Host '  .\tools\git\Git-Workflow.ps1 commit chore test "verify pre-push gate"' -ForegroundColor White
Write-Host "  .\tools\git\Git-Workflow.ps1 push" -ForegroundColor White
