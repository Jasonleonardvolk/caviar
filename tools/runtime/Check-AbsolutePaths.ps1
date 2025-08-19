# Check-AbsolutePaths.ps1
# PowerShell gate check to ensure no absolute paths in source code
# Excludes docs/conversations for historical preservation

param(
    [string]$RepoRoot = "D:\Dev\kha"
)

Write-Host "Checking for absolute path references..." -ForegroundColor Cyan
Write-Host "-" * 60

$pattern = 'C:\\Users\\jason\\Desktop\\tori\\kha'
$extensions = @("*.py","*.ts","*.tsx","*.js","*.jsx","*.svelte","*.wgsl","*.txt","*.json","*.md","*.yaml","*.yml")

$hitList = Get-ChildItem -Path $RepoRoot -Recurse -File -Include $extensions |
    Where-Object { 
        $_.Length -lt 2000000 -and
        $_.FullName -notmatch '\\docs\\conversations\\' -and 
        $_.FullName -notmatch '\\docs\\shiporder\.txt' -and
        $_.Name -ne 'RELEASE_GATE_COMPLETE.md' -and
        $_.FullName -notmatch '\\tools\\runtime\\README\.md' -and
        $_.FullName -notmatch '\\node_modules\\' -and 
        $_.FullName -notmatch '\\\.venv\\' -and
        $_.FullName -notmatch '\\venv\\' -and
        $_.FullName -notmatch '\\dist\\' -and 
        $_.FullName -notmatch '\\build\\' -and
        $_.FullName -notmatch '\\\.git\\' -and
        $_.FullName -notmatch '\\__pycache__\\' -and
        $_.FullName -notmatch '\\\.cache\\' -and
        $_.FullName -notmatch '\\tools\\dawn\\'
    } |
    Select-String -SimpleMatch $pattern -List

if ($hitList) {
    Write-Host "ERROR: Found hard-coded absolute paths in:" -ForegroundColor Red
    $hitList | ForEach-Object { 
        $relative = $_.Path.Replace("$RepoRoot\", "")
        Write-Host " - $relative" -ForegroundColor Yellow
    }
    Write-Host "-" * 60
    Write-Host "Total files with absolute paths: $($hitList.Count)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please use `${IRIS_ROOT} or {PROJECT_ROOT} instead." -ForegroundColor Cyan
    Write-Host "To fix, run: python tools\refactor\refactor_continue.py" -ForegroundColor Green
    exit 2
} else {
    Write-Host "OK: No absolute path references found!" -ForegroundColor Green
    Write-Host "(Excluding docs\conversations for historical preservation)" -ForegroundColor Gray
}
