# Quick refactoring command for large repos (no dry run)
param(
    [string]$BackupDir = ""
)

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "   PATH REFACTORING (23GB REPO)    " -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Replacing: C:\Users\jason\Desktop\tori\kha" -ForegroundColor Yellow
Write-Host "With: {PROJECT_ROOT} (Python) / `${IRIS_ROOT} (other)" -ForegroundColor Green
Write-Host ""

# Quick scan to show count
Write-Host "Quick count of files to process..." -ForegroundColor Gray
$count = (python -c "
import os
from pathlib import Path
root = Path(r'D:\Dev\kha')
old = r'C:\Users\jason\Desktop\tori\kha'
skip = {'.git','.venv','node_modules','dist','build','.cache','__pycache__'}
exts = {'.py','.ts','.tsx','.js','.jsx','.svelte','.json','.md','.txt','.yaml','.yml'}
c = 0
for d,dirs,files in os.walk(root):
    dirs[:] = [x for x in dirs if x not in skip]
    for f in files:
        p = Path(d)/f
        if p.suffix.lower() in exts:
            try:
                if p.stat().st_size < 2000000:
                    if old in p.read_text(encoding='utf-8',errors='ignore'):
                        c += 1
            except: pass
print(c)
" 2>$null)

Write-Host "Found approximately $count files to refactor" -ForegroundColor Cyan
Write-Host ""

if ($BackupDir -eq "") {
    Write-Host "WARNING: No backup directory specified!" -ForegroundColor Red
    $confirm = Read-Host "Continue without backup? (y/n)"
    if ($confirm -ne 'y') {
        Write-Host "Cancelled." -ForegroundColor Yellow
        exit
    }
} else {
    Write-Host "Backup directory: $BackupDir" -ForegroundColor Green
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
}

Write-Host ""
Write-Host "Starting refactoring..." -ForegroundColor Green
Write-Host "This will take some time for 23GB of data." -ForegroundColor Gray
Write-Host "Progress will be shown every 100 files." -ForegroundColor Gray
Write-Host ""

$args = @("tools\refactor\mass_refactor_simple.py")
if ($BackupDir -ne "") {
    $args += @("--backup-dir", $BackupDir)
}

& python @args

Write-Host ""
Write-Host "====================================" -ForegroundColor Green
Write-Host "        REFACTORING COMPLETE!       " -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""
Write-Host "Log files are in: tools\refactor\" -ForegroundColor Gray
