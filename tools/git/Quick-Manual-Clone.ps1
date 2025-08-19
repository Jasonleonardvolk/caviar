# Quick Manual Clone Fix
# Run these commands one by one in PowerShell

# 1. Save your current work (excluding .git and large files)
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = "D:\Dev\kha_backup_$ts"
Write-Host "Creating backup at $backup..."
robocopy D:\Dev\kha $backup /E /XD .git node_modules .venv venv .svelte-kit build dist /XF *.exe *.dll *.mp4 *.mov

# 2. Clone fresh from GitHub to a new location
$freshClone = "D:\Dev\kha_fresh_$ts"
Write-Host "Cloning fresh copy to $freshClone..."
git clone https://github.com/Jasonleonardvolk/Tori.git $freshClone

# 3. Switch to your branch in the fresh clone
cd $freshClone
git checkout -b feat/wowpack-prores-hdr10-pipeline

# 4. Copy your work back (excluding .git)
Write-Host "Copying your work back..."
robocopy $backup $freshClone /E /XD .git

# 5. Stage and check
git add .
git status

Write-Host ""
Write-Host "Fresh clone ready at: $freshClone" -ForegroundColor Green
Write-Host "Your backup is at: $backup" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. cd $freshClone"
Write-Host "2. git status  # Check what needs committing"
Write-Host "3. git commit -m 'feat(wowpack): Recovery after corruption'"
Write-Host "4. git push -u origin feat/wowpack-prores-hdr10-pipeline"