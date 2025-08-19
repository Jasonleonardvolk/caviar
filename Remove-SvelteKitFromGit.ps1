# PowerShell script to remove SvelteKit generated files from git tracking

Write-Host "Removing .svelte-kit directories from git tracking..." -ForegroundColor Yellow

# Remove from git index (keeps local files)
try {
    git rm -r --cached frontend/.svelte-kit/ 2>$null
    Write-Host "Removed frontend/.svelte-kit/ from tracking" -ForegroundColor Green
} catch {
    Write-Host "frontend/.svelte-kit/ not currently tracked" -ForegroundColor Gray
}

try {
    git rm -r --cached tori_ui_svelte/.svelte-kit/ 2>$null
    Write-Host "Removed tori_ui_svelte/.svelte-kit/ from tracking" -ForegroundColor Green
} catch {
    Write-Host "tori_ui_svelte/.svelte-kit/ not currently tracked" -ForegroundColor Gray
}

try {
    git rm -r --cached standalone-holo/.svelte-kit/ 2>$null
    Write-Host "Removed standalone-holo/.svelte-kit/ from tracking" -ForegroundColor Green
} catch {
    Write-Host "standalone-holo/.svelte-kit/ not currently tracked" -ForegroundColor Gray
}

Write-Host "`nUpdating .gitignore files..." -ForegroundColor Yellow

# Check if frontend/.gitignore exists and update it
$frontendGitignore = "frontend\.gitignore"
if (-not (Test-Path $frontendGitignore)) {
    Write-Host "Creating frontend/.gitignore..." -ForegroundColor Yellow
    @"
# SvelteKit build output
.svelte-kit/

# Node modules
node_modules/

# Build output
build/
dist/

# Environment variables
.env
.env.local
.env.*.local

# Logs
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*

# OS files
.DS_Store
Thumbs.db

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
coverage/
.nyc_output

# Temporary files
*.tmp
*.temp
"@ | Out-File -FilePath $frontendGitignore -Encoding UTF8
}

# Stage the .gitignore files
git add .gitignore
git add frontend/.gitignore

Write-Host "`nCommitting changes..." -ForegroundColor Yellow
git commit -m "Stop tracking .svelte-kit generated files and update .gitignore"

Write-Host "`nDone! The .svelte-kit directories are no longer tracked." -ForegroundColor Green
Write-Host "They will remain on your local filesystem but won't be pushed to the repository." -ForegroundColor Cyan
Write-Host "`nYou can now push these changes with: git push" -ForegroundColor Cyan
