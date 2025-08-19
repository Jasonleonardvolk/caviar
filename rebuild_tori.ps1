# TORI System: Rebuild Script
# This script resets your virtualenv, reinstalls from poetry, and checks TOML

Write-Host "🛠️  [TORI] Begin full rebuild & environment reset." -ForegroundColor Cyan

# 1. Clean up previous builds & trash existing venv
if (Test-Path .\.venv) {
    Write-Host "🚮 Moving old .venv to _TRASH_2025" -ForegroundColor Yellow
    Move-Item .\.venv .\_TRASH_2025\.venv -Force
} else {
    Write-Host "✅ No existing .venv found." -ForegroundColor Green
}

# 2. Remove old build artifacts (if any)
$buildFolders = @("build", "dist", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints")
foreach ($f in $buildFolders) {
    if (Test-Path $f) {
        Write-Host "🧹 Moving $f to _TRASH_2025" -ForegroundColor Yellow
        Move-Item $f .\_TRASH_2025\$f -Force
    }
}

# 3. Recreate virtualenv via poetry
Write-Host "⏳ Creating new poetry environment..." -ForegroundColor Cyan
poetry env use python

Write-Host "📦 Installing dependencies from poetry.lock/pyproject.toml..." -ForegroundColor Cyan
poetry install

# 4. Post-install: Confirm all key dependencies
Write-Host "🔍 Checking poetry dependency status:" -ForegroundColor Cyan
poetry show --tree

# 5. Final advice to user
Write-Host ""
Write-Host "====================================================="
Write-Host "    If you see any errors, copy them here and I'll fix the TOML for you."
Write-Host "=====================================================" -ForegroundColor Cyan

Write-Host "All done! Activate with 't' or 'poetry shell' and launch as usual." -ForegroundColor Green
