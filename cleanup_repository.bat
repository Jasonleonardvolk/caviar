@echo off
REM Git cleanup script to fix repository hygiene issues
REM Run this from the repository root

echo Starting repository cleanup...
echo.

REM 1. Remove cached files that should be ignored
echo Removing cached files that shouldn't be in git...

REM Remove Python bytecode and cache
git rm -r --cached **/__pycache__/ 2>nul
git rm -r --cached **/*.pyc 2>nul
git rm -r --cached **/*.pyo 2>nul
git rm -r --cached **/*.pyd 2>nul

REM Remove egg-info directories
git rm -r --cached **/*.egg-info/ 2>nul
git rm -r --cached alan_core.egg-info/ 2>nul
git rm -r --cached concept_mesh/concept_mesh.egg-info/ 2>nul

REM Remove logs
git rm -r --cached logs/ 2>nul
git rm --cached **/*.log 2>nul

REM Remove data files that shouldn't be in git
git rm --cached data/memory_vault/vault_live.jsonl 2>nul
git rm --cached **/*.pkl 2>nul
git rm --cached **/*.pkl.gz 2>nul

REM Remove temporary and backup files
git rm --cached **/*.bak 2>nul
git rm --cached **/*.backup_* 2>nul
git rm --cached **/*.OLD_DUPLICATE 2>nul

REM Remove duplicate fix scripts
git rm -r --cached fixes/soliton_500_fixes/ 2>nul
git rm --cached GREMLIN_*.ps1 2>nul
git rm --cached fix_soliton_*.ps1 2>nul
git rm --cached test_soliton_*.ps1 2>nul

REM Remove compiled Rust artifacts
git rm -r --cached concept_mesh/target/ 2>nul

REM 2. Check for .gitignore
if not exist .gitignore (
    echo Creating .gitignore...
    copy .gitignore.template .gitignore
    git add .gitignore
)

REM 3. Commit the cleanup
echo.
echo Committing cleanup...
git commit -m "chore: Clean up repository - remove logs, bytecode, and temporary files" -m "- Remove Python cache and bytecode files" -m "- Remove logs and session data" -m "- Remove egg-info directories" -m "- Remove duplicate fix scripts" -m "- Add comprehensive .gitignore" -m "- Keep only production code" -m "" -m "This addresses repository hygiene issues identified in audit."

echo.
echo Cleanup complete!
echo.
echo IMPORTANT: For existing history with sensitive data:
echo    If data/memory_vault/vault_live.jsonl contained sensitive info,
echo    you may need to use git filter-repo to remove it from history:
echo.
echo    pip install git-filter-repo
echo    git filter-repo --path data/memory_vault/vault_live.jsonl --invert-paths
echo.
echo    Then force-push: git push --force origin main
echo.
pause
