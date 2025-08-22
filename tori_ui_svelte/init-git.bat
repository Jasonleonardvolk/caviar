@echo off
REM init-git.bat - Quick Git initialization for iRis/TORI project

echo ========================================
echo Git Repository Setup for iRis/TORI
echo ========================================
echo.

REM Check if Git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in PATH
    pause
    exit /b 1
)

echo [INFO] Git found
echo.

REM Initialize repository if needed
if exist .git (
    echo [WARN] Repository already initialized
) else (
    echo [INFO] Initializing Git repository...
    git init
    echo [OK] Repository initialized
)

REM Set default branch to main
echo [INFO] Setting default branch to 'main'...
git config init.defaultBranch main
git branch -M main
echo [OK] Default branch set to 'main'

REM Configure line endings
echo [INFO] Configuring line ending handling...
git config core.autocrlf true
git config core.longpaths true
git config core.ignorecase false
echo [OK] Configuration complete

REM Stage configuration files
echo [INFO] Staging configuration files...
git add .gitignore .gitattributes GIT_CONVENTIONS.md
echo [OK] Configuration files staged

REM Normalize line endings
echo [INFO] Normalizing line endings...
git add --renormalize .
echo [OK] Line endings normalized

echo.
echo ========================================
echo Git Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Review staged changes: git status
echo   2. Create initial commit: git commit -m "chore(repo): initial Git setup"
echo   3. Add remote: git remote add origin [your-github-url]
echo   4. Push: git push -u origin main --follow-tags
echo.
echo For advanced workflow, use:
echo   PowerShell: .\Initialize-GitRepo.ps1
echo   Workflow helper: .\Git-Workflow.ps1 help
echo.
pause
