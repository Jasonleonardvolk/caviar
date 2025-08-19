@echo off
REM Windows version of quick sprint prep
REM Run this for 30-minute Albert sprint preparation

echo ======================================
echo    30-MINUTE SPRINT PREP - WINDOWS
echo ======================================
echo.

REM Step 1: Clean repository (10 min)
echo [Step 1/6] Cleaning repository...
echo.
call cleanup_repository.bat
if errorlevel 1 (
    echo Warning: Some cleanup commands failed, but continuing...
)

REM Step 2: Create release tag
echo.
echo [Step 2/6] Creating release tag v0.11.0-hotfix...
git tag -a v0.11.0-hotfix -m "Hotfix: Soliton API 500 errors resolved" -m "" -m "- Fixed import guards and error handling" -m "- Added rate limiting to frontend" -m "- Created comprehensive test suite" -m "- Added CI/CD for concept_mesh" -m "- Cleaned repository"
echo Tag created: v0.11.0-hotfix

REM Step 3: Optimize CI configuration
echo.
echo [Step 3/6] Optimizing CI for faster PR builds...
if exist ".github\workflows\build-concept-mesh-optimized.yml" (
    move ".github\workflows\build-concept-mesh.yml" ".github\workflows\build-concept-mesh-full.yml" >nul 2>&1
    move ".github\workflows\build-concept-mesh-optimized.yml" ".github\workflows\build-concept-mesh.yml" >nul 2>&1
    echo CI configuration optimized!
) else (
    echo CI optimization file not found, skipping...
)

REM Step 4: Reminder about README
echo.
echo [Step 4/6] README Badge Update
echo ========================================
echo MANUAL ACTION REQUIRED:
echo 1. Open README.md
echo 2. Find: USERNAME/REPO
echo 3. Replace with your actual GitHub username and repo name
echo Example: yourname/tori-kha
echo ========================================
pause

REM Step 5: Stage all changes
echo.
echo [Step 5/6] Staging all changes...
git add .
git add .github/workflows/build-concept-mesh.yml 2>nul
git add .gitignore
git add tests/test_pipeline_async.py 2>nul
git add tests/test_soliton_api.py 2>nul

REM Step 6: Commit
echo.
echo [Step 6/6] Creating commit...
git commit -m "chore: Final polish - CI optimization and cleanup for v0.11.0-hotfix" -m "" -m "- Optimized CI for faster PR builds" -m "- Added comprehensive .gitignore" -m "- Cleaned repository of temporary files" -m "- Ready for Albert sprint"

echo.
echo ======================================
echo    PREPARATION COMPLETE!
echo ======================================
echo.
echo Next steps:
echo.
echo 1. If you used git filter-repo to clean history:
echo    git push --force origin main
echo.
echo 2. Otherwise:
echo    git push origin main
echo.
echo 3. Push the tag:
echo    git push origin v0.11.0-hotfix
echo.
echo 4. Check GitHub Actions to see CI running
echo.
echo Ready for Albert sprint!
echo.
pause
