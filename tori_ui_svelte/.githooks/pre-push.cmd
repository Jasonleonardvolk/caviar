@echo off
REM iRis pre-push gate for Windows
setlocal enabledelayedexpansion

echo =======================================
echo      iRis Pre-Push Verification
echo =======================================
echo.

REM Get repository root
for /f "delims=" %%a in ('git rev-parse --show-toplevel 2^>nul') do set REPO_ROOT=%%a
if "%REPO_ROOT%"=="" (
    echo ERROR: Not in a Git repository
    exit /b 1
)

REM Convert forward slashes to backslashes for Windows
set REPO_ROOT=%REPO_ROOT:/=\%

REM Check for PowerShell Core vs Windows PowerShell
where pwsh >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PS=pwsh
) else (
    set PS=powershell
)

REM Set verify script path
set VERIFY_PS=%REPO_ROOT%\tools\release\verify-setup.ps1

REM Check for uncommitted changes
for /f "delims=" %%i in ('git status --porcelain 2^>nul') do (
    if not "%%i"=="" (
        echo WARNING: Uncommitted changes detected
        echo.
    )
)

REM Check if verify script exists
if exist "%VERIFY_PS%" (
    echo Running full verification script...
    echo.
    %PS% -NoProfile -ExecutionPolicy Bypass -File "%VERIFY_PS%"
    set EXIT_CODE=!ERRORLEVEL!
) else (
    echo WARN: %VERIFY_PS% not found.
    echo Running minimal endpoint checks...
    echo.
    
    REM Minimal checks inline
    %PS% -NoProfile -ExecutionPolicy Bypass -Command ^
        "$urls = @('http://localhost:3000/', 'http://localhost:3000/renderer', 'http://localhost:3000/upload', 'http://localhost:3000/api/list', 'http://localhost:3000/api/pdf/stats', 'http://localhost:3000/api/memory/state'); " ^
        "$failed = $false; " ^
        "foreach ($u in $urls) { " ^
        "  try { " ^
        "    $r = Invoke-WebRequest -Uri $u -UseBasicParsing -Method GET -TimeoutSec 5; " ^
        "    if ($r.StatusCode -eq 200) { " ^
        "      Write-Host \"  OK: $u\" -ForegroundColor Green " ^
        "    } else { " ^
        "      Write-Host \"  FAIL: $u\" -ForegroundColor Red; " ^
        "      $failed = $true " ^
        "    } " ^
        "  } catch { " ^
        "    if ($_.Exception.Message -like '*Unable to connect*') { " ^
        "      Write-Host \"  SKIP: $u (server not running)\" -ForegroundColor Yellow " ^
        "    } else { " ^
        "      Write-Host \"  FAIL: $u\" -ForegroundColor Red; " ^
        "      $failed = $true " ^
        "    } " ^
        "  } " ^
        "}; " ^
        "if ($failed) { exit 1 } else { exit 0 }"
    
    set EXIT_CODE=!ERRORLEVEL!
)

if !EXIT_CODE! equ 0 (
    echo.
    echo =======================================
    echo    Pre-Push Verification PASSED
    echo =======================================
) else (
    echo.
    echo =======================================
    echo    Pre-Push Verification FAILED
    echo =======================================
    echo Fix the issues above before pushing.
)

exit /b !EXIT_CODE!