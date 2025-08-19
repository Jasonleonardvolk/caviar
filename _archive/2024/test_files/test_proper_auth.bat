@echo off
REM TORI Complete Authentication Test Script for Windows
REM This script properly handles Bearer token authentication

echo 🔐 TORI Authentication ^& Upload Test
echo ====================================

REM Configuration
set TORI_HOST=localhost:8443
set PDF_FILE=%1
if "%PDF_FILE%"=="" set PDF_FILE=test_document.pdf

REM Check if we can reach TORI
echo 🌐 Testing connection to TORI...
curl.exe -s "http://%TORI_HOST%/health" > nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Cannot reach TORI server at http://%TORI_HOST%
    echo    Make sure TORI is running with:
    echo    python phase3_complete_production_system.py --host 0.0.0.0 --port 8443
    exit /b 1
)
echo ✅ TORI server is responding

REM Step 1: Authenticate and get token
echo.
echo 🔐 Step 1: Getting authentication token...

curl.exe -s -X POST "http://%TORI_HOST%/api/auth/login" ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"operator\",\"password\":\"operator123\"}" ^
  -o login_response.json

if %errorlevel% neq 0 (
    echo ❌ Error: Login request failed
    exit /b 1
)

echo 📋 Login Response:
type login_response.json

REM Extract token (Windows batch method)
for /f "tokens=2 delims=:" %%a in ('findstr "token" login_response.json') do (
    set TOKEN_PART=%%a
)

REM Clean token (remove quotes, spaces, and commas)
set TOKEN=%TOKEN_PART:"=%
set TOKEN=%TOKEN: =%
set TOKEN=%TOKEN:,=%

if "%TOKEN%"=="" (
    echo.
    echo ❌ Failed to extract token from response
    echo Please check if TORI is running and credentials are correct
    del login_response.json
    exit /b 1
)

echo.
echo ✅ Token extracted successfully

REM Check if PDF file exists
if not exist "%PDF_FILE%" (
    echo.
    echo 📄 PDF file '%PDF_FILE%' not found - creating a test file...
    echo This is a test PDF content for TORI authentication testing > "%PDF_FILE%"
    echo ✅ Created test file: %PDF_FILE%
)

REM Step 2: Upload with proper Authorization header
echo.
echo 📤 Step 2: Uploading PDF with Bearer token...

curl.exe -s -X POST "http://%TORI_HOST%/api/upload" ^
  -H "Authorization: Bearer %TOKEN%" ^
  -F "file=@%PDF_FILE%;type=application/pdf" ^
  -o upload_response.json

if %errorlevel% neq 0 (
    echo ❌ Error: Upload request failed
    del login_response.json
    exit /b 1
)

echo.
echo 📋 Upload Response:
type upload_response.json

REM Check response for success
findstr /i "success.*true" upload_response.json > nul
if %errorlevel% equ 0 (
    echo.
    echo 🎆 SUCCESS! PDF uploaded successfully!
    goto success
)

findstr /i "uploaded.*successfully" upload_response.json > nul
if %errorlevel% equ 0 (
    echo.
    echo 🎆 SUCCESS! PDF uploaded successfully!
    goto success
)

findstr /i "403\|Forbidden\|Not authenticated" upload_response.json > nul
if %errorlevel% equ 0 (
    echo.
    echo ❌ 403 Authentication Error - Token may be invalid
    echo    Check that the login was successful and token was extracted correctly
    goto cleanup
)

findstr /i "401\|Unauthorized" upload_response.json > nul
if %errorlevel% equ 0 (
    echo.
    echo ❌ 401 Authorization Error - Check credentials or token expiration
    goto cleanup
)

echo.
echo ⚠️ Unexpected response - check upload_response.json

:success
echo.
echo 🎯 Authentication test completed successfully!
goto show_manual

:cleanup
echo.
echo 🎯 Authentication test completed with errors

:show_manual
echo.
echo 💡 Manual commands for testing:
echo    1. Login: curl.exe -X POST "http://%TORI_HOST%/api/auth/login" -H "Content-Type: application/json" -d "{\"username\":\"operator\",\"password\":\"operator123\"}"
echo    2. Upload: curl.exe -X POST "http://%TORI_HOST%/api/upload" -H "Authorization: Bearer %%TOKEN%%" -F "file=@%PDF_FILE%"
echo.
echo 📋 Available user roles:
echo    - observer  / observer123  (read-only)
echo    - operator  / operator123  (can upload)
echo    - approver  / approver123  (can approve)
echo    - admin     / admin123     (full access)

REM Cleanup
del login_response.json 2>nul
echo.
echo 📁 Response saved in upload_response.json
