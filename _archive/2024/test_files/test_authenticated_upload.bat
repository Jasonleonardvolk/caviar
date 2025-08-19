@echo off
REM TORI Authentication & Upload Test Script for Windows
REM Usage: test_upload.bat [pdf_file_path]

echo 🚀 TORI Authentication ^& Upload Test
echo =====================================

REM Configuration
set TORI_HOST=localhost:8443
set PDF_FILE=%1
if "%PDF_FILE%"=="" set PDF_FILE=2407.15527v2.pdf

REM Check if PDF file exists
if not exist "%PDF_FILE%" (
    echo ❌ Error: PDF file '%PDF_FILE%' not found
    echo Usage: %0 [pdf_file_path]
    exit /b 1
)

echo 📄 PDF File: %PDF_FILE%
echo 🌐 TORI Host: %TORI_HOST%

REM Step 1: Login and get token
echo.
echo 🔐 Step 1: Authenticating with TORI...

curl.exe -s -X POST "http://%TORI_HOST%/api/auth/login" ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"operator\",\"password\":\"operator123\"}" ^
  -o login_response.json

if %errorlevel% neq 0 (
    echo ❌ Error: Failed to connect to TORI server
    echo Make sure TORI is running on http://%TORI_HOST%
    exit /b 1
)

REM Extract token (simplified for Windows batch)
for /f "tokens=4 delims=:" %%a in ('findstr "token" login_response.json') do (
    set TOKEN_RAW=%%a
)

REM Clean up token (remove quotes and comma)
set TOKEN=%TOKEN_RAW:"=%
set TOKEN=%TOKEN:,=%

if "%TOKEN%"=="" (
    echo ❌ Error: Failed to get authentication token
    type login_response.json
    del login_response.json
    exit /b 1
)

echo ✅ Authentication successful!
echo 🎫 Token obtained

REM Step 2: Upload PDF
echo.
echo 📤 Step 2: Uploading PDF...

curl.exe -X POST "http://%TORI_HOST%/api/upload" ^
  -H "Authorization: Bearer %TOKEN%" ^
  -F "file=@%PDF_FILE%;type=application/pdf" ^
  -o upload_response.json

if %errorlevel% neq 0 (
    echo ❌ Error: Upload failed
    del login_response.json
    exit /b 1
)

echo 📋 Upload Response:
type upload_response.json

REM Check if upload was successful
findstr "success.*true" upload_response.json >nul
if %errorlevel% equ 0 (
    echo.
    echo ✅ Upload successful!
    echo 🎆 Check upload_response.json for details
) else (
    echo.
    echo ❌ Upload may have failed. Check upload_response.json for details.
)

REM Cleanup
del login_response.json
echo.
echo 🎯 Test completed! Results saved in upload_response.json
