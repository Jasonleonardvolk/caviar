@echo off
REM Quick safety check before running full cleanup

echo ========================================
echo    SHADER CLEANUP SAFETY CHECK
echo ========================================
echo.

echo [1] Checking current distribution...
powershell -Command "& { $stats = @{}; 'frontend\lib\webgpu\shaders','frontend\shaders','frontend\hybrid','frontend\public\hybrid\wgsl' | ForEach-Object { $count = @(Get-ChildItem -Path $_ -Filter '*.wgsl' -ErrorAction SilentlyContinue).Count; Write-Host ('{0,-35} {1,3} files' -f $_, $count) -ForegroundColor $(if($_ -like '*lib\webgpu*'){'Green'}elseif($count -eq 0){'DarkGray'}else{'Yellow'}) } }"

echo.
echo [2] Checking for conflicts...
node tools\shaders\check_shader_dates.mjs 2>nul

echo.
echo ========================================
echo    READY TO CLEAN?
echo ========================================
echo.
echo If CANONICAL dates are newer (or equal), it's SAFE to run:
echo   cleanup_shaders_full.bat
echo.
echo If LEGACY dates are newer, MANUALLY REVIEW first!
echo.
pause
